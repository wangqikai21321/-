"""
R-CNN 核心模型模块（与 Girshick et al. 2014 论文一致）
=====================================================
- 使用 AlexNet 作为骨干网络（与原始论文相同）
- ImageNet 预训练 → 领域微调
- FC7 输出 4096 维特征
- 输入尺寸: 227×227
- 每类一个线性 SVM + 边界框回归器
"""

import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from config import (
    NUM_CLASSES, IMG_SIZE, FEATURE_DIM, DEVICE,
    CNN_EPOCHS, CNN_LR, CNN_MOMENTUM, CNN_WEIGHT_DECAY, BATCH_SIZE,
    SVM_C, SVM_HARD_NEG_ITER,
    BBOX_IOU_MIN, CNN_MODEL_PATH, SVM_MODEL_PATH, BBOX_REG_PATH,
)


class AlexNetFeatureExtractor(nn.Module):
    """
    AlexNet 特征提取器（与 R-CNN 论文一致）
    架构: Conv1→Pool1→Conv2→Pool2→Conv3→Conv4→Conv5→Pool5→FC6→FC7
    输出: 4096 维特征（FC7 层）
    """

    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision.models as models
        alexnet = models.alexnet(weights="IMAGENET1K_V1" if pretrained else None)

        self.features = alexnet.features   # Conv1 ~ Pool5
        self.avgpool = alexnet.avgpool      # AdaptiveAvgPool2d(6,6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # FC6
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),          # FC7
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: (B, 3, 227, 227)
        Returns: (B, 4096)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)  # 输出 4096 维（FC7）
        return x


class RCNNClassifier(nn.Module):
    """
    R-CNN 分类器（CNN 微调阶段使用）
    AlexNet 特征提取器 + FC8 分类头（N+1 类，含背景）

    架构（与论文一致）:
      Input: 227×227×3
      → Conv1~Pool5
      → FC6 (4096)
      → FC7 (4096)
      → FC8 (N+1) 替换原 ImageNet 的 1000 类
    """

    def __init__(self, num_classes=NUM_CLASSES + 1):
        super().__init__()
        self.extractor = AlexNetFeatureExtractor(pretrained=True)
        self.fc8 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """
        x: (B, 3, 227, 227)
        Returns: (B, N+1) 分类 logits
        """
        feat = self.extractor(x)
        return self.fc8(feat)


def preprocess_crop(crop_img):
    """
    预处理裁剪区域，与论文一致:
    1. 转为 3 通道 RGB（灰度图复制 3 次）
    2. Warp 到 227×227（各向异性缩放）
    3. 减去 ImageNet 均值 [103.939, 116.779, 123.68]（BGR 顺序的 OpenCV 均值）
    4. 返回 (3, 227, 227) tensor
    """
    # 如果是灰度图，转为 3 通道
    if len(crop_img.shape) == 2:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)

    # 各向异性缩放（论文方法：无视宽高比直接 warp）
    crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

    # 减去 ImageNet 均值（OpenCV 读取的是 BGR）
    crop_img[:, :, 0] -= 103.939  # B
    crop_img[:, :, 1] -= 116.779  # G
    crop_img[:, :, 2] -= 123.68   # R

    # (H, W, C) → (C, H, W)
    crop_tensor = torch.from_numpy(crop_img).permute(2, 0, 1)

    return crop_tensor


def train_cnn(train_data):
    """阶段 1: CNN 微调（对应论文 Section 2.3）"""
    from dataset import RCNNDataset

    print("\n" + "=" * 60)
    print("阶段 1: CNN 微调（AlexNet, 227x227, 4096-dim FC7）")
    print("=" * 60)

    dataset = RCNNDataset(train_data, positive_only=False, max_per_image=64)
    if len(dataset) < 10:
        print("[CNN] 样本太少，跳过训练")
        return RCNNClassifier().to(DEVICE)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = RCNNClassifier(num_classes=NUM_CLASSES + 1).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=CNN_LR,
        momentum=CNN_MOMENTUM,
        weight_decay=CNN_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    labels_count = {}
    for _, lbl in dataset:
        l = lbl.item() if isinstance(lbl, torch.Tensor) else lbl
        labels_count[l] = labels_count.get(l, 0) + 1
    print(f"[CNN] 样本分布（0=背景, 1-6=缺陷类）: {dict(sorted(labels_count.items()))}")
    print(f"[CNN] 训练 {len(dataset)} 样本, {CNN_EPOCHS} epoch...")
    print(f"[CNN] 与论文一致：SGD lr={CNN_LR}, momentum={CNN_MOMENTUM}, batch={BATCH_SIZE}")

    for epoch in range(CNN_EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, preds = output.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        acc = correct / total * 100 if total > 0 else 0
        print(f"  Epoch {epoch+1:2d}/{CNN_EPOCHS} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.2f}%")

    torch.save(model.state_dict(), CNN_MODEL_PATH)
    print(f"[CNN] 模型已保存: {CNN_MODEL_PATH}")
    return model


def get_feature_extractor():
    """加载微调后的 AlexNet 特征提取器（FC7 输出 4096 维）"""
    extractor = AlexNetFeatureExtractor(pretrained=True).to(DEVICE)
    extractor.eval()
    if CNN_MODEL_PATH.exists():
        print(f"[特征] 加载微调模型: {CNN_MODEL_PATH}")
        state_dict = torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True)
        extractor_state = {
            k.replace("extractor.", ""): v
            for k, v in state_dict.items()
            if k.startswith("extractor.")
        }
        if extractor_state:
            extractor.load_state_dict(extractor_state, strict=False)
    return extractor


def train_svm_classifiers(train_data, feature_extractor):
    """
    阶段 2: SVM 分类器训练（对应论文 Section 2.4）
    每个类别训练一个二分类线性 SVM
    正样本 = GT 框, 负样本 = IoU < 0.3 的候选框
    硬负样本挖掘（论文标准做法）
    """
    from dataset import SamplesForSVM

    print("\n" + "=" * 60)
    print("阶段 2: SVM 分类器训练（每类一个 LinearSVC）")
    print("=" * 60)

    samples = SamplesForSVM(train_data, feature_extractor)
    X = np.array([s["feature"] for s in samples.data])
    y = np.array([s["label"] for s in samples.data])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svms = {}
    print(f"[SVM] 训练 {NUM_CLASSES} 个二分类 SVM（论文使用 L2-regularized L2-loss SVC）...")
    for cls_id in range(1, NUM_CLASSES + 1):
        pos_mask = (y == cls_id)
        neg_mask = (y == 0)
        X_pos = X_scaled[pos_mask]
        X_neg_all = X_scaled[neg_mask]

        if len(X_pos) == 0:
            print(f"  [{cls_id}] 无正样本，跳过")
            svms[cls_id] = None
            continue

        svm = None
        n_neg_use = min(len(X_neg_all), max(len(X_pos) * 3, 50))
        X_neg = X_neg_all.copy()

        for iteration in range(SVM_HARD_NEG_ITER):
            if n_neg_use == 0:
                break
            neg_idxs = np.random.choice(len(X_neg), min(n_neg_use, len(X_neg)), replace=False)
            X_train = np.vstack([X_pos, X_neg[neg_idxs]])
            y_train = np.hstack([np.ones(len(X_pos)), np.zeros(len(neg_idxs))])

            svm = LinearSVC(C=SVM_C, max_iter=2000, dual="auto", random_state=42, loss="squared_hinge")
            svm.fit(X_train, y_train)

            # 硬负样本挖掘: 用当前 SVM 找出最难区分的负样本
            if iteration < SVM_HARD_NEG_ITER - 1 and len(X_neg) > n_neg_use:
                scores = svm.decision_function(X_neg)
                hard_idxs = np.argsort(scores)[-n_neg_use:]
                X_neg = X_neg[hard_idxs]

        svms[cls_id] = svm
        print(f"  SVM [{cls_id}]: pos={len(X_pos)}, neg={n_neg_use}")

    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump({"svms": svms, "scaler": scaler}, f)
    print(f"[SVM] 已保存: {SVM_MODEL_PATH}")
    return svms, scaler


def train_bbox_regressors(train_data, feature_extractor):
    """
    阶段 3: 边界框回归器训练（对应论文 Appendix C）
    每类一个岭回归器，学习 4 个偏移量 (t_x, t_y, t_w, t_h)

    论文中的回归目标（对数空间变换）:
      t_x = (G_x - P_x) / P_w
      t_y = (G_y - P_y) / P_h
      t_w = log(G_w / P_w)
      t_h = log(G_h / P_h)

    其中 P = 候选框, G = 真实框（均为中心点+宽高表示）
    """
    from dataset import compute_iou

    print("\n" + "=" * 60)
    print("阶段 3: 边界框回归器训练（Ridge Regression）")
    print("=" * 60)

    reg_samples = {i: [] for i in range(1, NUM_CLASSES + 1)}

    for item in train_data:
        img = cv2.imread(item["img_path"])
        if img is None:
            continue
        for (px, py, pw, ph) in item["proposals"]:
            p_box = (int(px), int(py), int(px + pw), int(py + ph))
            for (gx, gy, gxmax, gymax, cls_id) in item["gt_boxes"]:
                gt_box = (gx, gy, gxmax, gymax)
                if compute_iou(p_box, gt_box) < BBOX_IOU_MIN:
                    continue
                crop = img[p_box[1]:p_box[3], p_box[0]:p_box[2]]
                if crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue
                crop_t = preprocess_crop(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat = feature_extractor(crop_t).cpu().numpy()[0]

                # 转换为（中心点 + 宽高）形式
                px_c = px + pw / 2.0
                py_c = py + ph / 2.0
                pf_w, pf_h = float(pw), float(ph)
                gx_c = (gx + gxmax) / 2.0
                gy_c = (gy + gymax) / 2.0
                gw = float(gxmax - gx)
                gh = float(gymax - gy)
                if pf_w < 1 or pf_h < 1:
                    continue
                tx = (gx_c - px_c) / pf_w
                ty = (gy_c - py_c) / pf_h
                tw = np.log(gw / pf_w) if gw > 0 else 0.0
                th = np.log(gh / pf_h) if gh > 0 else 0.0
                reg_samples[cls_id + 1].append((feat, np.array([tx, ty, tw, th])))

    regressors = {}
    for cls_id in range(1, NUM_CLASSES + 1):
        n = len(reg_samples[cls_id])
        if n < 4:
            print(f"  [{cls_id}] 样本不足 ({n}), 跳过")
            regressors[cls_id] = None
            continue
        X_r = np.array([s[0] for s in reg_samples[cls_id]])
        t_r = np.array([s[1] for s in reg_samples[cls_id]])
        reg = Ridge(alpha=BBOX_IOU_MIN * 1000)
        reg.fit(X_r, t_r)
        regressors[cls_id] = reg
        mse = np.mean((reg.predict(X_r) - t_r) ** 2)
        print(f"  [{cls_id}]: {n} 样本, MSE={mse:.6f}")

    with open(BBOX_REG_PATH, "wb") as f:
        pickle.dump(regressors, f)
    print(f"[回归] 已保存: {BBOX_REG_PATH}")
    return regressors

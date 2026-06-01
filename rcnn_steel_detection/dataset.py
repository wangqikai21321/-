"""
R-CNN 数据集模块
===============
- 解析 VOC XML 标注文件
- IoU 计算
- Selective Search 候选框生成
- R-CNN 训练样本生成（正样本 IoU≥0.5, 负样本 IoU<0.5）
- SVM 训练样本准备（正样本=GT框, 负样本=IoU<0.3 候选框）
"""

import random
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    DATASET_PATH, CLASSES, CLASS_TO_IDX, NUM_CLASSES,
    IMG_SIZE, IOU_POSITIVE, IOU_NEGATIVE, TRAIN_RATIO,
    SS_MODE, SS_TOP_N, DEVICE, FEATURES_DIR,
)


def preprocess_crop(crop_img):
    """
    预处理裁剪区域（与 R-CNN 论文一致）:
    1. 转 3 通道 BGR（灰度图复制 3 次，保持 ImageNet 预训练权重兼容）
    2. 各向异性 Warp 到 227×227
    3. 减去 ImageNet BGR 均值 [103.939, 116.779, 123.68]
    返回: (3, 227, 227) float32 tensor
    """
    if len(crop_img.shape) == 2:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
    crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    crop_img[:, :, 0] -= 103.939  # B
    crop_img[:, :, 1] -= 116.779  # G
    crop_img[:, :, 2] -= 123.68   # R
    return torch.from_numpy(crop_img).permute(2, 0, 1)


def parse_voc_xml(xml_path):
    """解析 Pascal VOC XML，返回 [(xmin, ymin, xmax, ymax, class_id), ...]"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_TO_IDX:
            continue
        cls_id = CLASS_TO_IDX[name]
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        boxes.append((xmin, ymin, xmax, ymax, cls_id))
    return boxes


def compute_iou(boxA, boxB):
    """计算 IoU: boxA/boxB = (xmin, ymin, xmax, ymax)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def selective_search_proposals(img, mode="fast", top_n=2000):
    """Selective Search 候选框生成, 返回 [(x,y,w,h), ...]"""
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    return ss.process()[:top_n]


def load_dataset_paths():
    """扫描数据集，返回 [(img_path, xml_path), ...]"""
    images_dir = DATASET_PATH / "images"
    annotations_dir = DATASET_PATH / "annotations"
    pairs = []
    for cls_dir in images_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for img_file in cls_dir.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".bmp", ".png"]:
                continue
            xml_path = annotations_dir / (img_file.stem + ".xml")
            if xml_path.exists():
                pairs.append((str(img_file), str(xml_path)))
    return pairs


def generate_proposal_cache():
    """为所有图片预计算 Selective Search 候选框并缓存"""
    cache_path = FEATURES_DIR / "proposals_cache.pkl"
    if cache_path.exists():
        print(f"[数据] 加载候选框缓存: {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["train"], data["val"]

    print("[数据] 生成 Selective Search 候选框...")
    all_pairs = load_dataset_paths()
    print(f"[数据] 共 {len(all_pairs)} 个图像-标注对")

    # 按类别分组后划分
    class_pairs = {c: [] for c in CLASSES}
    for img_path, xml_path in all_pairs:
        boxes = parse_voc_xml(xml_path)
        if boxes:
            class_pairs[CLASSES[boxes[0][4]]].append((img_path, xml_path))

    train_raw, val_raw = [], []
    random.seed(42)
    for cls_name in CLASSES:
        items = class_pairs[cls_name]
        random.shuffle(items)
        split = int(len(items) * TRAIN_RATIO)
        train_raw.extend(items[:split])
        val_raw.extend(items[split:])

    print(f"[数据] 训练: {len(train_raw)}, 验证: {len(val_raw)}")

    def process_pair(img_path, xml_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        proposals = selective_search_proposals(img, mode=SS_MODE, top_n=SS_TOP_N)
        return {
            "img_path": img_path,
            "proposals": [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in proposals],
            "gt_boxes": parse_voc_xml(xml_path),
        }

    train_data, val_data = [], []
    for i, (imp, xp) in enumerate(train_raw):
        r = process_pair(imp, xp)
        if r:
            train_data.append(r)
        if (i + 1) % 100 == 0:
            print(f"  处理训练: {i+1}/{len(train_raw)}")
    for i, (imp, xp) in enumerate(val_raw):
        r = process_pair(imp, xp)
        if r:
            val_data.append(r)
        if (i + 1) % 100 == 0:
            print(f"  处理验证: {i+1}/{len(val_raw)}")

    print(f"[数据] 有效: 训练 {len(train_data)}, 验证 {len(val_data)}")
    with open(cache_path, "wb") as f:
        pickle.dump({"train": train_data, "val": val_data}, f)
    print(f"[数据] 缓存已保存: {cache_path}")
    return train_data, val_data


class RCNNDataset(Dataset):
    """
    R-CNN 训练用数据集（阶段 1 CNN 微调）
    每个样本 = 一个候选区域的 warped 图片 + 类别标签
    正样本: IoU ≥ 0.5, 标签 = 真实类别 (1~N)
    负样本: IoU < 0.5, 标签 = 0（背景）

    与论文一致:
      - Warp 到 227×227（各向异性）
      - 减去 ImageNet BGR 均值
      - 输入: (3, 227, 227)
    """

    def __init__(self, data_pairs, positive_only=False, max_per_image=64):
        self.samples = []

        for item in data_pairs:
            img_path = item["img_path"]
            proposals = item["proposals"]
            gt_boxes = item["gt_boxes"]
            gt_normalized = [(b[0], b[1], b[2], b[3], b[4]) for b in gt_boxes]

            pos, neg = [], []
            for (px, py, pw, ph) in proposals:
                p_box = (px, py, px + pw, py + ph)
                best_iou, best_cls = 0.0, 0
                for gt in gt_normalized:
                    iou = compute_iou(p_box, (gt[0], gt[1], gt[2], gt[3]))
                    if iou > best_iou:
                        best_iou = iou
                        best_cls = gt[4] + 1
                if best_iou >= IOU_POSITIVE:
                    pos.append((img_path, px, py, pw, ph, best_cls))
                elif not positive_only:
                    neg.append((img_path, px, py, pw, ph, 0))

            n_pos = min(len(pos), max_per_image // 4)
            n_neg = min(len(neg), max_per_image - n_pos)
            if pos:
                self.samples.extend(random.sample(pos, n_pos))
            if neg:
                self.samples.extend(random.sample(neg, n_neg))

        print(f"[数据集] {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, x, y, w, h, label = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.tensor(0, dtype=torch.long)

        h_img, w_img = img.shape[:2]
        x, y = max(0, int(x)), max(0, int(y))
        w, h = min(int(w), w_img - x), min(int(h), h_img - y)

        if w < 10 or h < 10:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.tensor(0, dtype=torch.long)

        crop = img[y:y+h, x:x+w]
        crop_tensor = preprocess_crop(crop)
        return crop_tensor, torch.tensor(label, dtype=torch.long)


class SamplesForSVM:
    """
    为 SVM 训练准备样本
    正样本: GT 框（IoU = 1.0）
    负样本: IoU < 0.3 的候选框

    使用 AlexNet FC7 提取 4096 维特征
    """

    def __init__(self, data_pairs, feature_extractor, device=DEVICE):
        self.data = []
        self.feature_extractor = feature_extractor
        self.device = device

        print("[SVM数据] 准备 SVM 训练样本（4096-dim FC7 特征）...")
        for idx, item in enumerate(data_pairs):
            img = cv2.imread(item["img_path"])
            if img is None:
                continue

            gt_boxes = item["gt_boxes"]
            proposals = item["proposals"]
            gt_formatted = [(b[0], b[1], b[2], b[3]) for b in gt_boxes]

            # 正样本: GT 框
            for (gx, gy, gxmax, gymax, cls_id) in gt_boxes:
                crop = img[gy:gymax, gx:gxmax]
                feat = self._extract_feat(crop)
                if feat is not None:
                    self.data.append({
                        "feature": feat,
                        "label": cls_id + 1,
                        "box": (gx, gy, gxmax, gymax),
                        "img_path": item["img_path"],
                        "gt_box": (gx, gy, gxmax, gymax, cls_id),
                    })

            # 负样本: IoU < 0.3
            neg_count = 0
            for (px, py, pw, ph) in proposals:
                if neg_count >= 20:
                    break
                p_box = (int(px), int(py), int(px + pw), int(py + ph))
                max_iou = max([compute_iou(p_box, gt) for gt in gt_formatted]) if gt_formatted else 0
                if max_iou < IOU_NEGATIVE:
                    crop = img[p_box[1]:p_box[3], p_box[0]:p_box[2]]
                    feat = self._extract_feat(crop)
                    if feat is not None:
                        self.data.append({
                            "feature": feat,
                            "label": 0,
                            "box": p_box,
                            "img_path": item["img_path"],
                            "gt_box": None,
                        })
                        neg_count += 1

            if (idx + 1) % 100 == 0:
                print(f"  SVM 样本: {idx+1}/{len(data_pairs)}")

        print(f"[SVM数据] 总样本: {len(self.data)}")

    def _extract_feat(self, crop):
        """提取 AlexNet FC7 特征（4096 维）"""
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        crop_t = preprocess_crop(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.feature_extractor(crop_t)
        return feat.cpu().numpy()[0]

"""R-CNN 环境快速测试（AlexNet + Selective Search）"""
import sys, os, time, cv2, numpy as np, torch
sys.path.insert(0, ".")

print("=" * 60)
print("R-CNN 环境测试（Girshick et al. 2014 原论文架构）")
print("=" * 60)

# 1. Selective Search
print("\n[1] Selective Search 候选框生成...")
img = cv2.imread(r"C:\Users\wang'q'k\.cache\kagglehub\datasets\kaustubhdikshit\neu-surface-defect-database\versions\1\NEU-DET\train\images\crazing\crazing_1.jpg")
t0 = time.time()
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()
rects = ss.process()
print(f"    ✓ 耗时: {time.time()-t0:.2f}s, 候选框: {len(rects)}")

# 2. AlexNet 特征提取器
print("\n[2] AlexNet 特征提取器 (FC7 → 4096-dim)...")
from model import AlexNetFeatureExtractor, RCNNClassifier, preprocess_crop
extractor = AlexNetFeatureExtractor(pretrained=True)
extractor.eval()
x = torch.randn(1, 3, 227, 227)
feat = extractor(x)
print(f"    ✓ 输入: (1, 3, 227, 227) → 输出: {feat.shape} (FC7)")

# 3. 预处理测试
print("\n[3] 图像预处理 (灰度→BGR→Warp→减均值)...")
fake_crop = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
tensor = preprocess_crop(fake_crop)
print(f"    ✓ ({fake_crop.shape[0]}x{fake_crop.shape[1]} 灰度) → {tensor.shape}")

# 4. 分类器测试
print("\n[4] R-CNN 分类器 (FC8: 7 classes)...")
classifier = RCNNClassifier(num_classes=7)
classifier.eval()
logits = classifier(torch.randn(1, 3, 227, 227))
print(f"    ✓ 输出: {logits.shape} (6 缺陷类 + 1 背景)")

# 5. 数据集
print("\n[5] 数据集加载...")
from dataset import load_dataset_paths, parse_voc_xml
pairs = load_dataset_paths()
print(f"    ✓ 共 {len(pairs)} 个图像-标注对")
boxes = parse_voc_xml(pairs[0][1])
print(f"    ✓ 样本标注: {boxes}")

# 6. IoU 计算
print("\n[6] IoU 计算...")
from dataset import compute_iou
iou_val = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
print(f"    ✓ (0,0,100,100) vs (50,50,150,150) IoU = {iou_val:.4f}")

print("\n" + "=" * 60)
print("✓ 全部测试通过！R-CNN 环境准备就绪")
print("=" * 60)
print("\n运行训练: python train.py")
print("运行检测: python detect.py <图片路径>")

# R-CNN 钢材缺陷检测 - 配置文件
# ================================

import os
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent

# 数据集路径
DATASET_PATH = Path(
    r"C:\Users\wang'q'k\.cache\kagglehub\datasets\kaustubhdikshit"
    r"\neu-surface-defect-database\versions\1\NEU-DET\train"
)

# 6 种钢材缺陷类别
CLASSES = [
    "crazing",        # 裂纹
    "inclusion",      # 夹杂物
    "patches",        # 斑块
    "pitted_surface", # 麻点
    "rolled-in_scale", # 氧化铁皮
    "scratches",      # 划痕
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}

# 图片参数（与原始 R-CNN 论文一致：AlexNet 输入 227×227）
IMG_SIZE = 227

# 特征维度
FEATURE_DIM = 4096  # AlexNet FC7 层输出维度（论文原始配置）

# Selective Search 参数
SS_MODE = "fast"      # "fast" 或 "quality"（200x200图推荐fast，速度差10倍以上）
SS_TOP_N = 2000       # 每张图候选框数量上限（200x200实际最多出50-60个）

# CNN 微调参数
BATCH_SIZE = 32
CNN_EPOCHS = 15
CNN_LR = 0.001
CNN_MOMENTUM = 0.9
CNN_WEIGHT_DECAY = 1e-4

# 正负样本 IoU 阈值
IOU_POSITIVE = 0.5    # IoU >= 0.5 为正样本（CNN 微调用）
IOU_NEGATIVE = 0.3    # IoU < 0.3 为负样本（SVM 硬负样本挖掘用）

# SVM 训练参数
SVM_C = 0.01
SVM_HARD_NEG_ITER = 3  # 硬负样本挖掘迭代次数

# BBox 回归参数
BBOX_IOU_MIN = 0.6     # IoU >= 0.6 才用于训练回归器
BBOX_REG_LAMBDA = 1000  # Smooth L1 损失权重

# 检测参数
DETECT_CONF_THRESH = 0.5  # SVM 置信度阈值
NMS_IOU_THRESH = 0.3      # NMS IoU 阈值

# 划分比例
TRAIN_RATIO = 0.8

# 模型保存路径
CHECKPOINT_DIR = ROOT / "checkpoints"
CNN_MODEL_PATH = CHECKPOINT_DIR / "cnn_finetuned.pth"
SVM_MODEL_PATH = CHECKPOINT_DIR / "svm_classifiers.pkl"
BBOX_REG_PATH = CHECKPOINT_DIR / "bbox_regressors.pkl"

# 预计算特征缓存（避免重复前向传播）
FEATURES_DIR = ROOT / "features"
FEATURES_DIR.mkdir(exist_ok=True)

# 设备
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CONFIG] 使用设备: {DEVICE}")

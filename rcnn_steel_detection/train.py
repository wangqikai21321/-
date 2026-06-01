"""
R-CNN 训练主脚本
===============
按论文三阶段训练:
  1. CNN 微调（候选框分类）
  2. SVM 分类器训练（硬负样本挖掘）
  3. 边界框回归器训练
  4. 验证集评估（mAP）

用法: python train.py [--eval] [--no-eval] [--conf 0.5] [--nms 0.3]
"""

import argparse
import sys
import torch

from config import DEVICE
from dataset import generate_proposal_cache
from model import (
    train_cnn,
    get_feature_extractor,
    train_svm_classifiers,
    train_bbox_regressors,
)


def main():
    parser = argparse.ArgumentParser(description="R-CNN 钢材缺陷检测 — 训练")
    parser.add_argument("--eval", action="store_true", default=True, help="训练后评估 mAP（默认开启）")
    parser.add_argument("--no-eval", action="store_true", help="跳过验证集评估")
    parser.add_argument("--conf", type=float, default=0.5, help="评估时的 SVM 阈值")
    parser.add_argument("--nms", type=float, default=0.3, help="评估时 NMS IoU 阈值")
    args = parser.parse_args()

    do_eval = args.eval and not args.no_eval

    print("=" * 60)
    print("R-CNN 钢材缺陷检测 — 训练流程")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    # ---- 0. 生成候选框缓存 ----
    train_data, val_data = generate_proposal_cache()
    print(f"\n训练集: {len(train_data)} 张, 验证集: {len(val_data)} 张")

    # ---- 1. CNN 微调 ----
    cnn_model = train_cnn(train_data)

    # ---- 2. SVM 训练 ----
    feature_extractor = get_feature_extractor()
    svms, scaler = train_svm_classifiers(train_data, feature_extractor)

    # ---- 3. BBox 回归器训练 ----
    regressors = train_bbox_regressors(train_data, feature_extractor)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  CNN:   checkpoints/cnn_finetuned.pth")
    print(f"  SVM:   checkpoints/svm_classifiers.pkl")
    print(f"  BBox:  checkpoints/bbox_regressors.pkl")
    print("=" * 60)

    # ---- 4. 验证集评估 ----
    if do_eval:
        print("\n")
        from eval import evaluate
        aps, mAP = evaluate(val_data, conf_thresh=args.conf, nms_thresh=args.nms)
        print(f"\n最终 mAP: {mAP:.4f}")

    print("\n运行检测: python detect.py <图片路径>")
    print("运行评估: python eval.py [--max-images N]")


if __name__ == "__main__":
    main()

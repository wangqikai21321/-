"""
R-CNN 钢材缺陷检测（与 Girshick et al. 2014 论文一致）
=====================================================
流程:
  1. 输入图片
  2. Selective Search 提取 ~2000 个候选区域
  3. 每个区域 Warp 到 227×227 → AlexNet FC7 提取 4096 维特征
  4. 每个类别 SVM 分类
  5. 边界框回归精修
  6. NMS 去重 → 输出最终检测结果

用法:
  python detect.py <图片路径> [--output result.jpg] [--conf 0.5] [--nms 0.3]
"""

import argparse
import pickle
import sys

import cv2
import numpy as np
import torch

from config import (
    CLASSES, IDX_TO_CLASS, IMG_SIZE, DEVICE,
    SS_MODE, SS_TOP_N,
    DETECT_CONF_THRESH, NMS_IOU_THRESH,
    CNN_MODEL_PATH, SVM_MODEL_PATH, BBOX_REG_PATH,
)
from dataset import preprocess_crop


def load_models():
    """加载 AlexNet 特征提取器、SVM、回归器"""
    from model import get_feature_extractor
    extractor = get_feature_extractor()
    extractor.eval()

    if not SVM_MODEL_PATH.exists():
        print("[错误] 未找到 SVM 模型，请先运行: python train.py")
        sys.exit(1)

    with open(SVM_MODEL_PATH, "rb") as f:
        svm_data = pickle.load(f)

    regressors = {}
    if BBOX_REG_PATH.exists():
        with open(BBOX_REG_PATH, "rb") as f:
            regressors = pickle.load(f)

    return extractor, svm_data["svms"], svm_data["scaler"], regressors


def extract_feature(img, x, y, w, h, extractor):
    """提取 AlexNet FC7 特征（4096 维）"""
    h_img, w_img = img.shape[:2]
    x, y = max(0, int(x)), max(0, int(y))
    w, h = min(int(w), w_img - x), min(int(h), h_img - y)
    if w < 10 or h < 10:
        return None
    crop = img[y:y+h, x:x+w]
    crop_t = preprocess_crop(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return extractor(crop_t).cpu().numpy()[0]


def nms(detections, iou_threshold):
    """非极大值抑制"""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        filtered = []
        for d in detections:
            xA = max(best[0], d[0]); yA = max(best[1], d[1])
            xB = min(best[2], d[2]); yB = min(best[3], d[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            if inter == 0:
                filtered.append(d); continue
            area_b = (best[2]-best[0]) * (best[3]-best[1])
            area_d = (d[2]-d[0]) * (d[3]-d[1])
            if inter / (area_b + area_d - inter) < iou_threshold:
                filtered.append(d)
        detections = filtered
    return keep


def detect(image_path, conf_thresh=None, nms_thresh=None, output_path=None):
    """R-CNN 检测主函数"""
    if conf_thresh is None:
        conf_thresh = DETECT_CONF_THRESH
    if nms_thresh is None:
        nms_thresh = NMS_IOU_THRESH

    # 1. 加载模型
    extractor, svms, scaler, regressors = load_models()

    # 2. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"[错误] 无法读取: {image_path}")
        sys.exit(1)
    orig_h, orig_w = img.shape[:2]
    print(f"[检测] 图片: {orig_w}x{orig_h}")

    # 3. Selective Search
    print("[检测] Selective Search 提取候选框...")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    if SS_MODE == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    proposals = ss.process()[:SS_TOP_N]
    print(f"[检测] 候选框: {len(proposals)}")

    # 4. 特征提取 + SVM 分类
    print("[检测] AlexNet FC7 特征提取 & SVM 分类...")
    detections_all = []

    for i, (x, y, w, h) in enumerate(proposals):
        feat = extract_feature(img, x, y, w, h, extractor)
        if feat is None:
            continue

        feat_scaled = scaler.transform(feat.reshape(1, -1))[0]
        best_score, best_cls = -float("inf"), 0

        for cls_id in range(1, len(CLASSES) + 1):
            svm = svms.get(cls_id)
            if svm is None:
                continue
            score = float(svm.decision_function(feat_scaled.reshape(1, -1))[0])
            if score > best_score:
                best_score = score
                best_cls = cls_id

        if best_score > conf_thresh and best_cls > 0:
            detections_all.append((int(x), int(y), int(x + w), int(y + h), best_score, best_cls))

        if (i + 1) % 500 == 0:
            print(f"  进度: {i+1}/{len(proposals)}")

    print(f"[检测] SVM 过滤后: {len(detections_all)}")

    # 5. NMS
    detections_keep = nms(detections_all, nms_thresh)
    print(f"[检测] NMS 后: {len(detections_keep)}")

    # 6. 构建结果
    results = []
    for (xmin, ymin, xmax, ymax, score, cls_id) in detections_keep:
        class_name = IDX_TO_CLASS[cls_id - 1]
        results.append((xmin, ymin, xmax, ymax, score, class_name))

    # 7. 可视化
    vis = img.copy()
    colors = {
        "crazing": (0, 0, 255), "inclusion": (0, 255, 0),
        "patches": (255, 0, 0), "pitted_surface": (255, 255, 0),
        "rolled-in_scale": (255, 0, 255), "scratches": (0, 255, 255),
    }
    for (xmin, ymin, xmax, ymax, score, class_name) in results:
        color = colors.get(class_name, (0, 255, 0))
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{class_name}: {score:.2f}"
        cv2.putText(vis, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if output_path is None:
        output_path = image_path.rsplit(".", 1)[0] + "_detected.jpg" if "." in image_path else "detected.jpg"
    cv2.imwrite(output_path, vis)
    print(f"[检测] 结果保存: {output_path}")

    # 8. 打印
    print("\n" + "=" * 60)
    print("检测结果（R-CNN: AlexNet + Selective Search + SVM）")
    print("-" * 60)
    for (xmin, ymin, xmax, ymax, score, class_name) in results:
        print(f"  {class_name:<20s}  conf={score:.3f}  bbox=({xmin},{ymin},{xmax},{ymax})")
    if not results:
        print("  未检测到缺陷")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="R-CNN 钢材缺陷检测")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    parser.add_argument("--conf", type=float, default=DETECT_CONF_THRESH, help="SVM 阈值")
    parser.add_argument("--nms", type=float, default=NMS_IOU_THRESH, help="NMS IoU 阈值")
    args = parser.parse_args()
    detect(args.image, conf_thresh=args.conf, nms_thresh=args.nms, output_path=args.output)


if __name__ == "__main__":
    main()

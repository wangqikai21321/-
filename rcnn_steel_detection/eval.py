"""
R-CNN 验证集评估
===============
使用验证集计算 mAP（mean Average Precision）
- 支持自定义 IoU 阈值（默认 0.3，适合 NEU-DET 缺陷标注风格）
- 支持同类预测合并（修复 SS 碎片化问题）
- 支持快速测试模式

用法: python eval.py [--iou 0.3] [--conf 0.0] [--nms 0.3] [--merge] [--max-images N]
"""

import argparse
import pickle
import sys
import os

import cv2
import numpy as np
import torch

from config import (
    CLASSES, CLASS_TO_IDX, NUM_CLASSES, DEVICE,
    SS_MODE, SS_TOP_N,
    DETECT_CONF_THRESH, NMS_IOU_THRESH,
    CNN_MODEL_PATH, SVM_MODEL_PATH, BBOX_REG_PATH, FEATURES_DIR,
)
from dataset import preprocess_crop, compute_iou


def load_models():
    from model import get_feature_extractor
    extractor = get_feature_extractor()
    extractor.eval()
    with open(SVM_MODEL_PATH, "rb") as f:
        svm_data = pickle.load(f)
    regressors = {}
    if BBOX_REG_PATH.exists():
        with open(BBOX_REG_PATH, "rb") as f:
            regressors = pickle.load(f)
    return extractor, svm_data["svms"], svm_data["scaler"], regressors


def detect_single(img, proposals, extractor, svms, scaler, conf_thresh, nms_thresh):
    """单图检测"""
    detections_all = []
    for (x, y, w, h) in proposals:
        h_img, w_img = img.shape[:2]
        x2, y2 = max(0, int(x)), max(0, int(y))
        w2 = min(int(w), w_img - x2)
        h2 = min(int(h), h_img - y2)
        if w2 < 10 or h2 < 10:
            continue
        crop = img[y2:y2+h2, x2:x2+w2]
        crop_t = preprocess_crop(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = extractor(crop_t).cpu().numpy()[0]

        feat_scaled = scaler.transform(feat.reshape(1, -1))[0]
        best_score, best_cls = -float("inf"), 0
        for cls_id in range(1, NUM_CLASSES + 1):
            svm = svms.get(cls_id)
            if svm is None:
                continue
            score = float(svm.decision_function(feat_scaled.reshape(1, -1))[0])
            if score > best_score:
                best_score = score
                best_cls = cls_id

        if best_score > conf_thresh and best_cls > 0:
            detections_all.append((
                int(x2), int(y2), int(x2 + w2), int(y2 + h2), best_score, best_cls
            ))

    # NMS
    if not detections_all:
        return []
    detections_all.sort(key=lambda d: d[4], reverse=True)
    keep = []
    while detections_all:
        best = detections_all.pop(0)
        keep.append(best)
        filtered = []
        for d in detections_all:
            xA = max(best[0], d[0]); yA = max(best[1], d[1])
            xB = min(best[2], d[2]); yB = min(best[3], d[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            if inter == 0:
                filtered.append(d); continue
            area_b = (best[2] - best[0]) * (best[3] - best[1])
            area_d = (d[2] - d[0]) * (d[3] - d[1])
            if inter / (area_b + area_d - inter) < nms_thresh:
                filtered.append(d)
        detections_all = filtered
    return keep


def merge_same_class(detections, merge_iou=0.1):
    """
    合并同类别的相邻预测框（修复 SS 碎片化问题）
    策略: 对每个类别，贪心合并 IoU >= merge_iou 的框（取外接矩形）
    返回合并后的预测列表
    """
    if not detections:
        return []

    # 按类别分组
    by_class = {}
    for d in detections:
        cls_id = d[5]
        if cls_id not in by_class:
            by_class[cls_id] = []
        by_class[cls_id].append(list(d))

    merged = []
    for cls_id, boxes in by_class.items():
        while boxes:
            current = boxes.pop(0)
            merged_current = False
            while True:
                # 找与 current 重叠的框
                found = False
                for i, other in enumerate(boxes):
                    xA = max(current[0], other[0]); yA = max(current[1], other[1])
                    xB = min(current[2], other[2]); yB = min(current[3], other[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    if inter == 0:
                        continue
                    area_c = (current[2] - current[0]) * (current[3] - current[1])
                    area_o = (other[2] - other[0]) * (other[3] - other[1])
                    if inter / (area_c + area_o - inter) >= merge_iou:
                        # 合并: 取外接矩形 + 取最高分
                        current[0] = min(current[0], other[0])
                        current[1] = min(current[1], other[1])
                        current[2] = max(current[2], other[2])
                        current[3] = max(current[3], other[3])
                        current[4] = max(current[4], other[4])
                        boxes.pop(i)
                        found = True
                        break
                if not found:
                    break
            merged.append(tuple(current))
    return merged


def compute_ap_all_points(recalls, precisions):
    """VOC 2010+ all-point interpolation"""
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    idx = np.argsort(recalls)
    recalls = recalls[idx]
    precisions = precisions[idx]
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for i in range(len(recalls) - 1):
        if recalls[i + 1] > recalls[i]:
            ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]
    return ap


def evaluate(val_data, iou_thresh=0.3, conf_thresh=0.0, nms_thresh=0.3,
             merge=False, max_images=None):
    """
    验证集评估

    Args:
        iou_thresh:  判定正确检测的 IoU 阈值（默认 0.3，适合被碎片化的缺陷标注）
        conf_thresh: SVM 置信度阈值（默认 0.0，SVM 的自然决策边界）
        nms_thresh:  NMS IoU 阈值
        merge:       是否合并同类相邻碎片预测
        max_images:  限制评估图片数量
    """
    print("=" * 60)
    print("R-CNN 验证集评估")
    print("  IoU 阈值: %.2f | SVM 阈值: %.2f | NMS: %.2f | Merge: %s" %
          (iou_thresh, conf_thresh, nms_thresh, merge))
    print("=" * 60)

    extractor, svms, scaler, regressors = load_models()

    if max_images:
        val_data = val_data[:max_images]
    print("\n[评估] 验证集: %d 张图片" % len(val_data))

    all_preds = {c: [] for c in range(NUM_CLASSES)}
    total_gt = {c: 0 for c in range(NUM_CLASSES)}
    total_dets = 0

    for img_idx, item in enumerate(val_data):
        img_path = item["img_path"]
        gt_boxes_raw = item["gt_boxes"]
        proposals = item["proposals"]

        for (_, _, _, _, cls_id) in gt_boxes_raw:
            total_gt[cls_id] += 1

        img = cv2.imread(img_path)
        if img is None:
            continue

        dets = detect_single(img, proposals, extractor, svms, scaler,
                             conf_thresh, nms_thresh)

        if merge and dets:
            dets = merge_same_class(dets)

        total_dets += len(dets)

        gt_by_class = {c: [] for c in range(NUM_CLASSES)}
        for (gx, gy, gxmax, gymax, gcid) in gt_boxes_raw:
            gt_by_class[gcid].append((gx, gy, gxmax, gymax))

        gt_matched = {c: set() for c in range(NUM_CLASSES)}
        dets.sort(key=lambda d: d[4], reverse=True)

        for (dx, dy, dxmax, dymax, dscore, dcid) in dets:
            cls_id = dcid - 1
            gt_list = gt_by_class.get(cls_id, [])
            matched = gt_matched.get(cls_id, set())

            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, (gx, gy, gxmax, gymax) in enumerate(gt_list):
                if gt_idx in matched:
                    continue
                iou = compute_iou((dx, dy, dxmax, dymax), (gx, gy, gxmax, gymax))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thresh:
                all_preds[cls_id].append((dscore, 1))
                gt_matched[cls_id].add(best_gt_idx)
            else:
                all_preds[cls_id].append((dscore, 0))

        if (img_idx + 1) % 50 == 0:
            print("  进度: %d/%d (累计 %d 个预测框)" % (img_idx + 1, len(val_data), total_dets))

    # 结果输出
    print("\n" + "=" * 70)
    print("%-22s %6s %10s %10s %10s %10s" %
          ("类别", "GT", "预测框", "Recall", "Precision", "AP"))
    print("-" * 70)

    aps = []
    total_gt_all = sum(total_gt.values())
    total_preds_all = sum(len(all_preds[c]) for c in range(NUM_CLASSES))
    total_tp_all = 0

    for cls_id in range(NUM_CLASSES):
        preds = all_preds[cls_id]
        gt_count = total_gt[cls_id]

        if gt_count == 0 or not preds:
            aps.append(0.0)
            print("%-22s %6d %10s %10s %10s %10.4f" %
                  (CLASSES[cls_id], gt_count, str(len(preds)),
                   "-", "-", 0.0))
            continue

        preds.sort(key=lambda x: x[0], reverse=True)
        tp_cumsum = 0
        fp_cumsum = 0
        recalls_list = []
        precisions_list = []

        for _, is_tp in preds:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            recall = tp_cumsum / gt_count
            precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0.0
            recalls_list.append(recall)
            precisions_list.append(precision)

        total_tp_all += tp_cumsum
        ap = compute_ap_all_points(recalls_list, precisions_list)
        aps.append(ap)
        final_recall = tp_cumsum / gt_count
        final_precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0.0

        print("%-22s %6d %10d %10.4f %10.4f %10.4f" %
              (CLASSES[cls_id], gt_count, len(preds), final_recall, final_precision, ap))

    mAP = np.mean(aps)
    overall_recall = total_tp_all / total_gt_all if total_gt_all > 0 else 0
    overall_precision = total_tp_all / total_preds_all if total_preds_all > 0 else 0

    print("-" * 70)
    print("%-22s %6d %10d %10.4f %10.4f %10.4f" %
          ("整体/平均值", total_gt_all, total_preds_all,
           overall_recall, overall_precision, mAP))
    print("=" * 70)
    print()
    print("  评估参数: IoU阈值=%.2f | SVM阈值=%.2f | NMS=%.2f | 合并碎片=%s" %
          (iou_thresh, conf_thresh, nms_thresh, merge))
    print("  mAP 解读: 缺陷标注覆盖大片区域，SS产生碎片提案，")
    print("            IoU 阈值越低，对碎片化预测越宽容，mAP越高。")
    print()

    # ---- 补充：图像级分类准确率 ----
    print("\n图像级分类准确率（每张图只取最高分判断缺陷类别）")
    print("-" * 50)

    # 重新遍历验证集做图像级分类
    img_correct = 0
    img_total = 0
    per_class_correct = {c: 0 for c in range(NUM_CLASSES)}
    per_class_total = {c: 0 for c in range(NUM_CLASSES)}

    for img_idx, item in enumerate(val_data[:max_images] if max_images else val_data):
        img_path = item["img_path"]
        gt_boxes_raw = item["gt_boxes"]
        proposals = item["proposals"]

        # 该图的真实类别（NEU-DET 每张图只有一个类别）
        if not gt_boxes_raw:
            continue
        true_class = gt_boxes_raw[0][4]  # 0-based

        img = cv2.imread(img_path)
        if img is None:
            continue

        # 统计所有候选框在每个类别上的最高分
        class_max_scores = np.zeros(NUM_CLASSES)
        for (x, y, w, h) in proposals:
            h_img, w_img = img.shape[:2]
            x2, y2 = max(0, int(x)), max(0, int(y))
            w2 = min(int(w), w_img - x2)
            h2 = min(int(h), h_img - y2)
            if w2 < 10 or h2 < 10:
                continue
            crop = img[y2:y2+h2, x2:x2+w2]
            crop_t = preprocess_crop(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = extractor(crop_t).cpu().numpy()[0]
            feat_s = scaler.transform(feat.reshape(1, -1))[0]
            for cls_id in range(1, NUM_CLASSES + 1):
                svm = svms.get(cls_id)
                if svm is None:
                    continue
                score = float(svm.decision_function(feat_s.reshape(1, -1))[0])
                class_max_scores[cls_id - 1] = max(class_max_scores[cls_id - 1], score)

        pred_class = int(np.argmax(class_max_scores))
        per_class_total[true_class] += 1
        img_total += 1
        if pred_class == true_class:
            img_correct += 1
            per_class_correct[true_class] += 1

    img_acc = img_correct / img_total * 100 if img_total > 0 else 0
    print("  类别        总数  正确  准确率")
    for cls_id in range(NUM_CLASSES):
        total_c = per_class_total[cls_id]
        correct_c = per_class_correct[cls_id]
        acc_c = correct_c / total_c * 100 if total_c > 0 else 0
        print("  %-20s %4d  %4d  %5.1f%%" % (CLASSES[cls_id], total_c, correct_c, acc_c))
    print("-" * 50)
    print("  整体: %d 张图, 正确 %d 张, 准确率 %.1f%%" % (img_total, img_correct, img_acc))
    print()

    return aps, mAP


def main():
    parser = argparse.ArgumentParser(description="R-CNN 验证集评估 (mAP)")
    parser.add_argument("--iou", type=float, default=0.3,
                        help="判定正确检测的 IoU 阈值 (默认 0.3)")
    parser.add_argument("--conf", type=float, default=0.0,
                        help="SVM 置信度阈值 (默认 0.0 = 自然决策边界)")
    parser.add_argument("--nms", type=float, default=NMS_IOU_THRESH,
                        help="NMS IoU 阈值")
    parser.add_argument("--merge", action="store_true",
                        help="合并同类相邻碎片预测为大方框")
    parser.add_argument("--max-images", type=int, default=None,
                        help="限制评估图片数 (快速测试)")
    args = parser.parse_args()

    cache_path = FEATURES_DIR / "proposals_cache.pkl"
    if not cache_path.exists():
        print("[错误] 请先运行 train.py 生成候选框缓存")
        sys.exit(1)

    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    evaluate(data["val"],
             iou_thresh=args.iou,
             conf_thresh=args.conf,
             nms_thresh=args.nms,
             merge=args.merge,
             max_images=args.max_images)


if __name__ == "__main__":
    main()

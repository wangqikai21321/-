from __future__ import annotations

from collections import defaultdict

import numpy as np

from .detect import Detection
from .geometry import iou
from .metrics import precision_recall_ap
from .voc import VOC_CLASSES, VOCDataset


def evaluate_detections(dataset: VOCDataset, detections: list[Detection], iou_threshold: float = 0.5, use_07_metric: bool = False) -> dict[str, object]:
    gt_by_class = {name: defaultdict(list) for name in VOC_CLASSES}
    npos = {name: 0 for name in VOC_CLASSES}
    for item in dataset:
        for obj in item.objects:
            if obj.difficult:
                continue
            gt_by_class[obj.name][item.image_id].append({"bbox": obj.bbox, "matched": False})
            npos[obj.name] += 1
    dets_by_class = {name: [] for name in VOC_CLASSES}
    for det in detections:
        dets_by_class[det.class_name].append(det)
    per_class = {}
    for name in VOC_CLASSES:
        labels: list[int] = []
        scores: list[float] = []
        for det in sorted(dets_by_class[name], key=lambda d: d.score, reverse=True):
            scores.append(det.score)
            candidates = gt_by_class[name].get(det.image_id, [])
            if not candidates:
                labels.append(0)
                continue
            gt_boxes = np.asarray([c["bbox"] for c in candidates], dtype=np.float32)
            overlaps = iou(det.box, gt_boxes)
            best = int(np.argmax(overlaps)) if len(overlaps) else -1
            if best >= 0 and overlaps[best] >= iou_threshold and not candidates[best]["matched"]:
                candidates[best]["matched"] = True
                labels.append(1)
            else:
                labels.append(0)
        recalls, precisions, ap = precision_recall_ap(labels, scores, npos[name], use_07_metric=use_07_metric)
        per_class[name] = {
            "ap": ap,
            "npos": npos[name],
            "detections": len(dets_by_class[name]),
            "recall": recalls.tolist(),
            "precision": precisions.tolist(),
        }
    return {"mAP": float(np.mean([v["ap"] for v in per_class.values()])), "per_class": per_class}

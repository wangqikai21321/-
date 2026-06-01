from __future__ import annotations

import numpy as np


def voc_ap(recalls: np.ndarray, precisions: np.ndarray, use_07_metric: bool = False) -> float:
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = 0 if np.sum(recalls >= t) == 0 else np.max(precisions[recalls >= t])
            ap += p / 11.0
        return float(ap)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def precision_recall_ap(labels: list[int], scores: list[float], total_positives: int, use_07_metric: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    if total_positives == 0:
        return np.array([]), np.array([]), 0.0
    order = np.argsort(-np.asarray(scores))
    sorted_labels = np.asarray(labels)[order]
    tp = (sorted_labels == 1).astype(np.float32)
    fp = (sorted_labels == 0).astype(np.float32)
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recalls = tp / float(total_positives)
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)
    return recalls, precisions, voc_ap(recalls, precisions, use_07_metric=use_07_metric)

from __future__ import annotations

import numpy as np


def iou(box: tuple[float, float, float, float], boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    area_a = max(0, box[2] - box[0] + 1) * max(0, box[3] - box[1] + 1)
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0] + 1) * np.maximum(0, boxes[:, 3] - boxes[:, 1] + 1)
    denom = area_a + area_b - inter
    return np.where(denom > 0, inter / denom, 0).astype(np.float32)


def bbox_transform(proposals: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    proposals = proposals.astype(np.float32)
    gt_boxes = gt_boxes.astype(np.float32)
    pw = proposals[:, 2] - proposals[:, 0] + 1
    ph = proposals[:, 3] - proposals[:, 1] + 1
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph
    gw = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
    gh = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
    gx = gt_boxes[:, 0] + 0.5 * gw
    gy = gt_boxes[:, 1] + 0.5 * gh
    return np.vstack([(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)]).T.astype(np.float32)


def bbox_inverse(proposals: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    proposals = proposals.astype(np.float32)
    deltas = deltas.astype(np.float32)
    pw = proposals[:, 2] - proposals[:, 0] + 1
    ph = proposals[:, 3] - proposals[:, 1] + 1
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph
    gx = deltas[:, 0] * pw + px
    gy = deltas[:, 1] * ph + py
    gw = np.exp(deltas[:, 2]) * pw
    gh = np.exp(deltas[:, 3]) * ph
    out = np.zeros_like(deltas, dtype=np.float32)
    out[:, 0] = gx - 0.5 * gw
    out[:, 1] = gy - 0.5 * gh
    out[:, 2] = gx + 0.5 * gw - 1
    out[:, 3] = gy + 0.5 * gh - 1
    return out


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        overlaps = iou(tuple(boxes[i]), boxes[order[1:]])
        order = order[1:][overlaps <= threshold]
    return keep

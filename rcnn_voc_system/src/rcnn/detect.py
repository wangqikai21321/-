from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch

from .features import extract_regions
from .geometry import bbox_inverse, clip_boxes, nms
from .model import RCNNAlexNet
from .proposals import selective_search
from .voc import VOC_CLASSES, VOCDataset


@dataclass(frozen=True)
class Detection:
    image_id: str
    class_name: str
    score: float
    box: tuple[float, float, float, float]


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def detect_image(
    image_bgr: np.ndarray,
    image_id: str,
    model: RCNNAlexNet,
    svms: dict,
    regressors: dict | None = None,
    proposals: np.ndarray | None = None,
    image_size: int = 227,
    proposal_mode: str = "fast",
    top_k: int = 2000,
    min_size: int = 16,
    score_threshold: float = 0.0,
    nms_iou: float = 0.3,
    max_detections_per_class: int = 100,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> list[Detection]:
    if proposals is None:
        proposals = selective_search(image_bgr, mode=proposal_mode, top_k=top_k, min_size=min_size)
    if len(proposals) == 0:
        return []
    feats = extract_regions(model, image_bgr, proposals, layer="fc7", image_size=image_size, batch_size=batch_size, device=device)
    height, width = image_bgr.shape[:2]
    output: list[Detection] = []
    for class_name in VOC_CLASSES:
        svm = svms.get(class_name)
        if svm is None:
            continue
        scores = svm.decision_function(feats)
        keep_mask = scores > score_threshold
        if not np.any(keep_mask):
            continue
        cls_boxes = proposals[keep_mask].astype(np.float32)
        cls_scores = scores[keep_mask].astype(np.float32)
        reg = regressors.get(class_name) if regressors else None
        if reg is not None:
            deltas = reg.predict(feats[keep_mask]).astype(np.float32)
            cls_boxes = clip_boxes(bbox_inverse(cls_boxes, deltas), width, height)
        keep = nms(cls_boxes, cls_scores, nms_iou)[:max_detections_per_class]
        for idx in keep:
            output.append(Detection(image_id, class_name, float(cls_scores[idx]), tuple(float(v) for v in cls_boxes[idx])))
    return output


def detect_dataset(
    dataset: VOCDataset,
    model: RCNNAlexNet,
    svms: dict,
    regressors: dict | None,
    proposal_cache: dict[str, np.ndarray] | None,
    out_dir: str | Path,
    **kwargs,
) -> list[Detection]:
    all_dets: list[Detection] = []
    for idx, item in enumerate(dataset):
        img = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        dets = detect_image(
            img,
            item.image_id,
            model,
            svms,
            regressors=regressors,
            proposals=proposal_cache.get(item.image_id) if proposal_cache else None,
            **kwargs,
        )
        all_dets.extend(dets)
        if (idx + 1) % 50 == 0:
            print(f"[detect] {idx + 1}/{len(dataset)}")
    write_voc_results(all_dets, out_dir)
    return all_dets


def write_voc_results(detections: list[Detection], out_dir: str | Path, prefix: str = "comp3_det_test") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_class = {name: [] for name in VOC_CLASSES}
    for det in detections:
        by_class[det.class_name].append(det)
    for class_name, dets in by_class.items():
        with open(out_dir / f"{prefix}_{class_name}.txt", "w", encoding="utf-8") as f:
            for d in sorted(dets, key=lambda x: x.score, reverse=True):
                x1, y1, x2, y2 = d.box
                f.write(f"{d.image_id} {d.score:.6f} {x1 + 1:.1f} {y1 + 1:.1f} {x2 + 1:.1f} {y2 + 1:.1f}\n")


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image_bgr.copy()
    rng = np.random.default_rng(123)
    colors = {name: tuple(int(v) for v in rng.integers(40, 255, size=3)) for name in VOC_CLASSES}
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.box]
        color = colors[det.class_name]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{det.class_name} {det.score:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out

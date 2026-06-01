from __future__ import annotations

from pathlib import Path
import pickle

import cv2
import numpy as np

from .voc import VOCDataset


def selective_search(image_bgr: np.ndarray, mode: str = "fast", top_k: int = 2000, min_size: int = 16) -> np.ndarray:
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError("opencv-contrib-python is required for cv2.ximgproc Selective Search.")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image_bgr)
    if mode == "quality":
        ss.switchToSelectiveSearchQuality()
    else:
        ss.switchToSelectiveSearchFast()
    rects = ss.process()
    boxes: list[tuple[int, int, int, int]] = []
    h, w = image_bgr.shape[:2]
    seen = set()
    for x, y, bw, bh in rects:
        if bw < min_size or bh < min_size:
            continue
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w - 1, int(x + bw - 1)), min(h - 1, int(y + bh - 1))
        box = (x1, y1, x2, y2)
        if box in seen:
            continue
        seen.add(box)
        boxes.append(box)
        if len(boxes) >= top_k:
            break
    return np.asarray(boxes, dtype=np.float32)


def proposal_cache_path(out_dir: str | Path, year: str, split: str) -> Path:
    return Path(out_dir) / f"voc{year}_{split}_proposals.pkl"


def generate_proposal_cache(dataset: VOCDataset, out_dir: str | Path, mode: str = "fast", top_k: int = 2000, min_size: int = 16, max_images: int | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = proposal_cache_path(out_dir, dataset.year, dataset.image_set)
    data: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print(f"[proposals] resume {len(data)} cached images from {cache_path}", flush=True)
    for idx, item in enumerate(dataset):
        if max_images is not None and idx >= max_images:
            break
        if item.image_id in data:
            continue
        img = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        data[item.image_id] = selective_search(img, mode=mode, top_k=top_k, min_size=min_size)
        if (idx + 1) % 100 == 0:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[proposals] {idx + 1}/{len(dataset)}", flush=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return cache_path


def load_proposals(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        return pickle.load(f)

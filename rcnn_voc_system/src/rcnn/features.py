from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .model import RCNNAlexNet
from .preprocess import preprocess_region
from .voc import VOCDataset


def extract_regions(
    model: RCNNAlexNet,
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    layer: str = "fc7",
    image_size: int = 227,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    feats: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(boxes), batch_size):
        batch_boxes = boxes[start : start + batch_size]
        x = torch.stack([preprocess_region(image_bgr, tuple(b), size=image_size) for b in batch_boxes]).to(device)
        with torch.no_grad():
            y = model.extract(x, layer)[layer]
            if y.ndim > 2:
                y = torch.flatten(y, 1)
        feats.append(y.cpu().numpy().astype(np.float32))
    if not feats:
        return np.empty((0, 4096), dtype=np.float32)
    return np.vstack(feats)


def build_feature_cache(
    dataset: VOCDataset,
    proposals: dict[str, np.ndarray],
    model: RCNNAlexNet,
    out_dir: str | Path,
    layer: str = "fc7",
    image_size: int = 227,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    max_images: int | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"voc{dataset.year}_{dataset.image_set}_{layer}_features.npz"
    arrays: dict[str, np.ndarray] = {}
    for idx, item in enumerate(dataset):
        if max_images is not None and idx >= max_images:
            break
        boxes = proposals.get(item.image_id, np.empty((0, 4), dtype=np.float32))
        img = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if img is None or len(boxes) == 0:
            continue
        arrays[f"{item.image_id}_boxes"] = boxes.astype(np.float32)
        arrays[f"{item.image_id}_features"] = extract_regions(model, img, boxes, layer=layer, image_size=image_size, batch_size=batch_size, device=device)
        if (idx + 1) % 50 == 0:
            print(f"[features] {idx + 1}/{len(dataset)}")
    np.savez_compressed(path, **arrays)
    return path


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}

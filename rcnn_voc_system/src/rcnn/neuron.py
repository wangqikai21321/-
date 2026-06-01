from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from .model import RCNNAlexNet
from .preprocess import preprocess_region
from .voc import VOCDataset


@dataclass
class ActivationHit:
    image_id: str
    image_path: str
    box: tuple[float, float, float, float]
    activation: float


def activation_value(tensor: torch.Tensor, channel: int | None = None, row: int | None = None, col: int | None = None, unit: int | None = None) -> float:
    if tensor.ndim == 4:
        c = 0 if channel is None else channel
        h = tensor.shape[2] // 2 if row is None else row
        w = tensor.shape[3] // 2 if col is None else col
        return float(tensor[0, c, h, w].item())
    if tensor.ndim == 2:
        u = 0 if unit is None else unit
        return float(tensor[0, u].item())
    raise ValueError(f"Unsupported activation tensor shape: {tuple(tensor.shape)}")


def activation_values(tensor: torch.Tensor, channel: int | None = None, row: int | None = None, col: int | None = None, unit: int | None = None) -> np.ndarray:
    if tensor.ndim == 4:
        c = 0 if channel is None else channel
        h = tensor.shape[2] // 2 if row is None else row
        w = tensor.shape[3] // 2 if col is None else col
        return tensor[:, c, h, w].detach().cpu().numpy().astype(float)
    if tensor.ndim == 2:
        u = 0 if unit is None else unit
        return tensor[:, u].detach().cpu().numpy().astype(float)
    raise ValueError(f"Unsupported activation tensor shape: {tuple(tensor.shape)}")


def scan_top_activations(
    dataset: VOCDataset,
    proposals: dict[str, np.ndarray],
    model: RCNNAlexNet,
    layer: str,
    top_k: int = 16,
    channel: int | None = None,
    row: int | None = None,
    col: int | None = None,
    unit: int | None = None,
    image_size: int = 227,
    max_images: int | None = None,
    device: str | torch.device = "cpu",
) -> list[ActivationHit]:
    hits: list[ActivationHit] = []
    model.eval()
    for idx, item in enumerate(dataset):
        if max_images is not None and idx >= max_images:
            break
        img = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        boxes = proposals.get(item.image_id, np.empty((0, 4), dtype=np.float32))
        for start in range(0, len(boxes), 64):
            batch_boxes = boxes[start : start + 64]
            x = torch.stack([preprocess_region(img, tuple(box), size=image_size) for box in batch_boxes]).to(device)
            with torch.no_grad():
                act = model.extract(x, layer)[layer].cpu()
            values = activation_values(act, channel=channel, row=row, col=col, unit=unit)
            for box, value in zip(batch_boxes, values):
                hits.append(ActivationHit(item.image_id, str(item.image_path), tuple(float(v) for v in box), float(value)))
            hits = sorted(hits, key=lambda h: h.activation, reverse=True)[:top_k]
    return hits


def save_hits(hits: list[ActivationHit], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(h) for h in hits], indent=2), encoding="utf-8")
    return out_path


def load_hits(path: str | Path) -> list[ActivationHit]:
    return [ActivationHit(**item) for item in json.loads(Path(path).read_text(encoding="utf-8"))]


def render_top_regions(hits: list[ActivationHit], out_path: str | Path, tile: int = 180, cols: int = 4) -> Path:
    rows = int(np.ceil(max(1, len(hits)) / cols))
    canvas = Image.new("RGB", (cols * tile, rows * tile), "white")
    for idx, hit in enumerate(hits):
        img = cv2.imread(hit.image_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in hit.box]
        crop = img[max(0, y1) : max(0, y2 + 1), max(0, x1) : max(0, x2 + 1)]
        if crop.size == 0:
            continue
        crop = cv2.cvtColor(cv2.resize(crop, (tile, tile)), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop)
        draw = ImageDraw.Draw(pil)
        draw.rectangle((2, 2, tile - 3, tile - 3), outline="white", width=3)
        draw.text((8, 8), f"{hit.activation:.3f}", fill="white")
        canvas.paste(pil, ((idx % cols) * tile, (idx // cols) * tile))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path

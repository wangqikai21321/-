from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import iou
from .preprocess import preprocess_region
from .voc import VOCDataset


@dataclass(frozen=True)
class RegionSample:
    image_path: str
    box: tuple[float, float, float, float]
    label: int


class FineTuneDataset(Dataset):
    def __init__(
        self,
        dataset: VOCDataset,
        proposals: dict[str, np.ndarray],
        positive_iou: float = 0.5,
        negative_iou: float = 0.5,
        max_regions_per_image: int = 128,
        image_size: int = 227,
        max_images: int | None = None,
    ):
        self.image_size = image_size
        self.samples: list[RegionSample] = []
        items = list(dataset)
        if max_images is not None:
            items = items[:max_images]
        for idx, item in enumerate(items, start=1):
            gt_boxes = np.asarray([obj.bbox for obj in item.objects if not obj.difficult], dtype=np.float32)
            gt_labels = np.asarray([obj.class_id + 1 for obj in item.objects if not obj.difficult], dtype=np.int64)
            if len(gt_boxes) == 0:
                continue
            boxes = proposals.get(item.image_id, np.empty((0, 4), dtype=np.float32))
            pos: list[RegionSample] = []
            neg: list[RegionSample] = []
            for box in boxes:
                overlaps = iou(tuple(box), gt_boxes)
                best = int(np.argmax(overlaps))
                if overlaps[best] >= positive_iou:
                    pos.append(RegionSample(str(item.image_path), tuple(box), int(gt_labels[best])))
                elif overlaps[best] < negative_iou:
                    neg.append(RegionSample(str(item.image_path), tuple(box), 0))
            pos.extend(RegionSample(str(item.image_path), tuple(b), int(l)) for b, l in zip(gt_boxes, gt_labels))
            random.shuffle(pos)
            random.shuffle(neg)
            n_pos = min(len(pos), max(1, max_regions_per_image // 4))
            n_neg = min(len(neg), max_regions_per_image - n_pos)
            self.samples.extend(pos[:n_pos] + neg[:n_neg])
            if idx % 500 == 0 or idx == len(items):
                print(f"[cnn dataset] {idx}/{len(items)} images samples={len(self.samples)}", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = cv2.imread(s.image_path, cv2.IMREAD_COLOR)
        if img is None:
            x = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        else:
            x = preprocess_region(img, s.box, size=self.image_size)
        return x, torch.tensor(s.label, dtype=torch.long)

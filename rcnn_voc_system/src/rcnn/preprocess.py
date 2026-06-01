from __future__ import annotations

import cv2
import numpy as np
import torch

TORCHVISION_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
TORCHVISION_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_and_warp(image_bgr: np.ndarray, box: tuple[float, float, float, float], size: int = 227, padding: int = 0) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), dtype=np.float32)
    crop = image_bgr[y1 : y2 + 1, x1 : x2 + 1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def preprocess_region(image_bgr: np.ndarray, box: tuple[float, float, float, float], size: int = 227) -> torch.Tensor:
    warped = crop_and_warp(image_bgr, box, size=size)
    warped = cv2.cvtColor(warped.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    warped = (warped - TORCHVISION_RGB_MEAN) / TORCHVISION_RGB_STD
    return torch.from_numpy(warped).permute(2, 0, 1).float()

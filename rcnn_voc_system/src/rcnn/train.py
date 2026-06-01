from __future__ import annotations

from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import FineTuneDataset
from .features import extract_regions
from .geometry import bbox_transform, iou
from .linear import LinearSVM, RidgeRegressor
from .model import RCNNAlexNet
from .voc import VOCDataset, VOC_CLASSES


def fine_tune_cnn(
    dataset: VOCDataset,
    proposals: dict[str, np.ndarray],
    out_path: str | Path,
    image_size: int = 227,
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    log_interval: int = 100,
    grad_clip_norm: float | None = 10.0,
    positive_iou: float = 0.5,
    negative_iou: float = 0.5,
    max_regions_per_image: int = 128,
    max_images: int | None = None,
    device: str | torch.device = "cpu",
) -> Path:
    train_ds = FineTuneDataset(
        dataset,
        proposals,
        positive_iou=positive_iou,
        negative_iou=negative_iou,
        max_regions_per_image=max_regions_per_image,
        image_size=image_size,
        max_images=max_images,
    )
    if len(train_ds) == 0:
        raise RuntimeError("No fine-tuning samples found. Check proposals and VOC annotations.")
    model = RCNNAlexNet(num_classes=len(VOC_CLASSES) + 1, pretrained=True).to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    print(f"[cnn] samples={len(train_ds)} batch_size={batch_size} epochs={epochs} lr={lr}", flush=True)
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for batch_idx, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite CNN loss at epoch={epoch + 1} batch={batch_idx}. "
                    "The run was stopped before saving a bad checkpoint."
                )
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()
            total_loss += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            total += len(y)
            if log_interval and (batch_idx % log_interval == 0):
                print(
                    f"[cnn] epoch {epoch + 1}/{epochs} batch={batch_idx}/{len(loader)} "
                    f"loss={total_loss / total:.4f} acc={correct / total:.3f}",
                    flush=True,
                )
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        print(f"[cnn] epoch {epoch + 1}/{epochs} done loss={epoch_loss:.4f} acc={epoch_acc:.3f}", flush=True)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "loss": epoch_loss, "acc": epoch_acc}, out_path.with_suffix(f".epoch{epoch + 1}.pt"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out_path)
    return out_path


def train_svms(
    dataset: VOCDataset,
    proposals: dict[str, np.ndarray],
    model: RCNNAlexNet,
    out_path: str | Path,
    image_size: int = 227,
    batch_size: int = 64,
    svm_c: float = 0.001,
    negative_iou: float = 0.3,
    positive_iou: float = 0.5,
    hard_negative_iterations: int = 3,
    negatives_per_image_per_class: int = 20,
    max_positives_per_class: int | None = None,
    max_negatives_per_class: int | None = None,
    max_images: int | None = None,
    device: str | torch.device = "cpu",
) -> Path:
    class_pos = {i: [] for i in range(len(VOC_CLASSES))}
    class_neg = {i: [] for i in range(len(VOC_CLASSES))}
    for idx, item in enumerate(dataset):
        if max_images is not None and idx >= max_images:
            break
        img = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gt_boxes = np.asarray([obj.bbox for obj in item.objects if not obj.difficult], dtype=np.float32)
        gt_labels = np.asarray([obj.class_id for obj in item.objects if not obj.difficult], dtype=np.int64)
        if len(gt_boxes) == 0:
            continue
        boxes = proposals.get(item.image_id, np.empty((0, 4), dtype=np.float32))
        boxes = np.vstack([boxes, gt_boxes])
        feats = extract_regions(model, img, boxes, layer="fc7", image_size=image_size, batch_size=batch_size, device=device)
        for box, feat in zip(boxes, feats):
            overlaps = iou(tuple(box), gt_boxes)
            best = int(np.argmax(overlaps))
            for cls_id in range(len(VOC_CLASSES)):
                if gt_labels[best] == cls_id and overlaps[best] >= positive_iou:
                    class_pos[cls_id].append(feat)
                elif overlaps[best] < negative_iou:
                    class_neg[cls_id].append(feat)
        for cls_id in range(len(VOC_CLASSES)):
            neg_limit = (idx + 1) * negatives_per_image_per_class
            if max_negatives_per_class is not None:
                neg_limit = min(neg_limit, max_negatives_per_class)
            if len(class_neg[cls_id]) > neg_limit:
                class_neg[cls_id] = class_neg[cls_id][-neg_limit:]
            if max_positives_per_class is not None and len(class_pos[cls_id]) > max_positives_per_class:
                class_pos[cls_id] = class_pos[cls_id][-max_positives_per_class:]
        if (idx + 1) % 50 == 0:
            print(f"[svm samples] {idx + 1}/{len(dataset)}", flush=True)
    svms = {}
    for cls_id, name in enumerate(VOC_CLASSES):
        pos = np.asarray(class_pos[cls_id], dtype=np.float32)
        neg = np.asarray(class_neg[cls_id], dtype=np.float32)
        if len(pos) == 0 or len(neg) == 0:
            svms[name] = None
            print(f"[svm] {name}: skipped pos={len(pos)} neg={len(neg)}")
            continue
        neg_pool = neg
        svm = None
        for it in range(max(1, hard_negative_iterations)):
            n_neg = min(len(neg_pool), max(len(pos) * 3, 100))
            choice = np.random.choice(len(neg_pool), size=n_neg, replace=False)
            x = np.vstack([pos, neg_pool[choice]])
            y = np.hstack([np.ones(len(pos)), -np.ones(n_neg)])
            svm = LinearSVM(c=svm_c, epochs=20, batch_size=256)
            svm.fit(x, y)
            scores = svm.decision_function(neg)
            neg_pool = neg[np.argsort(scores)[-n_neg:]]
        svms[name] = svm
        print(f"[svm] {name}: pos={len(pos)} neg={len(neg)}", flush=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(svms, f)
    return out_path


def train_bbox_regressors(
    dataset: VOCDataset,
    proposals: dict[str, np.ndarray],
    model: RCNNAlexNet,
    out_path: str | Path,
    image_size: int = 227,
    batch_size: int = 64,
    bbox_iou: float = 0.6,
    max_images: int | None = None,
    device: str | torch.device = "cpu",
) -> Path:
    samples = {name: {"x": [], "y": []} for name in VOC_CLASSES}
    for idx, item in enumerate(dataset):
        if max_images is not None and idx >= max_images:
            break
        img = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gt_boxes = np.asarray([obj.bbox for obj in item.objects if not obj.difficult], dtype=np.float32)
        gt_labels = np.asarray([obj.class_id for obj in item.objects if not obj.difficult], dtype=np.int64)
        if len(gt_boxes) == 0:
            continue
        boxes = proposals.get(item.image_id, np.empty((0, 4), dtype=np.float32))
        if len(boxes) == 0:
            continue
        keep_boxes = []
        targets = []
        names = []
        for box in boxes:
            overlaps = iou(tuple(box), gt_boxes)
            best = int(np.argmax(overlaps))
            if overlaps[best] >= bbox_iou:
                keep_boxes.append(box)
                targets.append(gt_boxes[best])
                names.append(VOC_CLASSES[int(gt_labels[best])])
        if not keep_boxes:
            continue
        keep_arr = np.asarray(keep_boxes, dtype=np.float32)
        target_arr = np.asarray(targets, dtype=np.float32)
        feats = extract_regions(model, img, keep_arr, layer="fc7", image_size=image_size, batch_size=batch_size, device=device)
        deltas = bbox_transform(keep_arr, target_arr)
        for name, feat, delta in zip(names, feats, deltas):
            samples[name]["x"].append(feat)
            samples[name]["y"].append(delta)
    regressors = {}
    for name in VOC_CLASSES:
        x = np.asarray(samples[name]["x"], dtype=np.float32)
        y = np.asarray(samples[name]["y"], dtype=np.float32)
        if len(x) < 4:
            regressors[name] = None
            print(f"[bbox] {name}: skipped n={len(x)}")
            continue
        reg = RidgeRegressor(alpha=1000.0)
        reg.fit(x, y)
        regressors[name] = reg
        print(f"[bbox] {name}: n={len(x)}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(regressors, f)
    return out_path

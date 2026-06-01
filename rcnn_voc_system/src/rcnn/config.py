from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Paths:
    root: Path
    checkpoints: Path
    proposals: Path
    features: Path
    detections: Path
    reports: Path
    neurons: Path


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_voc_root(cfg: dict[str, Any], voc_root: str | None = None) -> Path:
    value = voc_root or cfg.get("dataset", {}).get("voc_root") or ""
    if not value:
        raise ValueError("VOC root is required. Pass --voc-root or set dataset.voc_root.")
    return Path(value).expanduser().resolve()


def resolve_paths(cfg: dict[str, Any], project_root: str | Path | None = None) -> Paths:
    base = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    out_cfg = cfg.get("outputs", {})
    root = Path(out_cfg.get("root", "outputs"))
    if not root.is_absolute():
        root = base / root
    paths = Paths(
        root=root,
        checkpoints=root / "checkpoints",
        proposals=root / "proposals",
        features=root / "features",
        detections=root / "detections",
        reports=root / "reports",
        neurons=root / "neurons",
    )
    for p in paths.__dict__.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rcnn.config import load_config, resolve_paths, resolve_voc_root, seed_everything
from rcnn.voc import VOCDataset


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/rcnn_voc.yaml")
    parser.add_argument("--voc-root", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--device", default="cpu")


def runtime(args, split_key: str = "image_set_train"):
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("training", {}).get("seed", 42)))
    project_root = Path(args.project_root).resolve() if args.project_root else PROJECT_ROOT
    paths = resolve_paths(cfg, project_root)
    voc_root = resolve_voc_root(cfg, args.voc_root)
    ds_cfg = cfg["dataset"]
    dataset = VOCDataset(voc_root, year=str(ds_cfg.get("year", "2012")), image_set=ds_cfg.get(split_key, "trainval"))
    return cfg, paths, dataset

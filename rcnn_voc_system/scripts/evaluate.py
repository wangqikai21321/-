from __future__ import annotations

import argparse
import json

from common import add_common_args, runtime
from rcnn.detect import detect_dataset, load_pickle
from rcnn.evaluate import evaluate_detections
from rcnn.model import load_model
from rcnn.proposals import load_proposals, proposal_cache_path


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--split", default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    cfg, paths, dataset = runtime(args, split_key="image_set_eval")
    if args.split:
        from rcnn.voc import VOCDataset
        dataset = VOCDataset(dataset.voc_root, dataset.year, args.split)
    m_cfg, p_cfg, d_cfg = cfg["model"], cfg["proposals"], cfg["detection"]
    model = load_model(str(paths.checkpoints / "alexnet_rcnn_finetuned.pt"), pretrained=m_cfg.get("pretrained", True), device=args.device)
    svms = load_pickle(paths.checkpoints / "linear_svms.pkl")
    bbox_path = paths.checkpoints / "bbox_regressors.pkl"
    regressors = load_pickle(bbox_path) if bbox_path.exists() else {}
    cache_path = proposal_cache_path(paths.proposals, dataset.year, dataset.image_set)
    proposals = load_proposals(cache_path) if cache_path.exists() else None
    detections = detect_dataset(
        dataset,
        model,
        svms,
        regressors,
        proposals,
        paths.detections,
        image_size=m_cfg["image_size"],
        proposal_mode=p_cfg["mode"],
        top_k=p_cfg["top_k"],
        min_size=p_cfg["min_size"],
        score_threshold=d_cfg["score_threshold"],
        nms_iou=d_cfg["nms_iou"],
        max_detections_per_class=d_cfg["max_detections_per_class"],
        batch_size=cfg["training"]["batch_size"],
        device=args.device,
    )
    report = evaluate_detections(dataset, detections, iou_threshold=args.iou, use_07_metric=dataset.year == "2007")
    out = paths.reports / f"voc{dataset.year}_{dataset.image_set}_map.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"mAP": report["mAP"], "report": str(out)}, indent=2))


if __name__ == "__main__":
    main()

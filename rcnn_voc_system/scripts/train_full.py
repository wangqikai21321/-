from __future__ import annotations

import argparse
import platform

from common import add_common_args, runtime

platform.machine = lambda: "AMD64"
platform.system = lambda: "Windows"

from rcnn.model import load_model
from rcnn.proposals import generate_proposal_cache, load_proposals, proposal_cache_path
from rcnn.train import fine_tune_cnn, train_bbox_regressors, train_svms


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-svm", action="store_true")
    parser.add_argument("--skip-bbox", action="store_true")
    args = parser.parse_args()
    cfg, paths, dataset = runtime(args, split_key="image_set_train")
    p_cfg, t_cfg, m_cfg = cfg["proposals"], cfg["training"], cfg["model"]
    cache = proposal_cache_path(paths.proposals, dataset.year, dataset.image_set)
    if not cache.exists():
        generate_proposal_cache(dataset, paths.proposals, mode=p_cfg["mode"], top_k=p_cfg["top_k"], min_size=p_cfg["min_size"], max_images=t_cfg.get("max_images"))
    proposals = load_proposals(cache)
    print(f"[proposals] loaded {len(proposals)} cached images from {cache}", flush=True)
    cnn_path = paths.checkpoints / "alexnet_rcnn_finetuned.pt"
    if not args.skip_cnn:
        fine_tune_cnn(
            dataset,
            proposals,
            cnn_path,
            image_size=m_cfg["image_size"],
            batch_size=t_cfg["batch_size"],
            epochs=t_cfg["epochs"],
            lr=t_cfg["learning_rate"],
            momentum=t_cfg["momentum"],
            weight_decay=t_cfg["weight_decay"],
            log_interval=t_cfg.get("log_interval", 100),
            grad_clip_norm=t_cfg.get("grad_clip_norm", 10.0),
            positive_iou=t_cfg["positive_iou"],
            negative_iou=t_cfg["negative_iou"],
            max_regions_per_image=t_cfg["max_regions_per_image"],
            max_images=t_cfg.get("max_images"),
            device=args.device,
        )
    model = load_model(str(cnn_path) if cnn_path.exists() else None, pretrained=m_cfg.get("pretrained", True), device=args.device)
    if not args.skip_svm:
        train_svms(
            dataset,
            proposals,
            model,
            paths.checkpoints / "linear_svms.pkl",
            image_size=m_cfg["image_size"],
            batch_size=t_cfg["batch_size"],
            svm_c=t_cfg["svm_c"],
            negative_iou=t_cfg["svm_negative_iou"],
            positive_iou=t_cfg["positive_iou"],
            hard_negative_iterations=t_cfg["hard_negative_iterations"],
            negatives_per_image_per_class=t_cfg.get("svm_negatives_per_image_per_class", 20),
            max_positives_per_class=t_cfg.get("max_svm_positives_per_class"),
            max_negatives_per_class=t_cfg.get("max_svm_negatives_per_class"),
            max_images=t_cfg.get("max_images"),
            device=args.device,
        )
    if not args.skip_bbox:
        train_bbox_regressors(
            dataset,
            proposals,
            model,
            paths.checkpoints / "bbox_regressors.pkl",
            image_size=m_cfg["image_size"],
            batch_size=t_cfg["batch_size"],
            bbox_iou=t_cfg["bbox_iou"],
            max_images=t_cfg.get("max_images"),
            device=args.device,
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import cv2

from common import add_common_args, runtime
from rcnn.detect import detect_image, draw_detections, load_pickle
from rcnn.model import load_model


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("image")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg, paths, _ = runtime(args, split_key="image_set_eval")
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    model = load_model(str(paths.checkpoints / "alexnet_rcnn_finetuned.pt"), pretrained=cfg["model"].get("pretrained", True), device=args.device)
    svms = load_pickle(paths.checkpoints / "linear_svms.pkl")
    bbox_path = paths.checkpoints / "bbox_regressors.pkl"
    regressors = load_pickle(bbox_path) if bbox_path.exists() else {}
    dets = detect_image(img, "uploaded", model, svms, regressors=regressors, image_size=cfg["model"]["image_size"], proposal_mode=cfg["proposals"]["mode"], top_k=cfg["proposals"]["top_k"], min_size=cfg["proposals"]["min_size"], score_threshold=cfg["detection"]["score_threshold"], nms_iou=cfg["detection"]["nms_iou"], max_detections_per_class=cfg["detection"]["max_detections_per_class"], batch_size=cfg["training"]["batch_size"], device=args.device)
    out = args.output or str(paths.detections / "detected.jpg")
    cv2.imwrite(out, draw_detections(img, dets))
    for d in dets:
        print(d)
    print(out)


if __name__ == "__main__":
    main()

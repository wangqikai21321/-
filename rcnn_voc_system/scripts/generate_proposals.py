from __future__ import annotations

import argparse

from common import add_common_args, runtime
from rcnn.proposals import generate_proposal_cache


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--split-key", default="image_set_train")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()
    cfg, paths, dataset = runtime(args, split_key=args.split_key)
    p_cfg = cfg["proposals"]
    path = generate_proposal_cache(
        dataset,
        paths.proposals,
        mode=p_cfg.get("mode", "fast"),
        top_k=int(p_cfg.get("top_k", 2000)),
        min_size=int(p_cfg.get("min_size", 16)),
        max_images=args.max_images or cfg.get("training", {}).get("max_images"),
    )
    print(path)


if __name__ == "__main__":
    main()

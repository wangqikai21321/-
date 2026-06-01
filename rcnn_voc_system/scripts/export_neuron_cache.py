from __future__ import annotations

import argparse
import platform

from common import add_common_args, runtime

platform.machine = lambda: "AMD64"
platform.system = lambda: "Windows"

from rcnn.model import load_model
from rcnn.neuron import render_top_regions, save_hits, scan_top_activations
from rcnn.proposals import load_proposals, proposal_cache_path


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--layer", default="pool5")
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--col", type=int, default=None)
    parser.add_argument("--unit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()
    cfg, paths, dataset = runtime(args, split_key="image_set_eval")
    model = load_model(str(paths.checkpoints / "alexnet_rcnn_finetuned.pt") if (paths.checkpoints / "alexnet_rcnn_finetuned.pt").exists() else None, pretrained=cfg["model"].get("pretrained", True), device=args.device)
    proposals = load_proposals(proposal_cache_path(paths.proposals, dataset.year, dataset.image_set))
    top_k = args.top_k or cfg["visualization"]["top_k"]
    hits = scan_top_activations(dataset, proposals, model, args.layer, top_k=top_k, channel=args.channel, row=args.row, col=args.col, unit=args.unit, image_size=cfg["model"]["image_size"], max_images=args.max_images or cfg["visualization"].get("max_images"), device=args.device)
    stem = f"{args.layer}_c{args.channel}_r{args.row}_c{args.col}_u{args.unit}"
    json_path = save_hits(hits, paths.neurons / f"{stem}.json")
    png_path = render_top_regions(hits, paths.neurons / f"{stem}.png")
    print(json_path)
    print(png_path)


if __name__ == "__main__":
    main()

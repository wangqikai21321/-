from __future__ import annotations

import argparse
import json

from common import add_common_args, runtime
from rcnn.voc import dataset_summary


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--split-key", default="image_set_train")
    args = parser.parse_args()
    _, _, dataset = runtime(args, split_key=args.split_key)
    print(json.dumps(dataset_summary(dataset), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

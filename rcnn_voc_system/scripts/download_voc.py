from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Download Pascal VOC using torchvision.")
    parser.add_argument("--root", default=str(PROJECT_ROOT / "data"), help="Directory that will contain VOCdevkit.")
    parser.add_argument("--year", default="2012", choices=["2007", "2008", "2009", "2010", "2011", "2012"])
    parser.add_argument("--image-set", default="trainval")
    args = parser.parse_args()

    from torchvision.datasets import VOCDetection

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    VOCDetection(root=str(root), year=args.year, image_set=args.image_set, download=True)
    print(root / "VOCdevkit")


if __name__ == "__main__":
    main()

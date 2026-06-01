from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(VOC_CLASSES)}


@dataclass(frozen=True)
class VOCObject:
    name: str
    class_id: int
    bbox: tuple[int, int, int, int]
    difficult: bool = False


@dataclass(frozen=True)
class VOCImage:
    image_id: str
    image_path: Path
    annotation_path: Path | None
    objects: tuple[VOCObject, ...]


class VOCDataset:
    def __init__(self, voc_root: str | Path, year: str = "2012", image_set: str = "trainval"):
        self.voc_root = Path(voc_root)
        self.year = str(year)
        self.image_set = image_set
        self.dataset_dir = self._find_dataset_dir()
        self.image_ids = self._read_image_set()

    def _find_dataset_dir(self) -> Path:
        candidates = [
            self.voc_root / f"VOC{self.year}",
            self.voc_root / "VOCdevkit" / f"VOC{self.year}",
            self.voc_root,
        ]
        for candidate in candidates:
            if (candidate / "JPEGImages").exists() and (candidate / "ImageSets" / "Main").exists():
                return candidate
        raise FileNotFoundError(f"Could not find VOC{self.year} under {self.voc_root}")

    def _read_image_set(self) -> list[str]:
        path = self.dataset_dir / "ImageSets" / "Main" / f"{self.image_set}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing VOC image set file: {path}")
        return [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __iter__(self):
        for image_id in self.image_ids:
            yield self.get(image_id)

    def get(self, image_id: str) -> VOCImage:
        image_path = self.dataset_dir / "JPEGImages" / f"{image_id}.jpg"
        ann_path = self.dataset_dir / "Annotations" / f"{image_id}.xml"
        objects: tuple[VOCObject, ...] = tuple()
        if ann_path.exists():
            objects = tuple(parse_annotation(ann_path))
        return VOCImage(image_id=image_id, image_path=image_path, annotation_path=ann_path if ann_path.exists() else None, objects=objects)


def parse_annotation(path: str | Path) -> list[VOCObject]:
    root = ET.parse(path).getroot()
    objects: list[VOCObject] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_TO_IDX:
            continue
        difficult = obj.findtext("difficult", "0").strip() == "1"
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = int(float(bbox.findtext("xmin", "0"))) - 1
        ymin = int(float(bbox.findtext("ymin", "0"))) - 1
        xmax = int(float(bbox.findtext("xmax", "0"))) - 1
        ymax = int(float(bbox.findtext("ymax", "0"))) - 1
        objects.append(VOCObject(name=name, class_id=CLASS_TO_IDX[name], bbox=(xmin, ymin, xmax, ymax), difficult=difficult))
    return objects


def dataset_summary(dataset: VOCDataset) -> dict[str, object]:
    counts = {name: 0 for name in VOC_CLASSES}
    difficult = 0
    annotated = 0
    for item in dataset:
        if item.annotation_path is not None:
            annotated += 1
        for obj in item.objects:
            counts[obj.name] += 1
            difficult += int(obj.difficult)
    return {
        "year": dataset.year,
        "split": dataset.image_set,
        "images": len(dataset),
        "annotated_images": annotated,
        "objects": sum(counts.values()),
        "difficult_objects": difficult,
        "per_class": counts,
        "dataset_dir": str(dataset.dataset_dir),
    }

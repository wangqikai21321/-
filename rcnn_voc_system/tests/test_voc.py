from pathlib import Path

from rcnn.voc import VOCDataset, parse_annotation


def write_sample_voc(root: Path):
    base = root / "VOC2012"
    (base / "Annotations").mkdir(parents=True)
    (base / "JPEGImages").mkdir()
    (base / "ImageSets" / "Main").mkdir(parents=True)
    (base / "ImageSets" / "Main" / "train.txt").write_text("000001\n", encoding="utf-8")
    (base / "JPEGImages" / "000001.jpg").write_bytes(b"fake")
    (base / "Annotations" / "000001.xml").write_text(
        """
<annotation>
  <object><name>cat</name><difficult>0</difficult><bndbox>
    <xmin>1</xmin><ymin>2</ymin><xmax>11</xmax><ymax>12</ymax>
  </bndbox></object>
</annotation>
""",
        encoding="utf-8",
    )
    return base


def test_parse_voc_annotation(tmp_path):
    base = write_sample_voc(tmp_path)
    objs = parse_annotation(base / "Annotations" / "000001.xml")
    assert objs[0].name == "cat"
    assert objs[0].bbox == (0, 1, 10, 11)
    ds = VOCDataset(tmp_path, "2012", "train")
    item = ds.get("000001")
    assert item.image_id == "000001"
    assert len(item.objects) == 1

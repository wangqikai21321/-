import numpy as np
import torch

from rcnn.detect import Detection, write_voc_results
from rcnn.model import RCNNAlexNet


def test_alexnet_fc7_shape():
    model = RCNNAlexNet(pretrained=False)
    model.eval()
    with torch.no_grad():
        feat = model.extract(torch.randn(1, 3, 227, 227), "fc7")["fc7"]
    assert tuple(feat.shape) == (1, 4096)


def test_voc_result_format(tmp_path):
    dets = [Detection("000001", "cat", 0.75, (0, 1, 10, 11))]
    write_voc_results(dets, tmp_path)
    text = (tmp_path / "comp3_det_test_cat.txt").read_text(encoding="utf-8").strip()
    assert text == "000001 0.750000 1.0 2.0 11.0 12.0"

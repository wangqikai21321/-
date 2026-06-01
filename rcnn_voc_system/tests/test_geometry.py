import numpy as np

from rcnn.geometry import bbox_inverse, bbox_transform, iou, nms


def test_iou_and_nms():
    boxes = np.array([[0, 0, 99, 99], [50, 50, 149, 149], [200, 200, 250, 250]], dtype=np.float32)
    vals = iou((0, 0, 99, 99), boxes)
    assert vals[0] == 1.0
    assert 0.14 < vals[1] < 0.15
    assert vals[2] == 0.0
    keep = nms(boxes, np.array([1.0, 0.9, 0.8], dtype=np.float32), 0.3)
    assert keep == [0, 1, 2]


def test_bbox_transform_round_trip():
    proposals = np.array([[10, 10, 49, 49], [20, 30, 80, 100]], dtype=np.float32)
    gt = np.array([[12, 14, 51, 53], [18, 33, 84, 99]], dtype=np.float32)
    deltas = bbox_transform(proposals, gt)
    recovered = bbox_inverse(proposals, deltas)
    assert np.allclose(recovered, gt, atol=1e-4)

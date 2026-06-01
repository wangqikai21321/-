from rcnn.metrics import precision_recall_ap


def test_precision_recall_ap():
    recalls, precisions, ap = precision_recall_ap([1, 0, 1], [0.9, 0.8, 0.7], total_positives=2)
    assert recalls.tolist() == [0.5, 0.5, 1.0]
    assert precisions[0] == 1.0
    assert 0.83 < ap < 0.84

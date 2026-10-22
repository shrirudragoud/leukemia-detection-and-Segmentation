import numpy as np

from leukemia_pp.metrics import dice, iou


def test_dice_iou():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    b = np.zeros((10, 10), bool)
    b[:5, :5] = True
    assert dice(a, a) == 1.0 and iou(a, a) == 1.0
    assert abs(dice(a, b) - 2 * 25 / 75) < 1e-9
    assert abs(iou(a, b) - 0.5) < 1e-9
    assert dice(~a, a) == 0.0
    z = np.zeros((4, 4), bool)
    assert dice(z, z) == 1.0 and iou(z, z) == 1.0

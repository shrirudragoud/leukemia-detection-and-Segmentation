import cv2
import numpy as np
import pytest

BG = (235, 242, 238)        # BGR: pale background
RBC = (200, 175, 165)       # BGR: pale grey-blue red cell
WBC_CYTO = (215, 150, 185)  # BGR: lilac
WBC_NUC = (170, 60, 130)    # BGR: dark purple


def make_field(wbcs=(), rbcs=(), size=224, nucleus=True, seed=0, ellipses=()):
    """Synthetic smear. wbcs/rbcs: (y, x, radius). ellipses: (y, x, ax_major, ax_minor)."""
    img = np.empty((size, size, 3), np.uint8)
    img[:] = BG
    for y, x, r in rbcs:
        cv2.circle(img, (x, y), r, RBC, -1)
    for y, x, r in wbcs:
        cv2.circle(img, (x, y), r, WBC_CYTO, -1)
        if nucleus:
            cv2.circle(img, (x, y), int(r * 0.65), WBC_NUC, -1)
    for y, x, a, b in ellipses:
        cv2.ellipse(img, (x, y), (a, b), 0, 0, 360, WBC_CYTO, -1)
    noise = np.random.RandomState(seed).randint(-3, 4, img.shape)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def field():
    return make_field

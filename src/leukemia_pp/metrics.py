"""Segmentation metrics, for use once ground-truth masks are available."""
from __future__ import annotations

import numpy as np


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = pred.astype(bool), truth.astype(bool)
    total = pred.sum() + truth.sum()
    return 1.0 if total == 0 else float(2.0 * (pred & truth).sum() / total)


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = pred.astype(bool), truth.astype(bool)
    union = (pred | truth).sum()
    return 1.0 if union == 0 else float((pred & truth).sum() / union)

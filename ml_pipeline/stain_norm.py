"""Reinhard color normalization in LAB space.
Matches the mean/std of every input to a reference distribution — removes
scanner/slide colour drift so downstream models don't overfit to hue."""
import numpy as np
import cv2


def _lab_stats(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab.mean(axis=(0, 1)), lab.std(axis=(0, 1))


def fit_reference(image_paths):
    """Average LAB (mean, std) across the given reference images."""
    means, stds = [], []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        m, s = _lab_stats(img)
        means.append(m)
        stds.append(s)
    if not means:
        raise ValueError("No valid reference images found for stain fitting")
    return np.mean(means, axis=0), np.mean(stds, axis=0)


def normalize(bgr, ref_mean, ref_std):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean, src_std = lab.mean(axis=(0, 1)), lab.std(axis=(0, 1))
    out = (lab - src_mean) / (src_std + 1e-8) * ref_std + ref_mean
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

"""Per-image quality control: measurements plus human-readable flags.

Flags never drop an image; they are recorded so downstream code (or a human) can decide.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import QCConfig
from .segmentation import Segmentation
from .stain import bgr_to_lab


def measure(raw_bgr: np.ndarray, seg: Segmentation) -> dict[str, float]:
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(bgr_to_lab(raw_bgr)[..., 0].mean()),
        "clipped_frac": float((raw_bgr >= 254).all(axis=2).mean()),
        "wbc_area_frac": float(seg.cell_mask.mean()),
    }


def flags(m: dict[str, float], seg: Segmentation, n_border: int, cfg: QCConfig) -> list[str]:
    out = []
    if m["sharpness"] < cfg.min_sharpness:
        out.append("blurry")
    if m["brightness"] < cfg.min_brightness:
        out.append("dark")
    if seg.n_cells == 0:
        out.append("no_cells")
    if m["wbc_area_frac"] > cfg.max_wbc_fraction:
        out.append("wbc_fraction_implausible")
    if seg.n_cells and n_border / seg.n_cells > cfg.max_border_cell_frac:
        out.append("mostly_border_cells")
    if seg.n_cells and not any(i.found for i in seg.nucleus_info.values()):
        out.append("no_nuclei_resolved")
    return out

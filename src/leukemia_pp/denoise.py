"""Edge-preserving denoising.

Bilateral filtering is the default: microscopy noise here is additive, not multiplicative
speckle, which is what the HDS PDE of the prototype assumes (it diverges on this data; see
docs/paper.pdf). Non-local means is offered as a stronger alternative.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import DenoiseConfig


def denoise(bgr: np.ndarray, cfg: DenoiseConfig) -> np.ndarray:
    if cfg.method == "none":
        return bgr.copy()
    if cfg.method == "bilateral":
        return cv2.bilateralFilter(bgr, cfg.bilateral_d, cfg.bilateral_sigma_color,
                                   cfg.bilateral_sigma_space)
    return cv2.fastNlMeansDenoisingColored(bgr, None, cfg.nlm_h, cfg.nlm_h_color, 7, 21)

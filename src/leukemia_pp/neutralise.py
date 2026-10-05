"""Shortcut removal: a 'clean' view of the image that contains only the WBCs.

Everything outside the dilated cell mask (background, RBCs, debris, scale bar / text labels)
is replaced by one constant neutral colour, so a network cannot classify on slide background
tint, scale-bar style or RBC appearance. With `white_balance`, the colour cast of the slide is
also removed from the *cells*: a*/b* are shifted and L* scaled so the measured background
becomes the neutral fill colour.

Two outputs, so colour can be ablated: `rgb_clean` (uint8 RGB) and `gray_clean`
(L*/100, float32). Cell colour can still carry stain/session information even after white
balance: quantify it with `leukemia-pp audit`, and consider training on `gray_clean` and/or
with colour augmentation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .config import NeutraliseConfig
from .segmentation import Segmentation
from .stain import bgr_to_lab, lab_to_bgr


def keep_mask(seg: Segmentation, cfg: NeutraliseConfig) -> np.ndarray:
    mask = seg.cell_mask
    if cfg.dilate_px > 0 and mask.any():
        mask = ndi.binary_dilation(mask, iterations=cfg.dilate_px)
    return mask


def background_colour(lab: np.ndarray, keep: np.ndarray, min_px: int) -> np.ndarray | None:
    """Median Lab of the brighter half of the non-cell pixels (slide background, not RBC rims
    or debris); None if there is too little background to estimate it."""
    bg = ~keep
    if int(bg.sum()) < min_px:
        return None
    L = lab[..., 0]
    candidate = bg & (L >= np.percentile(L[bg], 50))
    return np.median(lab[candidate], axis=0)


def neutralise(norm_bgr: np.ndarray, seg: Segmentation, cfg: NeutraliseConfig
               ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb_clean uint8 HxWx3, gray_clean float32 HxW)."""
    lab = bgr_to_lab(norm_bgr)
    keep = keep_mask(seg, cfg)
    if cfg.white_balance:
        bg = background_colour(lab, keep, cfg.min_background_px)
        if bg is not None:
            gain = float(np.clip(cfg.fill_L / max(float(bg[0]), 1e-3), *cfg.gain_clip))
            lab[..., 0] = np.clip(lab[..., 0] * gain, 0.0, 100.0)
            lab[..., 1] -= bg[1]
            lab[..., 2] -= bg[2]
    lab[~keep] = (cfg.fill_L, 0.0, 0.0)
    rgb = np.ascontiguousarray(lab_to_bgr(lab)[..., ::-1])
    return rgb, (lab[..., 0] / 100.0).astype(np.float32)

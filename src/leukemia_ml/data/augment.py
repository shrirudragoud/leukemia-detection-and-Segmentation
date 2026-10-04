"""Training-time augmentation on uint8 HxWx3 crops (numpy; deterministic given the RNG).

Design rule: never alter cell *shape* (elastic/perspective/strong crops are excluded) because
shape is signal; vary what is nuisance: orientation, scale (slightly), stain colour, focus.
Random grayscale and stain jitter specifically discourage reliance on stain colour, which the
shortcut audit showed predicts the class on this dataset.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.color import hed2rgb, rgb2hed

from ..config import AugConfig

FILL = 232


def hed_jitter(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Per-image random scale/shift of the haematoxylin-eosin-DAB stain channels (Tellez-style)."""
    hed = rgb2hed(img)
    alpha = rng.uniform(1 - strength * 4, 1 + strength * 4, 3)
    beta = rng.uniform(-strength, strength, 3)
    out = hed2rgb(hed * alpha + beta)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def hsv_jitter(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-strength, strength) * 180) % 180
    hsv[..., 1] *= rng.uniform(1 - strength * 4, 1 + strength * 4)
    hsv[..., 2] *= rng.uniform(1 - strength * 2, 1 + strength * 2)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment(img: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    out = img
    if cfg.flips:
        if rng.random() < 0.5:
            out = out[:, ::-1]
        if rng.random() < 0.5:
            out = out[::-1]
    if cfg.rot90:
        out = np.rot90(out, int(rng.integers(0, 4)))
    out = np.ascontiguousarray(out)
    if cfg.scale_jitter > 0:
        s = 1.0 + rng.uniform(-cfg.scale_jitter, cfg.scale_jitter)
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, s)
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(FILL, FILL, FILL))
    if cfg.colour in ("hed", "hsv"):
        # Jitter stain colour on tissue pixels only: the neutral fill must stay EXACTLY neutral,
        # otherwise train crops get a different background from (un-augmented) test crops.
        tissue = (np.abs(out.astype(np.int16) - FILL) > 3).any(axis=2)
        jittered = (hed_jitter if cfg.colour == "hed" else hsv_jitter)(out, rng, cfg.colour_strength)
        out = np.where(tissue[..., None], jittered, out)
    if cfg.grayscale_p > 0 and rng.random() < cfg.grayscale_p:
        g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        out = np.repeat(g[..., None], 3, axis=2)
    if cfg.blur_p > 0 and rng.random() < cfg.blur_p:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, 1.0)))
    return out

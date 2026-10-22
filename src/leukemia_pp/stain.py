"""Reinhard colour normalisation in true (float) CIELAB.

Improvements over the textbook version used in v1:
  * statistics come from *foreground* pixels (cells), not the whole frame, so the amount of
    empty background in a field no longer changes how strongly cell colour is rescaled;
  * the per-channel gain is clipped, and frames with too little foreground are left
    unchanged (and flagged) instead of being normalised on unreliable statistics;
  * the fitted reference is a first-class object that is saved/loaded, so validation, test
    and future inference images are normalised with the *training* reference.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_otsu

from .config import StainConfig


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR -> float32 Lab with L in [0,100], a/b signed (neutral = 0)."""
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(np.rint(bgr * 255.0), 0, 255).astype(np.uint8)


def otsu(values: np.ndarray) -> float:
    """Otsu threshold that tolerates constant input."""
    values = np.asarray(values).ravel()
    if values.size == 0 or float(values.max() - values.min()) < 1e-6:
        return float(values.mean()) if values.size else 0.0
    return float(threshold_otsu(values))


MIN_L_SPREAD = 1.0   # L* std below this = blank/flat frame, nothing to normalise
ANNOTATION_L = 8.0   # near-black pixels (scale bars, label text) are overlays, not tissue


def foreground_stats(lab: np.ndarray, min_frac: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-channel (mean, std) over tissue pixels darker than the Otsu split of L*.

    Near-black pixels (L* < ANNOTATION_L) are annotation overlays such as the black "200 pix"
    scale bar, and are excluded: otherwise the bar's size changes the statistics and thereby
    the colour of every cell in the image, and the bar style is class-specific in this dataset.

    Returns None when the frame is blank or the foreground covers less than `min_frac` of it.
    There is deliberately no whole-frame fallback: the reference is a *foreground* statistic,
    so matching a sparse field's mostly-background statistics to it would map pale background
    onto cell colour."""
    L = lab[..., 0]
    valid = L >= ANNOTATION_L
    if int(valid.sum()) < 0.5 * L.size or float(L[valid].std()) < MIN_L_SPREAD:
        return None
    fg = valid & (L < otsu(L[valid]))
    if fg.mean() < min_frac:
        return None
    px = lab[fg]
    return px.mean(axis=0), px.std(axis=0)


@dataclass(frozen=True)
class ReinhardReference:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    n_images: int

    def to_dict(self) -> dict:
        return {"method": "reinhard-lab-foreground", "mean": list(self.mean),
                "std": list(self.std), "n_images": self.n_images}

    @classmethod
    def from_dict(cls, d: dict) -> ReinhardReference:
        return cls(tuple(d["mean"]), tuple(d["std"]), int(d["n_images"]))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ReinhardReference:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_reference(images: Iterable[np.ndarray], cfg: StainConfig) -> ReinhardReference:
    means, stds = [], []
    for bgr in images:
        stats = foreground_stats(bgr_to_lab(bgr), cfg.min_foreground_frac)
        if stats is None:          # too sparse to be informative; would bias the reference
            continue
        means.append(stats[0])
        stds.append(stats[1])
    if not means:
        raise ValueError("no usable reference images (all blank or with too little foreground)")
    return ReinhardReference(
        mean=tuple(float(x) for x in np.mean(means, axis=0)),
        std=tuple(float(x) for x in np.mean(stds, axis=0)),
        n_images=len(means),
    )


def normalize(bgr: np.ndarray, ref: ReinhardReference, cfg: StainConfig
              ) -> tuple[np.ndarray, bool]:
    """Returns (image, applied). `applied` is False, and the image is returned unchanged,
    when the frame has too little foreground to estimate its colour statistics."""
    lab = bgr_to_lab(bgr)
    stats = foreground_stats(lab, cfg.min_foreground_frac)
    if stats is None:
        return bgr.copy(), False
    src_mean, src_std = stats
    gain = np.clip(np.asarray(ref.std) / np.maximum(src_std, 1e-3), *cfg.scale_clip)
    out = (lab - src_mean) * gain + np.asarray(ref.mean)
    out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)
    return lab_to_bgr(out), True

"""Bidimensional Ensemble EMD (BEEMD) and pure-IMF selection.

Empirical mode decomposition splits an image into intrinsic mode functions (IMFs), ordered
from the finest spatial scale to the coarsest, plus a smooth residue:

        x = sum_k imf_k + residue                       (exact by construction)

Implementation notes (choices, since the reference papers differ in detail)
  * Envelopes: the fast order-statistic-filter BEMD (Bhuiyan et al., 2008): upper/lower
    envelope = mean-filtered max/min filter over a window whose size comes from the spacing of
    the detected extrema. No scattered-data interpolation, so it is fast and deterministic.
  * Sifting stops at `sd_tol` (normalised squared change) or `max_sifts`.
  * Ensemble: `ensemble` noise realisations, in antithetic +/- pairs so the injected noise
    cancels in the mean; the k-th IMFs are averaged. Seeded -> reproducible.
  * `residue` is defined as x - sum(mean IMFs), so reconstruction is exact regardless of how
    many modes each realisation produced.

Pure-IMF selection (`select_pure_imfs`) is an interpretable scale/energy rule: a mode is
"pure" detail if its characteristic spatial period lies in [min_scale_px, max_scale_px]
(rejects pixel noise and illumination drift) and it carries at least `min_energy_frac` of the
detail energy. Characteristic period = 1 / radial spectral centroid. If your paper defines a
different criterion, replace this one function; the rest of the pipeline only needs the mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .config import BEEMDConfig


def _extrema(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean maps of strict-ish local maxima / minima (3x3 neighbourhood)."""
    mx = (h == ndi.maximum_filter(h, size=3)) & (h > ndi.minimum_filter(h, size=3))
    mn = (h == ndi.minimum_filter(h, size=3)) & (h < ndi.maximum_filter(h, size=3))
    return mx, mn


def _nn_spacing(mask: np.ndarray) -> float:
    pts = np.argwhere(mask)
    if len(pts) < 3:
        return float("inf")
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(d[:, 1]))


def _window(mx: np.ndarray, mn: np.ndarray) -> int | None:
    """Odd window size from the finer of the two extrema spacings; None if too few extrema."""
    spacing = min(_nn_spacing(mx), _nn_spacing(mn))
    if not np.isfinite(spacing):
        return None
    w = int(round(spacing)) | 1                     # force odd
    return max(w, 3)


def _sift(x: np.ndarray, max_sifts: int, sd_tol: float) -> np.ndarray | None:
    """Extract one IMF from `x`; None if x has too few extrema to be decomposed further."""
    h = x.copy()
    imf = None
    for _ in range(max_sifts):
        mx, mn = _extrema(h)
        w = _window(mx, mn)
        if w is None:
            break
        upper = ndi.uniform_filter(ndi.maximum_filter(h, size=w), size=w, mode="reflect")
        lower = ndi.uniform_filter(ndi.minimum_filter(h, size=w), size=w, mode="reflect")
        h_new = h - 0.5 * (upper + lower)
        denom = float((h ** 2).sum()) + 1e-12
        sd = float(((h - h_new) ** 2).sum()) / denom
        h, imf = h_new, h_new
        if sd < sd_tol:
            break
    return imf


def emd(x: np.ndarray, n_imfs: int, max_sifts: int = 8, sd_tol: float = 0.2) -> list[np.ndarray]:
    """Plain BEMD: up to `n_imfs` IMFs, finest first."""
    r = np.asarray(x, dtype=np.float64).copy()
    imfs: list[np.ndarray] = []
    for _ in range(n_imfs):
        if float(r.std()) < 1e-9:
            break
        imf = _sift(r, max_sifts, sd_tol)
        if imf is None:
            break
        imfs.append(imf)
        r = r - imf
    return imfs


@dataclass
class Decomposition:
    imfs: np.ndarray        # (K, H, W) float32, finest first (K may be < cfg.n_imfs)
    residue: np.ndarray     # (H, W) float32, x - sum(imfs)
    selected: np.ndarray    # (K,) bool, pure-IMF mask
    scales: np.ndarray      # (K,) characteristic spatial period in px
    energy_frac: np.ndarray  # (K,) share of total detail energy

    @property
    def pure(self) -> np.ndarray:
        """Sum of the selected (pure) IMFs: a band-passed detail image (zero-mean-ish)."""
        if not self.selected.any():
            return np.zeros_like(self.residue)
        return self.imfs[self.selected].sum(axis=0)


def beemd(x: np.ndarray, cfg: BEEMDConfig) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble EMD. Returns (imfs (K,H,W), residue (H,W)) as float32."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(cfg.seed)
    amp = cfg.noise_std * float(x.std())
    n_real = cfg.ensemble + (cfg.ensemble % 2)       # antithetic pairs need an even count
    acc = np.zeros((cfg.n_imfs,) + x.shape)
    counts = np.zeros(cfg.n_imfs)
    for i in range(n_real):
        if i % 2 == 0:
            noise = rng.standard_normal(x.shape) * amp
        modes = emd(x + (noise if i % 2 == 0 else -noise), cfg.n_imfs, cfg.max_sifts, cfg.sd_tol)
        for k, m in enumerate(modes):
            acc[k] += m
            counts[k] += 1
    imfs = acc / n_real                               # missing modes contribute zero
    keep = counts > 0
    imfs = imfs[keep]
    residue = x - imfs.sum(axis=0) if len(imfs) else x.copy()
    return imfs.astype(np.float32), residue.astype(np.float32)


def characteristic_scale(img: np.ndarray) -> float:
    """Spatial period (px) of the radial spectral centroid; inf for an all-zero image."""
    # Hann window: without it the FFT treats the image as periodic, and a slow ramp (illumination
    # drift) acquires a wrap-around jump whose leakage makes it look like a fine-scale mode.
    window = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    f = np.abs(np.fft.fft2((img - img.mean()) * window)) ** 2
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    radius = np.hypot(fy, fx)
    total = float(f.sum())
    if total <= 1e-18:
        return float("inf")
    centroid = float((f * radius).sum() / total)
    return 1.0 / centroid if centroid > 0 else float("inf")


def select_pure_imfs(imfs: np.ndarray, cfg: BEEMDConfig
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (selected mask, characteristic scales in px, energy fractions)."""
    k = len(imfs)
    if k == 0:
        return np.zeros(0, bool), np.zeros(0), np.zeros(0)
    energy = np.array([float((m ** 2).sum()) for m in imfs])
    frac = energy / max(energy.sum(), 1e-18)
    scales = np.array([characteristic_scale(m) for m in imfs])
    selected = (scales >= cfg.min_scale_px) & (scales <= cfg.max_scale_px) \
        & (frac >= cfg.min_energy_frac)
    return selected, scales, frac


def decompose(x: np.ndarray, cfg: BEEMDConfig) -> Decomposition:
    imfs, residue = beemd(x, cfg)
    selected, scales, frac = select_pure_imfs(imfs, cfg)
    return Decomposition(imfs, residue, selected, scales, frac)

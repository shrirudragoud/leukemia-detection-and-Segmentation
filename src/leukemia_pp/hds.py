"""Stabilised Hybrid Diffusion-Steered (HDS) denoising + edge indicator.

The Block-2 prototype (and v1 `ml_pipeline/core.py`) uses the conduction coefficient

    c(s) = g * ( 1/(1 + s^2/h^2)  +  1/(|s| + eps) )          (Perona-Malik + total variation)

The TV term reaches 1/eps = 1e8 in flat regions, so the explicit update
u += dt*(div(c grad u) - lambda*...) violates the stability bound dt <= 1/(4*c_max) by
orders of magnitude and diverges on non-speckle noise (documented in docs/paper.pdf).

This version keeps the *hybrid* idea but is unconditionally stable:
  * both diffusivities are normalised to (0, 1]:
        PM(s) = 1/(1 + (s/h)^2)            TV_beta(s) = beta / sqrt(s^2 + beta^2)   (Charbonnier)
        c(s)  = w*PM(s) + (1 - w)*TV_beta(s)      so c_max = 1 and dt <= 0.25 is stable;
  * the data term is the additive-noise fidelity  -lambda*(u - f)  (the legacy log-prior is a
    multiplicative-speckle model and is the second source of divergence on this data);
  * iteration stops early once the mean update falls below `tol`.
Because every update is a convex combination of neighbouring values plus a pull towards f,
the discrete maximum principle holds: output stays within [min(f), max(f)].

The edge indicator is 1 - mean_k c(|grad_k u|) of the *denoised* image: ~0 in flat
regions, approaching 1 across strong edges, on a fixed (not per-image) scale.
"""
from __future__ import annotations

import numpy as np

from .config import HDSConfig


def _diffusivity(s: np.ndarray, h: float, beta: float, w: float) -> np.ndarray:
    return w / (1.0 + (s / h) ** 2) + (1.0 - w) * beta / np.sqrt(s * s + beta * beta)


def _neighbour_diffs(u: np.ndarray):
    p = np.pad(u, 1, mode="edge")
    return (p[:-2, 1:-1] - u, p[2:, 1:-1] - u, p[1:-1, :-2] - u, p[1:-1, 2:] - u)  # N S W E


def hds_denoise(f: np.ndarray, cfg: HDSConfig) -> tuple[np.ndarray, np.ndarray, int]:
    """Denoise a single-channel float image `f` (intensity roughly in [0, 1]).

    Returns (u, edge, n_iterations)."""
    f = np.asarray(f, dtype=np.float64)
    u = f.copy()
    n_done = 0
    for n_done in range(1, cfg.iterations + 1):
        diffs = _neighbour_diffs(u)
        flux = sum(_diffusivity(np.abs(d), cfg.h, cfg.beta, cfg.hybrid_weight) * d for d in diffs)
        u_next = u + cfg.dt * (flux - cfg.lam * (u - f))
        converged = float(np.abs(u_next - u).mean()) < cfg.tol
        u = u_next
        if converged:
            break
    diffs = _neighbour_diffs(u)
    c_mean = sum(_diffusivity(np.abs(d), cfg.h, cfg.beta, cfg.hybrid_weight) for d in diffs) / 4.0
    edge = np.clip(1.0 - c_mean, 0.0, 1.0)
    return u.astype(np.float32), edge.astype(np.float32), n_done

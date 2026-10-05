import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from leukemia_pp.config import HDSConfig
from leukemia_pp.hds import hds_denoise

ROOT = Path(__file__).resolve().parents[1]


def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))


def bench(sigma, seed=0):
    clean = np.full((128, 128), 0.2, np.float32)
    cv2.circle(clean, (64, 64), 30, 0.8, -1)
    cv2.rectangle(clean, (10, 10), (40, 40), 0.5, -1)
    noisy = clean + np.random.RandomState(seed).randn(*clean.shape).astype(np.float32) * sigma
    return clean, noisy.astype(np.float32)


@pytest.mark.parametrize("sigma", [0.05, 0.08, 0.12])
def test_denoises_additive_noise(sigma):
    clean, noisy = bench(sigma)
    u, _, _ = hds_denoise(noisy, HDSConfig())
    assert psnr(u, clean) - psnr(noisy, clean) > 8.0


def test_stable_and_bounded_even_when_run_to_exhaustion():
    """The legacy scheme diverges; this one satisfies the discrete maximum principle."""
    _, noisy = bench(0.3)
    u, e, n = hds_denoise(noisy, HDSConfig(iterations=3000, tol=0.0))
    assert n == 3000 and np.isfinite(u).all() and np.isfinite(e).all()
    assert u.min() >= noisy.min() - 1e-6 and u.max() <= noisy.max() + 1e-6


def test_legacy_hds_collapses_on_the_same_data():
    """Regression anchor for the paper's divergence finding (documents *why* this rewrite)."""
    spec = importlib.util.spec_from_file_location("legacy_core", ROOT / "ml_pipeline" / "core.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    clean, noisy = bench(0.08)
    old = legacy.hybrid_denoise(noisy, 200, 0.05, 0.15, 1.3, 1.0, 1e-8)
    new, _, _ = hds_denoise(noisy, HDSConfig())
    assert psnr(old, clean) < psnr(noisy, clean)             # legacy makes it worse
    assert psnr(new, clean) > psnr(noisy, clean) + 8


def test_constant_image_is_a_fixed_point_with_no_edges():
    flat = np.full((32, 32), 0.4, np.float32)
    u, e, _ = hds_denoise(flat, HDSConfig())
    assert np.allclose(u, 0.4, atol=1e-6) and e.max() < 1e-6


def test_edge_indicator_fixed_scale_and_localised():
    clean, _ = bench(0.0)
    _, e, _ = hds_denoise(clean, HDSConfig())
    assert 0.0 <= e.min() and e.max() <= 1.0
    assert e[34:39, 60:68].max() > 0.2          # the circle boundary
    assert e[100:120, 100:120].max() < 1e-3     # flat background
    # same absolute scale for a faint edge: weaker, not stretched back up to the same value
    faint = np.full((64, 64), 0.5, np.float32)
    faint[:, 32:] = 0.52
    assert hds_denoise(faint, HDSConfig())[1].max() < e.max()


def test_early_stopping():
    _, noisy = bench(0.05)
    _, _, n = hds_denoise(noisy, HDSConfig(iterations=500, tol=1e-2))
    assert n < 500


@pytest.mark.parametrize("bad", [{"dt": 0.3}, {"hybrid_weight": 1.5}, {"h": 0.0}, {"beta": -1}])
def test_invalid_params_rejected(bad):
    with pytest.raises(ValueError):
        HDSConfig(**bad)

import cv2
import numpy as np

from leukemia_pp.config import StainConfig
from leukemia_pp.stain import (ReinhardReference, bgr_to_lab, fit_reference, foreground_stats,
                               normalize)


def tinted(img, db, dg, dr):
    return np.clip(img.astype(np.int16) + np.array([db, dg, dr]), 0, 255).astype(np.uint8)


def test_normalisation_removes_global_tint(field):
    cfg = StainConfig()
    base = field(wbcs=[(60, 60, 20), (150, 150, 18)], rbcs=[(100, 100, 12)])
    ref = fit_reference([base], cfg)
    shifted = tinted(base, 25, -10, -20)          # simulated colour cast

    def dist(img):
        m, _ = foreground_stats(bgr_to_lab(img), cfg.min_foreground_frac)
        return np.linalg.norm(m - np.asarray(ref.mean))
    assert dist(shifted) > 8
    out, applied = normalize(shifted, ref, cfg)
    assert applied and dist(out) < 1.0


def test_self_reference_is_near_identity(field):
    cfg = StainConfig()
    img = field(wbcs=[(80, 80, 22)])
    out, applied = normalize(img, fit_reference([img], cfg), cfg)
    assert applied
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 2.0


def test_constant_image_does_not_explode(field):
    cfg = StainConfig()
    ref = fit_reference([field(wbcs=[(80, 80, 22)])], cfg)
    flat = np.full((64, 64, 3), 200, np.uint8)
    out, applied = normalize(flat, ref, cfg)
    assert not applied and np.array_equal(out, flat)


def test_reference_roundtrip(tmp_path, field):
    ref = fit_reference([field(wbcs=[(80, 80, 22)])], StainConfig())
    ref.save(tmp_path / "r.json")
    assert ReinhardReference.load(tmp_path / "r.json") == ref


def test_foreground_not_dominated_by_background(field):
    """Same cell, 3x more empty background (still above the cut-off) -> foreground stats should barely move."""
    cfg = StainConfig()
    small = field(wbcs=[(60, 60, 20)], size=128)
    big = cv2.copyMakeBorder(small, 0, 96, 0, 96, cv2.BORDER_CONSTANT, value=(235, 242, 238))
    m1, _ = foreground_stats(bgr_to_lab(small), cfg.min_foreground_frac)
    m2, _ = foreground_stats(bgr_to_lab(big), cfg.min_foreground_frac)
    assert np.linalg.norm(m1 - m2) < 3.0


def test_sparse_field_is_left_unchanged_not_misnormalised(field):
    """A field whose foreground is below min_foreground_frac must NOT have its background
    mapped onto the (foreground-based) reference: it is returned untouched and flagged."""
    cfg = StainConfig()
    ref = fit_reference([field(wbcs=[(60, 60, 20), (150, 150, 20)])], cfg)
    sparse = field(wbcs=[(60, 60, 8)])                   # ~0.4% of the frame
    out, applied = normalize(sparse, ref, cfg)
    assert not applied and np.array_equal(out, sparse)


def test_no_discontinuity_around_threshold(field):
    """Cells just above / just below the cut-off must not produce wildly different output
    for the shared background (the failure mode of a whole-frame fallback)."""
    cfg = StainConfig()
    ref = fit_reference([field(wbcs=[(60, 60, 20), (150, 150, 20)])], cfg)
    bg = []
    for r in (12, 14):                                    # fg fraction straddles 2%
        img = field(wbcs=[(60, 60, r)])
        out, _ = normalize(img, ref, cfg)
        bg.append(out[200, 200].astype(float))
    assert np.abs(bg[0] - bg[1]).max() < 40               # background stays background-like
    assert min(bg[0].min(), bg[1].min()) > 150

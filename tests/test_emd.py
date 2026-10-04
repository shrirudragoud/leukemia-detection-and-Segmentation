import numpy as np
import pytest

from leukemia_pp.config import BEEMDConfig
from leukemia_pp.emd import beemd, characteristic_scale, decompose, emd, select_pure_imfs

YY, XX = np.mgrid[:128, :128]
FINE = 0.15 * np.sin(2 * np.pi * XX / 4)
MID = 0.5 * np.sin(2 * np.pi * YY / 20)
DRIFT = 0.8 * (XX / 128.0)
SIGNAL = (FINE + MID + DRIFT + 0.5).astype(np.float64)


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_reconstruction_is_exact():
    d = decompose(SIGNAL, BEEMDConfig())
    assert np.abs(d.imfs.sum(0) + d.residue - SIGNAL).max() < 1e-4


def test_modes_are_ordered_fine_to_coarse():
    d = decompose(SIGNAL, BEEMDConfig())
    assert d.scales[0] < d.scales[-1]
    assert corr(d.imfs[0], FINE) > 0.9            # finest mode = fine texture
    assert corr(d.imfs[-1], MID) > 0.9            # coarsest mode = cell-scale structure
    assert corr(d.residue, DRIFT) > 0.5           # slow illumination ends up in the residue


def test_characteristic_scale_recovers_period():
    assert abs(characteristic_scale(MID) - 20) < 3
    assert abs(characteristic_scale(FINE) - 4) < 1
    assert characteristic_scale(np.zeros((16, 16))) == float("inf")


def test_deterministic_and_seed_sensitive():
    a, _ = beemd(SIGNAL, BEEMDConfig(seed=3))
    b, _ = beemd(SIGNAL, BEEMDConfig(seed=3))
    c, _ = beemd(SIGNAL, BEEMDConfig(seed=4))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_pure_selection_rejects_noise_and_drift_bands():
    imfs = np.stack([FINE, MID, DRIFT]).astype(np.float32)
    sel, scales, frac = select_pure_imfs(imfs, BEEMDConfig(min_scale_px=6, max_scale_px=40))
    assert list(sel) == [False, True, False] and abs(frac.sum() - 1) < 1e-6
    sel_all, _, _ = select_pure_imfs(imfs, BEEMDConfig(min_scale_px=2, max_scale_px=500))
    assert sel_all[:2].all()


def test_pure_is_sum_of_selected_modes():
    d = decompose(SIGNAL, BEEMDConfig())
    expect = d.imfs[d.selected].sum(0) if d.selected.any() else np.zeros_like(d.residue)
    assert np.allclose(d.pure, expect)


def test_flat_and_tiny_inputs_do_not_crash():
    assert emd(np.full((32, 32), 0.3), 4) == []
    d = decompose(np.full((32, 32), 0.3), BEEMDConfig())
    assert len(d.imfs) == 0 and not d.pure.any()
    small = decompose(np.random.RandomState(0).rand(24, 24), BEEMDConfig(ensemble=2))
    assert small.imfs.shape[1:] == (24, 24)


def test_ensemble_noise_cancels_in_the_mean():
    """Antithetic pairs: injected noise must not appear as extra energy in the reconstruction."""
    d = decompose(SIGNAL, BEEMDConfig(noise_std=0.3, ensemble=16))
    assert np.abs(d.imfs.sum(0) + d.residue - SIGNAL).max() < 1e-4


def test_config_validation():
    with pytest.raises(ValueError):
        BEEMDConfig(min_scale_px=10, max_scale_px=5)
    with pytest.raises(ValueError):
        BEEMDConfig(ensemble=0)


def test_white_noise_mode_is_rejected_but_cell_scale_structure_is_kept():
    """Calibration anchor: IMF1 of white noise sits at ~2.4 px, below min_scale_px=3.0."""
    rng = np.random.RandomState(0)
    img = 0.5 + MID + rng.randn(128, 128) * 0.08
    d = decompose(img, BEEMDConfig())
    assert d.scales[0] < 3.0 and not d.selected[0]               # noise mode rejected
    assert d.selected[-1] and corr(d.imfs[-1], MID) > 0.8        # structure kept

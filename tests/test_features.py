import math

import numpy as np

from leukemia_pp.config import FeatureConfig, SegmentationConfig
from leukemia_pp.features import (aggregate_image, extract_cell_features, masked_glcm)
from leukemia_pp.segmentation import segment
from leukemia_pp.stain import bgr_to_lab


def rows_for(img):
    seg = segment(bgr_to_lab(img), SegmentationConfig())
    rgb = np.ascontiguousarray(img[..., ::-1])
    return seg, extract_cell_features(rgb, seg, FeatureConfig())


def test_disc_geometry(field):
    seg, rows = rows_for(field(wbcs=[(112, 112, 25)]))
    r = rows[0]
    assert abs(r["cell_area"] - math.pi * 25 ** 2) / (math.pi * 25 ** 2) < 0.05
    assert abs(r["cell_eq_diameter"] - 50) < 2.5
    assert r["cell_circularity"] > 0.85 and r["cell_solidity"] > 0.97
    assert r["cell_aspect_ratio"] < 1.1
    assert 0.3 < r["nc_ratio"] < 0.55
    assert not r["touches_border"] and r["nucleus_found"]


def test_elongated_cell_has_lower_circularity_and_higher_aspect(field):
    _, disc = rows_for(field(wbcs=[(112, 112, 22)]))
    _, ell = rows_for(field(ellipses=[(112, 112, 36, 14)]))
    assert ell[0]["cell_aspect_ratio"] > 2.0 > disc[0]["cell_aspect_ratio"]
    assert ell[0]["cell_circularity"] < disc[0]["cell_circularity"]


def test_border_cells_flagged(field):
    _, rows = rows_for(field(wbcs=[(112, 112, 20), (110, 2, 20)]))
    flags = sorted(r["touches_border"] for r in rows)
    assert flags == [False, True]


def test_missing_nucleus_gives_nan_not_zero(field):
    _, rows = rows_for(field(wbcs=[(112, 112, 24)], nucleus=False))
    r = rows[0]
    assert not r["nucleus_found"]
    assert math.isnan(r["nc_ratio"]) and math.isnan(r["nucleus_area"])
    assert math.isnan(r["nucleus_glcm_contrast"])
    assert not math.isnan(r["cell_glcm_contrast"])


def test_glcm_ignores_surround():
    """v1 zeroed the surround, so contrast depended on how dark the masked-out area was."""
    rng = np.random.RandomState(0)
    cell = (120 + rng.randn(40, 40) * 8).clip(0, 255).astype(np.uint8)
    mask = np.zeros((60, 60), bool)
    mask[10:50, 10:50] = True
    cfg = FeatureConfig()
    outs = []
    for surround in (0, 128, 255):
        img = np.full((60, 60), surround, np.uint8)
        img[10:50, 10:50] = cell
        outs.append(masked_glcm(img, mask, cfg))
    for key in outs[0]:
        assert outs[0][key] == outs[1][key] == outs[2][key]


def test_glcm_separates_smooth_from_textured():
    rng = np.random.RandomState(1)
    mask = np.ones((40, 40), bool)
    smooth = np.full((40, 40), 120, np.uint8) + rng.randint(0, 3, (40, 40)).astype(np.uint8)
    rough = rng.randint(40, 200, (40, 40)).astype(np.uint8)
    cfg = FeatureConfig()
    s, t = masked_glcm(smooth, mask, cfg), masked_glcm(rough, mask, cfg)
    assert t["glcm_contrast"] > 10 * s["glcm_contrast"]
    assert t["entropy"] > s["entropy"]
    assert s["glcm_homogeneity"] > t["glcm_homogeneity"]


def test_glcm_tiny_region_is_nan():
    mask = np.zeros((20, 20), bool)
    mask[:3, :3] = True
    out = masked_glcm(np.full((20, 20), 100, np.uint8), mask, FeatureConfig())
    assert all(math.isnan(v) for v in out.values())


def test_aggregate_prefers_interior_cells(field):
    _, rows = rows_for(field(wbcs=[(112, 112, 20), (110, 2, 20)]))
    agg = aggregate_image(rows)
    assert agg["n_cells"] == 2 and agg["n_cells_interior"] == 1
    interior = next(r for r in rows if not r["touches_border"])
    assert agg["cell_area_mean"] == interior["cell_area"]
    assert agg["agg_uses_border_cells"] is False


def test_aggregate_empty_image():
    agg = aggregate_image([])
    assert agg["n_cells"] == 0 and math.isnan(agg["cell_area_mean"])

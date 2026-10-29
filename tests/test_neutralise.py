import numpy as np
import pytest

from leukemia_pp.config import NeutraliseConfig, PipelineConfig, SegmentationConfig
from leukemia_pp.neutralise import keep_mask, neutralise
from leukemia_pp.pipeline import process_array
from leukemia_pp.segmentation import segment
from leukemia_pp.stain import bgr_to_lab, fit_reference


def seg_of(img):
    return segment(bgr_to_lab(img), SegmentationConfig())


def tinted(img, db, dg, dr):
    return np.clip(img.astype(np.int16) + np.array([db, dg, dr]), 0, 255).astype(np.uint8)


def test_everything_outside_cells_is_one_constant(field):
    img = field(wbcs=[(70, 70, 20), (150, 150, 20)], rbcs=[(110, 110, 14), (30, 190, 12)])
    cfg = NeutraliseConfig()
    seg = seg_of(img)
    rgb, gray = neutralise(img, seg, cfg)
    outside = ~keep_mask(seg, cfg)
    assert outside.sum() > 0.7 * outside.size
    assert len(np.unique(rgb[outside].reshape(-1, 3), axis=0)) == 1       # one colour only
    assert np.unique(gray[outside]).size == 1
    assert rgb.dtype == np.uint8 and gray.dtype == np.float32 and 0 <= gray.min() <= gray.max() <= 1


def test_rbcs_and_background_colour_leave_no_trace(field):
    """Same cells, wildly different background / RBC colour -> identical clean image."""
    base = field(wbcs=[(70, 70, 20), (150, 150, 20)], rbcs=[(110, 110, 14)], seed=1)
    other = base.copy()
    mask = ~keep_mask(seg_of(base), NeutraliseConfig())
    other[mask] = (120, 200, 90)                                       # alien background
    cfg = NeutraliseConfig(white_balance=False)
    a, _ = neutralise(base, seg_of(base), cfg)
    b, _ = neutralise(other, seg_of(base), cfg)
    assert np.array_equal(a[mask], b[mask])


def test_scale_bar_label_is_removed(field):
    img = field(wbcs=[(70, 70, 20)])
    img[200:218, 150:215] = (0, 0, 0)                                  # black "200 pix" bar
    img[203:206, 160:200] = (255, 255, 255)
    rgb, _ = neutralise(img, seg_of(img), NeutraliseConfig())
    assert len(np.unique(rgb[196:222, 146:220].reshape(-1, 3), axis=0)) == 1


def test_cell_pixels_untouched_without_white_balance(field):
    img = field(wbcs=[(100, 100, 22)])
    seg = seg_of(img)
    rgb, _ = neutralise(img, seg, NeutraliseConfig(white_balance=False))
    inner = seg.cell_mask & (np.abs(bgr_to_lab(img)[..., 0] - 0) >= 0)
    assert np.abs(rgb[inner].astype(int) - img[..., ::-1][inner].astype(int)).max() <= 2


def test_white_balance_cancels_slide_tint_in_cells(field):
    """Two copies of the same cells under different colour casts: cell colour must agree far
    better after white balance (that is the point: stain/session colour is a shortcut)."""
    base = field(wbcs=[(70, 70, 20), (150, 150, 22)], rbcs=[(110, 110, 14)])
    cast = tinted(base, 30, -15, -25)

    def cell_lab(img, wb):
        seg = seg_of(img)
        rgb, _ = neutralise(img, seg, NeutraliseConfig(white_balance=wb))
        lab = bgr_to_lab(np.ascontiguousarray(rgb[..., ::-1]))
        return lab[seg.cell_mask].mean(axis=0)

    gap_raw = np.linalg.norm(cell_lab(base, False) - cell_lab(cast, False))
    gap_wb = np.linalg.norm(cell_lab(base, True) - cell_lab(cast, True))
    assert gap_raw > 10 and gap_wb < 0.5 * gap_raw


def test_no_cells_gives_uniform_image(field):
    img = field(rbcs=[(60, 60, 14)])
    rgb, gray = neutralise(img, seg_of(img), NeutraliseConfig())
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) == 1 and np.unique(gray).size == 1


def test_all_cell_frame_skips_white_balance_without_crashing(field):
    img = field(wbcs=[(112, 112, 160)])                               # cell covers the frame
    rgb, _ = neutralise(img, seg_of(img), NeutraliseConfig(min_background_px=10 ** 6))
    assert rgb.shape == img.shape


def test_pipeline_emits_clean_arrays_and_can_disable(field):
    cfg = PipelineConfig()
    img = field(wbcs=[(100, 100, 20), (150, 60, 20)])
    ref = fit_reference([img], cfg.stain)
    res = process_array(img, cfg, ref)
    assert {"rgb_clean", "gray_clean"} <= set(res.arrays)
    off = process_array(img, PipelineConfig.from_dict({"neutralise": {"enabled": False}}), ref)
    assert "rgb_clean" not in off.arrays


@pytest.mark.parametrize("bad", [{"nope": 1}])
def test_config_rejects_unknown(bad):
    with pytest.raises(ValueError):
        PipelineConfig.from_dict({"neutralise": bad})

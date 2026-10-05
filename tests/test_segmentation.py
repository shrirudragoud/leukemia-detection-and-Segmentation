import numpy as np
import pytest

from leukemia_pp.config import SegmentationConfig
from leukemia_pp.segmentation import segment, split_touching
from leukemia_pp.stain import bgr_to_lab

CFG = SegmentationConfig()


def seg_of(img):
    return segment(bgr_to_lab(img), CFG)


def test_rbcs_are_not_cells(field):
    """The v1 failure: grayscale-Otsu merged all RBCs into the cell mask."""
    img = field(rbcs=[(40 + 40 * i, 40 + 40 * j, 14) for i in range(4) for j in range(4)])
    assert seg_of(img).n_cells == 0


def test_wbcs_found_among_rbcs(field):
    rbcs = [(30, 30, 14), (30, 190, 14), (190, 30, 14), (110, 110, 14), (190, 190, 14)]
    wbcs = [(70, 70, 20), (150, 150, 18)]
    seg = seg_of(field(wbcs=wbcs, rbcs=rbcs))
    assert seg.n_cells == 2
    for (y, x, r) in wbcs:
        lab = seg.cell_labels[y, x]
        assert lab > 0
        area = (seg.cell_labels == lab).sum()
        assert abs(area - np.pi * r ** 2) / (np.pi * r ** 2) < 0.12
    assert not seg.cell_mask[30, 30]                         # RBC stays background


def test_touching_cells_are_split(field):
    seg = seg_of(field(wbcs=[(100, 85, 20), (100, 118, 20)]))      # centres 33 px apart, r=20
    assert seg.n_cells == 2
    areas = [(seg.cell_labels == i).sum() for i in (1, 2)]
    assert min(areas) / max(areas) > 0.8


def test_single_elongated_cell_is_not_oversplit(field):
    seg = seg_of(field(ellipses=[(112, 112, 34, 18)]))
    assert seg.n_cells == 1


def test_empty_field_gives_nothing(field):
    seg = seg_of(field())
    assert seg.n_cells == 0 and not seg.nucleus_mask.any()


def test_specks_are_dropped(field):
    seg = seg_of(field(wbcs=[(60, 60, 3), (150, 150, 20)]))
    assert seg.n_cells == 1


def test_nucleus_found_inside_cell_with_dark_core(field):
    seg = seg_of(field(wbcs=[(112, 112, 24)], nucleus=True))
    info = seg.nucleus_info[1]
    assert info.found
    nuc, cell = (seg.nucleus_labels == 1).sum(), (seg.cell_labels == 1).sum()
    assert 0.3 < nuc / cell < 0.55                          # true ratio (0.65)^2 = 0.42
    assert not (seg.nucleus_mask & ~seg.cell_mask).any()


def test_flat_cell_reports_no_nucleus_instead_of_inventing_one(field):
    seg = seg_of(field(wbcs=[(112, 112, 24)], nucleus=False))
    assert seg.n_cells == 1
    assert not seg.nucleus_info[1].found
    assert not seg.nucleus_mask.any()


def test_labels_are_consecutive_and_nucleus_ids_match(field):
    seg = seg_of(field(wbcs=[(60, 60, 18), (60, 160, 18), (160, 110, 18)]))
    assert sorted(np.unique(seg.cell_labels[seg.cell_labels > 0])) == [1, 2, 3]
    assert set(np.unique(seg.nucleus_labels)) <= {0, 1, 2, 3}
    assert (seg.nucleus_labels[seg.nucleus_labels > 0]
            == seg.cell_labels[seg.nucleus_labels > 0]).all()


def test_split_rejects_fragments_below_min_area():
    mask = np.zeros((60, 60), bool)
    yy, xx = np.mgrid[:60, :60]
    mask |= (yy - 30) ** 2 + (xx - 22) ** 2 < 15 ** 2
    mask |= (yy - 30) ** 2 + (xx - 40) ** 2 < 6 ** 2          # small lobe overlapping the disc
    whole = split_touching(mask, h=1.5, smooth=1.0, min_area=10_000)
    assert whole.max() == 1                                   # nothing may be split to < min_area


@pytest.mark.parametrize("seed", range(3))
def test_noise_robustness(field, seed):
    seg = seg_of(field(wbcs=[(70, 70, 20), (150, 150, 18)], rbcs=[(110, 110, 14)], seed=seed))
    assert seg.n_cells == 2

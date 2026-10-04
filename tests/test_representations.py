import csv

import cv2
import numpy as np
import pytest

from leukemia_pp.config import PipelineConfig
from leukemia_pp.dataset import (CHANNELS, FEATURE_GROUPS, PRESETS, compute_channel_stats,
                                 crop_cell, layer_feature_group, load_stack, load_stats,
                                 normalise, save_stats, stack_depth)
from leukemia_pp.features import CELL_COLUMNS
from leukemia_pp.pipeline import array_keys, cell_table_columns, image_columns, process_array, run
from leukemia_pp.stain import fit_reference

ALL_ON = {"hds": {"enabled": True}, "beemd": {"enabled": True, "ensemble": 2},
          "save_previews": False}


@pytest.fixture
def two_cells(field):
    return field(wbcs=[(70, 70, 20), (150, 150, 22)], rbcs=[(110, 110, 12)])


def test_default_config_computes_no_optional_arrays(field):
    cfg = PipelineConfig()
    img = field(wbcs=[(100, 100, 20)])
    res = process_array(img, cfg, fit_reference([img], cfg.stain))
    assert set(res.arrays) == set(array_keys(cfg))
    assert not any("imf" in k or "hds" in k for k in res.arrays)


def test_all_on_arrays_shapes_and_columns(two_cells):
    cfg = PipelineConfig.from_dict(ALL_ON)
    res = process_array(two_cells, cfg, fit_reference([two_cells], cfg.stain))
    assert set(res.arrays) == set(array_keys(cfg))
    h, w = two_cells.shape[:2]
    k = cfg.beemd.n_imfs
    assert res.arrays["imfs"].shape == (k, h, w) and res.arrays["imf_selected"].shape == (k,)
    assert res.arrays["hds"].dtype == res.arrays["imf_pure"].dtype == np.float32
    assert 0 <= res.arrays["hds_edge"].min() and res.arrays["hds_edge"].max() <= 1
    cols = set(cell_table_columns(cfg))
    for row in res.cell_rows:
        assert set(row) <= cols | {"image_id"}
        assert np.isfinite(row["cell_apc_mean"]) and np.isfinite(row["cell_hds_edge_rim_mean"])
        assert row["cell_imf1_energy"] >= 0
    assert set(res.image_row) <= set(image_columns(cfg))
    assert "cell_hds_edge_mean_mean" in image_columns(cfg)


def test_boundary_edge_stat_exceeds_interior_for_flat_cells(two_cells):
    """APC/HDS-edge 'rim' statistics should capture the cell boundary, not the interior."""
    cfg = PipelineConfig.from_dict(ALL_ON)
    res = process_array(two_cells, cfg, fit_reference([two_cells], cfg.stain))
    for row in res.cell_rows:
        assert row["cell_hds_edge_rim_mean"] > row["cell_hds_edge_mean"]


def test_pipeline_run_with_everything_enabled(tmp_path, field):
    root = tmp_path / "raw"
    for cname in ("Benign", "Early"):
        (root / cname).mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(root / cname / f"{cname}_{i}.png"),
                        field(wbcs=[(70, 70 + i * 5, 20), (150, 150, 20)], seed=i))
    cfg = PipelineConfig.from_dict(ALL_ON)
    out = tmp_path / "out"
    summary = run(root, out, cfg, workers=1)
    assert summary["n_ok"] == 6 and summary["n_failed"] == 0
    z = np.load(next((out / "cache").rglob("*.npz")))
    assert set(z.files) == set(array_keys(cfg))
    with open(out / "cells.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == cell_table_columns(cfg) and len(rows) == 12
    # deterministic, including the seeded ensemble
    out2 = tmp_path / "out2"
    run(root, out2, cfg, workers=1)
    assert (out / "cells.csv").read_bytes() == (out2 / "cells.csv").read_bytes()
    # cached tensors feed the dataset layer
    paths = sorted((out / "cache").rglob("*.npz"))
    names = PRESETS["full"]
    stats = compute_channel_stats(paths, names)
    assert len(stats["mean"]) == stack_depth(names) == 8
    stacked = np.stack([normalise(load_stack(p, names), stats) for p in paths])
    assert np.allclose(stacked.mean(axis=(0, 2, 3)), 0, atol=1e-4)
    assert np.allclose(stacked.std(axis=(0, 2, 3)), 1, atol=1e-3)
    save_stats(stats, tmp_path / "stats.json")
    assert load_stats(tmp_path / "stats.json") == stats


# ------------------------------------------------------------------------- dataset layer
@pytest.fixture
def cache_file(tmp_path, two_cells):
    cfg = PipelineConfig.from_dict(ALL_ON)
    res = process_array(two_cells, cfg, fit_reference([two_cells], cfg.stain))
    path = tmp_path / "c.npz"
    np.savez_compressed(path, **res.arrays)
    return path


def test_load_stack_scaling_and_order(cache_file):
    s = load_stack(cache_file, ("rgb_norm", "apc", "imf2"))
    assert s.shape[0] == 5 and s.dtype == np.float32
    assert 0 <= s[:3].min() and s[:3].max() <= 1.0              # uint8 scaled to [0, 1]
    z = np.load(cache_file)
    assert np.allclose(s[3], z["apc"]) and np.allclose(s[4], z["imfs"][1])


def test_missing_channel_gives_actionable_error(tmp_path, field):
    cfg = PipelineConfig()
    img = field(wbcs=[(100, 100, 20)])
    res = process_array(img, cfg, fit_reference([img], cfg.stain))
    np.savez_compressed(tmp_path / "d.npz", **res.arrays)
    with pytest.raises(KeyError, match="enable it in the pipeline config"):
        load_stack(tmp_path / "d.npz", ("rgb_norm", "hds_edge"))


def test_crop_cell_is_centred_square_and_deterministic(cache_file):
    z = np.load(cache_file)
    names = ("rgb_norm", "apc")
    a = crop_cell(cache_file, 1, names, size=64)
    assert a.shape == (4, 64, 64) and np.array_equal(a, crop_cell(z, 1, names, size=64))
    masked = crop_cell(cache_file, 1, ("a_score",), size=64, background="zero")
    assert masked[0][:4, :4].max() == 0                         # corner is blanked
    centre = crop_cell(cache_file, 1, ("a_score",), size=64)[0]
    ys, xs = np.nonzero(centre > centre.max() * 0.5)
    assert abs(ys.mean() - 32) < 6 and abs(xs.mean() - 32) < 6  # cell sits in the middle
    with pytest.raises(KeyError):
        crop_cell(cache_file, 99, names)
    with pytest.raises(ValueError):
        crop_cell(cache_file, 1, names, background="blur")


def test_crop_of_border_cell_is_padded_not_shifted(tmp_path, field):
    cfg = PipelineConfig()
    ref = fit_reference([field(wbcs=[(60, 60, 20), (150, 150, 20)])], cfg.stain)
    res = process_array(field(wbcs=[(110, 4, 20), (110, 150, 20)]), cfg, ref)
    np.savez_compressed(tmp_path / "e.npz", **res.arrays)
    border_id = next(r["cell_id"] for r in res.cell_rows if r["touches_border"])
    crop = crop_cell(tmp_path / "e.npz", border_id, ("rgb_norm",), size=64)
    assert crop.shape == (3, 64, 64)
    assert crop[:, :, :6].max() == 0                            # outside-the-image side is zero pad


def test_presets_and_feature_groups_are_consistent():
    for names in PRESETS.values():
        assert all(n in CHANNELS for n in names) and stack_depth(names) >= 3
    seen: list[str] = []
    for cols in FEATURE_GROUPS.values():
        assert set(cols) <= set(CELL_COLUMNS)
        seen += cols
    assert len(seen) == len(set(seen))                          # groups do not overlap
    assert set(layer_feature_group("apc")) == {"cell_apc_mean", "cell_apc_std", "cell_apc_rim_mean"}

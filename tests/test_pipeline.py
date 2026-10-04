import csv
import json

import cv2
import numpy as np
import pytest

from leukemia_pp import pipeline
from leukemia_pp.cli import main
from leukemia_pp.config import PipelineConfig
from leukemia_pp.pipeline import ARRAY_KEYS, process_array, run
from leukemia_pp.stain import fit_reference


def build_dataset(root, field, per_class=4):
    for ci, cname in enumerate(("Benign", "Early")):
        d = root / cname
        d.mkdir(parents=True)
        for i in range(per_class):
            img = field(wbcs=[(70 + 5 * i, 70, 18), (150, 150 - 4 * i, 20)],
                        rbcs=[(30, 30 + 20 * i, 12)], seed=ci * 100 + i)
            assert cv2.imwrite(str(d / f"{cname}_{i:02d}.png"), img)


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def dataset(tmp_path, field):
    root = tmp_path / "raw"
    build_dataset(root, field)
    return root


def test_end_to_end_outputs(dataset, tmp_path):
    out = tmp_path / "out"
    summary = run(dataset, out, PipelineConfig(), workers=1)
    assert summary["n_ok"] == summary["n_images"] == 8 and summary["n_failed"] == 0
    for name in ("config.json", "manifest.csv", "images.csv", "cells.csv",
                 "stain_reference.json", "run_summary.json"):
        assert (out / name).exists(), name
    assert not (out / "failures.csv").exists()

    manifest = read(out / "manifest.csv")
    assert {r["split"] for r in manifest} == {"train", "val", "test"}
    assert len(read(out / "images.csv")) == 8
    cells = read(out / "cells.csv")
    assert len(cells) == 16                                  # 2 WBCs per image, RBCs ignored
    assert {c["touches_border"] for c in cells} == {"False"}

    npz_files = sorted((out / "cache").rglob("*.npz"))
    assert len(npz_files) == 8
    z = np.load(npz_files[0])
    assert set(z.files) == set(ARRAY_KEYS)
    h, w = z["rgb"].shape[:2]
    assert z["rgb"].dtype == z["rgb_norm"].dtype == np.uint8
    assert z["apc"].dtype == z["log"].dtype == np.float32
    assert z["cell_labels"].dtype == np.int32 and z["cell_mask"].dtype == bool
    assert all(z[k].shape[:2] == (h, w) for k in ARRAY_KEYS)
    assert z["apc"].max() <= 1.0 and z["log"].max() <= 1.0
    assert len(list((out / "previews").rglob("*.png"))) == 8
    assert list(out.rglob("*.tmp")) == []                    # atomic writes leave no debris


def test_stain_reference_fitted_on_train_only(dataset, tmp_path):
    out = tmp_path / "out"
    cfg = PipelineConfig(save_previews=False)
    run(dataset, out, cfg, workers=1, dry_run=True)
    run(dataset, out, cfg, workers=1)
    ref_json = json.loads((out / "stain_reference.json").read_text())
    n_train = sum(r["split"] == "train" for r in read(out / "manifest.csv"))
    assert ref_json["n_images"] == min(cfg.stain.ref_sample_size, n_train)


def test_deterministic_across_runs_and_worker_counts(dataset, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run(dataset, a, PipelineConfig(save_previews=False), workers=1)
    run(dataset, b, PipelineConfig(save_previews=False), workers=2)
    for name in ("manifest.csv", "images.csv", "cells.csv", "stain_reference.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    za, zb = np.load(a / "cache/train" / next(p.name for p in (a / "cache/train").iterdir())), None
    zb = np.load(b / "cache/train" / next(p.name for p in (a / "cache/train").iterdir()))
    for k in ARRAY_KEYS:
        assert np.array_equal(za[k], zb[k]), k


def test_corrupt_image_is_reported_not_fatal(dataset, tmp_path):
    (dataset / "Early" / "broken.jpg").write_bytes(b"this is not a jpeg")
    (dataset / "Early" / "empty.png").write_bytes(b"")
    out = tmp_path / "out"
    summary = run(dataset, out, PipelineConfig(save_previews=False), workers=1)
    assert summary["n_failed"] == 2 and summary["n_ok"] == 8
    failures = read(out / "failures.csv")
    assert {f["image_id"] for f in failures} == {"broken", "empty"}
    assert all(f["error_type"] == "ImageReadError" for f in failures)


def test_changed_config_in_same_output_dir_is_refused(dataset, tmp_path):
    out = tmp_path / "out"
    run(dataset, out, PipelineConfig(save_previews=False), workers=1)
    other = PipelineConfig.from_dict({"segmentation": {"min_cell_area": 80},
                                      "save_previews": False})
    with pytest.raises(RuntimeError, match="different configuration"):
        run(dataset, out, other, workers=1)
    run(dataset, out, other, workers=1, force=True)


def test_duplicate_image_ids_rejected(dataset, tmp_path):
    (dataset / "Early" / "Benign_00.png").write_bytes((dataset / "Benign" / "Benign_00.png").read_bytes())
    with pytest.raises(ValueError, match="duplicate image id"):
        run(dataset, tmp_path / "out", PipelineConfig(), workers=1)


def test_unknown_class_folders_ignored_and_empty_input_errors(tmp_path):
    (tmp_path / "raw" / "Mystery").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "raw", tmp_path / "out", PipelineConfig())
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "missing", tmp_path / "out", PipelineConfig())


def test_sparse_field_is_flagged(field):
    cfg = PipelineConfig()
    ref = fit_reference([field(wbcs=[(60, 60, 20), (150, 150, 20)])], cfg.stain)
    res = process_array(field(wbcs=[(60, 60, 8)]), cfg, ref)
    assert "stain_not_applied" in res.flags


def test_empty_field_is_flagged_no_cells(field):
    cfg = PipelineConfig()
    ref = fit_reference([field(wbcs=[(60, 60, 20), (150, 150, 20)])], cfg.stain)
    res = process_array(field(), cfg, ref)
    assert "no_cells" in res.flags and res.image_row["n_cells"] == 0


def test_stain_disabled_needs_no_reference(field):
    cfg = PipelineConfig.from_dict({"stain": {"enabled": False}})
    res = process_array(field(wbcs=[(100, 100, 20)]), cfg, None)
    assert res.image_row["n_cells"] == 1


def test_max_side_downscales(dataset, tmp_path, field):
    big = field(wbcs=[(200, 200, 40)], size=448)
    cv2.imwrite(str(dataset / "Benign" / "big.png"), big)
    out = tmp_path / "out"
    run(dataset, out, PipelineConfig.from_dict({"data": {"max_side": 224}, "save_previews": False}),
        workers=1)
    row = next(r for r in read(out / "images.csv") if r["image_id"] == "big")
    assert (row["height"], row["width"]) == ("224", "224")


def test_cli(dataset, tmp_path, capsys):
    out = tmp_path / "out"
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"save_previews": False}))
    base = ["run", "--input", str(dataset), "--workers", "1", "--config", str(cfg_file)]
    assert main([*base, "--output", str(out)]) == 0
    assert main(["run", "--input", str(tmp_path / "nope"), "--output", str(out)]) == 2
    (dataset / "Early" / "broken.jpg").write_bytes(b"x")
    assert main([*base, "--output", str(tmp_path / "out2")]) == 3
    png = tmp_path / "qc.png"
    ref = out / "stain_reference.json"
    assert main(["inspect", str(dataset / "Benign" / "Benign_00.png"), "--out", str(png),
                 "--reference", str(ref)]) == 0
    assert png.stat().st_size > 1000
    assert "cell(s)" in capsys.readouterr().out


def test_pipeline_module_exports_documented_columns():
    cols = set(pipeline.CELL_TABLE_COLUMNS)
    assert {"image_id", "cell_id", "nc_ratio", "cell_area", "nucleus_glcm_contrast"} <= cols
    assert len(pipeline.CELL_TABLE_COLUMNS) == len(cols)      # no duplicate columns
    assert len(pipeline.IMAGE_COLUMNS) == len(set(pipeline.IMAGE_COLUMNS))

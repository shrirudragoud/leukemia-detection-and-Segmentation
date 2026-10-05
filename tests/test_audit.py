import json

import cv2
import numpy as np
import pytest

pytest.importorskip("sklearn")

from leukemia_pp.audit import corner_patch, run_audit, verdict  # noqa: E402
from leukemia_pp.cli import main  # noqa: E402
from leukemia_pp.config import PipelineConfig  # noqa: E402
from leukemia_pp.pipeline import run  # noqa: E402

TINTS = {"Benign": (0, 0, 0), "Early": (22, -6, -22), "Pre": (-24, 10, 20)}   # BGR casts


def tinted(img, bgr):
    return np.clip(img.astype(np.int16) + np.array(bgr), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def tint_only_dataset(tmp_path_factory):
    """Identical cell content in every class; the ONLY class signal is the slide colour cast
    (and a class-specific scale-bar mark in the corner)."""
    from conftest import make_field
    root = tmp_path_factory.mktemp("tint") / "raw"
    rng = np.random.RandomState(0)
    for cname, tint in TINTS.items():
        (root / cname).mkdir(parents=True)
        for i in range(1, 17):
            img = make_field(wbcs=[(60 + int(rng.randint(-8, 8)), 70, 20),
                                   (150, 140 + int(rng.randint(-8, 8)), 21)],
                             rbcs=[(110, 100, 13)], seed=i)
            img = tinted(img, tint)
            bar = {"Benign": 0, "Early": 1, "Pre": 2}[cname]
            img[205:218, 150:150 + 20 * (bar + 1)] = (0, 0, 0)               # scale-bar style
            cv2.imwrite(str(root / cname / f"WBC-{cname}-{i:03d}.png"), img)
    return root


@pytest.fixture(scope="module")
def audited(tint_only_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = PipelineConfig.from_dict({"split": {"sequence_block": 4}, "save_previews": False})
    run(tint_only_dataset, out, cfg, workers=1)
    report = run_audit(out, tint_only_dataset, seeds=(0,), n_trees=60)
    return out, {r["probe"].split(" (")[0]: r for r in report["results"]}, report


def test_audit_flags_background_and_scale_bar_shortcuts(audited):
    _, by, report = audited
    assert report["chance_balanced"] == pytest.approx(1 / 3)
    assert by["background colour, raw image"]["balanced_accuracy"] > 0.9
    assert by["scale-bar corner patch, raw image"]["balanced_accuracy"] > 0.9
    assert by["background colour, raw image"]["verdict"] == "VERY STRONG"


def test_clean_image_carries_no_background_or_corner_signal(audited):
    _, by, _ = audited
    for key in ("background colour of rgb_clean", "corner patch of rgb_clean"):
        assert by[key]["balanced_accuracy"] < 0.5                    # constant -> ~chance (1/3)


def test_white_balance_removes_cell_colour_cast(audited, tint_only_dataset):
    """Judge by the SIZE of the class gap, not by classifier accuracy: on noise-free synthetic
    cells a forest separates even a 1-unit gap perfectly."""
    import csv

    from leukemia_pp.audit import extract_probe_features
    out, _, _ = audited
    feats = extract_probe_features(out, tint_only_dataset)
    with open(out / "manifest.csv", newline="") as fh:
        y = np.array([r["class_name"] for r in csv.DictReader(fh)])

    def class_gap(block):
        means = np.array([block[y == c].mean(axis=0) for c in np.unique(y)])
        return max(np.linalg.norm(a - b) for a in means for b in means)

    assert class_gap(feats["bg_raw"]) > 10                    # the cast is large in the raw image
    assert class_gap(feats["cell_colour_clean"]) < 3.0        # and (almost) gone from the cells


def test_audit_writes_reports_and_cli(audited, tint_only_dataset, capsys):
    out, _, _ = audited
    data = json.loads((out / "audit.json").read_text())
    assert data["n_groups"] >= 5 and data["caveats"]
    md = (out / "audit.md").read_text()
    assert "Shortcut / confound audit" in md and "| acquisition |" in md
    assert main(["audit", "--run", str(out), "--input", str(tint_only_dataset), "--seeds", "1"]) == 0
    assert "Balanced acc." in capsys.readouterr().out


def test_verdict_thresholds():
    assert verdict(0.30, 0.25).startswith("none")
    assert verdict(0.40, 0.25) == "weak"
    assert verdict(0.60, 0.25) == "STRONG"
    assert verdict(0.85, 0.25) == "VERY STRONG"


def test_corner_patch_shape_and_range():
    img = np.random.RandomState(0).randint(0, 255, (224, 224, 3), dtype=np.uint8)
    p = corner_patch(img)
    assert p.shape == (192,) and 0 <= p.min() <= p.max() <= 1

import json
import tarfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_ml_data import dcfg, synth  # noqa: E402,F401

from leukemia_ml.bundle import make_bundle, verify_extracted  # noqa: E402
from leukemia_ml.config import (AugConfig, DataConfig, ExperimentConfig, ModelConfig,  # noqa: E402
                                TrainConfig)
from leukemia_ml.cv import run_cv  # noqa: E402
from leukemia_ml.data.crops import build_crop_cache  # noqa: E402
from leukemia_ml.data.index import load_index  # noqa: E402
from leukemia_ml.data.splits import PURGED, contiguous_split, sequence_numbers  # noqa: E402
from leukemia_ml.preflight import preflight  # noqa: E402
from leukemia_ml.train.loop import Trainer, amp_dtype  # noqa: E402


def exp(synth, split="contiguous", model=None, train=None, data=None):
    m = dict(encoder="timm:resnet10t", pretrained=False, input_size=32, freeze="frozen",
             head="mil", hidden=32, dropout=0.0)
    m.update(model or {})
    t = dict(epochs=4, batch_size=4, lr_head=3e-3, patience=10, seed=0, device="cpu")
    t.update(train or {})
    d = dict(split_scheme=split, n_folds=4, test_fold=0, val_offset=2, embargo=1)
    d.update(data or {})
    base = dcfg(synth)
    return ExperimentConfig(data=DataConfig(**{**base.__dict__, **d}), model=ModelConfig(**m),
                            aug=AugConfig(colour="hsv"), train=TrainConfig(**t), name="full")


# -------------------------------------------------------------------------- splits
def test_contiguous_split_segments_embargo_and_coverage(synth):
    _, out = synth
    idx = load_index(out, "4way", ())
    sp = contiguous_split(idx, n_folds=4, test_fold=0, val_offset=2, embargo=1)
    nums = sequence_numbers(idx)
    assert set(sp) <= {"train", "val", "test", PURGED}
    for c in range(4):
        m = idx.image_label == c
        t, v = np.sort(nums[m & (sp == "test")]), np.sort(nums[m & (sp == "val")])
        assert (np.diff(t) == 1).all() and (np.diff(v) == 1).all() and len(t) and len(v)
        held = np.concatenate([t, v])
        for n in nums[m & (sp == "train")]:
            assert np.abs(held - n).min() > 1                     # embargo respected
        for n in nums[m & (sp == PURGED)]:
            assert np.abs(held - n).min() <= 1                    # only edge neighbours purged
    # no embargo -> nothing purged, and every image is assigned
    sp0 = contiguous_split(idx, 4, 0, 2, 0)
    assert PURGED not in set(sp0) and len(sp0) == idx.n_images
    # rotating the test fold moves the test set
    assert set(np.nonzero(sp0 == "test")[0]) != set(np.nonzero(contiguous_split(idx, 4, 1, 2, 0) == "test")[0])


def test_split_config_validation():
    for bad in ({"n_folds": 2}, {"test_fold": 5}, {"val_offset": 0}, {"split_scheme": "x"}):
        with pytest.raises(ValueError):
            DataConfig(**bad)
    assert DataConfig().split_scheme == "contiguous"           # honest evaluation is the default


def test_trainer_uses_contiguous_split_and_never_trains_on_heldout(synth, tmp_path):
    cfg = exp(synth, train={"epochs": 1})
    t = Trainer(cfg, tmp_path / "t")
    idx = t.index
    tr, te = set(idx.image_indices("train")), set(idx.image_indices("test"))
    assert tr and te and not tr & te
    assert set(idx.split) == set(idx.image_split)                # per-cell mirror is consistent
    r = t.fit()
    assert r["split"]["scheme"] == "contiguous" and r["split"]["n_test_images"] == len(te)


# -------------------------------------------------------------------------- resume
def _same(a, b):
    return (a["test"] == b["test"]
            and [h["train_loss"] for h in a["history"]] == [h["train_loss"] for h in b["history"]]
            and a["best_val_macro_f1"] == b["best_val_macro_f1"])


def test_interrupted_and_resumed_frozen_run_equals_uninterrupted(synth, tmp_path):
    cfg = exp(synth)
    full = Trainer(cfg, tmp_path / "full").fit()
    part = Trainer(cfg, tmp_path / "res").fit(max_epochs_this_call=2)
    assert part == {"interrupted": True, "epochs_done": 2}
    assert (tmp_path / "res" / "last.pt").exists() and not (tmp_path / "res" / "metrics.json").exists()
    resumed = Trainer(cfg, tmp_path / "res").fit()
    assert _same(full, resumed)
    lines = (tmp_path / "res" / "train_log.jsonl").read_text().strip().splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == list(range(len(lines)))     # no duplicates


def test_interrupted_and_resumed_lora_run_with_augmentation_is_exact(synth, tmp_path):
    cfg = exp(synth, model={"encoder": "dinov2_s", "input_size": 28, "freeze": "lora",
                            "lora_rank": 2},
              train={"epochs": 3, "max_train_images": 8, "max_eval_images": 8, "lr_lora": 5e-3})
    full = Trainer(cfg, tmp_path / "full").fit()
    Trainer(cfg, tmp_path / "res").fit(max_epochs_this_call=1)
    resumed = Trainer(cfg, tmp_path / "res").fit()
    assert _same(full, resumed)


def test_finished_run_is_not_recomputed_and_other_config_does_not_resume(synth, tmp_path):
    cfg = exp(synth, train={"epochs": 2})
    out = tmp_path / "x"
    r1 = Trainer(cfg, out).fit()
    mtime = (out / "metrics.json").stat().st_mtime_ns
    assert Trainer(cfg, out).fit() == r1
    assert (out / "metrics.json").stat().st_mtime_ns == mtime
    other = exp(synth, train={"epochs": 2, "lr_head": 1e-2})
    (out / "metrics.json").unlink()
    r2 = Trainer(other, out).fit()                                # stale last.pt must be ignored
    assert r2["config_hash"] == other.hash() != r1["config_hash"]
    assert r2["history"][0]["train_loss"] != r1["history"][0]["train_loss"]


def test_amp_dtype_is_none_on_cpu():
    assert amp_dtype("auto", torch.device("cpu")) is None
    assert amp_dtype("fp16", torch.device("cpu")) is None


# ------------------------------------------------------------------------------- CV
def test_cv_tests_every_image_once_pools_and_resumes(synth, tmp_path):
    cfg = exp(synth, train={"epochs": 2})
    s = run_cv(cfg, tmp_path / "cv", n_boot=50)
    assert s["folds"] == [0, 1, 2, 3] and s["n_images"] == 48 and s["pooled"]["n"] == 48
    ci = s["pooled_ci"]["balanced_accuracy"]
    assert ci["lo"] <= ci["point"] <= ci["hi"] and ci["n_groups"] >= 8
    assert set(s["per_fold_mean"]) == {"balanced_accuracy", "macro_f1", "auroc_ovr", "ece"}
    for k in range(4):
        assert (tmp_path / "cv" / f"fold{k}" / "predictions.csv").exists()
    md = (tmp_path / "cv" / "cv_summary.md").read_text()
    assert "Pooled" in md and "Confusion" in md
    stamps = [(tmp_path / "cv" / f"fold{k}" / "metrics.json").stat().st_mtime_ns for k in range(4)]
    s2 = run_cv(cfg, tmp_path / "cv", n_boot=50)
    assert [(tmp_path / "cv" / f"fold{k}" / "metrics.json").stat().st_mtime_ns for k in range(4)] == stamps
    assert s2["pooled"] == s["pooled"]


def test_cv_subset_of_folds(synth, tmp_path):
    s = run_cv(exp(synth, train={"epochs": 1}), tmp_path / "cv", folds=[1, 2], n_boot=20)
    assert s["folds"] == [1, 2] and s["pooled"]["n"] < 48


# -------------------------------------------------------------------- preflight / bundle
def test_preflight_reports_timing_and_projects_epoch_time(synth):
    cfg = exp(synth, model={"freeze": "full"})
    info = preflight(cfg, steps=2)
    assert info["ok"] and info["step_seconds"] > 0 and info["cells_per_second"] > 0
    assert info["train_images"] > 0 and info["est_epoch_minutes"] >= 0
    assert info["est_total_hours_all_folds"] >= info["est_total_hours_one_fold"]
    bad = exp(synth, model={"encoder": "dinobloom_s", "weights_path": "/nope.pth",
                            "input_size": 28, "pretrained": True})
    res = preflight(bad, steps=1)                        # reports the problem instead of crashing
    assert not res["ok"] and any("DinoBloom weights" in p for p in res["problems"])


def test_bundle_roundtrip_and_corruption_detection(synth, tmp_path):
    cfg = exp(synth)
    idx = load_index(cfg.data.run_dir, "4way", ())
    build_crop_cache(cfg.data, idx, workers=1)
    m = make_bundle(cfg, tmp_path / "b.tar.gz")
    assert m["tar_bytes"] > 0 and any(k.endswith("crops.npy") for k in m["files"])
    with tarfile.open(tmp_path / "b.tar.gz") as tar:
        tar.extractall(tmp_path / "x")
    assert verify_extracted(tmp_path / "x" / "run") == []
    crops = next((tmp_path / "x" / "run" / "ml_cache").glob("*/crops.npy"))
    crops.write_bytes(crops.read_bytes()[:-1] + b"\x00")
    assert any("corrupt" in p for p in verify_extracted(tmp_path / "x" / "run"))
    with pytest.raises(FileNotFoundError):
        make_bundle(exp(synth, data={"margin": 0.9}), tmp_path / "c.tar.gz")

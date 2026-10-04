import csv
import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

from conftest import make_field  # noqa: E402

from leukemia_ml.config import AugConfig, DataConfig  # noqa: E402
from leukemia_ml.data import crops as crops_mod  # noqa: E402
from leukemia_ml.data.augment import augment  # noqa: E402
from leukemia_ml.data.crops import FILL, build_crop_cache, cache_key, cut_crop  # noqa: E402
from leukemia_ml.data.datasets import (BagDataset, EmbeddingBagDataset, FeatureScaler,  # noqa: E402
                                       collate_bags, load_crops)
from leukemia_ml.data.index import feature_columns, load_index  # noqa: E402
from leukemia_pp.config import PipelineConfig  # noqa: E402
from leukemia_pp.pipeline import run  # noqa: E402

CLASSES = ("Benign", "Early", "Pre", "Pro")


def make_dataset(root, per_class=12):
    """Class k has cells of radius 14+4k (so size is the signal) and 1+k..2+k cells."""
    rng = np.random.RandomState(0)
    for k, cname in enumerate(CLASSES):
        (root / cname).mkdir(parents=True)
        for i in range(1, per_class + 1):
            r = 14 + 4 * k
            n = 2 + (k % 2)
            cells = [(45 + 55 * j + int(rng.randint(-5, 5)), 60 + 70 * (j % 2) + int(rng.randint(-8, 8)), r)
                     for j in range(n)]
            img = make_field(wbcs=cells, rbcs=[(110, 150, 12)], seed=k * 100 + i)
            img[205:218, 150:150 + 20 * (k + 1)] = (0, 0, 0)          # class-specific scale bar
            cv2.imwrite(str(root / cname / f"WBC-{cname}-{i:03d}.png"), img)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    base = tmp_path_factory.mktemp("mlsynth")
    raw = base / "raw"
    make_dataset(raw)
    out = base / "run"
    cfg = PipelineConfig.from_dict({"split": {"sequence_block": 3}, "save_previews": False})
    run(raw, out, cfg, workers=1)
    return raw, out


def dcfg(synth, **kw):
    raw, out = synth
    kw.setdefault("split_scheme", "manifest")
    return DataConfig(run_dir=str(out), input_dir=str(raw), cache_size=48, **kw)


# ---------------------------------------------------------------------------- index
def test_index_matches_cells_csv_and_never_splits_a_group(synth):
    _, out = synth
    idx = load_index(out, "4way", ("cell_shape", "apc"))
    with open(out / "cells.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert idx.n_cells == len(rows) and idx.n_images == 48
    assert idx.features.shape == (len(rows), len(feature_columns(("cell_shape", "apc"))))
    seen: dict[str, set] = {}
    for g, s in zip(idx.image_group, idx.image_split):
        seen.setdefault(g, set()).add(s)
    assert all(len(v) == 1 for v in seen.values())
    assert set(idx.split) == {"train", "val", "test"}
    # cells of an image are contiguous and sorted by cell id
    for i in range(idx.n_images):
        c = idx.cells_of_image(i)
        assert (np.diff(idx.cell_id[c]) > 0).all() and len(set(idx.image_of_cell[c])) == 1
    assert load_index(out, "binary").n_classes == 2


def test_index_errors(synth):
    _, out = synth
    with pytest.raises(ValueError, match="unknown feature group"):
        load_index(out, "4way", ("nonsense",))
    with pytest.raises(ValueError, match="re-run the pipeline"):
        load_index(out, "4way", ("hds_edge",))


# ----------------------------------------------------------------------------- crops
def test_cut_crop_geometry_isolation_and_padding():
    img = np.full((100, 100, 3), 50, np.uint8)
    labels = np.zeros((100, 100), np.int32)
    labels[40:60, 40:60] = 1
    labels[40:60, 62:80] = 2                                           # a neighbour
    c = cut_crop(img, labels, 1, 32, 0.25, isolate=True)
    assert c.shape == (32, 32, 3)
    assert (c[:, -2:] == FILL).all()                                   # neighbour blanked
    assert c[16, 16].tolist() == [50, 50, 50]
    c2 = cut_crop(img, labels, 1, 32, 0.25, isolate=False)
    assert c2[:, -2:].min() == 50                                      # neighbour visible
    edge = np.zeros((100, 100), np.int32)
    edge[40:60, 0:15] = 1                                              # cell cut by the image edge
    e = cut_crop(img, edge, 1, 32, 0.25, isolate=True)
    assert (e[:, :6] == FILL).all()                                    # padded with the neutral fill


def test_crop_cache_build_reuse_and_key(synth):
    cfg = dcfg(synth)
    idx = load_index(cfg.run_dir, cfg.task, ())
    d = build_crop_cache(cfg, idx, workers=1)
    arr = load_crops(d)
    assert arr.shape == (idx.n_cells, 48, 48, 3) and arr.dtype == np.uint8
    mtime = (d / "crops.npy").stat().st_mtime_ns
    assert build_crop_cache(cfg, idx, workers=1) == d
    assert (d / "crops.npy").stat().st_mtime_ns == mtime               # reused, not rebuilt
    other = dcfg(synth, isolate=False)
    assert cache_key(cfg, "h") != cache_key(other, "h") != cache_key(cfg, "h2")
    d2 = build_crop_cache(cfg, idx, workers=2, force=True)             # parallel == serial
    assert np.array_equal(load_crops(d2), arr)


def test_crop_sources_differ_as_designed(synth):
    idx = load_index(dcfg(synth).run_dir, "4way", ())
    gray = load_crops(build_crop_cache(dcfg(synth, source="clean_gray"), idx, workers=1))
    assert (gray[..., 0] == gray[..., 1]).all() and (gray[..., 1] == gray[..., 2]).all()
    raw = load_crops(build_crop_cache(dcfg(synth, source="raw", isolate=False), idx, workers=1))
    norm = load_crops(build_crop_cache(dcfg(synth, source="norm", isolate=False), idx, workers=1))
    assert raw.shape == norm.shape and not np.array_equal(raw, norm)
    with pytest.raises(ValueError, match="input_dir"):
        crops_mod._image_pixels(DataConfig(source="raw", input_dir=None), None, "x", None)


def test_isolated_crops_leak_neither_scale_bar_nor_background_class(synth):
    """After isolation the border region of every crop is the same neutral colour for all
    classes; with isolate=False it is not (that is the shortcut we remove)."""
    idx = load_index(dcfg(synth).run_dir, "4way", ())
    iso = load_crops(build_crop_cache(dcfg(synth), idx, workers=1))
    vis = load_crops(build_crop_cache(dcfg(synth, source="norm", isolate=False), idx, workers=1))

    def border_mean(a):
        b = np.concatenate([a[:, :3].reshape(len(a), -1), a[:, -3:].reshape(len(a), -1)], axis=1)
        return b.mean(1)
    spread_iso = [border_mean(iso[idx.class_name == c]).mean() for c in CLASSES]
    assert np.ptp(spread_iso) < 2.0
    assert np.ptp([border_mean(vis[idx.class_name == c]).mean() for c in CLASSES]) > 0.0


# ------------------------------------------------------------------------- augment
def test_augment_is_deterministic_shape_preserving_and_keeps_fill():
    img = np.full((48, 48, 3), FILL, np.uint8)
    cv2.circle(img, (24, 24), 12, (150, 90, 200), -1)
    cfg = AugConfig(colour="hed", grayscale_p=0.0, blur_p=0.0)
    a = augment(img, cfg, np.random.default_rng(3))
    b = augment(img, cfg, np.random.default_rng(3))
    c = augment(img, cfg, np.random.default_rng(4))
    assert a.shape == img.shape and a.dtype == np.uint8 and np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert abs(float(a[:3, :3].mean()) - FILL) < 6                     # background stays neutral
    gray = augment(img, AugConfig(colour="none", grayscale_p=1.0, scale_jitter=0, blur_p=0),
                   np.random.default_rng(0))
    assert (gray[..., 0] == gray[..., 1]).all()
    for colour in ("none", "hsv", "hed"):
        assert augment(img, AugConfig(colour=colour), np.random.default_rng(0)).shape == img.shape


# ------------------------------------------------------------------------- datasets
def test_bag_dataset_caps_cells_is_deterministic_and_collates(synth):
    cfg = dcfg(synth)
    idx = load_index(cfg.run_dir, cfg.task, cfg.feature_groups)
    crops = load_crops(build_crop_cache(cfg, idx, workers=1))
    train_imgs = idx.image_indices("train")
    scaler = FeatureScaler(idx.features[np.isin(idx.image_of_cell, train_imgs)])
    ds = BagDataset(idx, crops, train_imgs, train=True, max_cells=2, aug=AugConfig(),
                    scaler=scaler, input_size=40)
    item = ds[0]
    assert item["crops"].shape[1:] == (3, 40, 40) and len(item["crops"]) <= 2
    ds.set_epoch(0)
    again = ds[0]
    assert torch_equal(item["crops"], again["crops"])
    ds.set_epoch(1)
    assert not torch_equal(ds[0]["crops"], item["crops"])
    ev = BagDataset(idx, crops, idx.image_indices("test"), train=False, max_cells=8, scaler=scaler)
    assert torch_equal(ev[0]["crops"], ev[0]["crops"])
    batch = collate_bags([ds[0], ds[1], ds[2]])
    assert batch["crops"].shape[0] == int(batch["n"].sum()) == len(batch["bag"]) == len(batch["pos"])
    assert batch["bag"].tolist() == sorted(batch["bag"].tolist()) and batch["feats"].shape[1] == idx.features.shape[1]
    assert batch["label"].tolist() == [int(idx.image_label[i]) for i in batch["image"].tolist()]


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)


def test_feature_scaler_uses_train_statistics_and_imputes():
    tr = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]], np.float32)
    s = FeatureScaler(tr)
    out = s(tr)
    assert np.isfinite(out).all() and abs(out[:, 0].mean()) < 1e-6
    new = s(np.array([[np.nan, 20.0]], np.float32))
    assert np.isfinite(new).all()
    empty = FeatureScaler(np.zeros((4, 0), np.float32))
    assert empty(np.zeros((2, 0), np.float32)).shape == (2, 0)


def test_embedding_dataset_matches_index(synth):
    cfg = dcfg(synth)
    idx = load_index(cfg.run_dir, cfg.task, ())
    emb = np.random.default_rng(0).normal(size=(idx.n_cells, 7)).astype(np.float32)
    ds = EmbeddingBagDataset(idx, emb, idx.image_indices("val"), train=False, max_cells=64)
    item = ds[0]
    assert np.allclose(item["crops"].numpy(), emb[item["rows"].numpy()])
    assert item["label"] == int(idx.image_label[item["image"]])
    assert json.dumps(len(ds))

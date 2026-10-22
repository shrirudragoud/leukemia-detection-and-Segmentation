"""Single-cell crop cache.

Crops are cut once from the `leukemia-pp` cache (or the raw images) and stored as one
memory-mapped uint8 array aligned with the `CellIndex` rows, so training never decompresses
.npz files or re-segments anything. The cache directory name contains a hash of every
parameter that influences the pixels (source, isolate, margin, size, run config hash), so a
stale cache cannot be reused by accident.

Geometry: square crop around the cell's bounding box, expanded by `margin`, padded with the
neutral fill colour where it leaves the image, resized to `cache_size`. With `isolate=True`
everything except this cell (dilated by 2 px) is replaced by the neutral colour, so neighbours,
RBCs, background and labels carry no information about the class.
"""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from leukemia_pp.io import read_image

from ..config import DataConfig
from .index import CellIndex, load_index

log = logging.getLogger(__name__)

FILL = 232          # neutral grey, ~ L* 92 (the pipeline's neutral fill)
DILATE = 2


def cache_key(cfg: DataConfig, run_hash: str) -> str:
    blob = json.dumps({"source": cfg.source, "isolate": cfg.isolate, "size": cfg.cache_size,
                       "margin": cfg.margin, "run": run_hash, "excl": cfg.exclude_border_cells},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def cut_crop(img: np.ndarray, labels: np.ndarray, cell_id: int, size: int, margin: float,
             isolate: bool) -> np.ndarray:
    """One (size, size, 3) uint8 crop of cell `cell_id` from an HxWx3 RGB image."""
    mask = labels == cell_id
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if isolate:
        keep = ndi.binary_dilation(mask, iterations=DILATE)
        img = np.where(keep[..., None], img, np.uint8(FILL))
    side = max(8, int(round(max(y1 - y0, x1 - x0) * (1.0 + 2.0 * margin))))
    top = int(round((y0 + y1) / 2.0 - side / 2.0))
    left = int(round((x0 + x1) / 2.0 - side / 2.0))
    h, w = labels.shape
    pad = max(0, -top, -left, top + side - h, left + side - w)
    if pad:
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), constant_values=FILL)
        top, left = top + pad, left + pad
    patch = img[top:top + side, left:left + side]
    interp = cv2.INTER_AREA if side >= size else cv2.INTER_CUBIC
    return cv2.resize(patch, (size, size), interpolation=interp)


def _image_pixels(cfg: DataConfig, run_dir: Path, rel_path: str, npz) -> np.ndarray:
    if cfg.source == "raw":
        if not cfg.input_dir:
            raise ValueError("source='raw' needs DataConfig.input_dir")
        bgr = read_image(Path(cfg.input_dir) / rel_path)
        return np.ascontiguousarray(bgr[..., ::-1])
    if cfg.source == "norm":
        return npz["rgb_norm"]
    if cfg.source == "clean_rgb":
        return npz["rgb_clean"]
    gray = np.clip(npz["gray_clean"] * 255.0, 0, 255).astype(np.uint8)       # clean_gray
    return np.repeat(gray[..., None], 3, axis=2)


def _work(job):
    cfg_dict, run_dir, image_id, split, rel_path, cell_ids = job
    cfg = DataConfig(**cfg_dict)
    npz = np.load(Path(run_dir) / "cache" / split / f"{image_id}.npz")
    img = _image_pixels(cfg, Path(run_dir), rel_path, npz)
    labels = npz["cell_labels"]
    return [cut_crop(img, labels, c, cfg.cache_size, cfg.margin, cfg.isolate) for c in cell_ids]


def build_crop_cache(cfg: DataConfig, index: CellIndex | None = None, workers: int = 4,
                     force: bool = False) -> Path:
    """Build (or reuse) the crop cache; returns its directory containing `crops.npy`."""
    import csv
    run_dir = Path(cfg.run_dir)
    run_hash = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config_hash"]
    out = run_dir / "ml_cache" / f"crops_{cache_key(cfg, run_hash)}"
    index = index or load_index(run_dir, cfg.task, (), cfg.exclude_border_cells)
    if (out / "crops.npy").exists() and not force:
        arr = np.load(out / "crops.npy", mmap_mode="r")
        if arr.shape[0] == index.n_cells:
            return out
        log.warning("stale crop cache (%d rows != %d cells); rebuilding", arr.shape[0], index.n_cells)
    out.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "manifest.csv", newline="", encoding="utf-8") as fh:
        man = {r["image_id"]: r for r in csv.DictReader(fh)}
    jobs, order = [], []
    for i, image_id in enumerate(index.images):
        rows = index.cells_of_image(i)
        if len(rows) == 0:
            continue
        m = man[image_id]
        jobs.append((cfg.__dict__, str(run_dir), image_id, m["split"], m["rel_path"],
                     [int(c) for c in index.cell_id[rows]]))
        order.append(rows)
    arr = np.lib.format.open_memmap(out / "crops.npy", mode="w+", dtype=np.uint8,
                                    shape=(index.n_cells, cfg.cache_size, cfg.cache_size, 3))
    if workers <= 1:
        results = map(_work, jobs)
    else:
        pool = mp.Pool(workers)
        results = pool.imap(_work, jobs, chunksize=8)
    for rows, crops in zip(order, results):
        arr[rows] = np.stack(crops)
    if workers > 1:
        pool.close()
        pool.join()
    arr.flush()
    (out / "meta.json").write_text(json.dumps({"config": cfg.__dict__, "run_hash": run_hash,
                                               "n_cells": index.n_cells}, indent=2, default=list))
    log.info("crop cache: %d crops -> %s", index.n_cells, out)
    return out

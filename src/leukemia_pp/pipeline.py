"""End-to-end orchestration: raw images -> tensors, cell table, image table, manifest.

Output layout (`<output>/`):
    config.json            resolved config (+ hash, package version)
    stain_reference.json   Reinhard reference fitted on the TRAIN split only
    manifest.csv           one row per input image, incl. split and sha256
    cache/<split>/<id>.npz tensors (see ARRAY_KEYS)
    previews/<split>/<id>.png  QC panels
    cells.csv              one row per segmented cell
    images.csv             one row per image: QC + aggregated features
    failures.csv           images that could not be processed (absent when none)
    run_summary.json
"""
from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import __version__, denoise, features, qc, representations, responses, stain
from .config import PipelineConfig
from .io import ImageReadError, Sample, discover, read_image
from .segmentation import Segmentation, segment
from .splits import assign_splits

log = logging.getLogger(__name__)

BASE_ARRAY_KEYS = ("rgb_norm", "rgb", "apc", "log", "canny", "a_score",
                   "cell_labels", "nucleus_labels", "cell_mask", "nucleus_mask")
ARRAY_KEYS = BASE_ARRAY_KEYS      # backwards-compatible alias; see array_keys(cfg)


def array_keys(cfg: PipelineConfig) -> tuple[str, ...]:
    extra: list[str] = []
    if cfg.hds.enabled:
        extra += ["hds", "hds_edge"]
    if cfg.beemd.enabled:
        extra += ["imfs", "imf_selected", "imf_pure", "imf_residue"]
    return BASE_ARRAY_KEYS + tuple(extra)


def cell_table_columns(cfg: PipelineConfig) -> list[str]:
    return (["image_id", "class_name", "class_id_binary", "class_id_4way", "split"]
            + features.CELL_COLUMNS + representations.channel_columns(cfg))


def image_columns(cfg: PipelineConfig) -> list[str]:
    return IMAGE_META_COLUMNS + QC_COLUMNS + features.aggregate_columns(_agg_extra(cfg))


def _agg_extra(cfg: PipelineConfig) -> list[str]:
    """Image-level aggregation of the per-cell channel statistics (means and IMF energies)."""
    return [c for c in representations.channel_columns(cfg)
            if c.endswith("_mean") and not c.endswith("rim_mean") or c.endswith("_energy")]
MANIFEST_COLUMNS = ["image_id", "rel_path", "class_name", "class_id_binary", "class_id_4way",
                    "split", "sha256"]
IMAGE_META_COLUMNS = ["image_id", "class_name", "class_id_binary", "class_id_4way", "split",
                      "height", "width"]
QC_COLUMNS = ["sharpness", "brightness", "clipped_frac", "wbc_area_frac", "a_threshold",
              "qc_flags"]
IMAGE_COLUMNS = IMAGE_META_COLUMNS + QC_COLUMNS + features.AGG_COLUMNS      # default config
CELL_TABLE_COLUMNS = ["image_id", "class_name", "class_id_binary", "class_id_4way", "split"
                      ] + features.CELL_COLUMNS                              # default config


@dataclass
class ImageResult:
    """Everything computed for one image. Pure data: no file I/O involved."""
    arrays: dict[str, np.ndarray]
    seg: Segmentation
    cell_rows: list[dict]
    image_row: dict
    flags: list[str]


def process_array(raw_bgr: np.ndarray, cfg: PipelineConfig,
                  reference: stain.ReinhardReference | None) -> ImageResult:
    """The scientific core, independent of files, splits and multiprocessing."""
    stain_applied = True
    if cfg.stain.enabled:
        if reference is None:
            raise ValueError("stain normalisation is enabled but no reference was provided")
        norm_bgr, stain_applied = stain.normalize(raw_bgr, reference, cfg.stain)
    else:
        norm_bgr = raw_bgr
    den_bgr = denoise.denoise(norm_bgr, cfg.denoise)

    gray = responses.gray_float(den_bgr)
    seg = segment(stain.bgr_to_lab(den_bgr), cfg.segmentation)

    rgb_norm = np.ascontiguousarray(norm_bgr[..., ::-1])
    arrays = {
        "rgb_norm": rgb_norm,
        "rgb": np.ascontiguousarray(den_bgr[..., ::-1]),
        "apc": responses.apc_response(gray, cfg.response.apc_sigma, cfg.response.apc_alpha,
                                       cfg.response.apc_gain),
        "log": responses.log_response(gray, cfg.response.log_sigma, cfg.response.log_gain),
        "canny": responses.canny_edges(gray, cfg.response.canny_low_ratio),
        "a_score": seg.a_score,
        "cell_labels": seg.cell_labels,
        "nucleus_labels": seg.nucleus_labels,
        "cell_mask": seg.cell_mask,
        "nucleus_mask": seg.nucleus_mask,
    }
    arrays.update(representations.compute(norm_bgr, cfg))
    cell_rows = features.extract_cell_features(rgb_norm, seg, cfg.features)
    chan = representations.channel_features(arrays, seg, cfg)
    for row in cell_rows:
        row.update(chan[row["cell_id"]])
    n_border = sum(r["touches_border"] for r in cell_rows)
    measures = qc.measure(raw_bgr, seg)
    flags = qc.flags(measures, seg, n_border, cfg.qc)
    if cfg.stain.enabled and not stain_applied:
        flags.append("stain_not_applied")
    image_row = {
        "height": raw_bgr.shape[0], "width": raw_bgr.shape[1],
        **measures, "a_threshold": seg.a_threshold, "qc_flags": ";".join(flags),
        **features.aggregate_image(cell_rows, _agg_extra(cfg)),
    }
    return ImageResult(arrays, seg, cell_rows, image_row, flags)


# ------------------------------------------------------------------------- worker side
_WORKER: dict = {}


def _init_worker(cfg: PipelineConfig, reference, output_dir: Path) -> None:
    cv2.setNumThreads(1)          # we parallelise over images; avoid thread oversubscription
    _WORKER.update(cfg=cfg, reference=reference, out=Path(output_dir))


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _process_sample(job: tuple[Sample, str]) -> dict:
    sample, split = job
    cfg: PipelineConfig = _WORKER["cfg"]
    out: Path = _WORKER["out"]
    meta = {
        "image_id": sample.image_id, "class_name": sample.class_name,
        "class_id_binary": cfg.class_map[sample.class_name][0],
        "class_id_4way": cfg.class_map[sample.class_name][1], "split": split,
    }
    try:
        raw = read_image(sample.path, cfg.data.max_side)
        res = process_array(raw, cfg, _WORKER["reference"])

        def _save_npz(tmp: Path) -> None:
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **res.arrays)
        _atomic_write(out / "cache" / split / f"{sample.image_id}.npz", _save_npz)

        if cfg.save_previews:
            from .preview import save_preview
            title = f"{sample.class_name} / {sample.image_id} / {split}"
            _atomic_write(out / "previews" / split / f"{sample.image_id}.png",
                          lambda tmp: save_preview(tmp, raw, res, cfg, title))
        cells = [{**meta, **r} for r in res.cell_rows]
        return {"ok": True, "image": {**meta, **res.image_row}, "cells": cells}
    except Exception as e:  # noqa: BLE001 - one bad image must not abort the whole run
        log.exception("failed: %s", sample.rel_path)
        return {"ok": False, "failure": {"image_id": sample.image_id, "rel_path": sample.rel_path,
                                         "error_type": type(e).__name__, "error": str(e)}}


# ------------------------------------------------------------------------------- run
def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    def _write(tmp: Path) -> None:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=columns, extrasaction="raise")
            w.writeheader()
            w.writerows(rows)
    _atomic_write(path, _write)


def _guard_config(output_dir: Path, cfg: PipelineConfig, force: bool) -> None:
    path = output_dir / "config.json"
    if path.exists() and not force:
        previous = json.loads(path.read_text(encoding="utf-8")).get("config_hash")
        if previous != cfg.hash():
            raise RuntimeError(
                f"{output_dir} was produced with a different configuration "
                f"({previous} != {cfg.hash()}). Use a new output directory or pass force=True.")
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"package_version": __version__, "config_hash": cfg.hash(),
                                "config": cfg.to_dict()}, indent=2), encoding="utf-8")


def fit_stain_reference(samples: list[Sample], splits: dict[str, str], cfg: PipelineConfig
                        ) -> stain.ReinhardReference:
    """Fit on TRAIN images of all classes (no val/test leakage, no class-colour bias).

    Candidates are visited in a seeded random order; unreadable files and frames with too
    little foreground are skipped until `ref_sample_size` usable images are collected."""
    pool = [s for s in samples if splits[s.image_id] == "train"] or list(samples)
    order = random.Random(cfg.split.seed).sample(pool, len(pool))

    def usable():
        taken = 0
        for s in order:
            if taken >= cfg.stain.ref_sample_size:
                return
            try:
                img = read_image(s.path, cfg.data.max_side)
            except ImageReadError as e:
                log.warning("skipping %s for stain reference: %s", s.rel_path, e)
                continue
            if stain.foreground_stats(stain.bgr_to_lab(img), cfg.stain.min_foreground_frac):
                taken += 1
                yield img
    return stain.fit_reference(usable(), cfg.stain)


def run(input_dir: Path, output_dir: Path, cfg: PipelineConfig | None = None,
        workers: int | None = None, dry_run: bool = False, force: bool = False) -> dict:
    cfg = cfg or PipelineConfig()
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    t0 = time.time()

    samples = discover(input_dir, cfg.class_map)
    if not samples:
        raise FileNotFoundError(
            f"no images under {input_dir}; expected subfolders {sorted(cfg.class_map)}")
    splits = assign_splits(samples, cfg.split)
    counts = Counter((s.class_name, splits[s.image_id]) for s in samples)
    for cname in sorted({s.class_name for s in samples}):
        log.info("%-8s train=%d val=%d test=%d", cname,
                 *(counts[(cname, sp)] for sp in ("train", "val", "test")))

    _guard_config(output_dir, cfg, force)
    _write_csv(output_dir / "manifest.csv", MANIFEST_COLUMNS, [
        {"image_id": s.image_id, "rel_path": s.rel_path, "class_name": s.class_name,
         "class_id_binary": cfg.class_map[s.class_name][0],
         "class_id_4way": cfg.class_map[s.class_name][1],
         "split": splits[s.image_id], "sha256": s.sha256} for s in samples])
    if dry_run:
        return {"n_images": len(samples), "dry_run": True}

    reference = None
    if cfg.stain.enabled:
        reference = fit_stain_reference(samples, splits, cfg)
        reference.save(output_dir / "stain_reference.json")
        log.info("stain reference fitted on %d image(s)", reference.n_images)

    n_workers = workers or os.cpu_count() or 1
    jobs = [(s, splits[s.image_id]) for s in samples]
    log.info("processing %d images with %d worker(s)", len(jobs), n_workers)
    if n_workers == 1:
        _init_worker(cfg, reference, output_dir)
        results = [_process_sample(j) for j in jobs]
    else:
        with mp.Pool(n_workers, initializer=_init_worker,
                     initargs=(cfg, reference, output_dir)) as pool:
            results = list(pool.imap_unordered(_process_sample, jobs, chunksize=4))

    ok = [r for r in results if r["ok"]]
    failures = sorted((r["failure"] for r in results if not r["ok"]), key=lambda f: f["image_id"])
    image_rows = sorted((r["image"] for r in ok), key=lambda r: r["image_id"])
    cell_rows = sorted((c for r in ok for c in r["cells"]),
                       key=lambda c: (c["image_id"], c["cell_id"]))
    _write_csv(output_dir / "images.csv", image_columns(cfg), image_rows)
    _write_csv(output_dir / "cells.csv", cell_table_columns(cfg), cell_rows)
    failures_path = output_dir / "failures.csv"
    if failures:
        _write_csv(failures_path, ["image_id", "rel_path", "error_type", "error"], failures)
    elif failures_path.exists():
        failures_path.unlink()

    flag_counts = Counter(f for r in image_rows for f in r["qc_flags"].split(";") if f)
    summary = {
        "package_version": __version__, "config_hash": cfg.hash(),
        "n_images": len(samples), "n_ok": len(ok), "n_failed": len(failures),
        "n_cells": len(cell_rows), "qc_flag_counts": dict(flag_counts),
        "seconds": round(time.time() - t0, 1),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("done: %d/%d ok, %d cells, %d failed", len(ok), len(samples), len(cell_rows),
             len(failures))
    return summary

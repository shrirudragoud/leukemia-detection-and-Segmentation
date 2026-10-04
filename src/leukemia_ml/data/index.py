"""Cell index: one row per segmented cell with label, split, patient-proxy group and features.

Everything is read from a `leukemia-pp run` output directory; groups come from the same
`assign_groups` the preprocessing split used, so cross-validation here can never separate
members of one group that preprocessing kept together.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from leukemia_pp import features as ppf
from leukemia_pp.config import PipelineConfig
from leukemia_pp.dataset import FEATURE_GROUPS, layer_feature_group
from leukemia_pp.io import Sample
from leukemia_pp.splits import assign_groups

CLASS_NAMES_4 = ("Benign", "Early", "Pre", "Pro")
CLASS_NAMES_2 = ("Benign", "Malignant")


def feature_columns(groups: tuple[str, ...]) -> list[str]:
    """Resolve feature-group names to `cells.csv` columns, keeping only existing ones in order."""
    cols: list[str] = []
    for g in groups:
        if g in FEATURE_GROUPS:
            cols += FEATURE_GROUPS[g]
        elif g in ("apc", "log", "hds_edge", "imf_pure", "imf_energy"):
            cols += layer_feature_group(g)
        elif g == "density":
            continue                              # image-level, deliberately not a cell feature
        else:
            raise ValueError(f"unknown feature group {g!r}")
    return list(dict.fromkeys(cols))


@dataclass
class CellIndex:
    image_id: np.ndarray            # (N,) str
    cell_id: np.ndarray             # (N,) int
    label: np.ndarray               # (N,) int, per `task`
    class_name: np.ndarray          # (N,) str
    split: np.ndarray               # (N,) str
    group: np.ndarray               # (N,) str patient-proxy group
    area: np.ndarray                # (N,) float
    touches_border: np.ndarray      # (N,) bool
    features: np.ndarray            # (N, F) float32, NaN where undefined
    feature_names: list[str]
    class_labels: tuple[str, ...]
    # per image
    images: np.ndarray              # (I,) str unique image ids, sorted
    image_of_cell: np.ndarray       # (N,) int index into `images`
    image_label: np.ndarray         # (I,) int
    image_split: np.ndarray         # (I,) str
    image_group: np.ndarray         # (I,) str
    image_class: np.ndarray         # (I,) str

    @property
    def n_cells(self) -> int:
        return len(self.image_id)

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def n_classes(self) -> int:
        return len(self.class_labels)

    def cells_of_image(self, i: int) -> np.ndarray:
        return self._by_image[i]

    def __post_init__(self) -> None:
        order = np.argsort(self.image_of_cell, kind="stable")
        counts = np.bincount(self.image_of_cell, minlength=len(self.images))
        self._by_image = np.split(order, np.cumsum(counts)[:-1])

    def image_indices(self, split: str | None = None, mask: np.ndarray | None = None) -> np.ndarray:
        keep = np.ones(self.n_images, bool)
        if split is not None:
            keep &= self.image_split == split
        if mask is not None:
            keep &= mask
        return np.nonzero(keep)[0]


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(v: str) -> float:
    return float("nan") if v in ("", "nan", "NaN") else float(v)


def load_index(run_dir: str | Path, task: str = "4way",
               feature_groups: tuple[str, ...] = ("cell_shape", "cell_texture", "apc", "log"),
               exclude_border: bool = False) -> CellIndex:
    run_dir = Path(run_dir)
    manifest = _read_csv(run_dir / "manifest.csv")
    cfg = PipelineConfig.from_dict(
        json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"])
    samples = [Sample(r["image_id"], Path(r["rel_path"]), r["rel_path"], r["class_name"],
                      r["sha256"]) for r in manifest]
    group_of = assign_groups(samples, cfg.split)
    label_key = "class_id_4way" if task == "4way" else "class_id_binary"
    class_labels = CLASS_NAMES_4 if task == "4way" else CLASS_NAMES_2
    meta = {r["image_id"]: r for r in manifest}

    cols = feature_columns(feature_groups)
    rows = sorted(_read_csv(run_dir / "cells.csv"), key=lambda r: (r["image_id"], int(r["cell_id"])))
    if cols and rows:
        missing = [c for c in cols if c not in rows[0]]
        if missing:
            raise ValueError(f"cells.csv lacks columns {missing[:4]}...; re-run the pipeline "
                             "with the matching representations enabled")
    if exclude_border:
        rows = [r for r in rows if r["touches_border"] != "True"]

    image_id = np.array([r["image_id"] for r in rows])
    images = np.array(sorted({r["image_id"] for r in manifest}))
    pos = {im: i for i, im in enumerate(images)}
    keep_images = np.array([im in meta for im in images])
    assert keep_images.all()
    return CellIndex(
        image_id=image_id,
        cell_id=np.array([int(r["cell_id"]) for r in rows]),
        label=np.array([int(meta[r["image_id"]][label_key]) for r in rows]),
        class_name=np.array([meta[r["image_id"]]["class_name"] for r in rows]),
        split=np.array([meta[r["image_id"]]["split"] for r in rows]),
        group=np.array([group_of[r["image_id"]] for r in rows]),
        area=np.array([_f(r["cell_area"]) for r in rows]),
        touches_border=np.array([r["touches_border"] == "True" for r in rows]),
        features=(np.array([[_f(r[c]) for c in cols] for r in rows], np.float32)
                  if cols else np.zeros((len(rows), 0), np.float32)),
        feature_names=cols,
        class_labels=class_labels,
        images=images,
        image_of_cell=np.array([pos[i] for i in image_id]),
        image_label=np.array([int(meta[i][label_key]) for i in images]),
        image_split=np.array([meta[i]["split"] for i in images]),
        image_group=np.array([group_of[i] for i in images]),
        image_class=np.array([meta[i]["class_name"] for i in images]),
    )


__all__ = ["CellIndex", "load_index", "feature_columns", "ppf"]

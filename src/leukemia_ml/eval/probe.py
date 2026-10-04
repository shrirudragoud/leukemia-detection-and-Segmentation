"""Grouped-CV linear probes on frozen embeddings (+ optional interpretable features).

The workhorse of the ablation study: cheap enough to repeat over every preprocessing
variant, encoder and feature set, with the SAME group-disjoint folds each time.
"""
from __future__ import annotations

import numpy as np

from ..data.index import CellIndex
from .folds import contiguous_folds, grouped_folds, purge_train
from .metrics import summarize


def image_features(index: CellIndex, emb: np.ndarray, feats: np.ndarray | None = None,
                   agg: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Per-image vectors: mean (or max) of cell embeddings [+ mean of cell features].
    Returns (X, image_idx) for images that have at least one cell."""
    rows, idx = [], []
    for i in range(index.n_images):
        cells = index.cells_of_image(i)
        if len(cells) == 0:
            continue
        e = emb[cells].astype(np.float32)
        v = [e.mean(0) if agg == "mean" else e.max(0)]
        if feats is not None:
            v.append(np.nan_to_num(feats[cells]).mean(0))
        rows.append(np.concatenate(v))
        idx.append(i)
    return np.array(rows), np.array(idx)


def image_numbers(index: CellIndex, image_idx: np.ndarray) -> np.ndarray:
    """Trailing integer of each image id (capture-order proxy); -1 if absent."""
    import re
    out = []
    for i in image_idx:
        m = re.search(r"(\d+)$", str(index.images[i]))
        out.append(int(m.group(1)) if m else -1)
    return np.array(out)


def run_probe(index: CellIndex, X: np.ndarray, image_idx: np.ndarray, n_splits: int = 5,
              seeds=(0, 1, 2), C: float = 1.0, embargo: int = 0, scheme: str = "grouped") -> dict:
    """Out-of-fold probabilities per seed + metric summary. Image-level, group-disjoint folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = index.image_label[image_idx]
    groups = index.image_group[image_idx]
    n_classes = index.n_classes
    oof = {s: np.full((len(y), n_classes), np.nan) for s in seeds}
    order = image_numbers(index, image_idx)
    n_train = []
    if scheme == "contiguous":
        # `seeds` select rotations of the segment boundaries; embargo is applied inside
        folds = (f for r in seeds for f in contiguous_folds(order, y, n_splits, embargo, rotate=r))
    elif scheme == "grouped":
        folds = ((seed, fo, purge_train(tr, te, y, order, embargo), te)
                 for seed, fo, tr, te in grouped_folds(groups, y, n_splits, seeds))
    else:
        raise ValueError("scheme must be 'grouped' or 'contiguous'")
    for seed, _fold, tr, te in folds:
        n_train.append(len(tr))
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, max_iter=500, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        proba = np.zeros((len(te), n_classes))
        proba[:, clf.classes_] = clf.predict_proba(X[te])
        oof[seed][te] = proba
    per_seed = [summarize(y, oof[s]) for s in seeds]
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "auroc_ovr", "ece")
    return {
        "mean": {k: float(np.mean([m[k] for m in per_seed])) for k in keys},
        "std": {k: float(np.std([m[k] for m in per_seed])) for k in keys},
        "per_seed": per_seed, "oof": oof, "y": y, "groups": groups, "image_idx": image_idx,
        "n_images": int(len(y)), "n_groups": int(len(set(groups))),
        "embargo": embargo, "scheme": scheme, "mean_train_size": float(np.mean(n_train)),
    }

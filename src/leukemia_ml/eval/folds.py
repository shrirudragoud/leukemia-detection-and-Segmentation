"""Repeated, class-stratified, GROUP-disjoint cross-validation folds."""
from __future__ import annotations

import numpy as np


def grouped_folds(groups: np.ndarray, y: np.ndarray, n_splits: int = 5, seeds=(0, 1, 2)):
    """Yield (seed, fold, train_idx, test_idx). No group ever appears in both train and test;
    classes are balanced across folds as well as the group structure allows."""
    from sklearn.model_selection import StratifiedGroupKFold
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            assert not set(groups[tr]) & set(groups[te])
            yield seed, fold, tr, te


def purge_train(train_idx: np.ndarray, test_idx: np.ndarray, y: np.ndarray, order: np.ndarray,
                embargo: int) -> np.ndarray:
    """Remove from `train_idx` every sample whose sequence number is within `embargo` of a test
    sample of the SAME class. Acquisition sessions are contiguous in the numbering, so this
    removes training samples that share a session/patient with a test sample even when the
    group blocks do not line up with the true patient boundaries."""
    if embargo <= 0 or len(test_idx) == 0:
        return train_idx
    keep = np.ones(len(train_idx), bool)
    for c in np.unique(y[test_idx]):
        t = np.sort(order[test_idx[y[test_idx] == c]])
        tr_mask = y[train_idx] == c
        pos = np.searchsorted(t, order[train_idx[tr_mask]])
        lo = np.abs(order[train_idx[tr_mask]] - t[np.clip(pos - 1, 0, len(t) - 1)])
        hi = np.abs(order[train_idx[tr_mask]] - t[np.clip(pos, 0, len(t) - 1)])
        keep[np.nonzero(tr_mask)[0][np.minimum(lo, hi) <= embargo]] = False
    return train_idx[keep]


def contiguous_folds(order: np.ndarray, y: np.ndarray, n_splits: int = 5, embargo: int = 0,
                     rotate: int = 0):
    """Session-aware folds: within each class, sort by sequence number and cut the sequence into
    `n_splits` CONSECUTIVE segments; fold k tests segment k of every class.

    Images are numbered in capture order and sessions/patients are contiguous in that numbering,
    so a test segment shares sessions with training images only at its two edges (at most one
    patient per edge); `embargo` then removes training images within that many numbers of a test
    image. Contrast with random assignment of fixed blocks, where almost every block has a
    same-patient neighbour in the training set. `rotate` shifts segment boundaries (a different
    but equally valid partition) so repeats are possible."""
    fold_of = np.zeros(len(y), int)
    for c in np.unique(y):
        members = np.nonzero(y == c)[0]
        ranked = members[np.argsort(order[members], kind="stable")]
        shift = (rotate * len(ranked)) // (n_splits * 3 if n_splits else 1)
        ranked = np.roll(ranked, -shift)
        for k, seg in enumerate(np.array_split(np.arange(len(ranked)), n_splits)):
            fold_of[ranked[seg]] = k
    for k in range(n_splits):
        te = np.nonzero(fold_of == k)[0]
        tr = np.nonzero(fold_of != k)[0]
        yield rotate, k, purge_train(tr, te, y, order, embargo), te

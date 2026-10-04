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

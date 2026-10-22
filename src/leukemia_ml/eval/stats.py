"""Uncertainty and significance for classifier comparison.

Why not plain image-level statistics: images of one patient/session are strongly correlated,
so resampling images treats ~3,000 correlated items as independent and gives intervals that
are far too narrow. We resample *groups* (cluster bootstrap). For repeated k-fold CV the usual
paired t-test is miscalibrated because training sets overlap; the Nadeau-Bengio correction
inflates the variance accordingly.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def cluster_bootstrap(metric: Callable[[np.ndarray], float], groups: np.ndarray, n_boot: int = 2000,
                      seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """`metric(idx)` evaluates the metric on the sample indices `idx`; groups are resampled
    with replacement and every member of a drawn group enters the sample."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inverse == g)[0] for g in range(len(uniq))]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.integers(0, len(uniq), len(uniq))
        vals[b] = metric(np.concatenate([members[g] for g in drawn]))
    vals = vals[np.isfinite(vals)]
    point = float(metric(np.arange(len(groups))))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "lo": float(lo), "hi": float(hi), "se": float(vals.std(ddof=1)),
            "n_groups": int(len(uniq))}


def paired_bootstrap(metric_a: Callable[[np.ndarray], float], metric_b: Callable[[np.ndarray], float],
                     groups: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Difference metric_a - metric_b on identical resampled groups; two-sided p from the
    bootstrap distribution of the difference (share of draws on the other side of zero)."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inverse == g)[0] for g in range(len(uniq))]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        diffs[b] = metric_a(idx) - metric_b(idx)
    diffs = diffs[np.isfinite(diffs)]
    point = float(metric_a(np.arange(len(groups))) - metric_b(np.arange(len(groups))))
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": point, "lo": float(lo), "hi": float(hi), "p": float(min(1.0, max(p, 1 / n_boot)))}


def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> dict[str, float]:
    """Nadeau & Bengio (2003): t = mean(d) / sqrt((1/J + n_test/n_train) * var(d))."""
    from scipy import stats
    d = np.asarray(diffs, float)
    j = len(d)
    var = d.var(ddof=1)
    if j < 2 or var == 0:
        return {"t": float("nan"), "p": float("nan"), "mean": float(d.mean()) if j else float("nan")}
    t = d.mean() / math.sqrt((1.0 / j + n_test / n_train) * var)
    return {"t": float(t), "p": float(2 * stats.t.sf(abs(t), j - 1)), "mean": float(d.mean())}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()

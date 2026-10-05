"""Classification metrics for image-level predictions (probabilities (N,C), labels (N,))."""
from __future__ import annotations

import numpy as np


def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), np.int64)
    np.add.at(m, (y, pred), 1)
    return m


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    return np.divide(np.diag(cm), support, out=np.full(len(cm), np.nan), where=support > 0)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    r = per_class_recall(confusion(y, pred, n_classes))
    return float(np.nanmean(r))


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    cm = confusion(y, pred, n_classes)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(axis=0), out=np.zeros(n_classes), where=cm.sum(axis=0) > 0)
    rec = np.divide(tp, cm.sum(axis=1), out=np.zeros(n_classes), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n_classes), where=(prec + rec) > 0)
    present = cm.sum(axis=1) > 0
    return float(f1[present].mean())


def auroc_ovr(y: np.ndarray, probs: np.ndarray) -> float:
    """Macro one-vs-rest AUROC (rank-based, ties averaged); classes absent in `y` are skipped."""
    from scipy.stats import rankdata
    aucs = []
    for c in range(probs.shape[1]):
        pos = y == c
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(probs[:, c])
        aucs.append((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(aucs)) if aucs else float("nan")


def ece(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top-label confidence."""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def binary_sens_spec(y: np.ndarray, pred: np.ndarray, benign_class: int = 0) -> dict[str, float]:
    """Benign-vs-malignant sensitivity/specificity from multi-class labels (malignant = not benign)."""
    pos, pred_pos = y != benign_class, pred != benign_class
    sens = float((pred_pos & pos).sum() / max(pos.sum(), 1))
    spec = float((~pred_pos & ~pos).sum() / max((~pos).sum(), 1))
    return {"sensitivity": sens, "specificity": spec}


def summarize(y: np.ndarray, probs: np.ndarray, benign_class: int = 0) -> dict:
    n_classes = probs.shape[1]
    pred = probs.argmax(axis=1)
    cm = confusion(y, pred, n_classes)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": balanced_accuracy(y, pred, n_classes),
        "macro_f1": macro_f1(y, pred, n_classes),
        "auroc_ovr": auroc_ovr(y, probs),
        "ece": ece(y, probs),
        "per_class_recall": [None if np.isnan(r) else float(r) for r in per_class_recall(cm)],
        "confusion": cm.tolist(),
        **binary_sens_spec(y, pred, benign_class),
    }

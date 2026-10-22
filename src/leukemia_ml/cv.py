"""Session-aware cross-validation: train one model per contiguous fold and pool the results.

This is the entry point for a "full run". Every fold uses the same preprocessing, config and
seed; fold k tests contiguous segment k of every class (with an embargo around the held-out
segments, see data/splits.py). Pooled out-of-fold predictions give the headline numbers with
group-level (cluster) bootstrap intervals; per-fold numbers show the spread. Re-running the
command resumes: finished folds are skipped, an interrupted fold continues from `last.pt`.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .eval import stats
from .eval.metrics import balanced_accuracy, macro_f1, summarize
from .train.loop import Trainer

log = logging.getLogger(__name__)


def _read_predictions(path: Path, n_classes: int):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([int(r["label"]) for r in rows])
    probs = np.array([[float(r[f"p_{k}"]) for k in range(n_classes)] for r in rows])
    return np.array([r["image_id"] for r in rows]), np.array([r["group"] for r in rows]), y, probs


def run_cv(cfg: ExperimentConfig, out_dir: str | Path, folds: list[int] | None = None,
           n_boot: int = 1000) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_folds = cfg.data.n_folds
    folds = list(range(n_folds)) if folds is None else folds
    per_fold = []
    for k in folds:
        fcfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, test_fold=k))
        log.info("=== fold %d/%d ===", k + 1, n_folds)
        res = Trainer(fcfg, out / f"fold{k}").fit()
        per_fold.append({"fold": k, "test": res["test"], "epochs": res["epochs_run"],
                         "seconds": res["seconds"], "split": res.get("split"),
                         "temperature": res["temperature"]})

    n_classes = len(per_fold[0]["test"]["confusion"])
    ids, groups, y, probs = [], [], [], []
    for k in folds:
        i, g, yy, pp = _read_predictions(out / f"fold{k}" / "predictions.csv", n_classes)
        ids.append(i), groups.append(g), y.append(yy), probs.append(pp)
    ids, groups, y, probs = map(np.concatenate, (ids, groups, y, probs))
    assert len(set(ids)) == len(ids), "an image was tested in more than one fold"
    pred = probs.argmax(1)
    pooled = summarize(y, probs)
    ci = {
        "balanced_accuracy": stats.cluster_bootstrap(
            lambda i: balanced_accuracy(y[i], pred[i], n_classes), groups, n_boot),
        "macro_f1": stats.cluster_bootstrap(
            lambda i: macro_f1(y[i], pred[i], n_classes), groups, n_boot),
    }
    keys = ("balanced_accuracy", "macro_f1", "auroc_ovr", "ece")
    summary = {
        "name": cfg.name, "config_hash": cfg.hash(), "folds": folds, "n_images": int(len(y)),
        "n_groups": int(len(set(groups))), "pooled": pooled, "pooled_ci": ci,
        "per_fold_mean": {k: float(np.mean([f["test"][k] for f in per_fold])) for k in keys},
        "per_fold_sd": {k: float(np.std([f["test"][k] for f in per_fold], ddof=1))
                        if len(per_fold) > 1 else float("nan") for k in keys},
        "per_fold": per_fold,
        "protocol": (f"{n_folds} contiguous session-aware folds per class, embargo "
                     f"{cfg.data.embargo} image numbers; one model per fold; pooled out-of-fold "
                     "predictions; 95% cluster-bootstrap CI over groups"),
    }
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out / "cv_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(s: dict) -> str:
    p, ci = s["pooled"], s["pooled_ci"]
    lines = [f"# CV summary: {s['name']}", "", s["protocol"], "",
             f"{s['n_images']} images, {s['n_groups']} groups, folds {s['folds']}.", "",
             "| Metric | Pooled | 95% CI | Per-fold mean +- SD |", "| --- | --- | --- | --- |"]
    for k, label in (("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1")):
        lines.append(f"| {label} | {p[k]:.3f} | [{ci[k]['lo']:.3f}, {ci[k]['hi']:.3f}] | "
                     f"{s['per_fold_mean'][k]:.3f} +- {s['per_fold_sd'][k]:.3f} |")
    lines.append(f"| AUROC (OvR) | {p['auroc_ovr']:.3f} | - | "
                 f"{s['per_fold_mean']['auroc_ovr']:.3f} +- {s['per_fold_sd']['auroc_ovr']:.3f} |")
    lines.append(f"| ECE | {p['ece']:.3f} | - | {s['per_fold_mean']['ece']:.3f} |")
    lines += ["", f"Benign-vs-malignant: sensitivity {p['sensitivity']:.3f}, specificity "
              f"{p['specificity']:.3f}.", "", "Per-class recall: "
              + ", ".join("n/a" if r is None else f"{r:.3f}" for r in p["per_class_recall"]),
              "", "Confusion (rows = true):", ""]
    lines += ["    " + " ".join(f"{v:5d}" for v in row) for row in p["confusion"]]
    return "\n".join(lines) + "\n"

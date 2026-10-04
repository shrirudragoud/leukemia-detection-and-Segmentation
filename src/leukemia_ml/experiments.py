"""Ablation driver: compare preprocessing / encoder / feature variants with one protocol.

Every variant is evaluated with the SAME group-disjoint, class-stratified, repeated folds (the
folds depend only on image groups and labels), on frozen embeddings (+ optional interpretable
features) with a logistic head. Uncertainty comes from a cluster bootstrap over groups; the
difference to the reference (first variant) from a paired cluster bootstrap on identical
resampled groups, Holm-corrected across variants. Out-of-fold predictions of seed 0 feed the
intervals; the point estimate in the table is the mean over seeds.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data.crops import build_crop_cache
from .data.datasets import load_crops
from .data.index import load_index
from .embed import extract_embeddings
from .eval import stats
from .eval.metrics import balanced_accuracy, macro_f1
from .eval.probe import image_features, run_probe

log = logging.getLogger(__name__)


def _apply(base: ExperimentConfig, variant: dict) -> ExperimentConfig:
    d = base.to_dict()
    for section in ("data", "model", "aug", "train"):
        d[section].update(variant.get(section, {}))
    return ExperimentConfig.from_dict(d)


def _variant_features(variant: dict, cfg: ExperimentConfig, device: str):
    """Image-level matrix X for one variant, aligned to `idx`."""
    idx = load_index(cfg.data.run_dir, cfg.data.task, cfg.data.feature_groups,
                     cfg.data.exclude_border_cells)
    if variant.get("tabular"):                         # interpretable features only, no pixels
        zeros = np.zeros((idx.n_cells, 1), np.float32)
        X, img = image_features(idx, zeros, idx.features)
        return idx, X[:, 1:], img
    d = build_crop_cache(cfg.data, idx, workers=4)
    emb, _ = extract_embeddings(load_crops(d), cfg.model, d, device)
    X, img = image_features(idx, np.asarray(emb), idx.features if variant.get("features") else None)
    return idx, X, img


def run_probe_ablation(base: ExperimentConfig, variants: list[dict], out_dir: str | Path,
                       seeds=(0, 1, 2), n_boot: int = 1000, device: str = "auto") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for v in variants:
        cfg = _apply(base, v)
        log.info("variant %s", v["name"])
        idx, X, img = _variant_features(v, cfg, device)
        res = run_probe(idx, X, img, seeds=seeds)
        runs.append({"name": v["name"], "idx": idx, "res": res, "cfg_hash": cfg.hash()})

    # align every variant on the images they all contain
    common = set.intersection(*[set(idx.images[r["res"]["image_idx"]]) for r in runs
                                for idx in [r["idx"]]])
    rows = []
    ref = None
    n_classes = runs[0]["idx"].n_classes
    pvals = []
    for r in runs:
        idx, res = r["idx"], r["res"]
        ids = idx.images[res["image_idx"]]
        keep = np.array([i in common for i in ids])
        order = np.argsort(ids[keep])
        y = res["y"][keep][order]
        groups = res["groups"][keep][order]
        oof = res["oof"][seeds[0]][keep][order]
        pred = oof.argmax(1)
        ba = lambda i, y=y, p=pred: balanced_accuracy(y[i], p[i], n_classes)   # noqa: E731
        f1 = lambda i, y=y, p=pred: macro_f1(y[i], p[i], n_classes)            # noqa: E731
        row = {
            "variant": r["name"], "cfg_hash": r["cfg_hash"], "n_images": int(keep.sum()),
            "balanced_accuracy": res["mean"]["balanced_accuracy"],
            "balanced_accuracy_sd_seeds": res["std"]["balanced_accuracy"],
            "balanced_accuracy_ci": stats.cluster_bootstrap(ba, groups, n_boot),
            "macro_f1": res["mean"]["macro_f1"],
            "macro_f1_ci": stats.cluster_bootstrap(f1, groups, n_boot),
            "auroc_ovr": res["mean"]["auroc_ovr"],
        }
        if ref is None:
            ref = (ba, groups)
        else:
            cmp = stats.paired_bootstrap(ba, ref[0], groups, n_boot)
            row["vs_reference"] = cmp
            pvals.append(cmp["p"])
        rows.append(row)
    adj = stats.holm(pvals) if pvals else []
    for row, p in zip([r for r in rows if "vs_reference" in r], adj):
        row["vs_reference"]["p_holm"] = p

    report = {"reference": rows[0]["variant"], "seeds": list(seeds), "n_boot": n_boot,
              "n_groups": int(len(set(runs[0]["res"]["groups"]))), "rows": rows,
              "protocol": ("image-level, group-disjoint stratified repeated 5-fold CV; logistic "
                           "head on frozen embeddings; cluster bootstrap over groups")}
    (out / "ablation.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    (out / "ablation.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    lines = [f"# Ablation (reference: {report['reference']})", "",
             f"{report['n_groups']} groups; seeds {report['seeds']}; {report['n_boot']} bootstrap "
             "resamples of groups. Balanced accuracy, 95% cluster-bootstrap CI.", "",
             "| Variant | Balanced acc. | 95% CI | Macro-F1 | AUROC | Diff. vs ref. (95% CI) | p (Holm) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in report["rows"]:
        ci = r["balanced_accuracy_ci"]
        cmp = r.get("vs_reference")
        diff = f"{cmp['diff']:+.3f} [{cmp['lo']:+.3f}, {cmp['hi']:+.3f}]" if cmp else "-"
        p = f"{cmp['p_holm']:.3f}" if cmp else "-"
        lines.append(f"| {r['variant']} | {r['balanced_accuracy']:.3f} | [{ci['lo']:.3f}, {ci['hi']:.3f}] "
                     f"| {r['macro_f1']:.3f} | {r['auroc_ovr']:.3f} | {diff} | {p} |")
    return "\n".join(lines) + "\n"


def load_variants(path: str | Path) -> tuple[ExperimentConfig, list[dict]]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    base = ExperimentConfig.from_dict(spec.get("base", {}))
    return base, copy.deepcopy(spec["variants"])


__all__ = ["run_probe_ablation", "load_variants", "render_markdown", "dataclasses"]

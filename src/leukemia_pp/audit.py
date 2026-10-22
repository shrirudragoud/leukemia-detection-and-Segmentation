"""Shortcut / confound audit: how well can the class be predicted from things that are NOT
leukemia biology?

Motivation: in this dataset each patient belongs to exactly one class and was imaged in a
separate session, so slide background tint, scale-bar style and stain colour can all predict
the class. A network that exploits them scores highly without learning anything clinical.
The audit measures, with grouped cross-validation (groups = patient proxy from the split
configuration, so no group is ever in both train and test folds):

  * "acquisition probes": background colour (raw / normalised), scale-bar corner patch;
  * "residual probes" on what a model would see after shortcut removal: cell colour (raw
    cache vs white-balanced `rgb_clean`), corner/background of `rgb_clean` (constant by
    construction, reported as a sanity check);
  * tabular baselines from `images.csv`: density, shape, texture, colour, combinations.

Everything is reported as balanced accuracy (mean +- std over RandomForest seeds) next to the
chance level, with a plain-language verdict. Results go to `<run>/audit.json` and
`<run>/audit.md` so a paper can cite exact numbers.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .io import Sample, read_image
from .splits import assign_groups
from .stain import bgr_to_lab

log = logging.getLogger(__name__)

CORNER_FRAC = (0.22, 0.30)          # bottom-right patch (height, width fractions): scale bar area
CORNER_GRID = (12, 16)              # patch is downsampled to this grid -> 192 features


# ------------------------------------------------------------------------- feature probes
def background_lab(bgr: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Median Lab of the brightest 30% of (non-`keep`) pixels."""
    lab = bgr_to_lab(bgr).reshape(-1, 3)
    if keep is not None:
        lab = lab[~keep.reshape(-1)]
    if lab.shape[0] == 0:
        return np.zeros(3)
    return np.median(lab[lab[:, 0] >= np.percentile(lab[:, 0], 70)], axis=0)


def corner_patch(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    ph, pw = max(8, round(h * CORNER_FRAC[0])), max(8, round(w * CORNER_FRAC[1]))
    gray = cv2.cvtColor(bgr[-ph:, -pw:], cv2.COLOR_BGR2GRAY)
    return (cv2.resize(gray, (CORNER_GRID[1], CORNER_GRID[0]), interpolation=cv2.INTER_AREA)
            .astype(np.float64).ravel() / 255.0)


def cell_colour(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.full(3, np.nan)
    lab = bgr_to_lab(np.ascontiguousarray(rgb[..., ::-1]))
    return lab[mask].mean(axis=0)


def _read_manifest(run_dir: Path) -> list[dict]:
    with open(run_dir / "manifest.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def extract_probe_features(run_dir: Path, input_dir: Path | None) -> dict[str, np.ndarray]:
    """One row per manifest image for every probe feature block that can be computed."""
    rows = _read_manifest(run_dir)
    feats: dict[str, list[np.ndarray]] = {k: [] for k in (
        "bg_raw", "corner_raw", "bg_norm", "cell_colour_norm",
        "cell_colour_clean", "bg_clean", "corner_clean")}
    have_clean = True
    for r in rows:
        z = np.load(run_dir / "cache" / r["split"] / f"{r['image_id']}.npz")
        if input_dir is not None:
            raw = read_image(Path(input_dir) / r["rel_path"])
            feats["bg_raw"].append(background_lab(raw))
            feats["corner_raw"].append(corner_patch(raw))
        norm_rgb = z["rgb_norm"]
        feats["bg_norm"].append(background_lab(np.ascontiguousarray(norm_rgb[..., ::-1])))
        mask = z["cell_mask"]
        feats["cell_colour_norm"].append(cell_colour(norm_rgb, mask))
        if "rgb_clean" in z.files:
            clean = z["rgb_clean"]
            feats["cell_colour_clean"].append(cell_colour(clean, mask))
            bgr = np.ascontiguousarray(clean[..., ::-1])
            feats["bg_clean"].append(background_lab(bgr))
            feats["corner_clean"].append(corner_patch(bgr))
        else:
            have_clean = False
    out = {k: np.array(v) for k, v in feats.items() if v}
    if not have_clean:
        for k in ("cell_colour_clean", "bg_clean", "corner_clean"):
            out.pop(k, None)
    return out


# --------------------------------------------------------------------------- evaluation
def _grouped_cv_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seeds, n_splits: int,
                       n_trees: int = 300) -> tuple[float, float]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import GroupKFold, cross_val_predict
    except ImportError as e:   # pragma: no cover
        raise RuntimeError("the audit needs scikit-learn: pip install 'leukemia-pp[audit]'") from e
    X = np.where(np.isfinite(X), X, np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0))
    scores = []
    for seed in seeds:
        clf = RandomForestClassifier(n_trees, random_state=seed, n_jobs=-1, class_weight="balanced")
        pred = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(n_splits))
        scores.append(balanced_accuracy_score(y, pred))
    return float(np.mean(scores)), float(np.std(scores))


def verdict(score: float, chance: float) -> str:
    gap = score - chance
    if gap < 0.10:
        return "none (near chance)"
    if gap < 0.25:
        return "weak"
    if gap < 0.45:
        return "STRONG"
    return "VERY STRONG"


TABULAR_SETS: dict[str, tuple[str, ...]] = {
    "density (n_cells, wbc_area_frac)": ("n_cells", "wbc_area_frac"),
    "cell shape": ("cell_area", "cell_circularity", "cell_solidity", "cell_aspect"),
    "cell texture (GLCM, entropy)": ("cell_glcm", "cell_entropy"),
    "cell colour": ("cell_L_", "cell_a_"),
    "shape + texture (no colour, no density)": (
        "cell_area", "cell_circularity", "cell_solidity", "cell_aspect", "cell_glcm",
        "cell_entropy"),
}


def _tabular_blocks(run_dir: Path, order: list[str]) -> dict[str, np.ndarray]:
    with open(run_dir / "images.csv", newline="", encoding="utf-8") as fh:
        rows = {r["image_id"]: r for r in csv.DictReader(fh)}
    cols = [c for c in next(iter(rows.values())) if c.endswith(("_mean", "_std", "_max"))]
    cols += ["n_cells", "wbc_area_frac"]

    def block(keys):
        use = [c for c in cols if any(c.startswith(k) or k in c for k in keys)]
        return np.array([[float(rows[i][c]) if rows[i][c] not in ("", "nan") else np.nan
                          for c in use] for i in order]) if use else None
    out = {name: block(keys) for name, keys in TABULAR_SETS.items()}
    out["ALL aggregated features"] = np.array(
        [[float(rows[i][c]) if rows[i][c] not in ("", "nan") else np.nan for c in cols]
         for i in order])
    return {k: v for k, v in out.items() if v is not None and v.size}


def run_audit(run_dir: Path, input_dir: Path | None = None, seeds=(0, 1, 2),
              n_splits: int = 5, n_trees: int = 300) -> dict:
    run_dir = Path(run_dir)
    manifest = _read_manifest(run_dir)
    cfg = PipelineConfig.from_json_dict(
        json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"])
    samples = [Sample(r["image_id"], Path(r["rel_path"]), r["rel_path"], r["class_name"],
                      r["sha256"]) for r in manifest]
    group_of = assign_groups(samples, cfg.split)
    order = [s.image_id for s in samples]
    y = np.array([s.class_name for s in samples])
    groups = np.array([group_of[i] for i in order])
    classes, counts = np.unique(y, return_counts=True)
    chance = 1.0 / len(classes)
    n_splits = min(n_splits, len(set(groups)))
    log.info("audit: %d images, %d groups, %d folds", len(samples), len(set(groups)), n_splits)

    probes: list[tuple[str, str, np.ndarray]] = []
    feats = extract_probe_features(run_dir, input_dir)
    names = {
        "bg_raw": ("acquisition", "background colour, raw image (3 numbers)"),
        "bg_norm": ("acquisition", "background colour, after stain normalisation (3 numbers)"),
        "corner_raw": ("acquisition", "scale-bar corner patch, raw image"),
        "bg_clean": ("after shortcut removal", "background colour of rgb_clean (sanity: constant)"),
        "corner_clean": ("after shortcut removal", "corner patch of rgb_clean (sanity: constant)"),
        "cell_colour_norm": ("residual", "mean cell colour, stain-normalised (3 numbers)"),
        "cell_colour_clean": ("residual", "mean cell colour, white-balanced rgb_clean (3 numbers)"),
    }
    for key, (group, label) in names.items():
        if key in feats:
            probes.append((group, label, feats[key]))
    for label, X in _tabular_blocks(run_dir, order).items():
        probes.append(("tabular baseline", label, X))

    results = []
    for group, label, X in probes:
        mean, std = _grouped_cv_scores(X, y, groups, seeds, n_splits, n_trees)
        results.append({"group": group, "probe": label, "balanced_accuracy": round(mean, 4),
                        "std": round(std, 4), "verdict": verdict(mean, chance)
                        if group != "tabular baseline" else ""})
        log.info("%-24s %-62s %.3f +- %.3f", group, label, mean, std)

    report = {
        "n_images": len(samples), "n_groups": int(len(set(groups))), "n_splits": n_splits,
        "classes": dict(zip(classes.tolist(), counts.tolist())), "chance_balanced": chance,
        "seeds": list(seeds), "results": results,
        "config_hash": cfg.hash(),
        "caveats": [
            "Groups are a patient PROXY (consecutive-number blocks + duplicate files); "
            "true patient ids are unavailable, so residual leakage is possible.",
            "RandomForest baseline, 3 seeds; fold assignment is fixed by the groups.",
            "Near-chance acquisition probes after cleaning only show that the cleaned image "
            "no longer carries them; cell colour can still encode stain/session.",
        ],
    }
    (run_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "audit.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Shortcut / confound audit", "",
        f"{report['n_images']} images, {report['n_groups']} patient-proxy groups, "
        f"{report['n_splits']}-fold grouped CV, classes {report['classes']}. "
        f"Chance (balanced accuracy) = {report['chance_balanced']:.2f}.", "",
        "| Group | Predictor | Balanced acc. | Shortcut risk |", "| --- | --- | --- | --- |"]
    for r in report["results"]:
        lines.append(f"| {r['group']} | {r['probe']} | {r['balanced_accuracy']:.2f} "
                     f"+- {r['std']:.2f} | {r['verdict']} |")
    lines += ["", "Caveats:"] + [f"- {c}" for c in report["caveats"]]
    return "\n".join(lines) + "\n"

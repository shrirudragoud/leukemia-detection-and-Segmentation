"""Figures that depend on the GPU runs. Run after the queue has produced results_for_paper/:

    python scripts/make_ft_figures.py                      # real results -> results_for_paper/figs/{FT-CURVES,FT-BARS,FT-CM}.png
    python scripts/make_ft_figures.py --sample             # SYNTHETIC data -> paper/figures/sample_{TAG}.png (watermarked; used as placeholders)

Missing runs are skipped with a message; the script never fails because one run is absent."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results_for_paper")
ap.add_argument("--out", default=None)
ap.add_argument("--sample", action="store_true")
a = ap.parse_args()
RES = Path("paper/sample_results" if a.sample else a.results)
OUT = Path(a.out or ("paper/figures" if a.sample else "results_for_paper/figs"))
OUT.mkdir(parents=True, exist_ok=True)
CL = ["Benign", "Early", "Pre", "Pro"]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
SHORT = {"g01": "g01 LoRA DinoBloom-S", "g02": "g02 LoRA gray", "g03": "g03 LoRA + features", "g04": "g04 ResNet-50 full", "g05": "g05 EfficientNet-B0 full",
         "g06": "g06 LoRA DINOv2-S", "g07": "g07 LoRA DinoBloom-B", "g08": "g08 LoRA raw + background"}


def save(fig, tag):
    if a.sample:
        fig.text(0.5, 0.5, "SAMPLE DATA - NOT RESULTS", ha="center", va="center", rotation=25, fontsize=26, color="0.55", alpha=0.45)
        fig.text(0.5, 0.01, f"tag {tag}: replace by running  python scripts/make_ft_figures.py  after the GPU runs", ha="center", fontsize=6.5, color="0.3")
        path = OUT / f"sample_{tag}.png"
    else:
        path = OUT / f"{tag}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); print("wrote", path)


def runs():
    out = {}
    for d in sorted(RES.glob("g0*")):
        f = d / "cv_summary.json"
        if f.exists():
            out[d.name.split("_")[0]] = (d, json.loads(f.read_text()))
    return out


R = runs()
if not R:
    print("no runs found in", RES); raise SystemExit(0)

# ---- FT-CURVES: training curves of the first available LoRA run (g01 preferred)
key = "g01" if "g01" in R else next(iter(R))
d, _ = R[key]
logs = [[json.loads(line) for line in open(f)] for f in sorted(d.glob("fold*_train_log.jsonl"))]
if logs:
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8))
    for k, L in enumerate(logs):
        st = dict(color=str(0.1 + 0.16 * k), ls=["-", "--", "-.", ":", (0, (5, 1))][k % 5], lw=1.1)
        ep = [r["epoch"] + 1 for r in L]
        ax[0].plot(ep, [r["train_loss"] for r in L], label=f"fold {k + 1}", **st)
        ax[1].plot(ep, [r["val_macro_f1"] for r in L], **st)
        ax[2].plot(ep, [r["lr"][0] for r in L], **st)
        b = int(np.argmax([r["val_macro_f1"] for r in L]))
        ax[1].plot(ep[b], L[b]["val_macro_f1"], "k*", ms=7)
    for x, t, y in zip(ax, ["(a) Training loss", "(b) Validation macro-F1 (star: best epoch)", "(c) Learning rate"], ["loss", "macro-F1", "rate"]):
        x.set_title(t, fontsize=8); x.set_xlabel("Epoch"); x.set_ylabel(y, fontsize=8); x.grid(ls=":")
    ax[0].legend(fontsize=6)
    fig.tight_layout(); save(fig, "FT-CURVES")
else:
    print("no train logs for", key)

# ---- FT-BARS: balanced accuracy with intervals for every run, sorted, reference line from the frozen headline model
ref = json.loads(Path("docs/results/cv_dinobloom_s_frozen_mil.json").read_text())["pooled"]["balanced_accuracy"]
rows = []
for k, (d, s) in R.items():
    ci = s["pooled_ci"]["balanced_accuracy"]
    rows.append((SHORT.get(k, k), s["pooled"]["balanced_accuracy"], ci["lo"], ci["hi"]))
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(rows) + 1.2))
for i, (nme, v, lo, hi) in enumerate(rows):
    ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="s", color="black", capsize=3)
ax.axvline(ref, color="0.4", ls="--", lw=1); ax.text(ref, len(rows) - 0.4, " frozen headline model", fontsize=7, color="0.3", va="bottom")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.set_xlabel("Balanced accuracy (95% cluster-bootstrap CI)"); ax.grid(axis="x", ls=":")
fig.tight_layout(); save(fig, "FT-BARS")

# ---- FT-CM: pooled confusion matrix of the best run
best = max(R.items(), key=lambda kv: kv[1][1]["pooled"]["balanced_accuracy"])
cm = np.array(best[1][1]["pooled"]["confusion"], float)
fig, ax = plt.subplots(figsize=(4.0, 3.6))
ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Greys", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black", fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(CL, fontsize=8); ax.set_yticks(range(4)); ax.set_yticklabels(CL, fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Run {best[0]}, pooled out-of-fold", fontsize=9)
fig.tight_layout(); save(fig, "FT-CM")

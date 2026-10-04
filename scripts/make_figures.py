"""Draw every paper figure from the committed result files (docs/results/*.json). Grayscale-safe."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

R = Path("docs/results")
F = Path("paper/figures")
F.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
CL = ["Benign", "Early", "Pre", "Pro"]


def load(n):
    return json.load(open(R / n))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    row1 = ["Raw RGB image\n224 x 224", "Stain\nnormalisation\n(Reinhard, Lab)", "Denoising\n(bilateral, NLM,\nstabilised HDS)",
            "Segmentation\n(Lab a* Otsu,\nwatershed)", "Per-cell crops\n(shortcut\nneutralised)"]
    row2 = ["Frozen encoder\n(DinoBloom-S)", "Image-level\nclassifier", "Per-cell\nfeatures", "Response layers\n(APC, LoG, Canny)"]
    w, h = 1.7, 1.5
    def box(x, y, t):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fc="#eeeeee", ec="black"))
        ax.text(x, y, t, ha="center", va="center", fontsize=7.5)
    xs = [1.0, 3.0, 5.0, 7.0, 9.0]
    for x, t in zip(xs, row1):
        box(x, 4.8, t)
    for i in range(4):
        ax.annotate("", xy=(xs[i + 1] - w / 2, 4.8), xytext=(xs[i] + w / 2, 4.8), arrowprops=dict(arrowstyle="->"))
    xs2 = [9.0, 7.0, 5.0, 3.0]
    for x, t in zip(xs2, row2):
        box(x, 1.2, t)
    ax.annotate("", xy=(9.0, 1.95), xytext=(9.0, 4.05), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(7.0 + w / 2, 1.2), xytext=(9.0 - w / 2, 1.2), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(7.0 - w / 2, 1.2), xytext=(5.0 + w / 2, 1.2), arrowprops=dict(arrowstyle="->", ls=":"))
    ax.annotate("", xy=(5.0 - w / 2, 1.2), xytext=(3.0 + w / 2, 1.2), arrowprops=dict(arrowstyle="->", ls=":"))
    ax.annotate("", xy=(3.2, 1.95), xytext=(4.8, 4.05), arrowprops=dict(arrowstyle="<-", ls=":"))
    ax.text(5.0, 0.05, "(solid: used for the headline model; dotted: tabular branch used in ablations)", ha="center", fontsize=7)
    fig.savefig(F / "fig_pipeline.png", bbox_inches="tight"); plt.close(fig)


def fig_examples():
    im = Image.open(F / "seg_examples_src.png")
    im.thumbnail((1000, 1500)); im.save(F / "fig_examples.png")
    c = Image.open(F / "classes_src.png"); c.thumbnail((1400, 1100)); c.save(F / "fig_classes.png")


def fig_ablation():
    d = load("ablation_sources_contiguous.json")["rows"]
    lab = [r["variant"].split(".")[0] for r in d]
    full = [r["variant"] for r in d]
    y = np.array([r["balanced_accuracy_ci"]["point"] for r in d])
    lo = y - np.array([r["balanced_accuracy_ci"]["lo"] for r in d])
    hi = np.array([r["balanced_accuracy_ci"]["hi"] for r in d]) - y
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.errorbar(range(len(d)), y, yerr=[lo, hi], fmt="s", color="black", capsize=4)
    ax.set_xticks(range(len(d))); ax.set_xticklabels(lab)
    ax.set_ylabel("Balanced accuracy"); ax.set_xlabel("Input variant (see Table legend)")
    ax.set_ylim(0.78, 1.0); ax.grid(axis="y", ls=":")
    fig.savefig(F / "fig_ablation.png", bbox_inches="tight"); plt.close(fig)
    return full


def fig_leakage():
    L = load("leakage_analysis.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    B = L["B_random_block_purge"]
    x = [b["embargo"] for b in B]
    ax[0].plot(x, [b["purged"] for b in B], "ko-", label="purge training images near test images")
    ax[0].plot(x, [b["random_same_size"] for b in B], "ks--", mfc="white", label="remove same number at random")
    ax[0].set_title("(a) Random-block folds", fontsize=9); ax[0].set_xlabel("Purge distance (capture numbers)")
    ax[0].set_ylabel("Balanced accuracy"); ax[0].legend(fontsize=7); ax[0].grid(ls=":")
    C = L["C_contiguous_gap"]
    x = [c["embargo"] for c in C]
    ax[1].plot(x, [c["dino"] for c in C], "ko-", label="DinoBloom-S, gap")
    ax[1].plot(x, [c["dino_rand"] for c in C], "ko--", mfc="white", label="DinoBloom-S, random")
    ax[1].plot(x, [c["hand"] for c in C], "k^-", label="hand-made, gap")
    ax[1].plot(x, [c["hand_rand"] for c in C], "k^--", mfc="white", label="hand-made, random")
    ax[1].set_title("(b) Contiguous folds", fontsize=9); ax[1].set_xlabel("Minimum gap to test segment")
    ax[1].legend(fontsize=7); ax[1].grid(ls=":")
    fig.tight_layout(); fig.savefig(F / "fig_leakage.png", bbox_inches="tight"); plt.close(fig)


def fig_confusion():
    d = load("cv_dinobloom_s_frozen_mil.json")
    cm = np.array(d["pooled"]["confusion"], float)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1, 1.1]})
    ax[0].imshow(cm / cm.sum(1, keepdims=True), cmap="Greys", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            v = cm[i, j] / cm[i].sum()
            ax[0].text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color="white" if v > 0.5 else "black", fontsize=8)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(CL, fontsize=8); ax[0].set_yticks(range(4)); ax[0].set_yticklabels(CL, fontsize=8)
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True"); ax[0].set_title("(a) Pooled out-of-fold", fontsize=9)
    pf = d["per_fold"]
    ax[1].bar(range(len(pf)), [p["test"]["balanced_accuracy"] for p in pf], color="0.5", edgecolor="black")
    ax[1].set_xticks(range(len(pf))); ax[1].set_xticklabels([f"Fold {p['fold']+1}" for p in pf], fontsize=8)
    ax[1].set_ylim(0.85, 1.0); ax[1].set_ylabel("Balanced accuracy"); ax[1].set_title("(b) Per fold", fontsize=9)
    fig.tight_layout(); fig.savefig(F / "fig_confusion.png", bbox_inches="tight"); plt.close(fig)


def fig_hds():
    h = load("hds_benchmark.json")["rows"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    s = [r["sigma"] for r in h]
    for k, m, ls in [("noisy", "x", ":"), ("gaussian", "s", "--"), ("bilateral", "^", "-"), ("stabilised_hds", "o", "-"), ("legacy_hds", "v", "-.")]:
        ax.plot(s, [r[k] for r in h], "k" + m, ls=ls, mfc="white" if k != "stabilised_hds" else "black", label=k.replace("_", " "))
    ax.set_xlabel("Noise SD"); ax.set_ylabel("PSNR (dB)"); ax.legend(fontsize=7); ax.grid(ls=":")
    fig.savefig(F / "fig_hds.png", bbox_inches="tight"); plt.close(fig)


def fig_session():
    s = load("session_structure.json")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(range(4), [s[c]["median_adjacent"] for c in CL], 0.38, color="0.35", edgecolor="black", label="adjacent capture numbers")
    ax.bar(np.arange(4) + 0.4, [s[c]["median_random"] for c in CL], 0.38, color="white", edgecolor="black", hatch="//", label="random pairs, same class")
    ax.set_xticks(np.arange(4) + 0.2); ax.set_xticklabels(CL); ax.set_ylabel("Median feature distance"); ax.legend(fontsize=7)
    fig.savefig(F / "fig_session.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_pipeline, fig_examples, fig_ablation, fig_leakage, fig_confusion, fig_hds, fig_session):
        f(); print("ok", f.__name__)

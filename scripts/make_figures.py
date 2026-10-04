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


# ---------------------------------------------------------------- figures from the headline cross-validation run
import csv  # noqa: E402

RUN = R / "cv_run"


def _preds():
    ids, y, P, nc, att = [], [], [], [], []
    for k in range(5):
        for r in csv.DictReader(open(RUN / f"fold{k}" / "predictions.csv")):
            ids.append(r["image_id"]); y.append(int(r["label"]))
            P.append([float(r[f"p_{j}"]) for j in range(4)]); nc.append(int(r["n_cells"])); att.append(float(r["top_attention"]))
    return np.array(ids), np.array(y), np.array(P), np.array(nc), np.array(att)


def fig_training():
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8))
    for k in range(5):
        h = [json.loads(line) for line in open(RUN / f"fold{k}" / "train_log.jsonl")]
        ep = [r["epoch"] + 1 for r in h]
        st = dict(color=str(0.1 + 0.16 * k), ls=["-", "--", "-.", ":", (0, (5, 1))][k], lw=1.1)
        ax[0].plot(ep, [r["train_loss"] for r in h], label=f"fold {k + 1}", **st)
        ax[1].plot(ep, [r["val_bal_acc"] for r in h], **st)
        ax[2].plot(ep, [r["lr"][0] for r in h], **st)
    for a, t, yl in zip(ax, ["(a) Training loss", "(b) Validation balanced accuracy", "(c) Learning rate"], ["loss", "balanced accuracy", "rate"]):
        a.set_title(t, fontsize=8); a.set_xlabel("Epoch"); a.set_ylabel(yl, fontsize=8); a.grid(ls=":")
    ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(F / "fig_training.png", bbox_inches="tight"); plt.close(fig)


def fig_roc_pr():
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    _, y, P, _, _ = _preds()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2))
    sty = ["-", "--", "-.", ":"]
    out = {}
    for j in range(4):
        fpr, tpr, _ = roc_curve(y == j, P[:, j]); pr, rc, _ = precision_recall_curve(y == j, P[:, j])
        out[CL[j]] = {"auroc": float(auc(fpr, tpr)), "auprc": float(auc(rc, pr))}
        ax[0].plot(fpr, tpr, "k" + sty[j], lw=1.2, label=f"{CL[j]} (AUC {out[CL[j]]['auroc']:.3f})")
        ax[1].plot(rc, pr, "k" + sty[j], lw=1.2, label=f"{CL[j]} (AP {out[CL[j]]['auprc']:.3f})")
    ax[0].plot([0, 1], [0, 1], color="0.6", lw=0.8); ax[0].set_xlabel("False-positive rate"); ax[0].set_ylabel("True-positive rate")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_ylim(0.5, 1.01)
    for a, t in zip(ax, ["(a) ROC, one-vs-rest", "(b) Precision-recall, one-vs-rest"]):
        a.set_title(t, fontsize=9); a.legend(fontsize=7, loc="lower right"); a.grid(ls=":")
    fig.tight_layout(); fig.savefig(F / "fig_roc_pr.png", bbox_inches="tight"); plt.close(fig)
    (R / "curve_summary.json").write_text(json.dumps(out, indent=1))


def fig_reliability():
    _, y, P, _, _ = _preds()
    conf, pred = P.max(1), P.argmax(1)
    bins = np.linspace(0.25, 1.0, 11)
    idx = np.digitize(conf, bins) - 1
    acc, cf, cnt = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum():
            acc.append((pred[m] == y[m]).mean()); cf.append(conf[m].mean()); cnt.append(int(m.sum()))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax[0].plot([0.25, 1], [0.25, 1], color="0.6"); ax[0].plot(cf, acc, "ko-")
    ax[0].set_xlabel("Mean confidence"); ax[0].set_ylabel("Accuracy"); ax[0].set_title("(a) Reliability diagram (calibrated)", fontsize=9); ax[0].grid(ls=":")
    ax[1].hist(conf[pred == y], bins=20, color="0.55", edgecolor="black", label="correct")
    ax[1].hist(conf[pred != y], bins=20, color="white", edgecolor="black", hatch="//", label="incorrect")
    ax[1].set_yscale("log"); ax[1].set_xlabel("Confidence"); ax[1].set_ylabel("Images (log scale)"); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) Confidence of correct and incorrect predictions", fontsize=9)
    fig.tight_layout(); fig.savefig(F / "fig_reliability.png", bbox_inches="tight"); plt.close(fig)


def fig_cells():
    _, y, P, nc, att = _preds()
    pred = P.argmax(1)
    edges = [0, 3, 5, 8, 12, 20, 100]
    labs, accs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (nc > lo) & (nc <= hi) if lo else (nc <= hi)
        if m.sum():
            labs.append(f"{lo + 1}-{hi}" if hi < 100 else f">{lo}"); accs.append((pred[m] == y[m]).mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax[0].bar(range(len(labs)), accs, color="0.55", edgecolor="black"); ax[0].set_xticks(range(len(labs))); ax[0].set_xticklabels(labs, fontsize=8)
    for i, (a, k) in enumerate(zip(accs, ns)):
        ax[0].text(i, a + 0.004, f"n={k}", ha="center", fontsize=6)
    ax[0].set_ylim(0.8, 1.02); ax[0].set_xlabel("Cells per image"); ax[0].set_ylabel("Accuracy"); ax[0].set_title("(a) Accuracy by cell count", fontsize=9)
    for j in range(4):
        ax[1].hist(att[y == j], bins=15, histtype="step", color="k", ls=["-", "--", "-.", ":"][j], label=CL[j])
    ax[1].set_xlabel("Largest attention weight in the image"); ax[1].set_ylabel("Images"); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) Concentration of attention", fontsize=9)
    fig.tight_layout(); fig.savefig(F / "fig_cells.png", bbox_inches="tight"); plt.close(fig)
    json.dump({"bins": labs, "acc": [float(a) for a in accs], "n": ns}, open(R / "cells_summary.json", "w"), indent=1)


if __name__ == "__main__":
    for f in (fig_training, fig_roc_pr, fig_reliability, fig_cells):
        f(); print("ok", f.__name__)

"""Explainability on the frozen DinoBloom-S + attention-MIL model (fold 0 of the session-aware CV).

1  attention statistics over ALL test images of the fold (is the pooling selective?)
2  Grad-CAM on the most-attended cell of example test images, and the parameter-randomisation sanity check
3  deletion curve (probability when the most salient vs random pixels are removed)
-> docs/results/xai_summary.json and paper/figures/fig_xai.png"""
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from leukemia_ml.config import ExperimentConfig  # noqa: E402
from leukemia_ml.data.crops import build_crop_cache  # noqa: E402
from leukemia_ml.data.datasets import load_crops, to_tensor  # noqa: E402
from leukemia_ml.data.index import load_index  # noqa: E402
from leukemia_ml.data.splits import contiguous_split  # noqa: E402
from leukemia_ml.embed import embedding_path  # noqa: E402
from leukemia_ml.models.model import LeukemiaModel, build_model  # noqa: E402
from leukemia_ml.xai.gradcam import deletion_curve, grad_cam, randomisation_sanity  # noqa: E402

cfg = ExperimentConfig.from_json("configs/ml/cv_dinobloom_s_frozen_mil.json")
run = Path(glob.glob("runs/cv_dinobloom_s_frozen_mil_*")[0])
state = torch.load(run / "fold0" / "model.pt", map_location="cpu", weights_only=False)
idx = load_index(cfg.data.run_dir, "4way", ())
idx.image_split = contiguous_split(idx, cfg.data.n_folds, 0, cfg.data.val_offset, cfg.data.embargo)
crops_dir = build_crop_cache(cfg.data, idx)
crops = load_crops(crops_dir)
emb = np.load(embedding_path(crops_dir, cfg.model)).astype(np.float32)
names = ["Benign", "Early", "Pre", "Pro"]

# ---------------------------------------------------------------- 1 attention statistics (embeddings only)
head_model = LeukemiaModel(None, 384, 4, cfg.model)
head_model.load_state_dict(state, strict=False)
head_model.eval()
ent, top, area_rho, n_cells = [], [], [], []
test_imgs = idx.image_indices("test")
with torch.no_grad():
    for i in test_imgs:
        rows = idx.cells_of_image(i)[: cfg.data.max_cells_eval]
        if len(rows) < 3:
            continue
        e = torch.from_numpy(emb[rows])
        n = torch.tensor([len(rows)])
        _, a = head_model(e, torch.zeros(len(rows), 0), torch.zeros(len(rows), dtype=torch.long),
                          torch.arange(len(rows)), n)
        a = a[0, : len(rows)].numpy()
        ent.append(float(-(a * np.log(a + 1e-12)).sum() / np.log(len(rows))))
        top.append(float(a.max() * len(rows)))
        n_cells.append(len(rows))
        if np.std(a) > 0 and np.std(idx.area[rows]) > 0:
            area_rho.append(float(spearmanr(a, idx.area[rows])[0]))
stats = {"n_images": len(ent), "normalised_entropy_mean": float(np.mean(ent)),
         "normalised_entropy_sd": float(np.std(ent, ddof=1)),
         "top_attention_over_uniform_mean": float(np.mean(top)),
         "attention_area_spearman_mean": float(np.mean(area_rho)),
         "attention_area_spearman_sd": float(np.std(area_rho, ddof=1)), "mean_cells": float(np.mean(n_cells))}
print(stats)

# ---------------------------------------------------------------- 2 Grad-CAM + sanity (needs the encoder)
full = build_model(cfg.model, 4)
full.load_state_dict(state, strict=False)
full.eval()
with open(run / "fold0" / "predictions.csv", newline="") as fh:
    preds = list(csv.DictReader(fh))
rng = np.random.default_rng(0)
chosen = []
for c in range(4):
    ok = [p for p in preds if int(p["label"]) == c and int(p["pred"]) == c and int(p["n_cells"]) >= 4]
    chosen.append(ok[int(rng.integers(0, len(ok)))])
fig, ax = plt.subplots(4, 3, figsize=(6.2, 8.0))
sanity_rows, deletion = [], []
for r, p in enumerate(chosen):
    i = int(np.nonzero(idx.images == p["image_id"])[0][0])
    rows = idx.cells_of_image(i)[: cfg.data.max_cells_eval]
    with torch.no_grad():
        e = torch.from_numpy(emb[rows])
        logits, a = head_model(e, torch.zeros(len(rows), 0), torch.zeros(len(rows), dtype=torch.long),
                               torch.arange(len(rows)), torch.tensor([len(rows)]))
    a = a[0, : len(rows)].numpy()
    best = int(np.argmax(a))
    x = to_tensor(np.asarray(crops[rows[best]]), cfg.model.input_size)[None]
    cam, cls = grad_cam(full, x, class_idx=int(p["label"]))
    img = x[0].permute(1, 2, 0).numpy()
    ax[r, 0].imshow(img)
    ax[r, 0].set_ylabel(names[int(p["label"])], fontsize=9)
    ax[r, 1].imshow(img)
    ax[r, 1].imshow(cam, cmap="gray_r", alpha=0.55, vmin=0, vmax=1)
    ax[r, 2].bar(range(len(a)), np.sort(a)[::-1], color="0.35")
    ax[r, 2].axhline(1 / len(a), color="k", lw=0.8, ls="--")
    ax[r, 2].set_xticks([])
    ax[r, 2].tick_params(labelsize=7)
    for k in range(2):
        ax[r, k].set_xticks([])
        ax[r, k].set_yticks([])
    if r == 0:
        ax[r, 0].set_title("A. Most-attended cell", fontsize=8)
        ax[r, 1].set_title("B. Grad-CAM", fontsize=8)
        ax[r, 2].set_title("C. Attention per cell", fontsize=8)
    san = randomisation_sanity(full, x)
    sanity_rows.append([s["spearman_with_original"] for s in san])
    stages = [s["randomised_through"] for s in san]
    d = deletion_curve(full, x, cam, int(p["label"]))
    deletion.append({"cam": d["cam"], "random": d["random"], "fractions": d["fractions"]})
plt.tight_layout()
Path("paper/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("paper/figures/fig_xai.png", dpi=300)

dm = np.mean([d["cam"] for d in deletion], axis=0)
rm = np.mean([d["random"] for d in deletion], axis=0)
summary = {"attention": stats, "gradcam_examples": [p["image_id"] for p in chosen],
           "sanity_stages": stages, "sanity_spearman_mean": np.mean(sanity_rows, axis=0).tolist(),
           "sanity_spearman_all": sanity_rows,
           "deletion": {"fractions": deletion[0]["fractions"], "cam_mean": dm.tolist(), "random_mean": rm.tolist()}}
Path("docs/results/xai_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["deletion"]), summary["sanity_spearman_mean"])

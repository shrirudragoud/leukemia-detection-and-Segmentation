"""Input/output figures for every processing stage, produced by running the real pipeline functions on real images.
Outputs: paper/figures/stage_*.png and docs/results/stage_examples.json (image ids, per-stage timings, IMF scales and selection, per-cell features of the example image)."""
import dataclasses
import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leukemia_pp import denoise, pipeline, representations, stain  # noqa: E402
from leukemia_pp.config import PipelineConfig  # noqa: E402
from leukemia_pp.emd import decompose  # noqa: E402
from leukemia_pp.hds import hds_denoise  # noqa: E402

F = Path("paper/figures")
RUN = Path("data/full_run_v2")
base = PipelineConfig.from_dict(json.loads((RUN / "config.json").read_text())["config"])
cfg = dataclasses.replace(base, hds=dataclasses.replace(base.hds, enabled=True), beemd=dataclasses.replace(base.beemd, enabled=True))
ref = stain.ReinhardReference.load(RUN / "stain_reference.json")
CL = ["Benign", "Early", "Pre", "Pro"]
imgs = pd.read_csv(RUN / "images.csv")
man = pd.read_csv(RUN / "manifest.csv").set_index("image_id")
plt.rcParams.update({"font.family": "serif", "font.size": 9})

# deterministic example images: the 10th image (by id) of each class whose cell count is within 1 of the class median
chosen = {}
for c in CL:
    med = imgs[imgs.class_name == c].n_cells.median()
    s = imgs[(imgs.class_name == c) & imgs.n_cells.between(med - 1, med + 1)].sort_values("image_id")
    chosen[c] = s.iloc[min(9, len(s) - 1)].image_id


def load(iid):
    return cv2.imread(str(Path("data/dataset/Original") / man.loc[iid, "rel_path"]))


def show(ax, im, title, cmap=None, vmin=None, vmax=None):
    ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


res = {c: pipeline.process_array(load(i), cfg, ref) for c, i in chosen.items()}
raw = {c: cv2.cvtColor(load(i), cv2.COLOR_BGR2RGB) for c, i in chosen.items()}
out = {"images": chosen}

# ---- stage 1: stain normalisation (4 classes)
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.9))
for j, c in enumerate(CL):
    show(ax[0, j], raw[c], f"{c}: input"); show(ax[1, j], res[c].arrays["rgb_norm"], "after normalisation")
fig.tight_layout(); fig.savefig(F / "stage_1_stain.png", bbox_inches="tight"); plt.close(fig)

# ---- stage 2: denoising (one image): input | bilateral | stabilised HDS | removed component
c0 = "Early"; r0 = res[c0]
lum = representations.luminance01(np.ascontiguousarray(r0.arrays["rgb_norm"][..., ::-1]))
den_bgr = denoise.denoise(np.ascontiguousarray(r0.arrays["rgb_norm"][..., ::-1]), cfg.denoise)
lum_bil = representations.luminance01(den_bgr)
hds_u, hds_edge, n_it = hds_denoise(lum, cfg.hds)
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.9))
z = (slice(60, 140), slice(60, 140))
for row, sl, name in [(0, (slice(None), slice(None)), "full image"), (1, z, "zoom")]:
    show(ax[row, 0], lum[sl], f"luminance ({name})", "gray", 0, 1)
    show(ax[row, 1], lum_bil[sl], "bilateral", "gray", 0, 1)
    show(ax[row, 2], hds_u[sl], f"stabilised HDS ({n_it} it.)", "gray", 0, 1)
    show(ax[row, 3], np.abs(lum - hds_u)[sl], "|input - HDS| (x4)", "gray", 0, 0.25)
fig.tight_layout(); fig.savefig(F / "stage_2_denoise.png", bbox_inches="tight"); plt.close(fig)
out["hds_iterations_example"] = int(n_it)

# ---- stage 3: HDS edge indicator (4 classes)
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.9))
for j, c in enumerate(CL):
    a = res[c].arrays
    show(ax[0, j], representations.luminance01(np.ascontiguousarray(a["rgb_norm"][..., ::-1])), f"{c}: luminance", "gray", 0, 1)
    show(ax[1, j], a["hds_edge"], "HDS edge indicator", "gray", 0, 1)
fig.tight_layout(); fig.savefig(F / "stage_3_hds_edge.png", bbox_inches="tight"); plt.close(fig)

# ---- stage 4: APC, LoG, Canny (one image)
a = r0.arrays
g = cv2.cvtColor(a["rgb"], cv2.COLOR_RGB2GRAY)
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1))
show(ax[0], g, "grey (denoised)", "gray"); show(ax[1], a["apc"], "APC (principal curvature)", "gray", 0, 1)
show(ax[2], a["log"], "LoG", "gray", 0, 1); show(ax[3], a["canny"], "Canny edges", "gray", 0, 1)
fig.tight_layout(); fig.savefig(F / "stage_4_layers.png", bbox_inches="tight"); plt.close(fig)

# ---- stage 5: BEEMD and pure-IMF selection (one image)
d = decompose(lum, cfg.beemd)
K = len(d.imfs)
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.9))
show(ax[0, 0], lum, "input luminance", "gray", 0, 1)
for k in range(min(K, 3)):
    s = "selected" if d.selected[k] else "rejected"
    show(ax[0, k + 1], d.imfs[k], f"IMF {k + 1}: {s}\n{d.scales[k]:.1f} px, {100 * d.energy_frac[k]:.0f}% energy", "gray", -0.15, 0.15)
for k in range(3, min(K, 4)):
    s = "selected" if d.selected[k] else "rejected"
    show(ax[1, 0], d.imfs[k], f"IMF {k + 1}: {s}\n{d.scales[k]:.1f} px, {100 * d.energy_frac[k]:.0f}% energy", "gray", -0.15, 0.15)
if K < 4:
    ax[1, 0].axis("off")
show(ax[1, 1], d.residue, "residue", "gray", 0, 1)
show(ax[1, 2], d.pure, "pure-IMF reconstruction", "gray", -0.15, 0.15)
show(ax[1, 3], lum - d.imfs.sum(0) - d.residue, "input - sum of modes (0)", "gray", -1e-5, 1e-5)
fig.tight_layout(); fig.savefig(F / "stage_5_beemd.png", bbox_inches="tight"); plt.close(fig)
out["beemd_example"] = {"n_imfs": int(K), "selected": d.selected.tolist(), "period_px": [float(x) for x in d.scales], "energy_frac": [float(x) for x in d.energy_frac],
                        "max_reconstruction_error": float(np.abs(lum - d.imfs.sum(0) - d.residue).max())}

# ---- stage 6: segmentation (4 classes): a* score, cells, nuclei
fig, ax = plt.subplots(3, 4, figsize=(7.2, 5.6))
rng = np.random.default_rng(1)
for j, c in enumerate(CL):
    a = res[c].arrays
    show(ax[0, j], a["a_score"], f"{c}: a* score", "gray")
    lab = a["cell_labels"]
    col = rng.uniform(0.2, 1, (lab.max() + 1, 3)); col[0] = 1
    gray3 = np.repeat(cv2.cvtColor(a["rgb"], cv2.COLOR_RGB2GRAY)[..., None], 3, 2).astype(float) / 255
    ov = np.where(lab[..., None] > 0, 0.4 * gray3 + 0.6 * col[lab], gray3)
    show(ax[1, j], np.clip(ov, 0, 1), f"{lab.max()} cells (instances)")
    img = a["rgb"].copy()
    cnt, _ = cv2.findContours((a["nucleus_mask"] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnt, -1, (255, 255, 0), 1)
    cnt, _ = cv2.findContours((a["cell_mask"] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnt, -1, (0, 160, 0), 1)
    show(ax[2, j], img, "cell: green, nucleus: yellow")
fig.tight_layout(); fig.savefig(F / "stage_6_segmentation.png", bbox_inches="tight"); plt.close(fig)

# ---- stage 7: neutralisation and crops
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.9))
for j, c in enumerate(CL):
    a = res[c].arrays
    show(ax[0, j], a["rgb_norm"], f"{c}: normalised"); show(ax[1, j], a["rgb_clean"], "neutralised (cells only)")
fig.tight_layout(); fig.savefig(F / "stage_7_neutralise.png", bbox_inches="tight"); plt.close(fig)
a = r0.arrays
fig, ax = plt.subplots(1, 6, figsize=(7.2, 1.6))
show(ax[0], a["rgb_clean"], "rgb_clean"); show(ax[1], a["gray_clean"], "gray_clean", "gray", 0, 1)
from leukemia_ml.data.crops import cut_crop  # noqa: E402
for k, cid in enumerate(range(1, 5)):
    if cid <= r0.seg.n_cells:
        show(ax[k + 2], cut_crop(a["rgb_clean"], a["cell_labels"], cid, 112, 0.25, True), f"crop of cell {cid}")
    else:
        ax[k + 2].axis("off")
fig.tight_layout(); fig.savefig(F / "stage_7b_crops.png", bbox_inches="tight"); plt.close(fig)

# ---- stage 8: morphological features of the example image
lab = r0.arrays["cell_labels"]
img = r0.arrays["rgb"].copy()
cnt, _ = cv2.findContours((lab > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, cnt, -1, (0, 160, 0), 1)
fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.imshow(img)
for row in r0.cell_rows:
    ax.text(row["centroid_x"], row["centroid_y"], str(row["cell_id"]), color="white", fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.15", fc="black", ec="none"))
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(F / "stage_8_cells.png", bbox_inches="tight"); plt.close(fig)
keys = ["cell_area", "cell_eq_diameter", "cell_aspect_ratio", "cell_solidity", "cell_circularity", "cell_a_mean", "cell_glcm_contrast", "cell_entropy", "cell_apc_mean", "cell_log_mean",
        "cell_hds_edge_mean", "cell_imf_pure_mean", "touches_border", "nucleus_found"]
out["example_cells"] = [{"cell_id": int(r["cell_id"]), **{k: float(r[k]) for k in keys if k in r}} for r in r0.cell_rows]

# ---- timings per stage (20 images)
t = {"stain": 0.0, "bilateral": 0.0, "hds": 0.0, "beemd": 0.0, "segment+features+layers": 0.0}
ids = list(imgs.image_id[::160][:20])
for iid in ids:
    im = load(iid)
    t0 = time.time(); nb, _ = stain.normalize(im, ref, cfg.stain); t["stain"] += time.time() - t0
    t0 = time.time(); denoise.denoise(nb, cfg.denoise); t["bilateral"] += time.time() - t0
    L = representations.luminance01(nb)
    t0 = time.time(); hds_denoise(L, cfg.hds); t["hds"] += time.time() - t0
    t0 = time.time(); decompose(L, cfg.beemd); t["beemd"] += time.time() - t0
    t0 = time.time(); pipeline.process_array(im, base, ref); t["segment+features+layers"] += time.time() - t0
out["seconds_per_image"] = {k: v / len(ids) for k, v in t.items()}
json.dump(out, open("docs/results/stage_examples.json", "w"), indent=1)
print(chosen, out["seconds_per_image"], out["beemd_example"])

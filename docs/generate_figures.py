"""Generate all figures for docs/paper.tex at 300 DPI.
Reads from ml_pipeline/ (algorithms), data/raw/ (originals), data/cache/ (processed)."""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.patches import ConnectionPatch

# make ml_pipeline importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_pipeline"))
import config as C
import stain_norm
from core import (apc_response, bilateral_denoise, canny_binary,
                  denoise_rgb, log_response)

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
NPZ  = ROOT / "data" / "cache"
OUT  = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True, parents=True)

IMAGES = [
    ("Early", "WBC-Malignant-Early-012", "train"),
    ("Pro",   "WBC-Malignant-Pro-002",   "test"),
    ("Pro",   "WBC-Malignant-Pro-003",   "train"),
]

DPI = 300

def load_raw(cls, stem):
    p = RAW / cls / f"{stem}.jpg"
    bgr = cv2.imread(str(p))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bgr

def load_npz(stem, split):
    return np.load(NPZ / split / f"{stem}.npz")

def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  -> {path.name}")

def overlay_mask(rgb, mask, colour, alpha=0.45):
    """Translucent-colour mask overlay on RGB."""
    out = rgb.copy().astype(np.float32)
    tint = np.array(colour, dtype=np.float32)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * tint
    return np.clip(out, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------
# F1 — Pipeline block diagram
# ------------------------------------------------------------------
def fig1_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    stages = [
        (0.3, 4.7, "Raw JPEG\n(RGB, H×W×3)",     "#EEEEEE"),
        (2.4, 4.7, "Stain Norm\n(Reinhard LAB)",            "#FFE0B2"),
        (4.5, 4.7, "Denoise\n(Bilateral)",                  "#FFCCBC"),
        (6.6, 5.6, "APC\n(Hessian)",                        "#C8E6C9"),
        (6.6, 4.7, "LOG\n(∇²G)",                  "#C8E6C9"),
        (6.6, 3.8, "Canny",                                 "#C8E6C9"),
        (6.6, 2.6, "Cell Mask\n(Otsu-inv)",                 "#BBDEFB"),
        (6.6, 1.5, "Nucleus Mask\n(Otsu-in-cell)",          "#BBDEFB"),
        (8.7, 4.7, "6-ch .npz\n(CNN input)",                "#D1C4E9"),
        (8.7, 2.0, "features.csv\n(XGBoost)",               "#D1C4E9"),
    ]
    boxes = {}
    for x, y, text, colour in stages:
        b = FancyBboxPatch((x, y-0.4), 1.7, 0.8, boxstyle="round,pad=0.06",
                           linewidth=1, edgecolor='#333', facecolor=colour)
        ax.add_patch(b)
        ax.text(x + 0.85, y, text, ha='center', va='center', fontsize=8.5)
        boxes[text.split('\n')[0]] = (x, y)
    def arrow(a, b, y_off_from=0, y_off_to=0):
        xa, ya = boxes[a]; xb, yb = boxes[b]
        ax.add_patch(FancyArrowPatch((xa + 1.7, ya + y_off_from), (xb, yb + y_off_to),
            arrowstyle='->', mutation_scale=12, color='#555', linewidth=1))
    arrow("Raw JPEG", "Stain Norm")
    arrow("Stain Norm", "Denoise")
    arrow("Denoise", "APC")
    arrow("Denoise", "LOG")
    arrow("Denoise", "Canny")
    arrow("Denoise", "Cell Mask")
    arrow("Cell Mask", "Nucleus Mask")
    arrow("APC", "6-ch .npz")
    arrow("LOG", "6-ch .npz", y_off_from=0, y_off_to=0)
    arrow("Canny", "6-ch .npz")
    arrow("Cell Mask", "features.csv", y_off_from=0)
    arrow("Nucleus Mask", "features.csv")
    ax.text(5, 0.4, "Denoised RGB fans out into 5 derived layers; the 6-channel tensor feeds CNNs,\n"
                    "morphology on the two masks feeds classical models.",
            ha='center', fontsize=8, style='italic', color='#555')
    fig.suptitle("Figure 1. Preprocessing pipeline overview", fontsize=11)
    save(fig, "fig01_pipeline.png")

# ------------------------------------------------------------------
# F3 — Stain normalization before / after
# ------------------------------------------------------------------
def fig3_stain_norm():
    cls, stem, _ = IMAGES[0]
    rgb, bgr = load_raw(cls, stem)
    # use self as reference (near-identity, expected)
    ref_mean, ref_std = stain_norm.fit_reference([RAW / cls / f"{stem}.jpg"])
    stained_bgr = stain_norm.normalize(bgr, ref_mean, ref_std)
    stained_rgb = cv2.cvtColor(stained_bgr, cv2.COLOR_BGR2RGB)

    lab_before = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_after  = cv2.cvtColor(stained_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4.5))
    ax[0].imshow(rgb);         ax[0].set_title("(a) Raw"); ax[0].axis('off')
    ax[1].imshow(stained_rgb); ax[1].set_title("(b) After Reinhard stain normalization"); ax[1].axis('off')
    stats = (f"LAB (μ, σ) per channel:\n"
             f"raw:    L={lab_before[...,0].mean():.1f}±{lab_before[...,0].std():.1f}  "
             f"a={lab_before[...,1].mean():.1f}±{lab_before[...,1].std():.1f}  "
             f"b={lab_before[...,2].mean():.1f}±{lab_before[...,2].std():.1f}\n"
             f"norm:   L={lab_after[...,0].mean():.1f}±{lab_after[...,0].std():.1f}  "
             f"a={lab_after[...,1].mean():.1f}±{lab_after[...,1].std():.1f}  "
             f"b={lab_after[...,2].mean():.1f}±{lab_after[...,2].std():.1f}")
    fig.suptitle(f"Figure 3. Stain normalization on {stem}\n{stats}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, "fig03_stain_norm.png")

# ------------------------------------------------------------------
# F5 — Bilateral filter effect (zoomed inset)
# ------------------------------------------------------------------
def fig5_bilateral_zoom():
    cls, stem, _ = IMAGES[0]
    rgb, bgr = load_raw(cls, stem)
    filt_bgr = bilateral_denoise(bgr, C.BILATERAL_D, C.BILATERAL_SIGMA_COLOR, C.BILATERAL_SIGMA_SPACE)
    filt_rgb = cv2.cvtColor(filt_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    y0, x0, sz = int(H * 0.35), int(W * 0.35), 60
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0,0].imshow(rgb);      ax[0,0].set_title("(a) Raw"); ax[0,0].axis('off')
    ax[0,0].add_patch(Rectangle((x0, y0), sz, sz, linewidth=2, edgecolor='red', facecolor='none'))
    ax[0,1].imshow(filt_rgb); ax[0,1].set_title("(b) Bilateral (d=9, σ=75)"); ax[0,1].axis('off')
    ax[0,1].add_patch(Rectangle((x0, y0), sz, sz, linewidth=2, edgecolor='red', facecolor='none'))
    ax[1,0].imshow(rgb[y0:y0+sz, x0:x0+sz]);      ax[1,0].set_title("(c) Raw — 4× zoom"); ax[1,0].axis('off')
    ax[1,1].imshow(filt_rgb[y0:y0+sz, x0:x0+sz]); ax[1,1].set_title("(d) Filtered — 4× zoom"); ax[1,1].axis('off')
    fig.suptitle("Figure 5. Bilateral filtering preserves cell edges while smoothing noise", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig05_bilateral_zoom.png")

# ------------------------------------------------------------------
# F6 — HDS divergence diagnostic
# ------------------------------------------------------------------
def fig6_hds_divergence():
    cls, stem, _ = IMAGES[0]
    rgb, bgr = load_raw(cls, stem)
    h20  = cv2.cvtColor(denoise_rgb(bgr,  20, C.HDS_LAMBDA, C.HDS_DT, C.HDS_G_PARAM, C.HDS_H_PARAM, C.HDS_EPSILON), cv2.COLOR_BGR2RGB)
    h200 = cv2.cvtColor(denoise_rgb(bgr, 200, C.HDS_LAMBDA, C.HDS_DT, C.HDS_G_PARAM, C.HDS_H_PARAM, C.HDS_EPSILON), cv2.COLOR_BGR2RGB)
    bilat = cv2.cvtColor(bilateral_denoise(bgr, C.BILATERAL_D, C.BILATERAL_SIGMA_COLOR, C.BILATERAL_SIGMA_SPACE), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 4, figsize=(14, 4.2))
    ax[0].imshow(rgb);   ax[0].set_title(f"(a) Raw\nμ={rgb.mean():.0f}, σ={rgb.std():.0f}")
    ax[1].imshow(h20);   ax[1].set_title(f"(b) HDS 20 iter\nμ={h20.mean():.0f}, σ={h20.std():.0f}")
    ax[2].imshow(h200);  ax[2].set_title(f"(c) HDS 200 iter\nμ={h200.mean():.0f}, σ={h200.std():.0f}")
    ax[3].imshow(bilat); ax[3].set_title(f"(d) Bilateral\nμ={bilat.mean():.0f}, σ={bilat.std():.0f}")
    for a in ax: a.axis('off')
    fig.suptitle("Figure 6. HDS (b, c) collapses colour toward a bimodal equilibrium; bilateral (d) preserves structure", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, "fig06_hds_divergence.png")

# ------------------------------------------------------------------
# F8/F10/F11 — response-map grids (APC, LOG, Canny)
# ------------------------------------------------------------------
def _response_grid(layer_key, cmap, title, filename, fignum):
    fig, ax = plt.subplots(3, 2, figsize=(7, 10))
    for i, (cls, stem, split) in enumerate(IMAGES):
        z = load_npz(stem, split)
        ax[i,0].imshow(z["rgb"]);            ax[i,0].axis('off')
        ax[i,0].set_title(f"{stem}\n(a{i+1}) Denoised RGB", fontsize=8)
        if z[layer_key].dtype == bool:
            ax[i,1].imshow(z[layer_key], cmap='gray')
        else:
            ax[i,1].imshow(z[layer_key], cmap=cmap)
        ax[i,1].axis('off')
        ax[i,1].set_title(f"(b{i+1}) {title}", fontsize=8)
    fig.suptitle(f"Figure {fignum}. {title} response on three sample images", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, filename)

def fig8_apc():
    _response_grid("apc", "inferno", f"APC (σ={C.APC_SIGMA}, α={C.APC_ALPHA})", "fig08_apc.png", 8)

def fig10_log():
    _response_grid("log", "inferno", f"LOG (σ={C.LOG_SIGMA})", "fig10_log.png", 10)

def fig11_canny():
    _response_grid("canny", "gray", f"Canny (low={C.CANNY_LOW}, high={C.CANNY_HIGH})", "fig11_canny.png", 11)

# ------------------------------------------------------------------
# F12/F13 — mask overlays (translucent colour on RGB)
# ------------------------------------------------------------------
def _mask_overlay_grid(mask_key, colour, title, filename, fignum):
    fig, ax = plt.subplots(3, 2, figsize=(7, 10))
    for i, (cls, stem, split) in enumerate(IMAGES):
        z = load_npz(stem, split)
        ax[i,0].imshow(z["rgb"]);                 ax[i,0].axis('off')
        ax[i,0].set_title(f"{stem}\n(a{i+1}) Denoised RGB", fontsize=8)
        ax[i,1].imshow(overlay_mask(z["rgb"], z[mask_key], colour))
        ax[i,1].axis('off')
        ax[i,1].set_title(f"(b{i+1}) {title}", fontsize=8)
    fig.suptitle(f"Figure {fignum}. {title} overlaid on denoised RGB", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, filename)

def fig12_cell_overlay():
    _mask_overlay_grid("cell_mask", (255, 60, 60), "Cell mask", "fig12_cell_overlay.png", 12)

def fig13_nucleus_overlay():
    _mask_overlay_grid("nucleus_mask", (60, 60, 255), "Nucleus mask", "fig13_nucleus_overlay.png", 13)

# ------------------------------------------------------------------
# F14/15/16 — full 6-panel per image
# ------------------------------------------------------------------
def _full_panel(cls, stem, split, fignum, filename):
    z = load_npz(stem, split)
    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    ax = ax.flat
    ax[0].imshow(z["rgb"]);                 ax[0].set_title("(a) Denoised RGB")
    ax[1].imshow(z["apc"], cmap='inferno'); ax[1].set_title(f"(b) APC σ={C.APC_SIGMA}")
    ax[2].imshow(z["log"], cmap='inferno'); ax[2].set_title(f"(c) LOG σ={C.LOG_SIGMA}")
    ax[3].imshow(z["canny"], cmap='gray');  ax[3].set_title("(d) Canny binary")
    ax[4].imshow(overlay_mask(z["rgb"], z["cell_mask"],    (255, 60, 60)))
    ax[4].set_title("(e) Cell mask overlay")
    ax[5].imshow(overlay_mask(z["rgb"], z["nucleus_mask"], (60, 60, 255)))
    ax[5].set_title("(f) Nucleus mask overlay")
    for a in ax: a.axis('off')
    fig.suptitle(f"Figure {fignum}. All six output layers for {stem} ({cls} class)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, filename)

def fig14_16_full_panels():
    _full_panel(*IMAGES[0], 14, "fig14_full_early012.png")
    _full_panel(*IMAGES[1], 15, "fig15_full_pro002.png")
    _full_panel(*IMAGES[2], 16, "fig16_full_pro003.png")

# ------------------------------------------------------------------
# F17 — Morphology bar chart
# ------------------------------------------------------------------
def _load_features():
    import csv
    with open(ROOT / "data" / "features.csv") as f:
        return list(csv.DictReader(f))

def fig17_morphology_bars():
    rows = _load_features()
    labels = [r["image_id"].replace("WBC-Malignant-", "") for r in rows]
    metrics = [
        ("nucleus_area",         "Nucleus area (px)"),
        ("nucleus_eccentricity", "Nucleus eccentricity"),
        ("nucleus_solidity",     "Nucleus solidity"),
        ("nc_ratio",             "N/C ratio"),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
    for i, (key, ylabel) in enumerate(metrics):
        vals = [float(r[key]) for r in rows]
        bars = ax[i].bar(range(len(vals)), vals,
                         color=['#C8E6C9', '#FFCCBC', '#FFCCBC'])
        ax[i].set_xticks(range(len(vals)))
        ax[i].set_xticklabels(labels, rotation=25, ha='right', fontsize=7)
        ax[i].set_ylabel(ylabel, fontsize=8)
        for b, v in zip(bars, vals):
            ax[i].text(b.get_x() + b.get_width()/2, v, f"{v:.2f}" if v < 100 else f"{int(v)}",
                       ha='center', va='bottom', fontsize=7)
    fig.suptitle("Figure 17. Nucleus morphology features across the three demo images", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig17_morphology_bars.png")

# ------------------------------------------------------------------
# F18 — GLCM texture bars
# ------------------------------------------------------------------
def fig18_glcm_bars():
    rows = _load_features()
    labels = [r["image_id"].replace("WBC-Malignant-", "") for r in rows]
    metrics = [
        ("glcm_contrast",    "Contrast"),
        ("glcm_energy",      "Energy"),
        ("glcm_homogeneity", "Homogeneity"),
        ("glcm_correlation", "Correlation"),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
    for i, (key, ylabel) in enumerate(metrics):
        vals = [float(r[key]) for r in rows]
        ax[i].bar(range(len(vals)), vals, color='#B3E5FC')
        ax[i].set_xticks(range(len(vals)))
        ax[i].set_xticklabels(labels, rotation=25, ha='right', fontsize=7)
        ax[i].set_ylabel(ylabel, fontsize=8)
        for j, v in enumerate(vals):
            ax[i].text(j, v, f"{v:.3f}" if v < 100 else f"{int(v)}",
                       ha='center', va='bottom', fontsize=7)
    fig.suptitle("Figure 18. GLCM texture features across the three demo images", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig18_glcm_bars.png")

# ------------------------------------------------------------------
# F19 — RBC contamination annotated
# ------------------------------------------------------------------
def fig19_rbc_contamination():
    cls, stem, split = IMAGES[2]  # Pro-003 clearly has one WBC + many RBCs
    z = load_npz(stem, split)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(z["rgb"]); ax[0].set_title(f"(a) {stem} — denoised RGB")
    ax[0].axis('off')
    ax[1].imshow(overlay_mask(z["rgb"], z["cell_mask"], (255, 60, 60)))
    ax[1].set_title("(b) Cell mask includes both WBCs and RBCs")
    ax[1].axis('off')
    # Annotate the WBC and an RBC
    ax[1].annotate("Diagnostic WBC\n(purple, large nucleus)", xy=(90, 130), xytext=(20, 40),
                   arrowprops=dict(arrowstyle='->', color='yellow'), color='yellow', fontsize=8)
    ax[1].annotate("Red blood cells\n(no nucleus, false positive)", xy=(180, 60), xytext=(140, 8),
                   arrowprops=dict(arrowstyle='->', color='cyan'), color='cyan', fontsize=8)
    fig.suptitle("Figure 19. Intensity-based cell segmentation cannot distinguish RBCs from WBCs",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig19_rbc_contamination.png")

# ------------------------------------------------------------------
# F20 — Comparison strips (pipeline flow per image)
# ------------------------------------------------------------------
def fig20_flow_strips():
    for i, (cls, stem, split) in enumerate(IMAGES):
        z = load_npz(stem, split)
        raw_rgb, _ = load_raw(cls, stem)
        panels = [
            (raw_rgb,                                                    "1. Raw", None),
            (z["rgb"],                                                   "2. Denoised", None),
            (z["apc"],                                                   "3. APC", "inferno"),
            (z["log"],                                                   "4. LOG", "inferno"),
            (z["canny"].astype(np.uint8) * 255,                          "5. Canny", "gray"),
            (overlay_mask(z["rgb"], z["cell_mask"],    (255, 60, 60)),   "6. Cell",     None),
            (overlay_mask(z["rgb"], z["nucleus_mask"], (60, 60, 255)),   "7. Nucleus",  None),
        ]
        fig, ax = plt.subplots(1, 7, figsize=(16, 2.6))
        for a, (img, ttl, cmap) in zip(ax, panels):
            if cmap:
                a.imshow(img, cmap=cmap)
            else:
                a.imshow(img)
            a.set_title(ttl, fontsize=8)
            a.axis('off')
        fig.suptitle(f"Figure 20.{chr(ord('a')+i)}. Pipeline flow for {stem}", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        save(fig, f"fig20{chr(ord('a')+i)}_flow_{stem.split('-')[-2].lower()}_{stem.split('-')[-1]}.png")


def main():
    print("Generating figures ->", OUT)
    fig1_pipeline_diagram()
    fig3_stain_norm()
    fig5_bilateral_zoom()
    fig6_hds_divergence()
    fig8_apc()
    fig10_log()
    fig11_canny()
    fig12_cell_overlay()
    fig13_nucleus_overlay()
    fig14_16_full_panels()
    fig17_morphology_bars()
    fig18_glcm_bars()
    fig19_rbc_contamination()
    fig20_flow_strips()
    print("\nAll figures written to", OUT)


if __name__ == "__main__":
    main()

"""QC preview panels (matplotlib is imported lazily so workers without a display stay light)."""
from __future__ import annotations

from pathlib import Path


def save_preview(path: Path, raw_bgr, res, cfg, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.measure import regionprops
    from skimage.segmentation import find_boundaries

    rgb = res.arrays["rgb_norm"]
    cells, nuclei = res.seg.cell_labels, res.seg.nucleus_labels
    overlay = rgb.copy()
    overlay[find_boundaries(cells, mode="inner")] = (0, 200, 0)
    overlay[find_boundaries(nuclei, mode="inner")] = (255, 0, 0)

    fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
    ax = ax.ravel()
    ax[0].imshow(raw_bgr[..., ::-1])
    ax[0].set_title("raw")
    ax[1].imshow(rgb)
    ax[1].set_title("stain-normalised")
    ax[2].imshow(overlay)
    ax[2].set_title(f"cells (green) / nuclei (red): {res.seg.n_cells}")
    for r in regionprops(cells):
        ax[2].text(r.centroid[1], r.centroid[0], str(r.label), color="yellow",
                   fontsize=8, ha="center", va="center")
    ax[3].imshow(res.arrays["a_score"], cmap="magma")
    ax[3].set_title(f"a* score (thr {res.seg.a_threshold:.1f})")
    ax[4].imshow(res.arrays["apc"], cmap="inferno", vmin=0, vmax=1)
    ax[4].set_title(f"APC σ={cfg.response.apc_sigma}")
    ax[5].imshow(res.arrays["log"], cmap="inferno", vmin=0, vmax=1)
    ax[5].set_title(f"LoG σ={cfg.response.log_sigma}")
    ax[6].imshow(res.arrays["canny"], cmap="gray")
    ax[6].set_title("Canny")
    ax[7].imshow(np.where(nuclei > 0, 2, np.where(cells > 0, 1, 0)), cmap="gray", vmin=0, vmax=2)
    ax[7].set_title("instance mask (cell / nucleus)")
    for a in ax:
        a.axis("off")
    flags = res.image_row["qc_flags"] or "none"
    fig.suptitle(f"{title}   |   QC flags: {flags}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=70, format="png")
    plt.close(fig)

"""Optional extra input representations (HDS, BEEMD/pure-IMF) computed on the
stain-normalised, *un-denoised* luminance, plus per-cell statistics of every response layer.

Both heavy stages are opt-in (`cfg.hds.enabled`, `cfg.beemd.enabled`). Their arrays have fixed
shapes for a given config, so cached tensors from one run are always stackable.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .config import PipelineConfig
from .emd import decompose
from .hds import hds_denoise
from .segmentation import Segmentation
from .stain import bgr_to_lab

RIM_PX = 2


def luminance01(norm_bgr: np.ndarray) -> np.ndarray:
    return (bgr_to_lab(norm_bgr)[..., 0] / 100.0).astype(np.float32)


def compute(norm_bgr: np.ndarray, cfg: PipelineConfig) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not (cfg.hds.enabled or cfg.beemd.enabled):
        return out
    lum = luminance01(norm_bgr)
    if cfg.hds.enabled:
        out["hds"], out["hds_edge"], _ = hds_denoise(lum, cfg.hds)
    if cfg.beemd.enabled:
        d = decompose(lum, cfg.beemd)
        k, (h, w) = cfg.beemd.n_imfs, lum.shape
        imfs = np.zeros((k, h, w), np.float32)
        sel = np.zeros(k, bool)
        imfs[: len(d.imfs)] = d.imfs
        sel[: len(d.imfs)] = d.selected
        out.update(imfs=imfs, imf_selected=sel, imf_pure=d.pure.astype(np.float32),
                   imf_residue=d.residue)
    return out


# ------------------------------------------------------------------ per-cell statistics
def layer_names(cfg: PipelineConfig) -> list[str]:
    names = ["apc", "log"]
    if cfg.hds.enabled:
        names.append("hds_edge")
    if cfg.beemd.enabled:
        names.append("imf_pure")
    return names


def channel_columns(cfg: PipelineConfig) -> list[str]:
    cols = [f"cell_{n}_{s}" for n in layer_names(cfg) for s in ("mean", "std", "rim_mean")]
    if cfg.beemd.enabled:
        cols += [f"cell_imf{k + 1}_energy" for k in range(cfg.beemd.n_imfs)]
    return cols


def channel_features(arrays: dict[str, np.ndarray], seg: Segmentation, cfg: PipelineConfig
                     ) -> dict[int, dict[str, float]]:
    """{cell_id: {column: value}}. `rim_mean` = mean over the outer RIM_PX pixels of the cell,
    i.e. edge strength at the cell boundary; `mean`/`std` summarise the whole cell."""
    out: dict[int, dict[str, float]] = {}
    for cid in range(1, seg.n_cells + 1):
        cell = seg.cell_labels == cid
        rim = cell & ~ndi.binary_erosion(cell, iterations=RIM_PX)
        row: dict[str, float] = {}
        for name in layer_names(cfg):
            layer = arrays[name]
            row[f"cell_{name}_mean"] = float(layer[cell].mean())
            row[f"cell_{name}_std"] = float(layer[cell].std())
            row[f"cell_{name}_rim_mean"] = float(layer[rim].mean()) if rim.any() else float("nan")
        if cfg.beemd.enabled:
            for k in range(cfg.beemd.n_imfs):
                row[f"cell_imf{k + 1}_energy"] = float((arrays["imfs"][k][cell] ** 2).mean())
        out[cid] = row
    return out

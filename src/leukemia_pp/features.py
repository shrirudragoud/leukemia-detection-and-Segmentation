"""Per-cell and per-image feature extraction.

Shape comes from the segmentation (denoised image); colour and texture are measured on the
stain-normalised but *not denoised* image, because bilateral smoothing flattens exactly the
chromatin texture we want to quantify. All lengths/areas are in pixels (pixel size is not
recorded in this dataset).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops

from .config import FeatureConfig
from .segmentation import Segmentation
from .stain import bgr_to_lab

NAN = float("nan")
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")

SHAPE_KEYS = ("area", "perimeter", "eq_diameter", "major_axis", "minor_axis", "aspect_ratio",
              "eccentricity", "solidity", "extent", "circularity")
COLOR_KEYS = tuple(f"{c}_{s}" for c in ("L", "a", "b") for s in ("mean", "std"))
TEXTURE_KEYS = tuple(f"glcm_{p}" for p in GLCM_PROPS) + ("entropy",)
TEXTURE_KEYS_RAW = TEXTURE_KEYS  # texture dict keys carry no cell_/nucleus_ prefix

CELL_COLUMNS = (
    ["cell_id", "centroid_y", "centroid_x", "touches_border"]
    + [f"cell_{k}" for k in SHAPE_KEYS]
    + [f"cell_{k}" for k in COLOR_KEYS]
    + ["nucleus_found", "nucleus_contrast", "nc_ratio"]
    + [f"nucleus_{k}" for k in SHAPE_KEYS]
    + [f"nucleus_{k}" for k in COLOR_KEYS]
    + ["cytoplasm_L_mean", "cytoplasm_a_mean", "cytoplasm_b_mean", "nuc_cyto_L_contrast"]
    + [f"nucleus_{k}" for k in TEXTURE_KEYS]
    + [f"cell_{k}" for k in TEXTURE_KEYS]
)


def _shape(mask: np.ndarray) -> dict[str, float]:
    """Shape descriptors of a binary mask (all components treated as one object)."""
    props = regionprops(mask.astype(np.uint8))[0]
    area = float(props.area)
    perim = float(props.perimeter_crofton)
    major, minor = float(props.axis_major_length), float(props.axis_minor_length)
    return {
        "area": area,
        "perimeter": perim,
        "eq_diameter": math.sqrt(4.0 * area / math.pi),
        "major_axis": major,
        "minor_axis": minor,
        "aspect_ratio": major / minor if minor > 0 else NAN,
        "eccentricity": float(props.eccentricity),
        "solidity": float(props.solidity),
        "extent": float(props.extent),
        "circularity": min(1.0, 4.0 * math.pi * area / perim ** 2) if perim > 0 else NAN,
    }


def _color(lab: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    px = lab[mask]
    return {f"{c}_{s}": float(fn(px[:, i]))
            for i, c in enumerate("Lab") for s, fn in (("mean", np.mean), ("std", np.std))}


def masked_glcm(gray_u8: np.ndarray, mask: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    """GLCM statistics over `mask` only, averaged over all distances x angles.

    Pixels outside the mask are mapped to a reserved level 0 whose rows/columns are dropped
    before normalising, so the black surround never contributes co-occurrences (the v1
    code zeroed the surround, which inflated contrast). Quantisation is global (fixed),
    so values are comparable between cells and images."""
    out = {f"glcm_{p}": NAN for p in GLCM_PROPS}
    out["entropy"] = NAN
    if int(mask.sum()) < cfg.min_texture_pixels:
        return out
    levels = cfg.glcm_levels
    q = (gray_u8.astype(np.int32) * levels // 256) + 1          # 1..levels
    q = np.where(mask, q, 0).astype(np.uint8)
    glcm = graycomatrix(q, distances=list(cfg.glcm_distances),
                        angles=[math.radians(a) for a in cfg.glcm_angles_deg],
                        levels=levels + 1, symmetric=True, normed=False).astype(np.float64)
    glcm[0, :, :, :] = 0.0
    glcm[:, 0, :, :] = 0.0
    sums = glcm.sum(axis=(0, 1), keepdims=True)
    valid = sums[0, 0] > 0
    if not valid.any():
        return out
    glcm = np.divide(glcm, sums, out=np.zeros_like(glcm), where=sums > 0)
    for p in GLCM_PROPS:
        vals = graycoprops(glcm, p)[valid]
        out[f"glcm_{p}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else NAN
    hist = np.bincount(q[mask], minlength=levels + 1)[1:].astype(np.float64)
    p_ = hist[hist > 0] / hist.sum()
    out["entropy"] = float(-(p_ * np.log2(p_)).sum())
    return out


def extract_cell_features(rgb_norm: np.ndarray, seg: Segmentation, cfg: FeatureConfig
                          ) -> list[dict[str, float]]:
    """One row per segmented cell. `rgb_norm` is the normalised, un-denoised RGB image."""
    bgr = np.ascontiguousarray(rgb_norm[..., ::-1])
    lab = bgr_to_lab(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rows: list[dict[str, float]] = []
    for region in regionprops(seg.cell_labels):
        cid, sl = region.label, region.slice
        cell = seg.cell_labels[sl] == cid
        nuc = seg.nucleus_labels[sl] == cid
        lab_c, gray_c = lab[sl], gray[sl]
        r0, c0, r1, c1 = region.bbox
        info = seg.nucleus_info.get(cid)
        found = bool(info and info.found and nuc.any())

        row: dict[str, float] = {
            "cell_id": cid,
            "centroid_y": float(region.centroid[0]),
            "centroid_x": float(region.centroid[1]),
            "touches_border": bool(r0 == 0 or c0 == 0 or r1 == h or c1 == w),
        }
        cell_shape = _shape(cell)
        row.update({f"cell_{k}": v for k, v in cell_shape.items()})
        row.update({f"cell_{k}": v for k, v in _color(lab_c, cell).items()})
        row["nucleus_found"] = found
        row["nucleus_contrast"] = float(info.contrast) if info else NAN

        nuc_shape = _shape(nuc) if found else dict.fromkeys(SHAPE_KEYS, NAN)
        row["nc_ratio"] = nuc_shape["area"] / cell_shape["area"] if found else NAN
        row.update({f"nucleus_{k}": v for k, v in nuc_shape.items()})
        nuc_col = _color(lab_c, nuc) if found else dict.fromkeys(COLOR_KEYS, NAN)
        row.update({f"nucleus_{k}": v for k, v in nuc_col.items()})

        cyto = cell & ~nuc
        has_cyto = found and int(cyto.sum()) >= cfg.min_texture_pixels
        cy = _color(lab_c, cyto) if has_cyto else dict.fromkeys(COLOR_KEYS, NAN)
        row["cytoplasm_L_mean"], row["cytoplasm_a_mean"], row["cytoplasm_b_mean"] = (
            cy["L_mean"], cy["a_mean"], cy["b_mean"])
        row["nuc_cyto_L_contrast"] = (cy["L_mean"] - nuc_col["L_mean"]) if has_cyto else NAN

        nt = masked_glcm(gray_c, nuc, cfg) if found else dict.fromkeys(TEXTURE_KEYS_RAW, NAN)
        row.update({f"nucleus_{k}": v for k, v in nt.items()})
        row.update({f"cell_{k}": v for k, v in masked_glcm(gray_c, cell, cfg).items()})
        rows.append(row)
    return rows


AGG_KEYS = (
    "cell_area", "cell_circularity", "cell_solidity", "cell_aspect_ratio", "nc_ratio",
    "nucleus_area", "nucleus_solidity", "nucleus_circularity", "cell_L_mean", "cell_a_mean",
    "nucleus_L_mean", "nuc_cyto_L_contrast", "nucleus_glcm_contrast", "nucleus_glcm_homogeneity",
    "nucleus_entropy", "cell_glcm_contrast", "cell_entropy",
)


def aggregate_keys(extra: list[str] | tuple[str, ...] = ()) -> list[str]:
    return list(AGG_KEYS) + [k for k in extra if k not in AGG_KEYS]


def aggregate_columns(extra: list[str] | tuple[str, ...] = ()) -> list[str]:
    return ["n_cells", "n_cells_interior", "n_cells_with_nucleus", "agg_uses_border_cells"] + [
        f"{k}_{s}" for k in aggregate_keys(extra) for s in ("mean", "std", "max")]


def aggregate_image(cell_rows: list[dict[str, float]],
                    extra: list[str] | tuple[str, ...] = ()) -> dict[str, float]:
    """Image-level descriptors for classical ML on whole-image labels.

    Cells cut by the image border have biased shape/size, so aggregation uses interior cells
    whenever at least one exists (`agg_uses_border_cells` records the fallback)."""
    interior = [r for r in cell_rows if not r["touches_border"]]
    pool = interior or cell_rows
    out: dict[str, float] = {
        "n_cells": len(cell_rows),
        "n_cells_interior": len(interior),
        "n_cells_with_nucleus": sum(bool(r["nucleus_found"]) for r in cell_rows),
        "agg_uses_border_cells": bool(cell_rows and not interior),
    }
    for key in aggregate_keys(extra):
        vals = np.array([r[key] for r in pool], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[f"{key}_mean"] = float(vals.mean()) if vals.size else NAN
        out[f"{key}_std"] = float(vals.std()) if vals.size > 1 else NAN
        out[f"{key}_max"] = float(vals.max()) if vals.size else NAN
    return out


AGG_COLUMNS = aggregate_columns()

"""WBC instance segmentation (cells + nuclei).

Why colour, not darkness: in Romanowsky-stained smears, red blood cells and the background
are pale and grey-blue, while leukocytes (and especially blasts) are strongly magenta/purple.
That is a large separation on the Lab a* axis, whereas on grayscale intensity RBCs overlap
with WBCs, which is why a global intensity Otsu merges all RBCs into the "cell" mask.

Pipeline
  1. a* score, lightly smoothed;  threshold = max(Otsu, a_floor)   -> foreground
  2. fill holes, open, drop specks
  3. split touching cells: distance transform + h-maxima markers + watershed. A split is only
     accepted if every resulting piece is a plausible cell (>= min_cell_area)
  4. per-cell nucleus: Otsu on L* over each cell's *interior* (rim excluded, since smoothing
     makes the rim brighter), accepted only if the nucleus and cytoplasm classes differ by
     >= nucleus_min_contrast L* units
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import disk, h_maxima, opening
from skimage.segmentation import relabel_sequential, watershed

from .config import SegmentationConfig
from .stain import otsu


@dataclass(frozen=True)
class NucleusInfo:
    found: bool
    threshold: float        # L* split inside the cell
    contrast: float         # L* gap between cytoplasm-class mean and nucleus-class mean


@dataclass
class Segmentation:
    cell_labels: np.ndarray        # int32, 0 = background, 1..N = cell instances
    nucleus_labels: np.ndarray     # int32, same ids as cell_labels
    a_score: np.ndarray            # float32 smoothed a*
    a_threshold: float
    nucleus_info: dict[int, NucleusInfo] = field(default_factory=dict)

    @property
    def cell_mask(self) -> np.ndarray:
        return self.cell_labels > 0

    @property
    def nucleus_mask(self) -> np.ndarray:
        return self.nucleus_labels > 0

    @property
    def n_cells(self) -> int:
        return int(self.cell_labels.max())


def _smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img


def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    areas = np.bincount(lab.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[lab]


def split_touching(mask: np.ndarray, h: float, smooth: float, min_area: int) -> np.ndarray:
    """Connected components -> instance labels, splitting necked-together cells."""
    cc, n = ndi.label(mask)
    out = np.zeros(mask.shape, np.int32)
    next_id = 1
    for idx, sl in enumerate(ndi.find_objects(cc), start=1):
        comp = np.pad(cc[sl] == idx, 1)
        dist = _smooth(ndi.distance_transform_edt(comp).astype(np.float32), smooth)
        markers, k = ndi.label(h_maxima(dist, h))
        pieces = comp.astype(np.int32)
        if k > 1:
            ws = watershed(-dist, markers, mask=comp)
            if np.bincount(ws.ravel())[1:].min() >= min_area:
                pieces = ws
        pieces = pieces[1:-1, 1:-1]
        for lab_id in np.unique(pieces[pieces > 0]):
            region = out[sl]
            region[pieces == lab_id] = next_id
            next_id += 1
    return out


def segment_cells(a_score: np.ndarray, cfg: SegmentationConfig) -> tuple[np.ndarray, float]:
    thr = max(otsu(a_score), cfg.a_floor)
    mask = a_score > thr
    if cfg.open_radius > 0:
        mask = opening(mask, disk(cfg.open_radius))
    mask = _remove_small(mask, cfg.min_cell_area)
    mask = ndi.binary_fill_holes(mask)
    labels = split_touching(mask, cfg.split_h, cfg.split_smooth, cfg.min_cell_area)
    labels, _, _ = relabel_sequential(labels)
    return labels.astype(np.int32), float(thr)


def _class_split(values: np.ndarray) -> tuple[float, float]:
    """Otsu split of `values`; returns (threshold, mean gap between the two classes)."""
    thr = otsu(values)
    lo, hi = values[values < thr], values[values >= thr]
    if lo.size == 0 or hi.size == 0:
        return thr, 0.0
    return thr, float(hi.mean() - lo.mean())


def segment_nuclei(L: np.ndarray, cell_labels: np.ndarray, cfg: SegmentationConfig
                   ) -> tuple[np.ndarray, dict[int, NucleusInfo]]:
    Ls = _smooth(L.astype(np.float32), cfg.nucleus_smooth_sigma)
    nucleus = np.zeros(cell_labels.shape, np.int32)
    info: dict[int, NucleusInfo] = {}
    for region in regionprops(cell_labels):
        sl, cid = region.slice, region.label
        cell = cell_labels[sl] == cid
        interior = ndi.binary_erosion(cell, disk(cfg.nucleus_rim_erode)) if cfg.nucleus_rim_erode \
            else cell
        if interior.sum() < max(cfg.nucleus_min_area, 1):
            info[cid] = NucleusInfo(False, float("nan"), 0.0)
            continue
        thr, contrast = _class_split(Ls[sl][interior])     # statistics from the interior only
        if contrast < cfg.nucleus_min_contrast:
            info[cid] = NucleusInfo(False, thr, contrast)
            continue
        nuc = cell & (Ls[sl] < thr)
        nuc = ndi.binary_fill_holes(opening(nuc, disk(1)))
        lab, n = ndi.label(nuc)
        if n:
            areas = np.bincount(lab.ravel())
            areas[0] = 0
            keep = areas >= max(cfg.nucleus_min_area, cfg.nucleus_keep_frac * areas.max())
            keep[0] = False
            nuc = keep[lab] & cell
        ok = nuc.sum() >= max(cfg.nucleus_min_area, cfg.nucleus_min_cell_frac * cell.sum())
        info[cid] = NucleusInfo(bool(ok), thr, contrast)
        if ok:
            nucleus[sl][nuc] = cid
    return nucleus, info


def segment(lab: np.ndarray, cfg: SegmentationConfig) -> Segmentation:
    """`lab` is a float Lab image (L 0-100, a/b signed), e.g. from `stain.bgr_to_lab`."""
    a_score = _smooth(lab[..., 1].astype(np.float32), cfg.smooth_sigma)
    cell_labels, thr = segment_cells(a_score, cfg)
    nucleus_labels, info = segment_nuclei(lab[..., 0], cell_labels, cfg)
    return Segmentation(cell_labels, nucleus_labels, a_score, thr, info)

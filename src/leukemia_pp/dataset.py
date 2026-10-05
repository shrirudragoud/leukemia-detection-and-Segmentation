"""Training-side access to the cached tensors (numpy only; wrap in a torch/tf Dataset later).

Everything a model needs to *consume* the pipeline output without re-implementing it:
  * named channels and ready-made channel stacks for ablations (`PRESETS`);
  * per-channel mean/std computed on the TRAIN split only (`compute_channel_stats`);
  * per-cell square crops (`crop_cell`) so the network sees one cell, not the background;
  * named tabular feature groups (`FEATURE_GROUPS`) for tabular ablations / fusion branches.

Notes for adapting pretrained networks (ResNet / ViT / EfficientNet) to extra channels
  * keep the first 3 channels = RGB so ImageNet weights stay valid;
  * initialise the extra input channels of the first conv / patch-embedding with the mean of
    the pretrained RGB filters scaled by ~0.1, or train them in a separate lightweight branch
    and fuse late - both avoid wrecking transfer learning;
  * decide whether a representation is worth keeping with the ablation ladder in README.md.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

from . import features, representations  # noqa: F401  (representations: documented dependency)

# name -> (npz key, channel width). 'imfN' (N>=1) is the N-th IMF of the 'imfs' array.
CHANNELS: dict[str, tuple[str, int]] = {
    "rgb_norm": ("rgb_norm", 3),   # stain-normalised, NOT denoised (keeps chromatin texture)
    "rgb": ("rgb", 3),             # stain-normalised + denoised
    "rgb_clean": ("rgb_clean", 3),     # WBCs only, neutral fill, white-balanced (shortcut-free)
    "gray_clean": ("gray_clean", 1),   # luminance of rgb_clean: no stain colour at all
    "apc": ("apc", 1), "log": ("log", 1), "canny": ("canny", 1),
    "a_score": ("a_score", 1),
    "hds": ("hds", 1), "hds_edge": ("hds_edge", 1),
    "imf_pure": ("imf_pure", 1), "imf_residue": ("imf_residue", 1),
}

PRESETS: dict[str, tuple[str, ...]] = {
    "rgb": ("rgb_norm",),
    "rgb_denoised": ("rgb",),
    "clean_rgb": ("rgb_clean",),
    "clean_gray": ("gray_clean",),
    "clean_rgb+edges": ("rgb_clean", "apc", "log"),
    "rgb+apc": ("rgb_norm", "apc"),
    "rgb+edges": ("rgb_norm", "apc", "log", "canny"),
    "rgb+hds": ("rgb_norm", "hds_edge"),
    "rgb+imf": ("rgb_norm", "imf_pure"),
    "full": ("rgb_norm", "apc", "log", "canny", "hds_edge", "imf_pure"),
}


def channel_width(name: str) -> int:
    if name.startswith("imf") and name[3:].isdigit():
        return 1
    return CHANNELS[name][1]


def stack_depth(names: Sequence[str]) -> int:
    return sum(channel_width(n) for n in names)


def _get(z, name: str) -> np.ndarray:
    """One named channel as float32 (C, H, W); RGB uint8 is scaled to [0, 1]."""
    if name.startswith("imf") and name[3:].isdigit():
        arr = z["imfs"][int(name[3:]) - 1]
    else:
        key, _ = CHANNELS[name]
        if key not in z.files if hasattr(z, "files") else key not in z:
            raise KeyError(f"channel {name!r} not in this cache; enable it in the pipeline config "
                           f"(hds.enabled / beemd.enabled) and re-run")
        arr = z[key]
    arr = arr.astype(np.float32) / (255.0 if arr.dtype == np.uint8 else 1.0)
    return arr.transpose(2, 0, 1) if arr.ndim == 3 else arr[None]


def load_stack(npz_path: Path | dict, names: Sequence[str]) -> np.ndarray:
    """Float32 (C, H, W) stack of the named channels, in order."""
    z = np.load(npz_path) if isinstance(npz_path, (str, Path)) else npz_path
    return np.concatenate([_get(z, n) for n in names], axis=0)


def compute_channel_stats(paths: Iterable[Path], names: Sequence[str]) -> dict:
    """Per-output-channel mean/std over `paths` (pass TRAIN files only). Exact, streaming."""
    n_ch = stack_depth(names)
    total = np.zeros(n_ch)
    total_sq = np.zeros(n_ch)
    count = 0
    for p in paths:
        s = load_stack(p, names).astype(np.float64)
        total += s.sum(axis=(1, 2))
        total_sq += (s ** 2).sum(axis=(1, 2))
        count += s.shape[1] * s.shape[2]
    if count == 0:
        raise ValueError("no files given to compute channel statistics")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 0.0))
    return {"channels": list(names), "mean": mean.tolist(), "std": np.maximum(std, 1e-6).tolist()}


def save_stats(stats: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_stats(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalise(stack: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"], np.float32)[:, None, None]
    std = np.asarray(stats["std"], np.float32)[:, None, None]
    return (stack - mean) / std


def crop_cell(npz_path: Path | dict, cell_id: int, names: Sequence[str], size: int = 96,
              margin: float = 0.25, background: str = "keep") -> np.ndarray:
    """Square crop around one cell -> float32 (C, size, size).

    The crop is the cell's bounding box expanded by `margin` (fraction of the box), made
    square, and resized (area interpolation). Regions beyond the image are zero-padded.
    `background='zero'` additionally blanks everything outside that cell (also neighbours)."""
    z = np.load(npz_path) if isinstance(npz_path, (str, Path)) else npz_path
    labels = z["cell_labels"]
    ys, xs = np.nonzero(labels == cell_id)
    if ys.size == 0:
        raise KeyError(f"cell {cell_id} not present")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(round(max(y1 - y0, x1 - x0) * (1.0 + 2.0 * margin)))
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    top, left = int(round(cy - side / 2.0)), int(round(cx - side / 2.0))

    stack = load_stack(z, names)
    if background == "zero":
        stack = stack * (labels == cell_id)[None]
    elif background != "keep":
        raise ValueError("background must be 'keep' or 'zero'")
    h, w = labels.shape
    pad = max(0, -top, -left, top + side - h, left + side - w)
    if pad:
        stack = np.pad(stack, ((0, 0), (pad, pad), (pad, pad)))
        top, left = top + pad, left + pad
    patch = stack[:, top:top + side, left:left + side]
    interp = cv2.INTER_AREA if side >= size else cv2.INTER_LINEAR
    return np.stack([cv2.resize(c, (size, size), interpolation=interp) for c in patch])


# ----------------------------------------------------------------- tabular feature groups
def _g(prefixes: Sequence[str], suffixes: Sequence[str] = ()) -> list[str]:
    return [c for c in features.CELL_COLUMNS
            if (not prefixes or c.startswith(tuple(prefixes)))
            and (not suffixes or c.endswith(tuple(suffixes)))]


FEATURE_GROUPS: dict[str, list[str]] = {
    "cell_shape": [f"cell_{k}" for k in features.SHAPE_KEYS],
    "cell_color": [f"cell_{k}" for k in features.COLOR_KEYS],
    "cell_texture": [f"cell_{k}" for k in features.TEXTURE_KEYS],
    "nucleus": [c for c in features.CELL_COLUMNS
                if c.startswith("nucleus_") or c in ("nc_ratio", "nuc_cyto_L_contrast")
                or c.startswith("cytoplasm_")],
}


def layer_feature_group(layer: str) -> list[str]:
    """Per-cell statistics of one response layer (apc, log, hds_edge, imf_pure) or 'imf_energy'."""
    if layer == "imf_energy":
        return [f"cell_imf{k}_energy" for k in range(1, 9)]
    return [f"cell_{layer}_{s}" for s in ("mean", "std", "rim_mean")]

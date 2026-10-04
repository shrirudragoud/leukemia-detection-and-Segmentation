"""Session-aware train/val/test assignment from capture-order numbering.

Images are numbered in capture order and acquisition sessions (patients) are contiguous in that
numbering (measured: neighbours share background/stain far more than random pairs). Randomly
assigning fixed blocks to folds therefore leaves a same-session neighbour in training for almost
every test block; on the real data this inflated balanced accuracy from ~0.74 to ~0.97. Here each
class's sequence is cut into consecutive segments, so only segment edges can leak, and an
embargo removes the training images next to those edges.
"""
from __future__ import annotations

import re

import numpy as np

from .index import CellIndex

PURGED = "purged"


def sequence_numbers(index: CellIndex) -> np.ndarray:
    """Trailing integer of each image id; images without one get their rank among sorted ids."""
    nums = np.array([int(m.group(1)) if (m := re.search(r"(\d+)$", str(i))) else -1
                     for i in index.images])
    if (nums < 0).any():
        nums = np.where(nums < 0, np.arange(len(nums)), nums)
    return nums


def contiguous_split(index: CellIndex, n_folds: int = 5, test_fold: int = 0, val_offset: int = 2,
                     embargo: int = 37) -> np.ndarray:
    """(n_images,) array of 'train' | 'val' | 'test' | 'purged'."""
    nums = sequence_numbers(index)
    labels = index.image_label
    split = np.full(index.n_images, "train", dtype=object)
    val_fold = (test_fold + val_offset) % n_folds
    held = np.zeros(index.n_images, bool)
    for c in np.unique(labels):
        members = np.nonzero(labels == c)[0]
        ranked = members[np.argsort(nums[members], kind="stable")]
        for k, seg in enumerate(np.array_split(np.arange(len(ranked)), n_folds)):
            if k == test_fold:
                split[ranked[seg]] = "test"
            elif k == val_fold:
                split[ranked[seg]] = "val"
    held = np.isin(split, ("val", "test"))
    if embargo > 0:
        for c in np.unique(labels):
            h = np.sort(nums[held & (labels == c)])
            tr = np.nonzero((split == "train") & (labels == c))[0]
            if len(h) == 0 or len(tr) == 0:
                continue
            pos = np.searchsorted(h, nums[tr])
            d = np.minimum(np.abs(nums[tr] - h[np.clip(pos - 1, 0, len(h) - 1)]),
                           np.abs(nums[tr] - h[np.clip(pos, 0, len(h) - 1)]))
            split[tr[d <= embargo]] = PURGED
    return split.astype(str)

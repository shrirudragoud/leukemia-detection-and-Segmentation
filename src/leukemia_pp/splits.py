"""Deterministic, stratified, leakage-aware train/val/test assignment."""
from __future__ import annotations

import logging
import random
import re
from collections import defaultdict

from .config import SplitConfig
from .io import Sample

log = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")


def allocate(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Split `n` units into (train, val, test) counts by largest remainder.

    n == 1 -> all train; n == 2 -> train + test; n >= 3 -> every split with a positive
    ratio gets at least one unit, so evaluation sets are never empty by rounding.
    """
    if n <= 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)
    if n == 2:
        return (1, 0, 1) if ratios[2] > 0 else (1, 1, 0)
    quotas = [n * r for r in ratios]
    counts = [int(q) for q in quotas]
    order = sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    for i in (2, 1):
        if ratios[i] > 0 and counts[i] == 0:
            counts[0] -= 1
            counts[i] += 1
    return counts[0], counts[1], counts[2]


def group_key(sample: Sample, regex: re.Pattern | None) -> str:
    if regex is not None:
        m = regex.search(sample.image_id)
        if m:
            return f"grp:{m.group(1) if m.groups() else m.group(0)}"
    return f"sha:{sample.sha256}"


def assign_splits(samples: list[Sample], cfg: SplitConfig) -> dict[str, str]:
    """Return {image_id: split}. Splits are made per class over *groups*, where a group is a
    patient/slide (via `cfg.group_regex`) or, by default, a set of byte-identical files.
    The result depends only on the sample ids, hashes, classes and seed."""
    regex = re.compile(cfg.group_regex) if cfg.group_regex else None
    ratios = (cfg.train, cfg.val, cfg.test)

    groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    cross_class: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        key = group_key(s, regex)
        groups[(s.class_name, key)].append(s)
        cross_class[key].add(s.class_name)
    for key, classes in cross_class.items():
        if len(classes) > 1:
            log.warning("group %s appears under several classes %s: possible label noise",
                        key, sorted(classes))

    by_class: dict[str, list[str]] = defaultdict(list)
    for (cname, key) in groups:
        by_class[cname].append(key)

    rng = random.Random(cfg.seed)
    out: dict[str, str] = {}
    for cname in sorted(by_class):
        keys = sorted(by_class[cname])
        rng.shuffle(keys)
        n_tr, n_va, _ = allocate(len(keys), ratios)
        for i, key in enumerate(keys):
            split = SPLITS[0] if i < n_tr else SPLITS[1] if i < n_tr + n_va else SPLITS[2]
            for s in groups[(cname, key)]:
                out[s.image_id] = split
    return out

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


_TRAILING_INT = re.compile(r"(\d+)$")


def group_keys(sample: Sample, regex: re.Pattern | None, block: int | None) -> list[str]:
    """Every grouping relation this sample participates in. Samples sharing ANY key end up in
    the same group (transitively). Byte-identical files always share a key."""
    keys = [f"sha:{sample.sha256}"]
    if block:
        m = _TRAILING_INT.search(sample.image_id)
        if m:
            keys.append(f"blk:{int(m.group(1)) // block}")
    if regex is not None:
        m = regex.search(sample.image_id)
        if m:
            keys.append(f"grp:{m.group(1) if m.groups() else m.group(0)}")
    return keys


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)        # deterministic root


def assign_splits(samples: list[Sample], cfg: SplitConfig) -> dict[str, str]:
    """Return {image_id: split}. Splits are made per class over *groups*. Two images share a
    group if they are byte-identical, fall in the same `cfg.sequence_block` of consecutive
    image numbers, or match the same `cfg.group_regex` group; groups are the transitive
    closure of those relations, so e.g. a duplicate pair straddling a block boundary stays
    together. The result depends only on ids, hashes, classes and the seed."""
    regex = re.compile(cfg.group_regex) if cfg.group_regex else None
    ratios = (cfg.train, cfg.val, cfg.test)

    cross_class: dict[str, set[str]] = defaultdict(set)
    uf = _UnionFind()
    for s in samples:
        cross_class[f"sha:{s.sha256}"].add(s.class_name)
        keys = [f"{s.class_name}|{k}" for k in group_keys(s, regex, cfg.sequence_block)]
        for k in keys[1:]:
            uf.union(keys[0], k)
        uf.find(keys[0])
    for key, classes in cross_class.items():
        if len(classes) > 1:
            log.warning("identical image content under several classes %s (%s): label noise",
                        sorted(classes), key)

    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        root = uf.find(f"{s.class_name}|{group_keys(s, regex, cfg.sequence_block)[0]}")
        groups[root].append(s)

    by_class: dict[str, list[str]] = defaultdict(list)
    for root, members in groups.items():
        by_class[members[0].class_name].append(root)

    rng = random.Random(cfg.seed)
    out: dict[str, str] = {}
    for cname in sorted(by_class):
        roots = sorted(by_class[cname])
        rng.shuffle(roots)
        n_tr, n_va, _ = allocate(len(roots), ratios)
        for i, root in enumerate(roots):
            split = SPLITS[0] if i < n_tr else SPLITS[1] if i < n_tr + n_va else SPLITS[2]
            for s in groups[root]:
                out[s.image_id] = split
    return out

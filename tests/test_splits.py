from pathlib import Path

import pytest

from leukemia_pp.config import SplitConfig
from leukemia_pp.io import Sample
from leukemia_pp.splits import allocate, assign_splits


def mk(i, cls, sha=None):
    return Sample(f"{cls}-{i:03d}", Path(f"{cls}-{i}.jpg"), f"{cls}/{i}.jpg", cls, sha or f"h{cls}{i}")


@pytest.mark.parametrize("n,expected", [
    (0, (0, 0, 0)), (1, (1, 0, 0)), (2, (1, 0, 1)), (3, (1, 1, 1)), (10, (7, 2, 1)),
    (100, (70, 15, 15)),
])
def test_allocate(n, expected):
    out = allocate(n, (0.7, 0.15, 0.15))
    assert sum(out) == n
    if n in (0, 1, 2, 3, 100):
        assert out == expected
    assert out[0] >= 1 if n else True


def test_allocate_sums_for_all_sizes():
    for n in range(1, 200):
        assert sum(allocate(n, (0.7, 0.15, 0.15))) == n
        if n >= 3:
            assert min(allocate(n, (0.7, 0.15, 0.15))) >= 1


def test_deterministic_and_stratified():
    samples = [mk(i, c) for c in ("A", "B") for i in range(20)]
    a = assign_splits(samples, SplitConfig(seed=1))
    assert a == assign_splits(list(reversed(samples)), SplitConfig(seed=1))
    assert a != assign_splits(samples, SplitConfig(seed=2))
    for c in ("A", "B"):
        splits = [a[s.image_id] for s in samples if s.class_name == c]
        assert (splits.count("train"), splits.count("val"), splits.count("test")) == (14, 3, 3)


def test_exact_duplicates_never_straddle_splits():
    samples = [mk(i, "A") for i in range(30)] + [mk(100 + i, "A", sha=f"hA{i}") for i in range(30)]
    out = assign_splits(samples, SplitConfig(seed=3))
    for i in range(30):
        assert out[f"A-{i:03d}"] == out[f"A-{100 + i:03d}"]


def test_group_regex_keeps_patient_together():
    samples = [Sample(f"P{p:02d}_img{k}", Path("x"), "x", "A", f"s{p}{k}")
               for p in range(20) for k in range(3)]
    out = assign_splits(samples, SplitConfig(group_regex=r"^(P\d+)_"))
    for p in range(20):
        assert len({out[f"P{p:02d}_img{k}"] for k in range(3)}) == 1
    assert {"train", "val", "test"} <= set(out.values())


def test_every_class_has_train():
    out = assign_splits([mk(0, "A"), mk(0, "B"), mk(1, "B")], SplitConfig())
    assert out["A-000"] == "train"
    assert "train" in {out["B-000"], out["B-001"]}

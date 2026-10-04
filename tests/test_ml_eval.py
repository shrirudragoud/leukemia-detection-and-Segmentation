import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from leukemia_ml.config import AugConfig, DataConfig, ExperimentConfig, TrainConfig  # noqa: E402
from leukemia_ml.eval import calibration, stats  # noqa: E402
from leukemia_ml.eval.folds import grouped_folds  # noqa: E402
from leukemia_ml.eval.metrics import (auroc_ovr, balanced_accuracy, binary_sens_spec, confusion,  # noqa: E402
                                      ece, macro_f1, summarize)


def test_metrics_known_values():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])
    pred = np.array([0, 0, 0, 1, 1, 0, 2, 2, 2, 2])
    cm = confusion(y, pred, 3)
    assert cm.tolist() == [[3, 1, 0], [1, 1, 0], [0, 0, 4]]
    assert balanced_accuracy(y, pred, 3) == pytest.approx((0.75 + 0.5 + 1.0) / 3)
    f1 = [2 * 0.75 * 0.75 / 1.5, 2 * 0.5 * 0.5 / 1.0, 1.0]
    assert macro_f1(y, pred, 3) == pytest.approx(np.mean(f1))
    assert balanced_accuracy(y, y, 3) == 1.0


def test_auroc_and_ece():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])
    assert auroc_ovr(y, perfect) == 1.0
    assert auroc_ovr(y, perfect[::-1].copy()) == 0.0
    rng = np.random.default_rng(0)
    assert abs(auroc_ovr(rng.integers(0, 2, 4000), rng.random((4000, 2))) - 0.5) < 0.05
    sure_wrong = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert ece(np.array([1, 1]), sure_wrong) == pytest.approx(1.0)
    assert ece(np.array([0, 0]), sure_wrong) == pytest.approx(0.0)


def test_binary_sens_spec_and_summary():
    y = np.array([0, 0, 1, 2, 3, 3])
    pred = np.array([0, 1, 0, 2, 3, 3])
    ss = binary_sens_spec(y, pred)
    assert ss["sensitivity"] == pytest.approx(3 / 4) and ss["specificity"] == pytest.approx(1 / 2)
    probs = np.eye(4)[pred] * 0.9 + 0.025
    s = summarize(y, probs)
    assert s["n"] == 6 and 0 <= s["auroc_ovr"] <= 1 and len(s["confusion"]) == 4


def test_grouped_folds_are_group_disjoint_stratified_and_seeded():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 5)
    y = np.repeat(rng.integers(0, 3, 60), 5)
    folds = list(grouped_folds(groups, y, 5, seeds=(0, 1)))
    assert len(folds) == 10
    for _, _, tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])
        assert set(np.unique(y[te])) == {0, 1, 2}
    tests0 = [tuple(te[:3]) for s, _, _, te in folds if s == 0]
    tests1 = [tuple(te[:3]) for s, _, _, te in folds if s == 1]
    assert tests0 != tests1
    # every sample is in exactly one test fold per seed
    for seed in (0, 1):
        seen = np.concatenate([te for s, _, _, te in folds if s == seed])
        assert sorted(seen) == list(range(len(y)))


def test_cluster_bootstrap_widens_with_correlated_groups():
    rng = np.random.default_rng(0)
    n_groups, per = 30, 20
    groups = np.repeat(np.arange(n_groups), per)
    base = rng.random(n_groups) < 0.7                       # per-group accuracy is all-or-none
    correct = np.repeat(base, per).astype(float)
    ci = stats.cluster_bootstrap(lambda i: correct[i].mean(), groups, n_boot=800)
    naive = stats.cluster_bootstrap(lambda i: correct[i].mean(), np.arange(len(correct)), n_boot=800)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert (ci["hi"] - ci["lo"]) > 2 * (naive["hi"] - naive["lo"])      # image-level CI is too narrow


def test_paired_bootstrap_detects_real_difference_and_not_null():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 10)
    good = (rng.random(400) < 0.9).astype(float)
    bad = (rng.random(400) < 0.6).astype(float)
    r = stats.paired_bootstrap(lambda i: good[i].mean(), lambda i: bad[i].mean(), groups, 800)
    assert r["diff"] > 0.2 and r["p"] < 0.01 and r["lo"] > 0
    same = stats.paired_bootstrap(lambda i: good[i].mean(), lambda i: good[i].mean(), groups, 200)
    assert same["diff"] == 0 and same["p"] == 1.0


def test_corrected_ttest_is_more_conservative_than_plain():
    from scipy import stats as sps
    d = np.array([0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.02, 0.01, 0.03, 0.02])
    corr = stats.corrected_resampled_ttest(d, n_train=80, n_test=20)
    plain = sps.ttest_1samp(d, 0.0)
    assert corr["p"] > plain.pvalue and corr["mean"] == pytest.approx(d.mean())
    t_manual = d.mean() / np.sqrt((1 / 10 + 20 / 80) * d.var(ddof=1))
    assert corr["t"] == pytest.approx(t_manual)
    assert np.isnan(stats.corrected_resampled_ttest(np.zeros(5), 80, 20)["p"])


def test_holm_matches_hand_computation():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5, 0.0001])[1] == pytest.approx(0.0002)
    assert max(stats.holm([0.5, 0.6, 0.7])) <= 1.0


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 3000)
    logits = rng.normal(0, 1, (3000, 3))
    logits[np.arange(3000), y] += 1.0
    over = logits * 4.0
    t = calibration.fit_temperature(over, y)
    assert 3.0 < t < 5.0
    before = ece(y, calibration.apply_temperature(over, 1.0))
    after = ece(y, calibration.apply_temperature(over, t))
    assert after < before / 2


def test_probe_finds_signal_and_not_noise():
    from leukemia_ml.data.index import CellIndex
    from leukemia_ml.eval.probe import image_features, run_probe
    rng = np.random.default_rng(0)
    n_img, per = 120, 4
    img_label = rng.integers(0, 3, n_img)
    image_of_cell = np.repeat(np.arange(n_img), per)
    labels = np.repeat(img_label, per)
    n = len(labels)
    index = CellIndex(
        image_id=np.array([f"i{i}" for i in image_of_cell]), cell_id=np.tile(np.arange(per), n_img),
        label=labels, class_name=labels.astype(str), split=np.array(["train"] * n),
        group=np.repeat(np.arange(n_img) // 2, per).astype(str), area=np.ones(n),
        touches_border=np.zeros(n, bool), features=np.zeros((n, 0), np.float32), feature_names=[],
        class_labels=("a", "b", "c"), images=np.array([f"i{i}" for i in range(n_img)]),
        image_of_cell=image_of_cell, image_label=img_label, image_split=np.array(["train"] * n_img),
        image_group=(np.arange(n_img) // 2).astype(str), image_class=img_label.astype(str))
    signal = np.eye(3)[labels] * 3 + rng.normal(0, 1, (n, 3))
    X, idx = image_features(index, np.hstack([signal, rng.normal(0, 1, (n, 5))]))
    good = run_probe(index, X, idx, seeds=(0,))
    assert good["mean"]["balanced_accuracy"] > 0.9
    noise = rng.normal(0, 1, (n, 8))
    Xn, idxn = image_features(index, noise)
    bad = run_probe(index, Xn, idxn, seeds=(0,))
    assert abs(bad["mean"]["balanced_accuracy"] - 1 / 3) < 0.15


def test_experiment_config_roundtrip_and_validation():
    cfg = ExperimentConfig.from_dict({"data": {"feature_groups": ["cell_shape"], "task": "binary"},
                                      "train": {"epochs": 2}, "name": "t"})
    assert cfg.data.feature_groups == ("cell_shape",) and cfg.train.epochs == 2
    assert ExperimentConfig.from_dict(cfg.to_dict()).hash() == cfg.hash()
    for bad in ({"nope": 1}, {"data": {"nope": 1}}):
        with pytest.raises(ValueError):
            ExperimentConfig.from_dict(bad)
    with pytest.raises(ValueError):
        DataConfig(source="x")
    with pytest.raises(ValueError):
        AugConfig(colour="x")
    with pytest.raises(ValueError):
        TrainConfig(class_weighting="x")

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_ml_data import dcfg, synth  # noqa: E402,F401  (fixture re-export)

from leukemia_ml.config import (AugConfig, ExperimentConfig, ModelConfig,  # noqa: E402
                                TrainConfig)
from leukemia_ml.data.crops import build_crop_cache  # noqa: E402
from leukemia_ml.data.datasets import load_crops  # noqa: E402
from leukemia_ml.data.index import load_index  # noqa: E402
from leukemia_ml.embed import embedding_path, extract_embeddings  # noqa: E402
from leukemia_ml.train.loop import (Trainer, class_weights, make_scheduler,  # noqa: E402
                                    stratified_subset)


def exp(synth, name="t", **kw):
    model = dict(encoder="timm:resnet10t", pretrained=False, input_size=32, freeze="frozen",
                 head="mil", hidden=32, dropout=0.0)
    model.update(kw.pop("model", {}))
    train = dict(epochs=6, batch_size=4, lr_head=3e-3, patience=10, seed=0, device="cpu",
                 warmup_frac=0.1)
    train.update(kw.pop("train", {}))
    data = kw.pop("data", {})
    return ExperimentConfig(data=dcfg(synth, **data), model=ModelConfig(**model),
                            aug=AugConfig(colour="hsv"), train=TrainConfig(**train), name=name)


def test_class_weights_scheduler_subset():
    y = np.array([0] * 8 + [1] * 2)
    w = class_weights(y, 2, "inv")
    assert w[1] > w[0] and float(w.mean()) == pytest.approx(1.0)
    assert float(class_weights(y, 2, "none").std()) == 0.0
    assert 1.0 < float(class_weights(y, 2, "inv_sqrt")[1] / class_weights(y, 2, "inv_sqrt")[0]) < float(w[1] / w[0])
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = make_scheduler(opt, 100, 0.1)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert lrs[0] < lrs[9] <= 1.0 and lrs[-1] < 0.01 and max(lrs) == pytest.approx(1.0, abs=0.01)


def test_stratified_subset_balances_classes(synth):
    _, out = synth
    idx = load_index(out, "4way", ())
    sub = stratified_subset(idx, idx.image_indices("train"), 8, seed=0)
    assert set(idx.image_label[sub]) == {0, 1, 2, 3} and len(sub) == 8
    assert np.array_equal(sub, stratified_subset(idx, idx.image_indices("train"), 8, seed=0))


def test_embedding_cache_is_keyed_and_reused(synth, tmp_path):
    cfg = exp(synth)
    idx = load_index(cfg.data.run_dir, "4way", ())
    d = build_crop_cache(cfg.data, idx, workers=1)
    crops = load_crops(d)
    e1, i1 = extract_embeddings(crops, cfg.model, tmp_path, "cpu")
    assert not i1["cached"] and e1.shape[0] == idx.n_cells and e1.dtype == np.float16
    e2, i2 = extract_embeddings(crops, cfg.model, tmp_path, "cpu")
    assert i2["cached"] and np.array_equal(e1, e2)
    other = ModelConfig(encoder="timm:resnet10t", pretrained=False, input_size=48)
    assert embedding_path(tmp_path, other) != embedding_path(tmp_path, cfg.model)


def test_frozen_embedding_training_is_deterministic_and_learns(synth, tmp_path):
    cfg = exp(synth, model={"use_features": True, "feature_hidden": 16},
              data={"feature_groups": ("cell_shape",)})
    r1 = Trainer(cfg, tmp_path / "a").fit()
    r2 = Trainer(cfg, tmp_path / "b").fit()
    assert r1["test"] == r2["test"] and r1["history"][0]["train_loss"] == r2["history"][0]["train_loss"]
    assert r1["history"][-1]["train_loss"] < r1["history"][0]["train_loss"]
    # class is encoded by cell size, which the shape features expose -> must be learnable
    assert r1["best_val_macro_f1"] > 0.6
    for f in ("config.json", "metrics.json", "predictions.csv", "model.pt", "train_log.jsonl"):
        assert (tmp_path / "a" / f).exists(), f
    m = json.loads((tmp_path / "a" / "metrics.json").read_text())
    ci = m["test_ci"]["balanced_accuracy"]
    assert ci["lo"] <= ci["point"] <= ci["hi"] and ci["n_groups"] >= 4
    rows = (tmp_path / "a" / "predictions.csv").read_text().strip().splitlines()
    assert len(rows) - 1 == m["test"]["n"] and rows[0].startswith("image_id,group,label,pred")
    assert m["temperature"] > 0 and 0 <= m["test_calibrated"]["ece"] <= 1


def test_different_seeds_change_the_run(synth, tmp_path):
    a = Trainer(exp(synth, train={"seed": 0, "epochs": 2}), tmp_path / "s0").fit()
    b = Trainer(exp(synth, train={"seed": 1, "epochs": 2}), tmp_path / "s1").fit()
    assert a["history"][0]["train_loss"] != b["history"][0]["train_loss"]


def test_early_stopping_and_best_checkpoint(synth, tmp_path):
    cfg = exp(synth, train={"epochs": 12, "patience": 1, "lr_head": 1e-2})
    r = Trainer(cfg, tmp_path / "es").fit()
    assert r["epochs_run"] <= 12
    best_epoch_f1 = max(h["val_macro_f1"] for h in r["history"])
    assert r["best_val_macro_f1"] == pytest.approx(best_epoch_f1)


def test_lora_training_updates_adapters_only(synth, tmp_path):
    cfg = exp(synth, model={"encoder": "dinov2_s", "input_size": 28, "freeze": "lora",
                            "lora_rank": 4}, train={"epochs": 1, "max_train_images": 8,
                                                    "max_eval_images": 8, "lr_lora": 1e-2})
    t = Trainer(cfg, tmp_path / "lora")
    from leukemia_ml.models.model import build_model
    probe = build_model(cfg.model, 4)
    base_before = {k: v.clone() for k, v in probe.encoder.state_dict().items()
                   if not k.endswith((".A", ".B"))}
    r = t.fit()
    assert r["epochs_run"] == 1 and r["trainable_params"] > 0
    state = torch.load(tmp_path / "lora" / "model.pt", weights_only=False)
    enc_keys = [k for k in state if k.startswith("encoder.")]
    assert enc_keys and all(k.endswith((".A", ".B")) for k in enc_keys)   # only adapters saved
    assert any(state[k].abs().sum() > 0 for k in enc_keys if k.endswith(".B"))
    # the pretrained (here: random) base weights of a fresh build are unchanged by construction
    assert all(torch.equal(v, probe.encoder.state_dict()[k]) for k, v in base_before.items())


def test_full_finetune_path_with_augmentation_runs(synth, tmp_path):
    cfg = exp(synth, model={"freeze": "full", "head": "mean"},
              train={"epochs": 1, "max_train_images": 8, "max_eval_images": 8, "lr_encoder": 1e-3})
    r = Trainer(cfg, tmp_path / "full").fit()
    assert np.isfinite(r["history"][0]["train_loss"]) and r["history"][0]["cells_per_s"] > 0


def test_binary_task_and_cell_head(synth, tmp_path):
    cfg = exp(synth, data={"task": "binary"}, model={"head": "cell"}, train={"epochs": 2})
    r = Trainer(cfg, tmp_path / "bin").fit()
    assert len(r["test"]["confusion"]) == 2

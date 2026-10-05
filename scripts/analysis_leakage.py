"""Evaluation-scheme analysis on cached DinoBloom-S embeddings -> docs/results/leakage_analysis.json

A  scheme comparison (random 37-image blocks vs contiguous folds, with/without embargo)
B  random-block folds: purge training images near test images, vs removing the SAME number at random
C  contiguous folds: accuracy vs minimum capture-number gap to the test segment, vs random removal
Features: frozen DinoBloom-S embeddings of isolated, white-balanced cell crops (mean per image),
          and hand-made shape/texture/curvature features. Logistic head, balanced class weights."""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from leukemia_ml.config import ExperimentConfig
from leukemia_ml.data.crops import build_crop_cache
from leukemia_ml.data.index import load_index
from leukemia_ml.embed import embedding_path
from leukemia_ml.eval.folds import contiguous_folds, grouped_folds, purge_train
from leukemia_ml.eval.metrics import balanced_accuracy
from leukemia_ml.eval.probe import image_features, image_numbers, run_probe

OUT = Path("docs/results/leakage_analysis.json")
cfg = ExperimentConfig.from_json("configs/ml/dinobloom_s_frozen_mil.json")
idx = load_index(cfg.data.run_dir, "4way", ("cell_shape", "cell_texture", "apc", "log"))
d = build_crop_cache(cfg.data, idx)
emb = np.load(embedding_path(d, cfg.model)).astype(np.float32)
Xe, img = image_features(idx, emb)
Xt, imt = image_features(idx, np.zeros((idx.n_cells, 1), np.float32), idx.features)
Xt = Xt[:, 1:]
y = idx.image_label[img]
groups = idx.image_group[img]
order = image_numbers(idx, img)
feats = {"DinoBloom-S": (Xe, img), "hand-made features": (Xt, imt)}


def fit_eval(X, tr, te):
    if len(set(y[tr])) < 4:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, class_weight="balanced"))
    clf.fit(X[tr], y[tr])
    return balanced_accuracy(y[te], clf.predict(X[te]), 4)


# ---- A
A = []
for name, kw in [("random 37-image blocks", dict(scheme="grouped", embargo=0)),
                 ("contiguous folds, embargo 0", dict(scheme="contiguous", embargo=0)),
                 ("contiguous folds, embargo 37", dict(scheme="contiguous", embargo=37)),
                 ("contiguous folds, embargo 75", dict(scheme="contiguous", embargo=75))]:
    for fname, (X, im) in feats.items():
        r = run_probe(idx, X, im, seeds=(0, 1, 2), **kw)
        A.append({"scheme": name, "features": fname, "train_size": r["mean_train_size"],
                  "balanced_accuracy": r["mean"]["balanced_accuracy"], "sd": r["std"]["balanced_accuracy"],
                  "macro_f1": r["mean"]["macro_f1"], "auroc": r["mean"]["auroc_ovr"]})
        print(A[-1])

# ---- B
rng = np.random.default_rng(0)
B = []
for e in (0, 37, 75, 150):
    pur, rnd, ns = [], [], []
    for _seed, _f, tr, te in grouped_folds(groups, y, 5, (0, 1)):
        trp = purge_train(tr, te, y, order, e)
        ns.append(len(trp))
        pur.append(fit_eval(Xe, trp, te))
        keep = []
        for c in range(4):
            pool = tr[y[tr] == c]
            keep.append(rng.choice(pool, min(int((y[trp] == c).sum()), len(pool)), replace=False))
        rnd.append(fit_eval(Xe, np.concatenate(keep), te))
    B.append({"embargo": e, "train_size": float(np.mean(ns)), "purged": float(np.nanmean(pur)),
              "random_same_size": float(np.nanmean(rnd))})
    print(B[-1])

# ---- C
C = []
for e in (0, 37, 75, 150, 225, 300):
    out = {"embargo": e, "n": [], "dino": [], "dino_rand": [], "hand": [], "hand_rand": []}
    for _r, _k, tr, te in contiguous_folds(order, y, 5, e, rotate=0):
        out["n"].append(len(tr))
        out["dino"].append(fit_eval(Xe, tr, te))
        out["hand"].append(fit_eval(Xt, tr, te))
        keep = []
        for c in range(4):
            pool = np.nonzero(y == c)[0]
            pool = pool[~np.isin(pool, te)]
            keep.append(rng.choice(pool, min(int((y[tr] == c).sum()), len(pool)), replace=False))
        keep = np.concatenate(keep)
        out["dino_rand"].append(fit_eval(Xe, keep, te))
        out["hand_rand"].append(fit_eval(Xt, keep, te))
    C.append({k: (float(np.nanmean(v)) if isinstance(v, list) else v) for k, v in out.items()})
    print(C[-1])

OUT.write_text(json.dumps({"A_schemes": A, "B_random_block_purge": B, "C_contiguous_gap": C,
                           "n_images": int(len(y)), "n_groups": int(len(set(groups)))}, indent=2))

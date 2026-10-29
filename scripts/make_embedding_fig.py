"""Projection of the frozen DinoBloom-S image embeddings (mean over cells), coloured by class and by capture-order segment.
-> paper/figures/fig_embedding.png and docs/results/embedding_summary.json (silhouette scores of class and of segment labels)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from leukemia_ml.config import ExperimentConfig  # noqa: E402
from leukemia_ml.data.crops import build_crop_cache  # noqa: E402
from leukemia_ml.data.index import load_index  # noqa: E402
from leukemia_ml.embed import embedding_path  # noqa: E402
from leukemia_ml.eval.folds import contiguous_folds  # noqa: E402
from leukemia_ml.eval.probe import image_features, image_numbers  # noqa: E402

cfg = ExperimentConfig.from_json("configs/ml/dinobloom_s_frozen_mil.json")
idx = load_index(cfg.data.run_dir, "4way", ())
d = build_crop_cache(cfg.data, idx)
emb = np.load(embedding_path(d, cfg.model)).astype(np.float32)
X, img = image_features(idx, emb)
y = idx.image_label[img]
order = image_numbers(idx, img)
seg = np.zeros(len(y), int)
for _, k, _, te in contiguous_folds(order, y, 5, 0):
    seg[te] = k
Z = StandardScaler().fit_transform(X)
P = PCA(50, random_state=0).fit_transform(Z)
T = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(P)
CL = ["Benign", "Early", "Pre", "Pro"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
for c, m in zip(range(4), ["o", "s", "^", "x"]):
    s = y == c
    ax[0].scatter(T[s, 0], T[s, 1], s=5, marker=m, color=str(0.15 + 0.22 * c), label=CL[c], linewidths=0.6)
for k, m in zip(range(5), ["o", "s", "^", "x", "+"]):
    s = seg == k
    ax[1].scatter(T[s, 0], T[s, 1], s=5, marker=m, color=str(0.1 + 0.18 * k), label=f"segment {k + 1}", linewidths=0.6)
for a, t in zip(ax, ["(a) Coloured by class", "(b) Coloured by capture-order segment"]):
    a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=6, markerscale=2, loc="best")
fig.tight_layout(); fig.savefig("paper/figures/fig_embedding.png", bbox_inches="tight")
out = {"n_images": int(len(y)), "silhouette_class_pca50": float(silhouette_score(P, y)), "silhouette_segment_pca50": float(silhouette_score(P, seg))}
json.dump(out, open("docs/results/embedding_summary.json", "w"), indent=1)
print(out)

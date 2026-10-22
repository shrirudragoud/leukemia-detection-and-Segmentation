"""What does each processing stage contribute to classification? Tabular logistic probe on image-level means of per-cell features computed by the
full run with HDS and BEEMD enabled (data/full_run_all). Contiguous folds, embargo 37, three rotations; 95% cluster-bootstrap CIs (first rotation).
Every variant sees ONLY the named feature groups (no pixels, no encoder). -> docs/results/stage_ablation.json
Also summarises the pure-IMF selection over all images."""
import json
from pathlib import Path

import numpy as np

from leukemia_ml.data.index import load_index
from leukemia_ml.eval import stats
from leukemia_ml.eval.metrics import balanced_accuracy
from leukemia_ml.eval.probe import image_features, run_probe

RUN = "data/full_run_all"
VARIANTS = [
    ("shape (morphological)", ("cell_shape",)),
    ("texture (GLCM, entropy)", ("cell_texture",)),
    ("APC response layer", ("apc",)),
    ("LoG response layer", ("log",)),
    ("HDS edge indicator", ("hds_edge",)),
    ("pure-IMF layer (BEEMD)", ("imf_pure",)),
    ("IMF energies (BEEMD)", ("imf_energy",)),
    ("shape + texture", ("cell_shape", "cell_texture")),
    ("shape + texture + APC + LoG", ("cell_shape", "cell_texture", "apc", "log")),
    ("shape + texture + APC + LoG + HDS edge", ("cell_shape", "cell_texture", "apc", "log", "hds_edge")),
    ("... + pure-IMF layer and IMF energies", ("cell_shape", "cell_texture", "apc", "log", "hds_edge", "imf_pure", "imf_energy")),
]
rows = []
base = None
for name, groups in VARIANTS:
    idx = load_index(RUN, "4way", groups)
    X, img = image_features(idx, np.zeros((idx.n_cells, 1), np.float32), idx.features)
    X = X[:, 1:]
    r = run_probe(idx, X, img, seeds=(0, 1, 2), embargo=37, scheme="contiguous")
    y, g = r["y"], r["groups"]
    pred = r["oof"][0].argmax(1)
    ci = stats.cluster_bootstrap(lambda i: balanced_accuracy(y[i], pred[i], 4), g, 1000)
    row = {"variant": name, "groups": list(groups), "n_features": int(X.shape[1]), "balanced_accuracy": r["mean"]["balanced_accuracy"],
           "sd_rotations": r["std"]["balanced_accuracy"], "ci": ci, "macro_f1": r["mean"]["macro_f1"], "auroc": r["mean"]["auroc_ovr"]}
    if name == "shape + texture":
        base = (y, g, pred)
    if base is not None and groups[:2] == ("cell_shape", "cell_texture") and len(groups) > 2:
        yb, gb, pb = base
        row["vs_shape_texture"] = stats.paired_bootstrap(lambda i: balanced_accuracy(y[i], pred[i], 4), lambda i: balanced_accuracy(yb[i], pb[i], 4), g, 1000)
    rows.append(row)
    print(name, round(row["balanced_accuracy"], 3))
hp = [v for v in rows if "vs_shape_texture" in v]
adj = stats.holm([v["vs_shape_texture"]["p"] for v in hp])
for v, a in zip(hp, adj):
    v["vs_shape_texture"]["p_holm"] = a

# pure-IMF selection statistics over all images (from the cached arrays)
import glob  # noqa: E402

sel, per_class = [], {}
for f in glob.glob(f"{RUN}/cache/*/*.npz"):
    z = np.load(f)
    if "imf_selected" in z:
        s = z["imf_selected"]
        sel.append(s)
sel = np.array(sel)
imf = {"n_images": int(len(sel)), "selected_fraction_per_mode": sel.mean(0).tolist(), "all_selected_fraction": float(sel.all(1).mean()),
       "none_selected_fraction": float((~sel.any(1)).mean())}
Path("docs/results/stage_ablation.json").write_text(json.dumps({"rows": rows, "imf_selection": imf, "protocol": "contiguous folds, embargo 37, 3 rotations, logistic head on image-mean features"}, indent=1))
print(imf)

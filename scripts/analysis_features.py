"""Per-class descriptive statistics of per-cell features (median, IQR) -> docs/results/feature_stats.json.
Descriptive only: cells of one image are not independent, so no significance tests are reported."""
import json

import numpy as np
import pandas as pd

cells = pd.read_csv("data/full_run_v2/cells.csv")
cells = cells[cells["touches_border"] == 0]
cols = ["cell_area", "cell_eq_diameter", "cell_aspect_ratio", "cell_solidity", "cell_circularity", "cell_L_mean", "cell_a_mean", "cell_b_mean",
        "cell_L_std", "cell_glcm_contrast", "cell_glcm_homogeneity", "cell_entropy", "cell_apc_mean", "cell_log_mean"]
out = {"n_interior_cells": int(len(cells)), "per_class_n": {}, "features": {}}
for k, g in cells.groupby("class_name"):
    out["per_class_n"][k] = int(len(g))
for c in cols:
    out["features"][c] = {}
    for k, g in cells.groupby("class_name"):
        v = g[c].dropna().to_numpy()
        q = np.quantile(v, [0.25, 0.5, 0.75])
        out["features"][c][k] = {"q25": float(q[0]), "median": float(q[1]), "q75": float(q[2])}
json.dump(out, open("docs/results/feature_stats.json", "w"), indent=1)
print(out["per_class_n"])

# ---- per-class medians of the response-layer statistics (full run with HDS and BEEMD enabled)
full = pd.read_csv("data/full_run_all/cells.csv")
full = full[full["touches_border"] == 0]
lcols = ["cell_apc_mean", "cell_apc_rim_mean", "cell_log_mean", "cell_log_rim_mean", "cell_hds_edge_mean", "cell_hds_edge_rim_mean",
         "cell_imf_pure_mean", "cell_imf_pure_std", "cell_imf1_energy", "cell_imf2_energy", "cell_imf3_energy", "cell_imf4_energy"]
layers = {"n_cells": {k: int(len(g)) for k, g in full.groupby("class_name")}, "medians": {}}
for c in lcols:
    layers["medians"][c] = {}
    for k, g in full.groupby("class_name"):
        q = np.quantile(g[c].dropna().to_numpy(), [0.25, 0.5, 0.75])
        layers["medians"][c][k] = {"q25": float(q[0]), "median": float(q[1]), "q75": float(q[2])}
json.dump(layers, open("docs/results/stage_layer_stats.json", "w"), indent=1)
print(layers["n_cells"])

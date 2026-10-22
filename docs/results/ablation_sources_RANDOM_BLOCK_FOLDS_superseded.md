> **Superseded.** Computed with the old random 37-block folds, which leak sessions. Re-run with contiguous folds + embargo before citing.

# Ablation (reference: A. clean RGB, isolated cell (reference))

90 groups; seeds [0, 1, 2]; 1000 bootstrap resamples of groups. Balanced accuracy, 95% cluster-bootstrap CI.

| Variant | Balanced acc. | 95% CI | Macro-F1 | AUROC | Diff. vs ref. (95% CI) | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- |
| A. clean RGB, isolated cell (reference) | 0.969 | [0.961, 0.980] | 0.969 | 0.998 | - | - |
| B. clean grayscale, isolated cell (no stain colour) | 0.959 | [0.945, 0.969] | 0.958 | 0.995 | -0.014 [-0.022, -0.006] | 0.006 |
| C. normalised RGB, background visible | 0.988 | [0.976, 0.992] | 0.988 | 0.999 | +0.014 [+0.006, +0.023] | 0.006 |
| D. raw RGB, background visible | 0.992 | [0.982, 0.995] | 0.992 | 1.000 | +0.018 [+0.011, +0.027] | 0.006 |
| E. tabular shape+texture+curvature (no pixels) | 0.854 | [0.830, 0.874] | 0.848 | 0.959 | -0.119 [-0.140, -0.098] | 0.004 |

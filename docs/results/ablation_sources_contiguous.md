# Ablation (reference: A. clean RGB, isolated cell (reference))

90 groups; contiguous folds, embargo 37; seeds [0, 1, 2]; 1000 bootstrap resamples of groups. Balanced accuracy, 95% cluster-bootstrap CI.

| Variant | Balanced acc. | 95% CI | Macro-F1 | AUROC | Diff. vs ref. (95% CI) | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- |
| A. clean RGB, isolated cell (reference) | 0.959 | [0.944, 0.973] | 0.958 | 0.995 | - | - |
| B. clean grayscale, isolated cell (no stain colour) | 0.940 | [0.923, 0.958] | 0.939 | 0.991 | -0.018 [-0.027, -0.010] | 0.004 |
| C. normalised RGB, background visible | 0.975 | [0.964, 0.987] | 0.973 | 0.998 | +0.016 [+0.007, +0.028] | 0.004 |
| D. raw RGB, background visible | 0.978 | [0.967, 0.988] | 0.977 | 0.999 | +0.019 [+0.010, +0.029] | 0.004 |
| E. tabular shape+texture+curvature (no pixels) | 0.825 | [0.797, 0.852] | 0.818 | 0.944 | -0.135 [-0.158, -0.114] | 0.004 |

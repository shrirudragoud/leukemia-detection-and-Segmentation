# CV summary: cv_dinobloom_s_frozen_mil

5 contiguous session-aware folds per class, embargo 37 image numbers; one model per fold; pooled out-of-fold predictions; 95% cluster-bootstrap CI over groups

3256 images, 90 groups, folds [0, 1, 2, 3, 4].

| Metric | Pooled | 95% CI | Per-fold mean +- SD |
| --- | --- | --- | --- |
| Balanced accuracy | 0.968 | [0.955, 0.978] | 0.968 +- 0.019 |
| Macro-F1 | 0.967 | [0.950, 0.978] | 0.967 +- 0.024 |
| AUROC (OvR) | 0.995 | - | 0.996 +- 0.004 |
| ECE | 0.010 | - | 0.121 |

Benign-vs-malignant: sensitivity 0.990, specificity 0.960.

Per-class recall: 0.960, 0.973, 0.957, 0.980

Confusion (rows = true):

      484     9    10     1
       14   958     7     6
       10    16   922    15
        3     4     9   788

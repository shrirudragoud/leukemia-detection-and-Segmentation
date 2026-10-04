# Shortcut / confound audit

3256 images, 90 patient-proxy groups, 5-fold grouped CV, classes {'Benign': 504, 'Early': 985, 'Pre': 963, 'Pro': 804}. Chance (balanced accuracy) = 0.25.

| Group | Predictor | Balanced acc. | Shortcut risk |
| --- | --- | --- | --- |
| acquisition | background colour, raw image (3 numbers) | 0.82 +- 0.00 | VERY STRONG |
| acquisition | background colour, after stain normalisation (3 numbers) | 0.78 +- 0.00 | VERY STRONG |
| acquisition | scale-bar corner patch, raw image | 0.72 +- 0.00 | VERY STRONG |
| after shortcut removal | background colour of rgb_clean (sanity: constant) | 0.25 +- 0.00 | none (near chance) |
| after shortcut removal | corner patch of rgb_clean (sanity: constant) | 0.44 +- 0.00 | weak |
| residual | mean cell colour, stain-normalised (3 numbers) | 0.77 +- 0.00 | VERY STRONG |
| residual | mean cell colour, white-balanced rgb_clean (3 numbers) | 0.80 +- 0.00 | VERY STRONG |
| tabular baseline | density (n_cells, wbc_area_frac) | 0.59 +- 0.00 |  |
| tabular baseline | cell shape | 0.68 +- 0.00 |  |
| tabular baseline | cell texture (GLCM, entropy) | 0.65 +- 0.00 |  |
| tabular baseline | cell colour | 0.83 +- 0.00 |  |
| tabular baseline | shape + texture (no colour, no density) | 0.78 +- 0.00 |  |
| tabular baseline | ALL aggregated features | 0.91 +- 0.00 |  |

Caveats:
- Groups are a patient PROXY (consecutive-number blocks + duplicate files); true patient ids are unavailable, so residual leakage is possible.
- RandomForest baseline, 3 seeds; fold assignment is fixed by the groups.
- Near-chance acquisition probes after cleaning only show that the cleaned image no longer carries them; cell colour can still encode stain/session.

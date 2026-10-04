import csv as _csv
import json as _json

import matplotlib as _mpl

_mpl.use("Agg")
import matplotlib.pyplot as _plt  # noqa: E402

CURVES = _json.loads((R / "curve_summary.json").read_text())
CELLS = _json.loads((R / "cells_summary.json").read_text())
RUNDIR = R / "cv_run"
LOGS = [[_json.loads(line) for line in open(RUNDIR / f"fold{k}" / "train_log.jsonl")] for k in range(5)]
GPU_NAMES = [("g01_lora_dinobloom_s", "g01", "DinoBloom-S, low-rank adaptation (RGB, isolated cells)"),
             ("g02_lora_dinobloom_s_gray", "g02", "DinoBloom-S, low-rank adaptation (grayscale)"),
             ("g03_lora_dinobloom_s_hybrid", "g03", "DinoBloom-S, low-rank adaptation + hand-made feature branch"),
             ("g04_full_resnet50", "g04", "ResNet-50, full fine-tuning"),
             ("g05_full_efficientnet_b0", "g05", "EfficientNet-B0, full fine-tuning"),
             ("g06_lora_dinov2_s", "g06", "DINOv2-S (generic), low-rank adaptation"),
             ("g07_lora_dinobloom_b", "g07", "DinoBloom-B, low-rank adaptation"),
             ("g08_lora_dinobloom_s_raw_background", "g08", "DinoBloom-S, low-rank adaptation, raw image with background")]


def _gpu(name):
    f = ROOT / "results_for_paper" / name / "cv_summary.json"
    return _json.loads(f.read_text()) if f.exists() else None


GPU_RES = {nm: _gpu(nm) for nm, _, _ in GPU_NAMES}
SLOT = "[[GPU]]"
N_SLOTS = sum(v is None for v in GPU_RES.values())


def PH(key, fname, title, desc, width=6.0):
    """Placeholder figure: a grey panel naming what must replace it. If `fname` exists the real figure is used instead."""
    real = ROOT / fname
    path = ROOT / "paper" / "figures" / f"ph_{key}.png"
    if real.exists():
        Fg(key, fname, title + " " + desc, width)
        return
    fig, ax = _plt.subplots(figsize=(6.0, 3.2))
    ax.set_facecolor("0.9"); ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.62, "PLACEHOLDER", ha="center", va="center", fontsize=20, color="0.35", transform=ax.transAxes)
    ax.text(0.5, 0.40, "replace with:\n" + fname, ha="center", va="center", fontsize=8, color="0.25", transform=ax.transAxes)
    fig.savefig(path, bbox_inches="tight"); _plt.close(fig)
    Fg(key, f"paper/figures/ph_{key}.png", "**[PLACEHOLDER - replace before submission]** " + title + " " + desc, width)


# ------------------------------------------------------------------ real figures from the CPU run
Fg("training", "paper/figures/fig_training.png", "Training of the attention head (frozen DinoBloom-S) in each of the five folds. (a) Training loss (label-smoothed, class-weighted cross-entropy), "
   "(b) validation balanced accuracy and (c) learning rate (linear warm-up followed by cosine decay) by epoch. Training of each fold stopped early when the validation "
   "macro-F1 had not improved for eight epochs.", 6.4)
Fg("roc", "paper/figures/fig_roc_pr.png", "Discrimination of each class from the other three on the pooled out-of-fold predictions. (a) Receiver operating characteristic curves and "
   "(b) precision-recall curves of the headline model, with the area under each curve in the legend.", 6.0)
Fg("rel", "paper/figures/fig_reliability.png", "Calibration of the temperature-scaled probabilities on the pooled out-of-fold predictions. (a) Reliability diagram: observed accuracy "
   "against mean confidence in ten confidence bins; the diagonal is perfect calibration. (b) Histogram of the confidence of correct and incorrect predictions (log scale).", 6.0)
Fg("cells", "paper/figures/fig_cells.png", "Dependence of the accuracy of the headline model on the number of cells that the pipeline found in the image. (a) Accuracy by number of "
   "cells per image (n above each bar is the number of images). (b) Distribution of the largest attention weight in an image, by true class.", 6.0)

# ------------------------------------------------------------------ fine-tuning table (slots until result files exist)
_rows = []
for nm, short, desc in GPU_NAMES:
    r = GPU_RES[nm]
    if r is None:
        _rows.append([short, desc, SLOT, SLOT, SLOT, SLOT])
    else:
        _rows.append([short, desc, f3(r["pooled"]["balanced_accuracy"]), ci(r["pooled_ci"]["balanced_accuracy"]), f3(r["pooled"]["macro_f1"]), f3(r["pooled"]["auroc_ovr"])])
_rows.insert(0, ["ref.", "DinoBloom-S, frozen encoder, attention head (headline model)", f3(ba), ci(bal), f3(mf), f3(pool["auroc_ovr"])])
T("ft", "Fine-tuning and encoder comparison under the same five contiguous session-aware folds (embargo 37). Entries marked [[GPU]] are produced by the graphics-processor "
        "experiments and are inserted from the result files by the build script; the headline model is shown for reference.",
  [{"w": 0.9, "align": "left"}, {"w": 5, "align": "left"}, {"w": 1.2, "align": "right"}, {"w": 1.7, "align": "right"}, {"w": 1.1, "align": "right"}, {"w": 1.1, "align": "right"}],
  ["Run", "Configuration", "Balanced accuracy", "95% CI", "Macro-F1", "AUROC"], _rows)

PH("ft_curves", "results_for_paper/g01_lora_dinobloom_s/fig_training_curves.png", "Training curves of the low-rank adaptation of DinoBloom-S (run g01).",
   "Expected content: for each fold, training loss and validation macro-F1 against epoch, with the epoch of the best validation score marked and the learning-rate schedule. "
   "Purpose: to show whether adaptation converges, overfits or stops early.", 6.2)
PH("ft_bars", "results_for_paper/fig_encoder_comparison.png", "Balanced accuracy of the fine-tuned configurations.",
   "Expected content: one marker per configuration of Table {T:ft} with its 95% cluster-bootstrap interval, sorted by accuracy, with the frozen headline model as a reference line. "
   "Purpose: to show which differences exceed the sampling uncertainty.", 5.8)
PH("ft_cm", "results_for_paper/g01_lora_dinobloom_s/fig_confusion.png", "Confusion matrix of the best fine-tuned configuration.",
   "Expected content: pooled out-of-fold confusion matrix, as in Fig. {F:cm}a. Purpose: to show whether adaptation changes which classes are confused.", 4.6)
PH("ft_attn", "results_for_paper/g01_lora_dinobloom_s/fig_attention_examples.png", "Attention weights and saliency maps for correctly and incorrectly classified images.",
   "Expected content: for each class, example images with the attention weight of each cell and a Grad-CAM map, for one correct and one incorrect prediction, with a sanity check "
   "in which the model weights are randomised. Purpose: to show where the model looks and whether it looks at cells or at the periphery.", 6.2)
PH("ft_embed", "results_for_paper/fig_embedding_projection.png", "Projection of the image embeddings.",
   "Expected content: a two-dimensional projection of the pooled embeddings of the test images, coloured by class in one panel and by capture-number segment in the other. "
   "Purpose: to show whether the embedding separates capture sessions as strongly as it separates classes.", 6.0)


FS = _json.loads((R / "feature_stats.json").read_text())
_FN = [("cell_area", "Cell area (px)", 0), ("cell_eq_diameter", "Equivalent diameter (px)", 1), ("cell_aspect_ratio", "Aspect ratio", 2), ("cell_solidity", "Solidity", 2),
       ("cell_circularity", "Circularity", 2), ("cell_L_mean", "Mean L*", 1), ("cell_a_mean", "Mean a*", 1), ("cell_b_mean", "Mean b*", 1),
       ("cell_glcm_contrast", "GLCM contrast", 2), ("cell_glcm_homogeneity", "GLCM homogeneity", 2), ("cell_entropy", "Entropy", 2),
       ("cell_apc_mean", "Mean curvature response", 2), ("cell_log_mean", "Mean LoG response", 2)]


def _q(d, dec):
    return f"{d['median']:.{dec}f} ({d['q25']:.{dec}f}-{d['q75']:.{dec}f})"


T("feat", "Per-cell features by class: median and interquartile range (in parentheses) over cells that do not touch the image border. The statistics are descriptive; "
          "cells of the same image are not independent, so no significance tests are given.",
  [{"w": 2.6, "align": "left"}] + [{"w": 2, "align": "right"}] * 4,
  ["Feature"] + [f"{k} (n = {FS['per_class_n'][k]})" for k in CL],
  [[lab] + [_q(FS["features"][key][k], dec) for k in CL] for key, lab, dec in _FN])


def features_section():
    f = FS["features"]
    S("Cell features by class")
    P(f"Table {{T:feat}} summarises the per-cell measurements for the {FS['n_interior_cells']} cells that do not touch the border. Cell size differed between classes: the median area was "
      f"{f['cell_area']['Benign']['median']:.0f} px for Benign, {f['cell_area']['Early']['median']:.0f} for Early, {f['cell_area']['Pre']['median']:.0f} for Pre and {f['cell_area']['Pro']['median']:.0f} for Pro, "
      f"with the Benign cells the smallest. Measures of the outline - aspect ratio, solidity and circularity - were close to identical in the four classes (for example, the median solidity was "
      f"{f['cell_solidity']['Benign']['median']:.2f} to {f['cell_solidity']['Early']['median']:.2f} in all of them), so that the shape of the outline of a segmented cell carries little class information in this dataset. "
      "Texture measures differed modestly.")
    P(f"The colour of the cells differed much more. The median a* value was {f['cell_a_mean']['Pre']['median']:.1f} in Pre cells and {f['cell_a_mean']['Pro']['median']:.1f} in Pro cells, with "
      f"{f['cell_a_mean']['Benign']['median']:.1f} in Benign and {f['cell_a_mean']['Early']['median']:.1f} in Early cells, and the median b* value ranged from {f['cell_b_mean']['Pre']['median']:.1f} (Pre) to "
      f"{f['cell_b_mean']['Early']['median']:.1f} (Early). The Pre class stands out from the other three in lightness and in both chromatic channels. Because the classes are not expected to differ so strongly in "
      "the colour of their cytoplasm and chromatin, and because the colour statistics were computed after stain normalisation, this pattern is consistent with the class-specific acquisition "
      "conditions that the shortcut audit detected. It cannot be separated from a true biological difference with these data.")


@section("results_extra")
def results_extra():
    h0 = [min(L, key=lambda r: -r["val_bal_acc"]) for L in LOGS]
    l_first = float(sum(L[0]["train_loss"] for L in LOGS) / 5)
    l_last = float(sum(L[-1]["train_loss"] for L in LOGS) / 5)
    features_section()
    S("Training dynamics of the headline model")
    P(f"Training the attention head converged quickly (Fig. {{F:training}}). The mean training loss over the five folds fell from {l_first:.2f} in the first epoch to {l_last:.2f} in the last, "
      f"and most of the decrease occurred within the first three epochs. The validation balanced accuracy of the best epoch was between "
      f"{f3(min(h['val_bal_acc'] for h in h0))} and {f3(max(h['val_bal_acc'] for h in h0))} across folds, and the best epoch was reached after {min(h['epoch'] + 1 for h in h0)} to "
      f"{max(h['epoch'] + 1 for h in h0)} epochs. The runs stopped after {min(n_epochs)} to {max(n_epochs)} epochs by the early-stopping rule, much earlier than the maximum of "
      f"{CVC['train']['epochs']}. The validation score of the folds differed by several points at the same epoch (Fig. {{F:training}}b), which reflects that validation stretches contain different "
      "sessions and is the same variation that the per-fold results show.")
    S("Discrimination and precision")
    P(f"The one-vs-rest area under the ROC curve of the pooled out-of-fold probabilities was {f3(CURVES['Benign']['auroc'])} for Benign, {f3(CURVES['Early']['auroc'])} for Early, "
      f"{f3(CURVES['Pre']['auroc'])} for Pre and {f3(CURVES['Pro']['auroc'])} for Pro (Fig. {{F:roc}}a). The area under the precision-recall curve was "
      f"{f3(CURVES['Benign']['auprc'])}, {f3(CURVES['Early']['auprc'])}, {f3(CURVES['Pre']['auprc'])} and {f3(CURVES['Pro']['auprc'])} (Fig. {{F:roc}}b). The Pre class had the lowest area under the ROC curve "
      "although its precision-recall area was similar to that of the other classes, which indicates that it is confused with its neighbours in the ordering of the disease stages without being poorly "
      "ranked overall. These values are computed from probabilities pooled over the folds, so they combine models that were fitted on different training sets.")
    S("Calibration")
    P(f"Before temperature scaling the probabilities of the model were under-confident: the expected calibration error averaged {f3(CV['per_fold_mean']['ece'])} over folds, and the fitted "
      f"temperature was below 1 in every fold (range {min(p['temperature'] for p in pf):.2f}-{max(p['temperature'] for p in pf):.2f}). After scaling, the pooled error was "
      f"{f3(pool['ece'])} (Fig. {{F:rel}}a). The median confidence was {f3(MED_OK)} for correct and {f3(MED_BAD)} for incorrect predictions (Fig. {{F:rel}}b), so the confidence can be used to flag uncertain images for review. "
      "A temperature fitted on one stretch of one dataset is not guaranteed to transfer, and the calibration should be re-checked on any new data.")
    S("Where the model fails")
    accs = CELLS["acc"]
    P(f"Accuracy depended on the number of cells in the image (Fig. {{F:cells}}a). Images in which the pipeline found one to three cells were classified correctly in {pct(accs[0])}% of cases "
      f"(n = {CELLS['n'][0]}), against {pct(accs[2])}% for six to eight cells (n = {CELLS['n'][2]}) and {pct(accs[3])}% for nine to twelve (n = {CELLS['n'][3]}). Images with few cells provide the model with little "
      "evidence, and some of them may contain cells that the segmentation missed; I did not check this image by image. The largest attention weight of an image had a median of "
      f"{f2(MED_ATT)}, which is about {MED_ATT / MED_UNI:.1f} times the weight that uniform attention would give (Fig. {{F:cells}}b), so the head weights cells unequally but does not rely on a single cell. "
      f"In total {sum(1 for _ in ERR_ROWS)} of {N_IMG} images were misclassified; Appendix VI lists them with their true class, predicted class and confidence.")
    P(f"The errors were concentrated between neighbouring stages. Of the {sum(cm[1]) - cm[1][1]} Early images that were misclassified, {cm[1][0]} were called Benign, {cm[1][2]} Pre and {cm[1][3]} Pro. Of the "
      f"{sum(cm[2]) - cm[2][2]} misclassified Pre images, {cm[2][0]} were called Benign, {cm[2][1]} Early and {cm[2][3]} Pro. Benign images were called Early ({cm[0][1]}), Pre ({cm[0][2]}) or Pro ({cm[0][3]}) "
      f"in {sum(cm[0]) - cm[0][0]} cases, and {cm[3][0] + cm[3][1] + cm[3][2]} of {sum(cm[3])} Pro images were misclassified. Because the dataset labels the stages of a continuous maturation process, confusion between "
      "neighbouring stages is expected even for expert observers; I did not have access to a second reading of the images and cannot quantify human agreement.")
    S("Fine-tuning and encoder comparison")
    P("The preceding results use a frozen encoder. Adapting the encoder to the task can improve the fit to the cell appearance, but it also gives the model more capacity to learn the acquisition cues that "
      "the audit identified. Eight configurations were therefore defined (Methods, Hyperparameters of the planned fine-tuning experiments): low-rank adaptation of DinoBloom-S with colour (g01), in grayscale "
      "(g02) and with a feature branch (g03); full fine-tuning of two convolutional networks (g04, g05); low-rank adaptation of a generic DINOv2 model (g06) and of the larger DinoBloom-B (g07); and "
      "adaptation with the background left in the image (g08), which is a deliberate shortcut control. Table {T:ft} lists the results of the runs that have been completed.")
    if N_SLOTS:
        P(f"**[[GPU: {N_SLOTS} of {len(GPU_NAMES)} configurations have no result file yet. Write here: (1) the balanced accuracy of g01 compared with the frozen reference, with the paired-bootstrap difference "
          "and Holm-adjusted p value; (2) the effect of colour (g01 vs g02); (3) the effect of the feature branch (g03 vs g01); (4) the ordering of the encoders (g01, g04, g05, g06, g07); (5) the shortcut control "
          "(g08 vs g01). Delete this paragraph if the runs are not performed.]]**")
    PB_ALL = None
    P(f"Figure {{F:ft_curves}} shows the training curves of the adapted model and Fig. {{F:ft_bars}} compares the configurations with their intervals. The confusion matrix of the best configuration "
      "is shown in Fig. {F:ft_cm}.")
    S("Attention, saliency and embedding structure")
    P("Two further analyses address what the adapted models use. Attention weights over the cells of an image (Methods) indicate which cells drive the bag representation, and gradient-weighted class "
      f"activation maps {c('selvaraju2017')} indicate which regions of a cell drive the score. Because saliency maps can look plausible for untrained networks, a model-randomisation check "
      f"{c('adebayo2018')} is part of the protocol. Fig. {{F:ft_attn}} is reserved for the examples. A two-dimensional projection of the embeddings coloured by class and by capture segment "
      "(Fig. {F:ft_embed}) addresses the question of whether the embedding encodes the session as strongly as the class.")
    if N_SLOTS:
        P("**[[GPU: after running scripts/analysis_xai.py on a checkpoint, describe in two to three paragraphs what the attention and saliency maps show, including at least one failure case; "
          "report the model-randomisation result. Delete this paragraph and the two figures above if the analysis is not performed.]]**")


# misclassified images, read from the prediction files of the CPU run
ERR_ROWS = []
_ok, _bad, _att, _nc = [], [], [], []
for _k in range(5):
    for _r in _csv.DictReader(open(RUNDIR / f"fold{_k}" / "predictions.csv")):
        _p = [float(_r[f"p_{j}"]) for j in range(4)]
        (_bad if int(_r["label"]) != _p.index(max(_p)) else _ok).append(max(_p))
        _att.append(float(_r["top_attention"])); _nc.append(int(_r["n_cells"]))
        if int(_r["label"]) != _p.index(max(_p)):
            ERR_ROWS.append([_r["image_id"], str(_k + 1), CL[int(_r["label"])], CL[_p.index(max(_p))], f2(max(_p)), _r["n_cells"]])
ERR_ROWS.sort(key=lambda r: r[0])

import numpy as _np  # noqa: E402

MED_OK, MED_BAD = float(_np.median(_ok)), float(_np.median(_bad))
MED_ATT = float(_np.median(_att))
MED_UNI = float(_np.median(1.0 / _np.array(_nc)))

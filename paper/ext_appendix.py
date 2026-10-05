@section("appendix_extra")
def appendix_extra():
    PB()
    H("Appendix VI. Images misclassified by the headline model")
    P(f"Table A7 lists the {len(ERR_ROWS)} of {N_IMG} images that the headline model classified incorrectly in the out-of-fold evaluation, with the fold in which each was tested, the true and predicted class, the confidence "
      "of the prediction and the number of cells found in the image. The list is sorted by file name.", noindent=True)
    BODY.append({"t": "table", "num": "A7", "caption": "Misclassified images (pooled out-of-fold predictions).",
                 "cols": [{"w": 3, "align": "left"}, {"w": 0.8, "align": "right"}, {"w": 1.3, "align": "left"}, {"w": 1.3, "align": "left"}, {"w": 1.2, "align": "right"}, {"w": 1, "align": "right"}],
                 "header": ["Image", "Fold", "True class", "Predicted", "Confidence", "Cells"], "rows": ERR_ROWS, "foot": None})

    PB()
    H("Appendix VII. Training logs of the five folds")
    BODY.append({"t": "p", "noindent": True, "text": "Tables A8 to A12 list the training loss and the validation scores at each epoch of each fold of the headline model."})
    for k, L in enumerate(LOGS):
        BODY.append({"t": "table", "num": f"A{8 + k}", "caption": f"Epoch log of fold {k + 1} (train loss, validation macro-F1, validation balanced accuracy, seconds per epoch).",
                     "cols": [{"w": 1, "align": "right"}, {"w": 1.5, "align": "right"}, {"w": 1.5, "align": "right"}, {"w": 1.5, "align": "right"}, {"w": 1.2, "align": "right"}],
                     "header": ["Epoch", "Train loss", "Val macro-F1", "Val bal. acc.", "Seconds"],
                     "rows": [[str(r["epoch"] + 1), f"{r['train_loss']:.4f}", f3(r["val_macro_f1"]), f3(r["val_bal_acc"]), f"{r['train_seconds']:.1f}"] for r in L], "foot": None})

    PB()
    H("Appendix VIII. Configurations of the fine-tuning experiments")
    P("Table A13 lists the settings that differ between the eight prepared configurations; all other settings are identical and are given in the configuration listing of Appendix IV and in the "
      "files configs/ml/gpu/g01 to g08 of the repository.", noindent=True)
    cfgs = [json.loads(Path(ROOT / "configs" / "ml" / "gpu" / f"{nm}.json").read_text()) for nm, _, _ in GPU_NAMES]
    rows = []
    for (nm, short, desc), cfg in zip(GPU_NAMES, cfgs):
        m, d, t = cfg["model"], cfg["data"], cfg["train"]
        rows.append([short, m["encoder"].replace("timm:", ""), m["freeze"], d["source"], "yes" if d["isolate"] else "no", "yes" if m["use_features"] else "no",
                     str(t["batch_size"]), f"{t['lr_lora'] if m['freeze'] == 'lora' else t['lr_encoder']:g}"])
    BODY.append({"t": "table", "num": "A13", "caption": "Settings that differ between the fine-tuning configurations.",
                 "cols": [{"w": 0.8, "align": "left"}, {"w": 2.6, "align": "left"}, {"w": 1, "align": "left"}, {"w": 1.4, "align": "left"}, {"w": 1, "align": "left"}, {"w": 1, "align": "left"}, {"w": 0.9, "align": "right"}, {"w": 1.2, "align": "right"}],
                 "header": ["Run", "Encoder", "Adaptation", "Input", "Isolated", "Features", "Batch", "Encoder learning rate"], "rows": rows,
                 "foot": "Adaptation: low-rank (lora) or full. The learning rate shown is that of the adapter weights for low-rank adaptation and of the encoder for full fine-tuning; the head uses 0.001 in all runs."})
    BODY.append({"t": "table", "num": "A14", "caption": "Settings shared by all fine-tuning configurations.",
                 "cols": [{"w": 3, "align": "left"}, {"w": 3, "align": "left"}],
                 "header": ["Setting", "Value"],
                 "rows": [["Task", "four classes"], ["Folds", "5 contiguous, validation offset 2, embargo 37"], ["Cells per bag (train / evaluation)", "16 / 64"],
                          ["Crop (stored / network input)", "128 x 128 / 112 x 112 pixels"], ["Epochs (maximum) / patience", "12 / 4"], ["Head", "gated attention, 256 hidden units, dropout 0.2"],
                          ["Optimiser", "AdamW, weight decay 0.05, warm-up 10% then cosine"], ["Loss", "cross-entropy, label smoothing 0.1, inverse-square-root class weights"],
                          ["Gradient clipping", "norm 1"], ["Mixed precision", "bfloat16 where supported, otherwise float16"], ["Low-rank adaptation", "rank 8, scaling 16, attention qkv and projection layers"],
                          ["Seed", "0"]], "foot": None})

    PB()
    H("Appendix IX. Algorithms in pseudocode")
    P("The three procedures that determine the evaluation are given in pseudocode. They correspond to the functions contiguous_folds, purge_train, cluster_bootstrap and holm of the code.", noindent=True)
    BODY.append({"t": "small", "text": "Algorithm 1. Contiguous session-aware folds with embargo (variant used for the probe analyses; the headline model additionally holds out a validation stretch, see Methods)."})
    BODY.append({"t": "code", "lines": [
        "input: labels y, capture numbers n, number of folds K, embargo e",
        "for each class c:",
        "    sort the images of class c by n; cut the sorted list into K consecutive segments S_1..S_K",
        "    assign every image of segment S_k to fold k",
        "for k = 1..K:",
        "    test  = images of fold k;   train = images of all other folds",
        "    for each class c:  remove from train every image of class c with |n_i - n_j| <= e for some test image j of class c",
        "    yield (train, test)"]})
    BODY.append({"t": "small", "text": "Algorithm 2. Cluster bootstrap interval for a metric m."})
    BODY.append({"t": "code", "lines": [
        "input: groups g(i), metric m(indices), number of resamples B",
        "for b = 1..B:",
        "    draw G groups with replacement, G = number of distinct groups",
        "    idx = concatenation of the images of the drawn groups (a group drawn twice contributes twice)",
        "    v_b = m(idx)",
        "return the 2.5th and 97.5th percentiles of v_1..v_B, and the standard deviation of v"]})
    BODY.append({"t": "small", "text": "Algorithm 3. Paired comparison and Holm adjustment."})
    BODY.append({"t": "code", "lines": [
        "input: metrics m_A and m_B on the same groups, B resamples",
        "for b = 1..B:  draw groups as above; d_b = m_A(idx) - m_B(idx)",
        "p = 2 * min( share of d_b <= 0, share of d_b >= 0 ),  floored at 1/B",
        "Holm: sort the p values ascending p_(1) <= ... <= p_(M);  p'_(r) = max_(s <= r) min(1, (M - s + 1) * p_(s))"]})

    PB()
    H("Appendix X. Glossary of terms and abbreviations")
    gl = [["a*, b*, L*", "Axes of the CIE Lab colour space: green-red, blue-yellow and lightness."],
          ["APC", "Adaptive principal curvature: a response built from the eigenvalues of the image Hessian."],
          ["Attention (MIL)", "A learned weighting of the cells of an image that sums to one."],
          ["AUROC", "Area under the receiver operating characteristic curve."],
          ["Balanced accuracy", "Mean of the recalls of the classes."],
          ["Bag", "The set of cells of one image, treated as one training example."],
          ["Bootstrap (cluster)", "Resampling of whole groups of images with replacement."],
          ["ECE", "Expected calibration error."],
          ["Embargo", "Number of capture numbers around a test image from which training images are removed."],
          ["Foundation model", "A large model pre-trained on broad data and adapted to many tasks."],
          ["Frozen encoder", "An encoder whose weights are not changed during training."],
          ["GLCM", "Grey-level co-occurrence matrix, the basis of Haralick texture features."],
          ["Group", "A set of images assumed to share a session or patient (here consecutive blocks of 37 capture numbers and duplicates)."],
          ["HDS", "Hybrid diffusion-steered denoising (Perona-Malik plus total variation)."],
          ["LoG", "Laplacian of Gaussian."],
          ["LoRA", "Low-rank adaptation of the weights of a network."],
          ["MIL", "Multiple-instance learning."],
          ["Neutralisation", "Replacement of everything outside the cells by a constant colour."],
          ["PBS", "Peripheral blood smear."],
          ["Purge", "Removal of training images that lie close to the test images."],
          ["Shortcut", "A feature that predicts the label in the dataset without being causally related to it."],
          ["Temperature scaling", "Division of the logits by a scalar fitted on validation data to calibrate the probabilities."]]
    BODY.append({"t": "table", "num": "A15", "caption": "Glossary.", "cols": [{"w": 2, "align": "left"}, {"w": 7, "align": "left"}],
                 "header": ["Term", "Meaning"], "rows": gl, "foot": None})

    PB()
    H("Appendix XI. Source listings of the core algorithms")
    P("The listings below reproduce the source code of the modules that implement the operations described in Materials and Methods, as of the commit of the repository from which the results were produced. "
      "The complete code, with the remaining modules and the tests, is in the repository.", noindent=True)
    for rel in ["src/leukemia_pp/stain.py", "src/leukemia_pp/segmentation.py", "src/leukemia_pp/hds.py", "src/leukemia_pp/neutralise.py",
                "src/leukemia_ml/models/heads.py", "src/leukemia_ml/eval/folds.py", "src/leukemia_ml/eval/stats.py", "src/leukemia_ml/eval/calibration.py", "src/leukemia_ml/eval/metrics.py"]:
        BODY.append({"t": "small", "text": f"Listing: {rel}"})
        BODY.append({"t": "code", "lines": [ln.replace("\t", "    ").rstrip() or " " for ln in (ROOT / rel).read_text().splitlines()]})

    PB()
    H("Appendix XII. Test suite")
    P(f"The repository has {ntests} automated tests, all of which passed in the run used for this paper (python -m pytest, {int(json.loads((R / 'test_summary.json').read_text())['seconds'])} seconds on a CPU). They are "
      "listed by file; each name states the behaviour that is checked.", noindent=True)
    import subprocess
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True).stdout
    names = [ln.strip() for ln in out.splitlines() if "::" in ln]
    BODY.append({"t": "code", "lines": [nm.replace("tests/", "") for nm in names]})

    PB()
    H("Appendix XIII. Files released with the paper")
    inv = [["docs/results/cv_dinobloom_s_frozen_mil.json", "Headline cross-validation: pooled and per-fold metrics, intervals, temperatures (Table {T:cv})."],
           ["docs/results/cv_run/", "Per-fold predictions, training logs and metrics of the headline model (Appendices I, VI, VII; Figs. of Results)."],
           ["docs/results/ablation_sources_contiguous.json", "Representation ablation (Table {T:abl})."],
           ["docs/results/ablation_sources_RANDOM_BLOCK_FOLDS_superseded.json", "The same ablation with random blocks (Appendix II)."],
           ["docs/results/audit_full_dataset.json", "Shortcut audit (Table {T:audit})."],
           ["docs/results/leakage_analysis.json", "Scheme comparison, purge and gap analyses (Tables {T:scheme}, {T:purge}, {T:gap})."],
           ["docs/results/dataset_stats.json", "Dataset and segmentation statistics (Table {T:dataset})."],
           ["docs/results/session_structure.json", "Adjacent-versus-random similarity analysis."],
           ["docs/results/hds_benchmark.json", "Denoising benchmark."],
           ["docs/results/feature_stats.json", "Per-class feature statistics."],
           ["docs/results/curve_summary.json, cells_summary.json", "ROC and precision-recall areas; accuracy by cell count."],
           ["scripts/analysis_*.py, scripts/benchmark_hds.py, scripts/make_figures.py", "Scripts that produce the above files and the figures."],
           ["paper/build_paper.py", "Script that assembles this manuscript from the result files."]]
    BODY.append({"t": "table", "num": "A16", "caption": "Result files and the scripts that produce them.", "cols": [{"w": 4, "align": "left"}, {"w": 5, "align": "left"}],
                 "header": ["File", "Content"], "rows": inv, "foot": None})

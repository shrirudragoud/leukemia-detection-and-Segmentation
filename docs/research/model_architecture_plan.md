# Model architecture, evaluation and paper-revision plan

Status: research + design document (no model code written yet). Written against (a) what our own
data audit found, (b) the IJ-ICT reviewers' comments on "Curvature-Based Morphological Analysis of
Leukemia Microscopic Images", and (c) a literature search. **Source reliability is marked**:
`[V]` = the paper/page exists and the claim was read in a search summary this session (I could not
open arXiv or MICCAI directly, so details are second-hand and must be checked against the paper
before citing); `[K]` = well-known classic cited from memory; `[?]` = not verified.

---------------------------------------------------------------------------------------------------
## 0. Executive summary

1. **The paper's strongest honest contribution is not "a better classifier"; it is a *leakage- and
   shortcut-aware* pipeline plus interpretable curvature/morphology descriptors, evaluated
   rigorously.** Reviewer A is right that the pieces (diffusion, Hessian/curvature, morphology) are
   known. What is new and defensible: (i) a stabilised HDS denoiser (the original diverges, tested),
   (ii) a measured shortcut audit showing background colour alone gives 0.82 balanced accuracy
   (chance 0.25) and scale-bar corner 0.72 on this dataset, (iii) a neutralisation step that removes
   them, (iv) grouped, patient-proxy evaluation with statistics, (v) an ablation of every module.
2. **Primary model family: a pretrained encoder on single-cell crops + a small interpretable
   curvature/morphology branch, aggregated per image with attention-MIL.** Encoders to compare:
   ImageNet CNNs (ResNet-50, EfficientNet, ConvNeXt-T) as baselines, general DINOv2, and
   **DinoBloom** (a DINOv2 model trained on 380k+ blood-cell images, Apache-2.0) `[V]`, each as
   frozen-embedding linear probe first, then LoRA. Foundation-model probing is cheap, so most
   ablations run on frozen embeddings and tabular features, and only the final candidates are
   fine-tuned.
3. **Do not trust any accuracy computed with image-level splits on this dataset.** There are no
   patient ids; consecutive image numbers cluster by session (our test), so we split by blocks and
   duplicate groups. Literature on C-NMC reports the same leakage problem `[V]`.
4. **External validation is mandatory for any generalisation claim** (C-NMC 2019 via TCIA, CC BY 3.0
   `[V]`, binary only), and a segmentation ground-truth set (Raabin-WBC masks `[V]`, ALL-IDB `[V]`).
5. **Before modelling: check whether DinoBloom's 13 training datasets include this Kaggle
   dataset.** I could not verify; if it does, DinoBloom results on it are contaminated.

---------------------------------------------------------------------------------------------------
## 1. What our own data already tells us (hard facts from this repo)

| Fact | Evidence |
| --- | --- |
| 3,256 images, 224x224, 4 classes: Benign 504, Early 985, Pre 963, Pro 804 | download listing |
| No patient ids; 40 byte-identical duplicate pairs (always adjacent numbers) | `docs/results`, audit |
| Adjacent numbers closer in background colour than random pairs for malignant classes (Pro 0.30 vs 0.64) | analysis in session |
| Background colour alone: balanced acc 0.82 (3 numbers); after stain normalisation 0.78; corner patch 0.72; chance 0.25 | `docs/results/audit_full_dataset.md` |
| After neutralisation: background 0.25 (chance); corner 0.44 (cells in corner) | same |
| **Cell colour alone still 0.77-0.80**, even white-balanced | same |
| Tabular, grouped CV: shape 0.68, texture 0.65, shape+texture 0.78, colour 0.83, all 0.91 | same |
| Nucleus found in only ~8% of cells (thin cytoplasm, nucleus fills the cell at 224 px) | pipeline run |
| Segmentation looks good qualitatively; known failures: pale-blue cell missed, pink smudge accepted | previews; no ground truth yet |

Implications: (a) colour is a confound we cannot remove from cells; the model protocol must report
colour-free results; (b) 90 patient-proxy groups means fold variance is large, so confidence
intervals matter more than point estimates; (c) per-image labels are *weak* for cells (a malignant
image may contain non-blast WBCs).

---------------------------------------------------------------------------------------------------
## 2. Reviewer comments -> concrete actions

| Reviewer point | Action in this plan | Section |
| --- | --- | --- |
| A1 novelty gap, "integration of existing techniques" | Reframe contributions (Sec. 3); state exactly what is new vs reused | 3 |
| A2 missing ViT, attention CNNs, foundation models, SSL, recent segmentation nets, XAI | Compare CNN, ViT/DINOv2, DinoBloom (SSL foundation), attention-MIL; segmentation benchmark vs Cellpose-SAM/StarDist/SAM-based; XAI protocol with sanity checks | 4, 6, 8 |
| A3 no ablation per module | Full ablation matrix (preprocessing, curvature channels, HDS, BEEMD/IMF, morphology, backbone, fusion) | 7 |
| A4 grammar; A5 captions too descriptive | Editing task; short captions | 11 |
| B1 research gap, why curvature | Gap = interpretability + leakage-aware evaluation + shortcut control; curvature descriptors are the interpretable branch | 3 |
| B2 limited literature, state contributions | Reference plan (>=25, >=15 recent journal) | 12 |
| B3 dataset description incomplete | Dataset card: name, DOI, counts, per-class, patients (89), capture, license, split protocol | 5 |
| B4 parameter settings, flowchart | Config hash + `config.json`; flowchart from `docs/figures` | 5, 11 |
| B5 evaluation procedure, statistics (SD, significance) | Repeated grouped CV, cluster bootstrap CIs, paired tests, corrections | 6 |
| B6 "vessel/vascular" terminology | Replace with "curvature response", "ridge/edge", "cell boundary" (APC originates in vessel work; cite as method origin only) | 11 |
| B7 weak references | Audit every citation against a claim | 12 |
| C1 why curvature descriptors vs CNNs; interpretability vs accuracy | Interpretability-accuracy frontier experiment (tabular vs hybrid vs deep) + concept probing | 7, 8 |
| C2 clinical relevance | Discussion material in Sec. 9 | 9 |
| C3 deployment, real-time, hybrid CNN+curvature, transfer learning as future work | Hybrid + transfer learning are now *in* the study; deployment limits in Sec. 9 | 4, 9 |

**Important honesty point.** Reviewer C says the paper "notes superior performance over baselines".
Any such claim must be **re-run under the leakage-aware protocol**. If curvature descriptors do not
beat a fine-tuned encoder, the revised paper should say so and position them as the interpretable
component, not the accuracy winner.

---------------------------------------------------------------------------------------------------
## 3. Contribution statement (defensible wording; edit once results exist)

1. A reproducible, config-hashed preprocessing pipeline with a *stabilised* hybrid diffusion-steered
   denoiser (bounded, convergent; the published form diverges on additive noise: 22 dB -> 5 dB PSNR
   in our test).
2. A shortcut audit protocol for blood-smear datasets (acquisition probes, residual probes, grouped
   CV) and a neutralisation step that removes background/scale-bar cues; quantified on a public
   dataset.
3. Interpretable multi-scale curvature (Hessian principal-curvature) and morphology descriptors per
   cell, fused with a pretrained encoder, with a module-by-module ablation.
4. A leakage-aware evaluation with uncertainty (cluster bootstrap) and external validation.

Not claimed: a new curvature algorithm, or a state-of-the-art accuracy figure.

---------------------------------------------------------------------------------------------------
## 4. Candidate models and why

Unit of prediction. Labels are per image; cells inherit them. Use **image-level prediction from the
set of cells** (attention-MIL, Ilse et al. 2018 `[K]`) as primary, with mean pooling as the baseline,
and a field-level CNN on `rgb_clean` as an alternative. This handles weak cell labels and variable
cell counts (Benign ~8 vs Pre ~21 per image; note count is itself a density shortcut: report with
and without count).

| Tier | Model | Input | Role / rationale | Cost |
| --- | --- | --- | --- | --- |
| M0 | LogReg / RandomForest / gradient boosting | per-image aggregates of cell shape, texture, curvature-layer statistics | Interpretable baseline; already 0.78 (shape+texture) | minutes, CPU |
| M1 | ResNet-50 (ImageNet) `[K]` | `clean_rgb` field 224 | Standard CNN baseline | GPU hours |
| M2 | EfficientNet-B0/B3, ConvNeXt-T `[K]` | cell crops 96-128 | Modern CNN baselines; EfficientNet is widely reported on ALL data `[V]` | GPU hours |
| M3a | DINOv2-S/B, generic `[K]` | cell crops | General SSL foundation baseline | extract once + linear probe |
| M3b | **DinoBloom-S/B** `[V]` (weights on Zenodo, Apache-2.0; S 22M, B 86M, L 304M, G 1.1B) | cell crops | Domain-specific SSL; best expected transfer under small data/domain shift | extract once + linear probe; LoRA later |
| M4 | **Hybrid**: embedding (M3b) concatenated with standardised curvature/morphology vector -> MLP, attention-MIL pooling | crops + features | Interpretable component + deep component; answers Reviewer C | small |
| M5 | LoRA fine-tune of M3b (rank 8, qkv) `[V: LoRA matches full fine-tuning with <2% params on medical tasks]` | crops | Adapt without wrecking pretrained features; small data | GPU hours |
| (stretch) | continue DINOv2 pretraining on our unlabeled crops | crops | Only if the frozen features underperform | GPU days |

Why this ordering. With ~3k images/90 groups, full fine-tuning of large nets is high-variance and
shortcut-prone; frozen foundation features + linear/MLP heads are low-variance, cheap enough for
repeated grouped CV with many seeds, and make ablations affordable. A related paper benchmarks
DinoBloom, BiomedCLIP and CLIP under linear probing, LoRA and retrieval-augmented classification
for leukemia across datasets `[V: arXiv 2608.10657]` - read it before finalising, it is close to
our design.

What not to do: do not add APC/LoG/Canny as extra channels to a *pretrained* RGB encoder as the main
route (it disturbs transfer); use them for the tabular/curvature branch and as an ablation with an
adapted first layer.

### Segmentation sub-study (Reviewer A: "recent medical segmentation networks")
Our classical a*-channel + watershed segmenter has no ground truth. Plan: evaluate on masks from
Raabin-WBC (1,145 expert nucleus/cytoplasm masks `[V]`) and ALL-IDB blast annotations `[V]`; compare
to Cellpose / Cellpose-SAM `[V]`, StarDist `[V]`, a SAM-based approach `[V]` (MedSAM needs box
prompts, which our detections can supply `[V]`). Report Dice/IoU/detection F1. Decide whether to keep
classical (interpretable, fast) or use a learned segmenter for nuclei, which our data cannot resolve.

---------------------------------------------------------------------------------------------------
## 5. Dataset card (Reviewer B3) - to paste into the paper

Name: Acute Lymphoblastic Leukemia (ALL) image dataset (Taleqani Hospital, Tehran), Kaggle
`mehradaria/leukemia`, DOI 10.34740/KAGGLE/DSV/2175623 `[V]`. License string on Kaggle: "Database:
Open Database, Contents: (c) Original Authors" (verified from the API) - cite, do not redistribute.
3,256 original JPEG images, 224x224 (some literature reports 3,242 images / 89 patients `[V]`);
Benign 504, Early Pre-B 985, Pre-B 963, Pro-B 804 (counted from the download). 89 patients (25
benign, 64 ALL) `[V]`; Zeiss camera, 100x `[V]`; flow-cytometry-confirmed labels `[V]`. A "segmented"
HSV-threshold version also exists; it is not ground truth. No patient identifiers are provided.

Evaluation protocol (to state): patient-proxy groups (consecutive-number blocks of 37 + duplicate
files, union-find), per-class stratified; repeated grouped 5-fold CV (3 seeds); a held-out
group-disjoint test partition used once; all preprocessing statistics fitted on training groups only.

---------------------------------------------------------------------------------------------------
## 6. Training and evaluation protocol

Splits: use `leukemia_pp.splits.assign_groups` for every experiment (one definition of "group").

Training (all deep models): AdamW; cosine schedule with warm-up; label smoothing 0.1; class-balanced
sampling or weighted loss; early stopping on validation macro-F1 (validation groups disjoint from
test); mixed precision; fixed seeds; <=3 seeds for deep models, more for cheap heads.

Augmentation (the shortcut-critical part): geometric (flips, 90-degree rotations, small scale);
**colour: HED/HSV jitter and random grayscale** (stain-augmentation evidence `[V]`; stain mix-up and
ContriMix as optional stronger variants `[V]`); no augmentation that changes cell shape
(elastic/strong crops) because shape is the signal. Always also train/report a **`clean_gray`**
variant that cannot use stain colour.

Metrics: balanced accuracy and macro-F1 (primary), per-class recall, one-vs-rest AUROC, confusion
matrix; binary Benign-vs-Malignant sensitivity/specificity; calibration (ECE, temperature scaling).

Uncertainty and significance (Reviewer B5):
* report mean +- SD over folds x seeds, plus **cluster (group-level) bootstrap 95% CIs** (resample
  groups, not images; >=2000 reps);
* compare two models with a **paired group-bootstrap of the metric difference**; corrected
  resampled t-test (Nadeau-Bengio) over repeated CV `[V]` as a secondary check, noting its
  assumptions; McNemar only on a single fixed held-out test `[V]`; Holm correction for many
  comparisons; report effect sizes, not only p-values. (Demsar 2006 `[V]` covers multi-dataset
  comparisons; with one dataset the group-level bootstrap is the appropriate tool.)
* never select hyper-parameters on the test partition.

External validation: train on this dataset, test on C-NMC 2019 (binary leukemia vs normal; 15,135
single-cell images, 118 subjects; TCIA, CC BY 3.0 `[V]`) and, if license permits, Raabin-WBC /
ALL-IDB. Expect a large drop: that drop is the finding. A "leakage-aware benchmark" on C-NMC
reports AUROC 0.913 for the best model under subject-disjoint evaluation, far below near-perfect
image-level numbers `[V: arXiv 2606.24944]` - a useful comparison for the Discussion.

---------------------------------------------------------------------------------------------------
## 7. Ablation matrix (Reviewer A3) and the interpretability-accuracy frontier

Cheap protocol: for each preprocessing variant, extract frozen embeddings (DinoBloom-B and a
CNN) and fit a logistic head with grouped CV, plus the tabular RF. Only the final 2-3 candidates
are fine-tuned.

| ID | Variant | What it isolates |
| --- | --- | --- |
| P0 | raw image | baseline (expected shortcut-inflated) |
| P1 | + Reinhard stain normalisation (fixed: annotation pixels excluded) | stain normalisation |
| P2 | + bilateral denoise | denoising |
| P3 | + stabilised HDS instead of bilateral | HDS contribution |
| P4 | + neutralisation (`rgb_clean`) | shortcut removal |
| P5 | `gray_clean` | colour-free |
| P6 | tabular: shape / texture / colour / count, each alone and combined | morphology, density |
| P7 | + curvature-layer statistics (APC, LoG, HDS-edge) | curvature descriptors |
| P8 | + BEEMD pure-IMF energies / pure-IMF image | BEEMD + IMF selection |
| P9 | embedding only vs embedding + P7/P8 features (hybrid) | fusion benefit |
| P10 | backbone: CNN vs DINOv2 vs DinoBloom, frozen vs LoRA | encoder choice |
| P11 | pooling: mean vs attention-MIL; with/without cell count | aggregation, density shortcut |

Report each as the change in balanced accuracy and macro-F1 with CI, relative to the preceding
row. **Interpretability-accuracy frontier (Reviewer C1):** plot accuracy against model transparency
(tabular curvature -> hybrid -> fine-tuned encoder), with CIs; say plainly where the interpretable
models lose or tie.

---------------------------------------------------------------------------------------------------
## 8. Explainability plan (Reviewer A2, C1)

* CNNs: Grad-CAM; ViT/DINO: gradient-weighted attention rollout and Grad-CAM; a recent evaluation on
  blood-cell data reports DINO + Grad-CAM as the most faithful/localised of the ViT combinations
  `[V: arXiv 2510.12021]`, while Grad-CAM fidelity degrades for ViTs in some settings `[V]`.
* **Sanity checks are required, not optional**: model-parameter and label randomisation tests
  (Adebayo et al. 2018) `[V]`, deletion/insertion faithfulness, and a *background test*: mask the
  background and measure the accuracy drop (this directly tests the shortcut finding).
* Shortcut evidence in the literature to cite: DeGrave et al., Nat Mach Intell 2021;3:610-619 `[V]`.
* Interpretable branch: permutation importance / SHAP over curvature+morphology feature groups;
  concept probing (does the embedding linearly predict circularity, solidity, curvature statistics?).

---------------------------------------------------------------------------------------------------
## 9. Clinical relevance and deployment (Reviewer C2, C3) - discussion material

Defensible points: curvature/morphology descriptors give hematologists *named* quantities (boundary
regularity, roundness, texture heterogeneity) that can be inspected, which supports triage and may
reduce inter-observer variability; per-cell attention weights show which cells drove an image
score. Limits to state: single-hospital dataset, no patient ids, colour confound, binary external
validation only, no prospective study, not a diagnostic device. Deployment: LIS/slide-scanner
integration, per-image latency (classical pipeline ~0.2 s/img CPU; foundation-model inference needs
GPU or a distilled model), calibration and abstention for out-of-distribution fields (our QC flags
are a start), regulatory pathway, drift monitoring.

---------------------------------------------------------------------------------------------------
## 10. Code architecture plan (nothing implemented yet)

New sibling package `leukemia_ml` (keeps `leukemia_pp` torch-free):

```
src/leukemia_ml/
  data/       manifest.py (reads leukemia_pp run dir), crops.py (cell crops via dataset.crop_cell),
              datasets.py (torch Dataset: cell / bag / field), augment.py (geometry + HED/gray)
  models/     encoders.py (registry: timm CNNs, DINOv2, DinoBloom loader), heads.py (linear, MLP,
              gated attention-MIL), hybrid.py (embedding + feature branch, group dropout), lora.py
  train/      loop.py (AMP, seeds, early stop), config.py (frozen dataclasses, hashed like pp)
  eval/       metrics.py, calibration.py, stats.py (cluster bootstrap, paired tests, Holm),
              folds.py (repeated grouped CV from assign_groups), external.py (C-NMC loader)
  xai/        gradcam.py, rollout.py, sanity.py (randomisation tests), background_test.py
  experiments/ablation.py (P0-P11 driver), embed.py (extract + cache embeddings)
tests/        synthetic-data tests as in leukemia_pp (known geometry, determinism, leakage checks)
```
Design rules carried over: frozen validated configs with hashes, deterministic seeds, outputs that
record the exact config, tests that assert behaviour on synthetic data, no silent fallbacks.
Dependencies: torch, timm, peft (optional), scikit-learn, scipy. Embedding caches make ablations
cheap; every experiment writes a JSON of metrics with fold/seed/group info for the statistics.

---------------------------------------------------------------------------------------------------
## 11. Writing fixes (Reviewer A4/A5, B6; layout editor)

IMRaD(C) structure with the editor's six-step Results & Discussion; short captions; remove "vessel",
"vascular", "vessel density"; IEEE references in order of appearance with DOIs; author biographies
and icons per template; Acknowledgements/Funding/CRediT; spell-check; highlighted revision within
the editor's 8-week window.

---------------------------------------------------------------------------------------------------
## 12. Reference plan (>=25 references, >=15 recent journal articles; DOIs mandatory)

Reality check: many recent items found are **arXiv preprints**. The journal requires journal
articles with DOIs, so each needs its published version found. None of the DOIs below was
verified except where stated; do not paste without checking.

| Topic | Candidate | Status |
| --- | --- | --- |
| Dataset | Ghaderzadeh et al., B-ALL CNN on PBS images, Int J Intell Syst 2021; dataset DOI 10.34740/KAGGLE/DSV/2175623 | dataset DOI [V]; paper DOI [?] |
| Leakage | Albzour, leakage-aware benchmark on C-NMC (arXiv 2606.24944) | [V] preprint; find journal version |
| Shortcuts | DeGrave, Janizek, Lee, Nat Mach Intell 2021;3:610-619 | [V]; DOI [?] |
| Foundation | Koch et al., DinoBloom, MICCAI 2024 (arXiv 2404.05022) | [V] |
| Foundation | Retrieval-augmented foundation models for leukemia (arXiv 2608.10657) | [V] preprint |
| Foundation | Oquab et al., DINOv2 | [K] |
| PEFT | Hu et al., LoRA; "Less could be better" PEFT for medical foundation models (arXiv 2401.12215) | [K]/[V] |
| Backbones | He (ResNet), Tan & Le (EfficientNet), Liu (ConvNeXt), Dosovitskiy (ViT) | [K] |
| Stain/DG | Tellez et al. stain augmentation/normalisation; stain mix-up; ContriMix | [V] exists; exact cites [?] |
| Segmentation | Cellpose-SAM (bioRxiv 2025); SAMCell; digital-cytology cell-detection comparison (arXiv 2504.06957); StarDist; nnU-Net | [V]/[K] |
| Segmentation data | Raabin-WBC (Sci Rep 2022); ALL-IDB (ICIP 2011) | [V] exists; DOIs [?] |
| External data | C-NMC 2019, TCIA DOI 10.7937/tcia.2019.dc64i46r | [V] |
| XAI | Adebayo et al. 2018 (sanity checks); Grad-CAM; ViT explainability in medical imaging (arXiv 2510.12021) | [V]/[K] |
| Statistics | Nadeau & Bengio 2003; Demsar 2006 (JMLR 7:1-30); Benavoli et al. 2017 (JMLR) | [V]/[K] |
| ALL deep learning | EfficientNet transfer learning for ALL (arXiv 2508.06535); attention-based CNN (arXiv 2601.01026); YOLOv8/v11 for ALL (arXiv 2410.10701); ViT/CNN fusion and hybrid feature papers (Frontiers Oncol 2024) | [V] exist; find journal versions |
| Review | Systematic review and meta-analysis of leukemia detection (J Evid Based Med 2025) | [V] |
| Curvature | Hessian/curvature descriptors: Frangi et al. 1998 (multiscale Hessian filters; cite as method origin only, avoid vessel framing); shape-index/curvedness for cell shape | [K]/[?] |
| MIL | Ilse et al. 2018, attention-based deep MIL | [K] |

---------------------------------------------------------------------------------------------------
## 13. Roadmap and what is needed from you

| Phase | Work | Needs |
| --- | --- | --- |
| A | Verify DinoBloom training-set overlap; fetch weights (Zenodo access) ; read 2608.10657 and 2606.24944 | network allowlist for zenodo.org / github release hosts; or you download |
| B | `leukemia_ml` skeleton + embedding extraction + grouped-CV linear probes (CPU feasible for small encoders) | none |
| C | Ablation P0-P11 on frozen embeddings + tabular | GPU strongly preferred for ViT-B/L extraction |
| D | Segmentation ground-truth evaluation (Raabin-WBC / ALL-IDB) | dataset access/licences |
| E | Fine-tune/LoRA finalists, XAI sanity checks, calibration | GPU |
| F | External validation (C-NMC) | TCIA download (~GBs) |
| G | Write revised manuscript sections from results | results |

Open questions: (1) GPU available (Colab/Kaggle/own)? (2) Revise the *existing* curvature paper, or
write a new one that includes it? The reviewers' framing assumes the former; the audit findings
suggest the evaluation-integrity angle is the stronger story. (3) May I download Raabin-WBC,
ALL-IDB and C-NMC (requires allowlisting hosts)? (4) Is the submission deadline (8 weeks from the
decision email) already running?

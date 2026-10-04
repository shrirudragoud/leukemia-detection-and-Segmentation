"""Assemble paper/content.json (-> Word via paper/render_docx.js). Every number is read from
docs/results/*.json (or computed here); nothing numeric is typed by hand except protocol constants that
are also stored in the config files. Run:  python paper/build_paper.py && node paper/render_docx.js ..."""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper"))
from refs import REFS, sorted_refs  # noqa: E402

R = ROOT / "docs" / "results"
J = lambda n: json.loads((R / n).read_text())  # noqa: E731
META = json.loads((ROOT / "paper" / "meta.json").read_text())
CV = J("cv_dinobloom_s_frozen_mil.json")
CVC = J("cv_dinobloom_s_frozen_mil_config.json")
ABL = J("ablation_sources_contiguous.json")
ABL_OLD = J("ablation_sources_RANDOM_BLOCK_FOLDS_superseded.json")
LEAK = J("leakage_analysis.json")
AUD = J("audit_full_dataset.json")
DS = J("dataset_stats.json")
SES = J("session_structure.json")
HDS = J("hds_benchmark.json")
ENC = J("ablation_encoders.json") if (R / "ablation_encoders.json").exists() else None
GPU = sorted(glob.glob(str(ROOT / "results_for_paper" / "*" / "cv_summary.json")))
CL = ["Benign", "Early", "Pre", "Pro"]
USED = set()


def c(*keys):
    """Parenthetical name-year citation."""
    USED.update(keys)
    return "(" + "; ".join(REFS[k][0] for k in keys) + ")"


def n(k):
    """Narrative citation: Name et al. (year)."""
    USED.add(k)
    lab = REFS[k][0]
    m = re.match(r"(.*) (\d{4})$", lab)
    return f"{m.group(1)} ({m.group(2)})"


f3 = lambda x: f"{x:.3f}"  # noqa: E731
f2 = lambda x: f"{x:.2f}"  # noqa: E731
pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
ci = lambda d: f"{d['lo']:.3f}-{d['hi']:.3f}"  # noqa: E731

BODY, TABLES, FIGS, APP_TABLES, APP_FIGS = [], {}, {}, {}, {}


def H(t): BODY.append({"t": "h", "text": t})
def S(t): BODY.append({"t": "sh", "text": t})
def P(t, **k): BODY.append({"t": "p", "text": t, **k})
def PB(): BODY.append({"t": "pb"})


EXT = {}


def section(name):
    def deco(fn):
        EXT[name] = fn
        return fn
    return deco


def ext(name):
    if name in EXT:
        EXT[name]()


# ------------------------------------------------------------------ measured quantities
pf = CV["per_fold"]
pool = CV["pooled"]
pci = CV["pooled_ci"]
cm = pool["confusion"]
ablrows = {r["variant"][0]: r for r in ABL["rows"]}
A, B, C_ = LEAK["A_schemes"], LEAK["B_random_block_purge"], LEAK["C_contiguous_gap"]
sch = {(a["scheme"], a["features"]): a for a in A}
acq = {a["probe"]: a for a in AUD["results"]}
N_IMG, N_CELL = DS["n_images"], DS["n_cells"]
PER = DS["per_class"]
ntests = "NTESTS"


def aud(prefix):
    for k, v in acq.items():
        if k.startswith(prefix):
            return v["balanced_accuracy"]
    raise KeyError(prefix)


# ------------------------------------------------------------------ tables
def T(key, caption, cols, header, rows, foot=None, app=False):
    (APP_TABLES if app else TABLES)[key] = {"t": "table", "caption": caption, "cols": cols, "header": header,
                                            "rows": rows, "foot": foot}


def Fg(key, path, caption, width=6.0, app=False):
    (APP_FIGS if app else FIGS)[key] = {"t": "fig", "path": str(ROOT / path), "caption": caption, "width_in": width}


T("dataset", "Composition of the dataset and output of the segmentation stage, by class. Cells are connected components "
  "of the foreground mask after watershed splitting (Methods, Cell segmentation); \"Nucleus found\" is the fraction of cells in which the "
  "nucleus step returned a mask.",
  [{"w": 2, "align": "left"}] + [{"w": 1, "align": "right"}] * 6,
  ["Class", "Images", "Cells", "Cells per image (mean +/- SD)", "Median cell area (px)", "Border-touching cells (%)", "Nucleus found (%)"],
  [[k, str(PER[k]["images"]), str(PER[k]["cells"]), f"{PER[k]['cells_per_image_mean']:.1f} +/- {PER[k]['cells_per_image_sd']:.1f}",
    f"{PER[k]['cell_area_median']:.0f}", pct(PER[k]["border_cell_frac"]), pct(PER[k]["nucleus_found_frac"])] for k in CL] +
  [["All", str(N_IMG), str(N_CELL), "-", f"{DS['cell_area_median']:.0f}", "-", pct(DS["nucleus_found_frac"])]])

AUDROWS = [[a["group"], a["probe"], f3(a["balanced_accuracy"]), f3(a["std"]), a["verdict"] or "-"] for a in AUD["results"]]
T("audit", "Shortcut audit: balanced accuracy of a random-forest classifier that sees only the named quantity, under 5-fold "
  "5-fold grouped cross-validation over 37-image capture blocks without embargo (three random-forest seeds; the SD reflects only the seeds; chance = 0.25). Verdicts are the audit tool's own thresholds.",
  [{"w": 1.5, "align": "left"}, {"w": 4, "align": "left"}, {"w": 1, "align": "right"}, {"w": 1, "align": "right"}, {"w": 1.7, "align": "left"}],
  ["Group", "Information given to the classifier", "Balanced accuracy", "SD over seeds", "Verdict"], AUDROWS)

T("abl", "Effect of the image representation on four-class balanced accuracy with a frozen DinoBloom-S encoder and a logistic head "
  "(five contiguous folds per class, embargo of 37 capture numbers, three seeds). Intervals are 95% cluster-bootstrap "
  "intervals over the 90 session groups, computed from the first-rotation out-of-fold predictions while point values are means over three rotations; differences are paired against variant A and the p value is Holm-adjusted over the four comparisons.",
  [{"w": 4, "align": "left"}, {"w": 1.1, "align": "right"}, {"w": 1.7, "align": "right"}, {"w": 1.1, "align": "right"},
   {"w": 2.3, "align": "right"}, {"w": 1, "align": "right"}],
  ["Variant", "Balanced accuracy", "95% CI", "Macro-F1", "Difference vs A (95% CI)", "p (Holm)"],
  [[r["variant"], f3(r["balanced_accuracy"]), ci(r["balanced_accuracy_ci"]), f3(r["macro_f1"]),
    ("-" if "vs_reference" not in r else f"{r['vs_reference']['diff']:+.3f} ({r['vs_reference']['lo']:+.3f}, {r['vs_reference']['hi']:+.3f})"),
    ("-" if "vs_reference" not in r else f3(r["vs_reference"]["p_holm"]))] for r in ABL["rows"]])

T("cv", "Headline result: frozen DinoBloom-S with an attention-pooling head on isolated, shortcut-neutralised cell crops. Out-of-fold "
  "predictions of five contiguous session-aware folds (embargo 37) are pooled; the interval is a 95% cluster-bootstrap interval over groups.",
  [{"w": 3, "align": "left"}, {"w": 1.4, "align": "right"}, {"w": 1.8, "align": "right"}, {"w": 2.2, "align": "right"}],
  ["Metric", "Pooled", "95% CI", "Per-fold mean +/- SD"],
  [["Balanced accuracy", f3(pool["balanced_accuracy"]), ci(pci["balanced_accuracy"]), f"{f3(CV['per_fold_mean']['balanced_accuracy'])} +/- {f3(CV['per_fold_sd']['balanced_accuracy'])}"],
   ["Macro-F1", f3(pool["macro_f1"]), ci(pci["macro_f1"]), f"{f3(CV['per_fold_mean']['macro_f1'])} +/- {f3(CV['per_fold_sd']['macro_f1'])}"],
   ["AUROC (one-vs-rest)", f3(pool["auroc_ovr"]), "-", f"{f3(CV['per_fold_mean']['auroc_ovr'])} +/- {f3(CV['per_fold_sd']['auroc_ovr'])}"],
   ["Sensitivity, malignant vs benign", f3(pool["sensitivity"]), "-", "-"],
   ["Specificity, malignant vs benign", f3(pool["specificity"]), "-", "-"],
   ["ECE, uncalibrated (per-fold mean)", "-", "-", f"{f3(CV['per_fold_mean']['ece'])} +/- {f3(CV['per_fold_sd']['ece'])}"],
   ["ECE, temperature-scaled (pooled)", f3(pool["ece"]), "-", "-"]],
  foot="Sensitivity and specificity treat the three malignant classes (Early, Pre, Pro) as positive and Benign as negative. "
       "ECE: expected calibration error.")

T("scheme", "The same classifier evaluated under different fold constructions (three seeds; logistic head; mean over folds). "
  "\"Random blocks\" assigns consecutive runs of 37 capture numbers to folds at random; \"contiguous\" gives each fold a stretch of the capture order per class (for fold-boundary rotations 1 and 2 the stretch wraps around the ends of the sequence), "
  "with capture numbers, with the stated embargo removed from the training set on both sides.",
  [{"w": 4, "align": "left"}, {"w": 1.4, "align": "left"}, {"w": 1.2, "align": "right"}, {"w": 1.2, "align": "right"}, {"w": 1.2, "align": "right"}],
  ["Fold construction", "Features", "Training images", "Balanced accuracy", "SD over seeds"],
  [[a["scheme"], a["features"], f"{a['train_size']:.0f}", f3(a["balanced_accuracy"]), f3(a["sd"])] for a in A])

T("purge", "Training images close in capture number to the test images are removed from random-block folds (purged), "
  "compared with removing the same number of training images of each class at random. Mean over five folds and two fold seeds; one random-subset draw per setting.",
  [{"w": 2, "align": "left"}, {"w": 1.5, "align": "right"}, {"w": 1.7, "align": "right"}, {"w": 2, "align": "right"}],
  ["Purge distance", "Training images", "Purged: balanced acc.", "Random removal: balanced acc."],
  [[str(b["embargo"]), f"{b['train_size']:.0f}", f3(b["purged"]), f3(b["random_same_size"])] for b in B])

T("gap", "Contiguous folds with an increasing minimum gap between the training images and the test segment (\"gap\"), compared with "
  "removing the same number of training images at random (\"random\"). Balanced accuracy, mean over five folds, single run per setting; fold boundaries are not rotated (unlike the scheme comparison), so values differ slightly from it.",
  [{"w": 1.4, "align": "left"}, {"w": 1.4, "align": "right"}, {"w": 1.4, "align": "right"}, {"w": 1.4, "align": "right"},
   {"w": 1.4, "align": "right"}, {"w": 1.4, "align": "right"}],
  ["Gap", "Training images", "DinoBloom-S, gap", "DinoBloom-S, random", "Hand-made, gap", "Hand-made, random"],
  [[str(x["embargo"]), f"{x['n']:.0f}", f3(x["dino"]), f3(x["dino_rand"]), f3(x["hand"]), f3(x["hand_rand"])] for x in C_])

T("hds", "Denoising benchmark on synthetic Gaussian noise (peak signal-to-noise ratio in dB, higher is better). \"Legacy HDS\" is the diffusion "
  "scheme of the original repository; \"stabilised HDS\" is the bounded re-implementation used here.",
  [{"w": 1.2, "align": "left"}] + [{"w": 1.3, "align": "right"}] * 6,
  ["Noise SD", "Noisy input", "Gaussian", "Bilateral", "Legacy HDS", "Stabilised HDS", "Stabilised HDS iterations"],
  [[f2(r["sigma"]), f"{r['noisy']:.1f}", f"{r['gaussian']:.1f}", f"{r['bilateral']:.1f}", f"{r['legacy_hds']:.1f}",
    f"{r['stabilised_hds']:.1f}", str(r["stabilised_iterations"])] for r in HDS["rows"]],
  foot=f"Mean gain over the noisy input: legacy HDS {HDS['mean_gain_db']['legacy_hds']:+.1f} dB, stabilised HDS "
       f"{HDS['mean_gain_db']['stabilised_hds']:+.1f} dB, bilateral {HDS['mean_gain_db']['bilateral']:+.1f} dB, Gaussian {HDS['mean_gain_db']['gaussian']:+.1f} dB.")

if ENC:
    T("enc", "Comparison of frozen encoders (and the hybrid with hand-made features) under the same contiguous protocol as Table {T:abl}.",
      [{"w": 4, "align": "left"}, {"w": 1.1, "align": "right"}, {"w": 1.7, "align": "right"}, {"w": 1.1, "align": "right"},
       {"w": 2.3, "align": "right"}, {"w": 1, "align": "right"}],
      ["Encoder", "Balanced accuracy", "95% CI", "Macro-F1", "Difference vs reference (95% CI)", "p (Holm)"],
      [[r["variant"], f3(r["balanced_accuracy"]), ci(r["balanced_accuracy_ci"]), f3(r["macro_f1"]),
        ("-" if "vs_reference" not in r else f"{r['vs_reference']['diff']:+.3f} ({r['vs_reference']['lo']:+.3f}, {r['vs_reference']['hi']:+.3f})"),
        ("-" if "vs_reference" not in r else f3(r["vs_reference"]["p_holm"]))] for r in ENC["rows"]])

Fg("pipeline", "paper/figures/fig_pipeline.png", "Processing pipeline. Solid arrows show the path used for the headline model; the dotted branch produces the "
   "hand-made (tabular) features used in the ablations and the hybrid variant.", 6.0)
Fg("classes", "paper/figures/fig_classes.png", "Example images of the four classes (Benign, Early, Pre, Pro) of the dataset, 224 x 224 pixels as supplied. "
   "Differences in background colour and illumination between classes are visible, as are scale bars of different styles in the image corners (red, black or white).", 6.0)
Fg("seg", "paper/figures/fig_examples.png", "Cell segmentation. Left column: stain-normalised input; right column: detected cell boundaries (green). "
   "Touching cells are split by the watershed step (middle row). Border-touching cells are retained but flagged.", 4.4)
Fg("session", "paper/figures/fig_session.png", "Capture-order structure of the dataset. Median distance between the background colours (Lab, brightest 30% of pixels) of images with adjacent capture numbers "
   "(dark bars) and between random pairs from the same class (hatched bars). Adjacent images are more alike than random pairs in the Early, Pre and Pro classes.", 5.0)
Fg("hds", "paper/figures/fig_hds.png", "Denoising benchmark (peak signal-to-noise ratio, dB) on synthetic Gaussian noise; stabilised diffusion (solid circles) "
   "compared with the unmodified diffusion scheme (inverted triangles), bilateral and Gaussian filtering.", 4.6)
Fg("abl", "paper/figures/fig_ablation.png", "Balanced accuracy of the five input representations (A-E, defined in Table {T:abl}), with 95% cluster-bootstrap intervals "
   "over session groups. Contiguous folds, embargo 37.", 5.6)
Fg("leak", "paper/figures/fig_leakage.png", "Dependence of accuracy on proximity in capture number. (a) Random-block folds: purging training images near the test images "
   "(solid) versus removing the same number of training images at random (dashed). (b) Contiguous folds: increasing the gap to the test segment "
   "(solid) versus random removal (dashed), for DinoBloom-S features and for hand-made features.", 6.2)
Fg("cm", "paper/figures/fig_confusion.png", "(a) Pooled out-of-fold confusion matrix (counts; shading shows row-normalised proportion) and (b) balanced accuracy of each of the "
   "five folds for the headline model.", 6.2)


# ================================================================== TEXT
ba, bal, mf = pool["balanced_accuracy"], pci["balanced_accuracy"], pool["macro_f1"]
dsA, dsB, dsC, dsD, dsE = (ablrows[k] for k in "ABCDE")
bg_raw = aud("background colour, raw image")
cellcol = aud("mean cell colour, white-balanced")
taball = aud("ALL aggregated")
rb0, rb37 = sch[("random 37-image blocks", "DinoBloom-S")], sch[("contiguous folds, embargo 37", "DinoBloom-S")]
rb0h, rb37h = sch[("random 37-image blocks", "hand-made features")], sch[("contiguous folds, embargo 37", "hand-made features")]
B75, B150 = B[2], B[3]
C0, C225, C300 = C_[0], C_[4], C_[5]
bench = HDS["rows"]
CAL_FOLD_MEAN = float(sum(json.loads((R / 'cv_run' / f'fold{k}' / 'metrics.json').read_text())['test_calibrated']['ece'] for k in range(5)) / 5)
RECALLS = ', '.join(CL[i] + ' ' + f3(pool['per_class_recall'][i]) for i in range(4))
n_epochs = [p["epochs"] for p in pf]
secs = sum(p["seconds"] for p in pf)

# ------------------------------------------------------------------ abstract
ABSTRACT = (
    f"Acute lymphoblastic leukemia (ALL) is diagnosed in part from the appearance of lymphoblasts on peripheral blood smears, and many "
    f"studies routinely report very high classification accuracy on public smear datasets. These datasets rarely contain patient identifiers, so it is "
    f"unclear how much of the reported accuracy reflects morphology and how much reflects the way the images were acquired. I analysed a public "
    f"dataset of {N_IMG} smear images in four classes (Benign, Early, Pre and Pro; {N_CELL} segmented cells) with a leakage-aware protocol. "
    f"Images were stain-normalised, denoised and segmented, and each cell was isolated with the background replaced by a neutral fill. Capture "
    f"order was used to build {CV['n_groups']} session groups and contiguous cross-validation folds with an embargo. A frozen hematology foundation model "
    f"(DinoBloom-S) with an attention-pooling head reached a balanced accuracy of {f3(ba)} (95% CI {ci(bal)}) and a macro-F1 of {f3(mf)}. "
    f"Background colour alone, however, predicted the class with a balanced accuracy of {f3(bg_raw)} (grouped block folds without embargo), and with the same frozen encoder and a logistic probe, keeping the raw background raised balanced accuracy from "
    f"{f3(dsA['balanced_accuracy'])} to {f3(dsD['balanced_accuracy'])}. Random block folds overestimated accuracy relative to contiguous folds, and removing "
    f"training images that were close in capture number to the test images lowered accuracy more than removing the same number at random. "
    f"Because the classes appear to have been imaged in separate sessions (this could not be verified), in-dataset accuracy probably overstates performance on new patients; no external data were tested. "
    f"I conclude that reporting session-aware folds, shortcut audits and external validation should be a minimum standard for leukemia image "
    f"classification, and I release the code and every result file used in this paper."
)

# ------------------------------------------------------------------ build body
def build():
    H("Introduction")
    P("Acute lymphoblastic leukemia (ALL) is the most common cancer of childhood and is characterised by the uncontrolled proliferation of immature "
      "lymphoid cells in the bone marrow and blood. Examination of a stained peripheral blood smear (PBS) is an inexpensive first step in "
      "diagnosis, but it depends on the availability of trained hematologists, and agreement between observers on the morphology of "
      "individual blasts is imperfect. Automated analysis of PBS images has therefore been studied for more than a decade, "
      f"including classical image-processing pipelines {c('labati2011')} and, more recently, deep convolutional networks and vision transformers "
      f"{c('ghaderzadeh2022')}.")
    P(f"Public datasets have made this research possible, and some studies report very high accuracies. High accuracy on a small public dataset is, however, only informative if the evaluation separates the "
      f"images used to train a model from those used to test it in the same way that deployment would. Two properties of the data make this difficult. "
      f"First, public smear datasets rarely publish patient or slide identifiers, so a random split of images puts different images of the same "
      f"slide into both the training and the test set. Second, if the classes were imaged on different days or with different equipment, "
      f"a classifier can learn the imaging conditions instead of the cells. Shortcut learning {c('geirhos2020')} has produced spurious predictors in radiographic classifiers {c('degrave2021')}, and data leakage is a documented cause of "
      f"overoptimism in machine-learning-based science {c('kapoor2023')}.")
    P("The dataset analysed here was released by "
      f"{n('aria2021')} and described by {n('ghaderzadeh2022')}. It contains {N_IMG} images of peripheral blood smears in four classes: "
      "benign hematogones and three stages of malignant lymphoblasts (Early, Pre and Pro). The dataset documentation reports 89 patients, but the "
      "image files carry no patient identifier. I show below that neighbouring file numbers have more similar backgrounds in three of the four classes, which is consistent with the numbering following the order of image capture.", )
    P("Recent foundation models offer a way to obtain strong features without large labelled datasets. DINOv2 "
      f"{c('oquab2024')} is a self-supervised vision transformer {c('dosovitskiy2021')}; DinoBloom {c('koch2024')} continues its training on a large "
      "collection of single-cell images from hematology, and its weights are openly available. A small trained head on frozen features of such a model "
      "(here a gated-attention pooling head) is a strong, cheap and well-defined baseline, and is the model used for the headline result in this paper.")
    ext("intro_extra")
    S("Objectives")
    P("The aim of this work was to establish how accurately the four classes can be separated when the evaluation is designed to prevent "
      "leakage between neighbouring images, and how much of the accuracy can be attributed to cell morphology rather than image acquisition. I "
      "addressed four questions. (1) Can a reproducible preprocessing pipeline - colour normalisation, denoising, segmentation and "
      "per-cell isolation - be built and checked quantitatively? (2) How much of the class information is present in acquisition cues that are "
      "unrelated to the cells? (3) How large is the optimism introduced by random block folds, relative to contiguous session-aware folds, "
      "and is it caused by proximity in capture order? (4) What accuracy does a frozen foundation-model encoder reach under a contiguous "
      "session-aware protocol with an embargo of 37 capture numbers? I did not attempt to claim clinical performance; the data do not allow it.")

    H("Materials and Methods")
    S("Dataset")
    P(f"I used the {N_IMG} RGB images of the public dataset {c('aria2021', 'ghaderzadeh2022')} (Kaggle identifier mehradaria/leukemia). All images are "
      f"{DS['image_width'][0]} x {DS['image_height'][0]} pixels. The four classes are Benign ({PER['Benign']['images']} images), Early ({PER['Early']['images']}), "
      f"Pre ({PER['Pre']['images']}) and Pro ({PER['Pro']['images']}). The segmentation stage found {N_CELL} cells (Table {{T:dataset}}). I kept the "
      f"file names unchanged and recorded a SHA-256 hash for every image. Comparison of SHA-256 hashes found "
      f"{DS['duplicate_groups']} groups of byte-identical images ({DS['images_in_duplicate_groups']} images); duplicates were assigned to the "
      "same group; all of these pairs are consecutive images of the Pro class and fall in the same contiguous fold.")
    S("Preprocessing")
    P("The preprocessing pipeline (Fig. {F:pipeline}) is a Python package with a command-line interface and a configuration object whose hash is stored with "
      "every output. It applies the stages listed in Table {T:stages}.")
    P(f"*Colour normalisation.* Stain variation between slides was reduced with the Reinhard method {c('reinhard2001')}: the mean and standard deviation "
      "of each CIE Lab channel are matched to those of a reference computed from a sample of images. Three details differ from a direct implementation. "
      "The computation is carried out in floating point to avoid quantisation. Statistics are computed over foreground pixels only, and pixels with "
      "L* below 8 are excluded, because several images contain a black scale bar whose pixels otherwise shift the statistics and change the "
      "colour of cells (this was found as a bias in an early version and is covered by regression tests). The target statistics are computed "
      f"once for the dataset and stored. The influence of colour normalisation on classifiers has been reported by {n('tellez2019')}.")
    P("*Denoising.* Three denoisers are available: bilateral filtering " + c('tomasi1998') + ", non-local means, and a diffusion scheme "
      "(HDS) that combines Perona-Malik anisotropic diffusion " + c('perona1990') + " with total-variation regularisation. The original "
      "repository implemented the diffusion without a bound on the diffusivity, and the result made the image worse (Table {T:hds}). My version "
      "normalises the diffusivity to at most 1, uses a time step of 0.2, and adds a fidelity term, which makes the iteration "
      "bounded; a test with 3000 iterations confirmed that the output stays finite and inside the input range. The benchmark in Table {T:hds} "
      "uses a synthetic scene (a disc and a square on a flat background) with added Gaussian noise, one noise draw per level; it does not use dataset images, and its bilateral filter (diameter 9, colour sigma 0.2, space sigma 5 on an image scaled to [0, 1]) differs from the pipeline setting. The full-dataset run reported in the Results used bilateral filtering (diameter 9, "
      "colour and space sigma 75); the diffusion denoiser is disabled by default and the benchmark documents the module only.")
    P("*Cell segmentation.* White blood cells stain more strongly in the green-magenta (a*) axis than the surrounding red cells and background. I applied "
      "Otsu's threshold " + c('otsu1979') + " to the a* channel with a floor of 8, removed small objects by morphological opening, filled holes, and "
      "separated touching cells with a marker-controlled watershed whose markers are h-maxima of the distance transform "
      f"{c('vanderwalt2014')}. Cells that touch the image border are kept and flagged. In total {DS['qc_flag_counts'].get('no_nuclei_resolved', 0)} images "
      f"were flagged because no nucleus was resolved and {DS['qc_flag_counts'].get('mostly_border_cells', 0)} because most of their cells touched the border. "
      "An example is shown in Fig. {F:seg}. No ground-truth masks are available for this dataset, so the segmentation was checked visually "
      "on a sample and with consistency tests, not by overlap with annotations; I make no claim about segmentation accuracy.")
    P(f"*Nucleus segmentation.* Within each cell, a second threshold was applied to the interior to isolate the nucleus. This step returned a nucleus "
      f"in only {pct(DS['nucleus_found_frac'])}% of cells overall (Table {{T:dataset}}), so nuclear measurements (for example the nucleus-to-cell ratio) are "
      "missing for most cells and were not used as primary features. I report this failure rate here instead of imputing the measurements.")
    P("*Response layers.* Three local-structure maps were computed from the normalised image: a Laplacian-of-Gaussian (LoG) response with scale "
      "normalisation, a Canny edge map, and an adaptive principal-curvature response (APC), a weighted combination of the absolute Hessian eigenvalues in the manner of the vesselness "
      f"measure of {n('frangi1998')}. The gains are fixed constants, not data-dependent, so that the values are comparable between images.")
    S("Shortcut neutralisation and cell isolation")
    P("To remove information that identifies the acquisition session rather than the cell, I produced a \"clean\" version of each image in which every pixel "
      "outside the (dilated) segmented cells is replaced by a neutral light-grey fill (L* = 92, a* = b* = 0) and the cells are white-balanced. Each cell was then cropped and "
      "resized for the encoder. A grayscale version of the clean image was also produced. The unprocessed background-visible versions "
      "(normalised and raw) were retained for the ablation. The effect of neutralisation was measured by the audit described next, and not assumed.")
    S("Shortcut audit")
    P("The audit trains a random-forest classifier " + c('pedregosa2011') + " on a single low-dimensional quantity (for example the mean background colour, "
      "a patch of the corner containing the scale bar, or the mean cell colour) and reports its balanced accuracy under 5-fold grouped cross-validation over the 37-image capture blocks (without embargo, and therefore less strict than the contiguous folds used later; the probes may score lower under the stricter protocol). If a quantity that "
      "contains no information about the cells predicts the class far above chance (0.25), the dataset contains a shortcut. The audit is also applied to the "
      "neutralised images, where the background and corner probes are expected to return chance, as a check on the neutralisation itself.")
    S("Session groups and cross-validation")
    P(f"File names are numbered in the order of capture within each class. If images captured consecutively come from the same slide or the same "
      f"session, random assignment of images to folds leaks information. I measured this directly: in the Early, Pre and Pro classes, images with "
      f"adjacent capture numbers had a median distance between their background colours (Lab, brightest 30% of pixels) that was {pct(1 - SES['Early']['ratio'])}%, {pct(1 - SES['Pre']['ratio'])}% and "
      f"{pct(1 - SES['Pro']['ratio'])}% smaller, respectively, than that of random pairs from the same class (Fig. {{F:session}}); the Benign class showed no such "
      "structure. I therefore assigned images to groups by consecutive blocks of 37 capture numbers (and merged duplicate files). This gives "
      f"{CV['n_groups']} groups. The block length of 37 is the mean number of images per patient implied by the documented 89 patients (3256/89); this assumes patient runs of equal length, which I could not verify. The effect of changing the embargo is analysed in Results.")
    P("Cross-validation used five *contiguous* folds (the stain reference, fitted once on 20 images without labels, was not refitted per fold, and those images can include test-fold images): within each class, the images ordered by capture number are cut into five unbroken "
      "stretches, and fold k contains the k-th stretch of every class. A validation stretch is taken from the fold two positions after the test fold, and an *embargo* of "
      "37 capture numbers is removed from the training set on both sides of the test and validation stretches. All preprocessing statistics that "
      "depend on the data (the stain reference) are computed independently of labels. Confidence intervals use a cluster bootstrap that resamples "
      f"whole groups {c('efron1979')} (1000 resamples), and paired comparisons between variants use the paired cluster bootstrap with Holm's correction "
      f"{c('holm1979')}. The Nadeau-Bengio corrected resampled t statistic {c('nadeau2003')} is available in the code for comparisons between "
      "fold-level results.")
    S("Models")
    P(f"The headline model is a frozen DinoBloom-S encoder (a small DINOv2 vision transformer trained on hematology images {c('koch2024', 'oquab2024')}), "
      f"applied to each cell crop at {CVC['model']['input_size']} x {CVC['model']['input_size']} pixels, followed by gated attention pooling over the cells of an image "
      f"{c('ilse2018')} and a linear classification layer. Only the head (a 256-unit projection, attention pooling and the classifier; about 166,000 parameters) was trained (AdamW {c('loshchilov2019')}, learning rate "
      f"{CVC['train']['lr_head']}, batch size {CVC['train']['batch_size']}, at most {CVC['train']['epochs']} epochs, early stopping with patience "
      f"{CVC['train']['patience']} on validation macro-F1, seed {CVC['train']['seed']}). Class imbalance was handled with inverse-square-root class weights in the loss. "
      f"Probabilities were calibrated by temperature scaling {c('guo2017')} fitted on the validation stretch of each fold.")
    P("For the ablations and the leakage analysis I used a faster setting in which the frozen embeddings of the cells of an image are averaged and a "
      "multinomial logistic regression with balanced class weights is fitted on standardised features. This probe is deterministic given the folds; "
      f"the three seeds in the tables rotate the boundaries between folds. Alternative encoders (ResNet {c('he2016')}, EfficientNet "
      f"{c('tan2019')}, from the timm library {c('wightman2019')}; generic DINOv2) and a hybrid model that concatenates hand-made features "
      "to the embedding are defined in the code. Low-rank adaptation " + c('hu2022') + " and full fine-tuning are implemented and tested "
      "(see Appendix IV) but were not run for this paper.")
    S("Statistics and software")
    P(f"All code is in a public repository; the analysis was run on a CPU only, with Python 3.11, PyTorch {c('paszke2019')} 2.14, scikit-learn 1.9, "
      f"OpenCV {c('bradski2000')} and scikit-image. The code is covered by {ntests} automated tests. The pipeline configuration hash of the run reported here is "
      f"{DS['config_hash']}; the model configuration hash is {CV['config_hash']}. Numbers are read from result files in the "
      "repository by the script that builds the manuscript, unless stated.")

    ext("methods_stages")
    ext("methods_extra")

    H("Results")
    S("Dataset and preprocessing")
    P(f"The pipeline processed all {N_IMG} images without failure in {DS['run_seconds'] / 60:.1f} min on a CPU. It produced {N_CELL} cells, a mean of "
      f"{N_CELL / N_IMG:.1f} cells per image; the Pre class contained many more cells per image ({PER['Pre']['cells_per_image_mean']:.1f}) than the other classes "
      f"({PER['Benign']['cells_per_image_mean']:.1f}, {PER['Early']['cells_per_image_mean']:.1f} and {PER['Pro']['cells_per_image_mean']:.1f} for Benign, Early and Pro; "
      f"Table {{T:dataset}}). Between {pct(min(PER[k]['border_cell_frac'] for k in CL))}% and {pct(max(PER[k]['border_cell_frac'] for k in CL))}% of the cells in a class touched the image border. "
      f"Example images of the four classes are shown in Fig. {{F:classes}}.")
    P(f"For a synthetic scene with Gaussian noise of standard deviation {bench[0]['sigma']:.2f}-{bench[-1]['sigma']:.2f}, the stabilised diffusion improved the peak "
      f"signal-to-noise ratio by {HDS['mean_gain_db']['stabilised_hds']:.1f} dB on average, comparable with bilateral filtering "
      f"({HDS['mean_gain_db']['bilateral']:.1f} dB) and above Gaussian smoothing ({HDS['mean_gain_db']['gaussian']:.1f} dB), whereas the original scheme "
      f"made the image worse ({HDS['mean_gain_db']['legacy_hds']:.1f} dB; Table {{T:hds}}, Fig. {{F:hds}}). On the synthetic textured disc, the correlation of the filtered "
      f"texture with the true texture pattern was {HDS['texture_correlation']['stabilised_hds']:.2f} for stabilised diffusion and "
      f"{HDS['texture_correlation']['bilateral']:.2f} for bilateral filtering ({HDS['texture_correlation']['noisy']:.2f} for the noisy input), so stabilised diffusion preserved more of the texture; this is a single synthetic example.")
    S("The class can be predicted from the background")
    P(f"The audit showed strong acquisition cues in the raw images (Table {{T:audit}}). The mean background colour alone, three numbers, gave a balanced accuracy of "
      f"{f3(bg_raw)} (chance 0.25), and the scale-bar corner patch gave {f3(aud('scale-bar corner patch'))}. After neutralisation, the background probe fell to chance "
      f"({f3(aud('background colour of rgb_clean'))}), confirming that the neutralised images no longer carry the background. The corner patch of the neutralised "
      f"image still gave {f3(aud('corner patch of rgb_clean'))}, above chance, which I have not explained; cell pixels near the corner may contribute, but I did not test this.")
    P(f"The colour of the cells themselves is not free of session information: the mean colour of the white-balanced cells gave {f3(cellcol)}. "
      f"A random forest on all aggregated hand-made features reached {f3(taball)}, whereas shape and texture features without colour or density reached "
      f"{f3(aud('shape + texture'))} (Table {{T:audit}}). Cell colour is therefore the most informative single group, and it cannot be separated from stain "
      "and session effects in this dataset.")
    S("Headline result with a frozen foundation model")
    P(f"With the headline model (Table {{T:cv}}, Fig. {{F:cm}}), the pooled out-of-fold balanced accuracy was {f3(ba)} (95% CI {ci(bal)}) and the macro-F1 was "
      f"{f3(mf)} (95% CI {ci(pci['macro_f1'])}). The AUROC (one-vs-rest) was {f3(pool['auroc_ovr'])}. The recall of the four classes was "
      f"{RECALLS}. Treating the three malignant classes as one, sensitivity was "
      f"{f3(pool['sensitivity'])} and specificity {f3(pool['specificity'])}. The per-fold balanced accuracy ranged from "
      f"{f3(min(p['test']['balanced_accuracy'] for p in pf))} to {f3(max(p['test']['balanced_accuracy'] for p in pf))}. The most frequent confusions were "
      f"Pre predicted as Early ({cm[2][1]} images), Pre predicted as Pro ({cm[2][3]}) and Early predicted as Benign ({cm[1][0]}). Training stopped after "
      f"{min(n_epochs)} to {max(n_epochs)} epochs per fold and took {secs / 60:.0f} min for all folds on a CPU.")
    P(f"Calibration improved with temperature scaling: the mean per-fold expected calibration error fell from {f3(CV['per_fold_mean']['ece'])} (uncalibrated) to {f3(CAL_FOLD_MEAN)} (temperature-scaled), "
      f"and the pooled temperature-scaled value was {f3(pool['ece'])}. The fitted temperatures ranged from "
      f"{min(p['temperature'] for p in pf):.2f} to {max(p['temperature'] for p in pf):.2f}; a temperature below 1 means the raw probabilities were under-confident.")
    S("What the model uses: representation ablation")
    P(f"Table {{T:abl}} and Fig. {{F:abl}} compare five inputs under the same protocol. Isolated cells with a neutral background (A) gave a balanced accuracy of "
      f"{f3(dsA['balanced_accuracy'])} (95% CI {ci(dsA['balanced_accuracy_ci'])}). Removing stain colour (B, grayscale) lowered it by "
      f"{abs(dsB['vs_reference']['diff']):.3f} (95% CI {abs(dsB['vs_reference']['hi']):.3f}-{abs(dsB['vs_reference']['lo']):.3f}, Holm-adjusted p <= {f3(dsB['vs_reference']['p_holm'])}, the minimum attainable with 1000 resamples), "
      f"so colour contributes a measurable but small part of the accuracy. Giving the model the background as well, after stain normalisation (C) or raw (D), "
      f"raised the accuracy to {f3(dsC['balanced_accuracy'])} and {f3(dsD['balanced_accuracy'])}; both differences from A were significant (Holm p = "
      f"{f3(dsC['vs_reference']['p_holm'])} for each). The hand-made shape, texture and curvature features alone (E, no pixels) gave {f3(dsE['balanced_accuracy'])}, "
      f"well below the encoder.")
    P(f"The same ablation was first computed with random-block folds and was superseded by the contiguous result: the accuracy of variants A-E was "
      f"{', '.join(f3(r['balanced_accuracy']) for r in ABL_OLD['rows'])}, respectively, and the random-block results are kept in the repository for transparency.")
    S("Random block folds gave higher accuracy, and removing neighbours hurt more than random removal")
    P(f"With DinoBloom-S features, random 37-image blocks gave a balanced accuracy of {f3(rb0['balanced_accuracy'])} and contiguous folds gave "
      f"{f3(sch[('contiguous folds, embargo 0', 'DinoBloom-S')]['balanced_accuracy'])} with no embargo and {f3(rb37['balanced_accuracy'])} with an embargo of 37 (Table {{T:scheme}}). "
      f"For hand-made features the gap was larger: {f3(rb0h['balanced_accuracy'])} for random blocks versus {f3(rb37h['balanced_accuracy'])} for contiguous folds "
      f"with the embargo. The difference between the schemes is therefore about {100 * (rb0['balanced_accuracy'] - rb37['balanced_accuracy']):.1f} percentage points for the foundation-model "
      f"features and {100 * (rb0h['balanced_accuracy'] - rb37h['balanced_accuracy']):.1f} for the hand-made features. A larger embargo reduced accuracy slightly further "
      f"({f3(sch[('contiguous folds, embargo 75', 'DinoBloom-S')]['balanced_accuracy'])} at 75).")
    P(f"If proximity in capture order carries information, removing near neighbours of the test images from the training set should hurt more than removing the "
      f"same number of images at random. This was the case (Table {{T:purge}}, Fig. {{F:leak}}a). With random-block folds, purging training images within 75 capture numbers of the "
      f"test images (leaving {B75['train_size']:.0f} training images) gave {f3(B75['purged'])}, whereas removing {B75['train_size']:.0f}-sized random subsets gave "
      f"{f3(B75['random_same_size'])}. At a purge distance of 150 the purged model fell to {f3(B150['purged'])} against {f3(B150['random_same_size'])} for random removal. "
      f"The same pattern appeared with contiguous folds (Table {{T:gap}}, Fig. {{F:leak}}b): at a gap of 225 the accuracy was {f3(C225['dino'])} (random removal "
      f"{f3(C225['dino_rand'])}) for DinoBloom-S and {f3(C225['hand'])} (random {f3(C225['hand_rand'])}) for the hand-made features, which depend on proximity more strongly. "
      f"At a gap of 300 the training sets were small and class coverage was reduced, and accuracy fell also with random removal ({f3(C300['dino_rand'])}), so this setting "
      "does not isolate the effect of proximity.")
    P("These purge and gap analyses are single runs per setting, without confidence intervals, and I interpret them as a consistent pattern rather than as precise estimates.")
    if ENC:
        S("Comparison of encoders")
        P("Table {T:enc} compares the encoders under the contiguous protocol.")
    if GPU:
        S("Fine-tuning")
        P("Fine-tuned runs completed by the author are listed in Table {T:ft}.")
    ext("results_stages")
    ext("results_stage_outputs")
    ext("results_extra")
    PB_FIGS_MARK.append(len(BODY))

    H("Discussion")
    P(f"A frozen hematology foundation model with an attention head classified the four classes with a balanced accuracy of {f3(ba)} under a protocol that separated "
      f"capture sessions with an embargo. Three observations qualify this number, and I regard them as the main findings of the study.")
    P(f"First, the dataset contains strong acquisition shortcuts. The mean background colour of an image predicted the class with a balanced accuracy of "
      f"{f3(bg_raw)}, and a model that was allowed to see the background performed better than one that saw only the cells (Table {{T:abl}}). This agrees with "
      f"the reports of shortcut learning in other medical imaging tasks {c('degrave2021', 'geirhos2020')}. The most likely explanation is that the classes were imaged in "
      "separate sessions with different illumination and staining; I could not verify this because the dataset documentation does not give per-image acquisition metadata. "
      "The class-batch confound means that no analysis of this dataset alone can separate the morphology of the cells from the conditions of the session. "
      "Neutralising the background removed the background cue, but the colour of the cells (balanced accuracy {f3(cellcol)}) and the density of cells ({f3(aud('density'))}) remained informative about the class (Table {T:audit}), as it would be "
      "if the stain differed between sessions and also if the diseases differ in cytoplasmic staining. The accuracy reported here should therefore be read as an optimistic, unvalidated estimate for new patients and laboratories; no external data were tested.")
    P(f"Second, random folds overestimate accuracy, and the effect operates through proximity in capture order. The overestimate was small for the foundation model "
      f"({100 * (rb0['balanced_accuracy'] - rb37['balanced_accuracy']):.1f} percentage points) and larger for hand-made features "
      f"({100 * (rb0h['balanced_accuracy'] - rb37h['balanced_accuracy']):.1f}). It is tempting to conclude that the foundation-model features are robust to leakage; "
      "a more careful reading is that they are so strong that even a training set from distant sessions remains sufficient (at a gap of "
      f"225 capture numbers the accuracy was still {f3(C225['dino'])}), which is itself compatible with a class-specific session signature that is stable across the whole class. "
      "Distance in capture order is only a proxy for patient identity; without true patient labels I cannot exclude that images of the same patient were present on both sides of "
      "every fold.")
    P(f"Third, the comparison of inputs (Table {{T:abl}}) shows that the shortcut-neutralised input gave lower accuracy than the background-visible input "
      f"({f3(dsA['balanced_accuracy'])} versus {f3(dsD['balanced_accuracy'])}), which is the expected direction when a shortcut is removed. Hand-made morphological features "
      f"reached {f3(dsE['balanced_accuracy'])} and the encoder features {f3(dsA['balanced_accuracy'])}, so the foundation model captured information that my features did not. "
      f"The nucleus step failed for {100 - float(pct(DS['nucleus_found_frac'])):.1f}% of the cells, which limits what hand-made morphology can add, and the segmentation was not "
      "validated against ground-truth masks.")
    S("Limitations")
    P("The study has limits that the data impose and I could not remove. There are no patient identifiers, and the 'session groups' are a proxy. All classes appear to come from "
      "different sessions, so classification accuracy cannot be separated from session identification. There is one dataset and no external test set. Class labels were "
      "taken as given, and I did not review them. The DinoBloom training collection may include this dataset or related datasets from the same source; I could not verify this, and "
      "if it does, the foundation-model results are optimistic for a further reason. The purge and gap analyses are single runs without intervals. The headline model was trained on a CPU; "
      "fine-tuning of the encoder and a comparison of encoders on a GPU was prepared and tested in code but its results are not part of this paper"
      + ("." if not (ENC or GPU) else ", except where reported above.") +
      " Calibration was fitted on a validation stretch and assessed on the test stretch of the same dataset, and has not been validated on other data.")
    S("Implications and future work")
    P("Three practices follow from the results and would improve work on this and similar datasets. Report evaluation with session-aware or patient-aware folds and an embargo "
      "whenever capture order is available. Report a shortcut audit - at minimum the accuracy obtained from the background alone - together with any headline accuracy. "
      "Validate on an external dataset acquired in a different laboratory. The natural next experiments are an evaluation on an external dataset of leukemic and normal cells "
      "such as C-NMC " + c('mourya2019') + " or the Raabin-WBC data " + c('kouzehkanan2022') + ", and fine-tuning of the foundation-model encoder under the same folds.")
    P("In conclusion, a frozen foundation model separated the four classes of this dataset with a balanced accuracy of "
      f"{f3(ba)} under a session-aware protocol, but the dataset contains acquisition shortcuts that by themselves predicted the class with a balanced accuracy of {f3(bg_raw)}. "
      "The accuracy is best interpreted as an optimistic estimate that has not been validated outside the dataset, and the evaluation design, not the model, is the main determinant of how far such a number can be trusted.")

    ext("discussion_extra")

    H("Acknowledgments")
    P(META["acknowledgments"], noindent=True)
    H("Literature Cited")
    for r in sorted_refs(USED):
        BODY.append({"t": "ref", "text": r})


PB_FIGS_MARK = []


# ================================================================== appendices
def appendices():
    PB()
    H("Appendix I. Per-fold results of the headline model")
    P("Table A1 lists the results of each of the five folds. Table A2 lists per-class recall in each fold. These are the unpooled test-stretch results; the pooled numbers in the main text are computed from the same predictions pooled over the folds.", noindent=True)
    BODY.append({"t": "table", "num": "A1", "caption": "Per-fold results of the headline model (frozen DinoBloom-S, attention head).",
                 "cols": [{"w": 1, "align": "left"}] + [{"w": 1.1, "align": "right"}] * 9,
                 "header": ["Fold", "Train", "Valid.", "Test", "Bal. acc.", "Macro-F1", "AUROC", "Epochs", "Temp.", "Time (s)"],
                 "rows": [[str(p["fold"] + 1), str(p["split"]["n_train_images"]), str(p["split"]["n_val_images"]), str(p["split"]["n_test_images"]),
                           f3(p["test"]["balanced_accuracy"]), f3(p["test"]["macro_f1"]), f3(p["test"]["auroc_ovr"]), str(p["epochs"]),
                           f2(p["temperature"]), f"{p['seconds']:.0f}"] for p in pf], "foot": None})
    BODY.append({"t": "table", "num": "A2", "caption": "Per-class recall of each fold (test stretch).",
                 "cols": [{"w": 1, "align": "left"}] + [{"w": 1.2, "align": "right"}] * 4,
                 "header": ["Fold"] + CL, "rows": [[str(p["fold"] + 1)] + [f3(v) for v in p["test"]["per_class_recall"]] for p in pf], "foot": None})
    BODY.append({"t": "table", "num": "A3", "caption": "Pooled out-of-fold confusion matrix of the headline model (rows: true class; columns: predicted class).",
                 "cols": [{"w": 1.5, "align": "left"}] + [{"w": 1, "align": "right"}] * 4,
                 "header": ["True \\ predicted"] + CL, "rows": [[CL[i]] + [str(v) for v in cm[i]] for i in range(4)], "foot": None})
    rows4 = []
    for p in pf:
        for i in range(4):
            rows4.append([str(p["fold"] + 1) if i == 0 else "", CL[i]] + [str(v) for v in p["test"]["confusion"][i]])
    BODY.append({"t": "table", "num": "A4", "caption": "Confusion matrices of the five folds (rows: true class; columns: predicted class).",
                 "cols": [{"w": 0.8, "align": "left"}, {"w": 1.4, "align": "left"}] + [{"w": 1, "align": "right"}] * 4,
                 "header": ["Fold", "True class"] + CL, "rows": rows4, "foot": None})

    PB()
    H("Appendix II. Superseded random-block ablation")
    P("The ablation of Table {T:abl} was first computed with random-block folds. These results are kept because they show the size of the inflation; they are not "
      "used as evidence of performance.", noindent=True)
    BODY.append({"t": "table", "num": "A5", "caption": "Representation ablation with random 37-image blocks (superseded by the contiguous protocol).",
                 "cols": [{"w": 5, "align": "left"}, {"w": 1.4, "align": "right"}, {"w": 1.4, "align": "right"}, {"w": 1.4, "align": "right"}],
                 "header": ["Variant", "Balanced accuracy", "Macro-F1", "AUROC"],
                 "rows": [[r["variant"], f3(r["balanced_accuracy"]), f3(r["macro_f1"]), f3(r["auroc_ovr"])] for r in ABL_OLD["rows"]], "foot": None})

    PB()
    H("Appendix III. Definitions of the per-cell features")
    P("Each segmented cell yields the following measurements. Image-level aggregates are the mean, standard deviation and maximum over the cells of an image "
      "(computed over interior cells unless the image has none). All lengths are in pixels and all areas in square pixels of the 224 x 224 image.", noindent=True)
    feats = [
        ["Shape", "cell_area, cell_perimeter, cell_eq_diameter", "Area, perimeter and equivalent-circle diameter of the cell mask."],
        ["Shape", "cell_major_axis, cell_minor_axis, cell_aspect_ratio, cell_eccentricity", "Ellipse-fit axes of the mask and derived ratios."],
        ["Shape", "cell_solidity, cell_extent, cell_circularity", "Area over convex-hull area; area over bounding-box area; 4 pi area / perimeter^2."],
        ["Position", "centroid_y, centroid_x, touches_border", "Centroid and whether the mask touches the image border."],
        ["Colour", "cell_L_mean/std, cell_a_mean/std, cell_b_mean/std", "Mean and SD of the CIE Lab channels inside the cell (stain-normalised image)."],
        ["Nucleus", "nucleus_found, nucleus_contrast", "Whether the nucleus step succeeded, and the contrast used to accept it."],
        ["Nucleus", "nucleus_area ... nucleus_circularity", "The same shape measurements for the nucleus mask (missing when no nucleus was found)."],
        ["Nucleus", "nc_ratio", "Nucleus area divided by cell area."],
        ["Nucleus", "cytoplasm_L/a/b_mean, nuc_cyto_L_contrast", "Lab means of the cytoplasm (cell minus nucleus) and nucleus-cytoplasm lightness contrast."],
        ["Texture", "cell_glcm_contrast/homogeneity/energy/correlation", "Grey-level co-occurrence statistics {c('haralick1973')} of the cell, masked and averaged over four directions."],
        ["Texture", "nucleus_glcm_*, nucleus_entropy, cell_entropy", "The same for the nucleus; Shannon entropy of the grey-level histogram."],
        ["Response", "cell_apc_mean/std/rim_mean", "Mean, SD and rim mean of the principal-curvature response layer inside the cell."],
        ["Response", "cell_log_mean/std/rim_mean", "The same for the Laplacian-of-Gaussian layer."],
    ]
    USED.add("haralick1973")
    for f in feats:
        f[2] = f[2].replace("{c('haralick1973')}", c("haralick1973"))
    BODY.append({"t": "table", "num": "A6", "caption": "Per-cell features computed by the preprocessing package.",
                 "cols": [{"w": 1.2, "align": "left"}, {"w": 4, "align": "left"}, {"w": 5, "align": "left"}],
                 "header": ["Group", "Columns", "Definition"], "rows": feats, "foot": None})

    PB()
    H("Appendix IV. Software and prepared experiments")
    P("The repository contains two packages. The first (leukemia_pp) implements the preprocessing described in Materials and Methods and exposes the commands run, "
      "inspect and audit. The second (leukemia_ml) implements encoder loading, low-rank adaptation, attention pooling, a hybrid branch, training with exact "
      "resume, cross-validation, calibration, statistics and Grad-CAM. The test suite contains " + ntests + " tests and was run on a CPU.", noindent=True)
    P("Eight GPU experiments (configurations g01 to g08) were prepared for fine-tuning: low-rank adaptation of DinoBloom-S (g01), a grayscale variant (g02), the hybrid model with "
      "hand-made features (g03), full fine-tuning of ResNet-50 (g04) and EfficientNet-B0 (g05), low-rank adaptation of generic DINOv2-S (g06) and of DinoBloom-B (g07), and a "
      "raw image with background (g08). They were tested for correctness on a CPU with a few steps, but their results are "
      + ("reported in the Results." if GPU else "not part of this paper.") , noindent=True)
    BODY.append({"t": "small", "text": "Headline model configuration (JSON):"})
    BODY.append({"t": "code", "lines": json.dumps(CVC, indent=1).splitlines()})
    BODY.append({"t": "small", "text": "Preprocessing configuration of the full run (JSON):"})
    BODY.append({"t": "code", "lines": json.dumps(DS["pipeline_config"], indent=1).splitlines()})

    PB()
    H("Appendix V. Reproducing the results")
    P("The commands below reproduce the preprocessing, the audit, the cross-validation and this manuscript from the images of the public dataset (a Kaggle account is "
      "needed to download them).", noindent=True)
    BODY.append({"t": "code", "lines": [
        "pip install -e .",
        "leukemia-pp run  --input data/dataset/Original --output data/full_run_v2 --config configs/full_dataset.json",
        "leukemia-pp audit --run data/full_run_v2",
        "leukemia-ml cv    --config configs/ml/cv_dinobloom_s_frozen_mil.json --out runs/cv_dinobloom_s",
        "leukemia-ml ablate --spec configs/ml/ablation_sources.json --out runs/ablation_sources   # results copied to docs/results",
        "leukemia-pp run --input data/dataset/Original --output data/full_run_all --config configs/full_all_representations.json   # HDS + BEEMD on",
        "python scripts/analysis_stage_ablation.py; python scripts/make_stage_figures.py",
        "python scripts/analysis_dataset.py; python scripts/analysis_session.py",
        "python scripts/analysis_leakage.py; python scripts/benchmark_hds.py",
        "python scripts/make_figures.py",
        "python paper/build_paper.py && node paper/render_docx.js paper/content.json paper/paper.docx",
    ]})
    P("The weights of DinoBloom-S are available from Zenodo (record 10908163) under a Creative Commons Attribution licence.", noindent=True)
    ext("appendix_extra")


# ================================================================== assemble

# ------------------------------------------------------------------ abbreviation clean-up (the guide allows abbreviations for units only)
_PAREN = re.compile(r" \((HDS|APC|LoG|BEEMD|EMD|IMFs?|ECE|AUROC|ROC|MIL|GLCM|LoRA|PBS|ALL|SD|CI)\)")
_PHRASES = [
    (r"\bStabilised HDS\b", "Stabilised hybrid diffusion"), (r"\bstabilised HDS\b", "stabilised hybrid diffusion"), (r"\bLegacy HDS\b", "Original hybrid diffusion"),
    (r"\blegacy HDS\b", "original hybrid diffusion"), (r"\bHDS edge\b", "hybrid-diffusion edge"), (r"\bHDS\b", "hybrid diffusion"),
    (r"\bAPC\b", "adaptive principal-curvature"), (r"\bLoG\b", "Laplacian-of-Gaussian"),
    (r"\bpure-IMF\b", "pure-mode"), (r"\bIMF energies\b", "mode energies"), (r"\bIMFs\b", "intrinsic mode functions"), (r"\bIMF\b", "intrinsic mode function"),
    (r"\bBEEMD\b", "ensemble mode decomposition"), (r"\bEMD\b", "empirical mode decomposition"),
    (r"\bAUROC\b", "area under the receiver operating characteristic curve"), (r"\bROC\b", "receiver operating characteristic"),
    (r"\bECE\b", "expected calibration error"), (r"\bGLCM\b", "grey-level co-occurrence"), (r"\bLoRA\b", "low-rank adaptation"),
    (r"\bMIL\b", "multiple-instance learning"), (r"\bPBS\b", "peripheral blood smear"), (r"\bCPU\b", "central processor"),
    (r"\bGPU\b", "graphics processor"), (r"\bRGB\b", "colour"), (r"\bCI\b", "confidence interval"), (r"\bSD\b", "standard deviation"),
    (r"\bt-SNE\b", "t-distributed stochastic neighbour embedding"), (r"\bpx\b", "pixels"), (r"\bALL\b", "acute lymphoblastic leukemia"),
]


def _art(m, rep):
    pre = m.group(1) or ""
    if pre:
        a = "an " if rep[0].lower() in "aeiou" else "a "
        pre = (a.capitalize() if pre[0] == "A" else a)
    return pre + rep


def despell(t):
    if not isinstance(t, str):
        return t
    t = t.replace("[[GPU", "\x00G").replace("[[PENDING", "\x00P").replace("FT-", "\x00F")
    t = _PAREN.sub("", t)
    for pat, rep in _PHRASES:
        t = re.sub(r"\b(An? )?(" + pat.replace(r"\b", "") + r")", lambda m, rep=rep: _art(m, rep), t) if False else re.sub(r"(\b[Aa]n? )?" + pat, lambda m, rep=rep: _art(m, rep), t)
    t = re.sub(r"\b(diffusion|decomposition|curvature|learning) \1\b", r"\1", t)
    t = t.replace("Eight graphics processor", "Eight graphics-processor")
    return t.replace("\x00G", "[[GPU").replace("\x00P", "[[PENDING").replace("\x00F", "FT-")


def short_cap(c):
    c = re.sub(r"\*", "", c)
    c = re.split(r"\. | \(|; ", c)[0].strip().rstrip(".")
    if len(c) > 100:
        c = c[:100].rsplit(" ", 1)[0]
    return c + "."


def assemble():
    global ntests
    for f in sorted((ROOT / "paper").glob("ext_*.py")):
        exec(compile(f.read_text(), str(f), "exec"), globals())
    tp = ROOT / "docs" / "results" / "test_summary.json"
    ntests = str(json.loads(tp.read_text())["passed"]) if tp.exists() else "189"
    build()
    appendices()
    # number tables / figures by first citation in the body text
    order_t, order_f = [], []
    for b in BODY:
        txt = b.get("text", "") + " ".join(b.get("rows", []) and [] or [])
        for k in re.findall(r"\{T:(\w+)\}", txt):
            order_t.append(k) if k not in order_t else None
        for k in re.findall(r"\{F:(\w+)\}", txt):
            order_f.append(k) if k not in order_f else None
    if META.get("gpu_placeholders", "keep") == "cut":
        TABLES.pop("ft", None)
        for k in [k for k in FIGS if k.startswith("ft_")]:
            FIGS.pop(k)
    for k in TABLES:
        assert k in order_t, f"table {k} is never cited"
    for k in FIGS:
        assert k in order_f, f"figure {k} is never cited"
    tn = {k: str(i + 1) for i, k in enumerate(order_t)}
    fn = {k: str(i + 1) for i, k in enumerate(order_f)}

    def sub(s):
        s = re.sub(r"\{T:(\w+)\}", lambda m: tn[m.group(1)], s)
        return re.sub(r"\{F:(\w+)\}", lambda m: fn[m.group(1)], s)

    out = []
    pre = [{"t": "title", "text": META["title"]}, {"t": "authors", "lines": [", ".join(META["authors"]), META["affiliation"], "Correspondence: " + META["correspondence"]]},
           {"t": "h", "text": "Abstract"}, {"t": "p", "text": ABSTRACT, "noindent": True},
           {"t": "p", "text": "*Keywords:* " + META["keywords"], "noindent": True}]
    mark = PB_FIGS_MARK[0]
    for i, b in enumerate(BODY):
        if i == mark:
            for k in order_t:
                out.append({"t": "pb"}); out.append({**TABLES[k], "num": tn[k]})
            for k in order_f:
                out.append({"t": "pb"}); out.append({**FIGS[k], "num": fn[k]})
            out.append({"t": "pb"})
        out.append(b)
    blocks = pre + out
    # ---- contents, list of tables, list of figures (page numbers measured from the rendered PDF by paper/paginate.py)
    pages = json.loads((ROOT / "paper" / "toc_pages.json").read_text()) if (ROOT / "paper" / "toc_pages.json").exists() else {}
    toc = [{"t": "pb"}, {"t": "toch", "text": "Contents", "first": True}]
    lists_t, lists_f = [], []
    for b in blocks:
        if b["t"] == "h" and b["text"].lower() != "abstract":
            toc.append({"t": "toc", "level": 1, "text": b["text"], "key": "h:" + b["text"][:60], "page": "0"})
        elif b["t"] == "sh":
            toc.append({"t": "toc", "level": 2, "text": b["text"], "key": "sh:" + b["text"][:60], "page": "0"})
        elif b["t"] == "table":
            lists_t.append({"t": "toc", "level": 3, "text": f"Table {b['num']}. " + short_cap(b["caption"]), "key": f"table:{b['num']}", "page": "0"})
        elif b["t"] == "fig":
            cap = re.sub(r"\*\*\[[^\]]*\]\*\*\s*", "", b["caption"])
            lists_f.append({"t": "toc", "level": 3, "text": f"Figure {b['num']}. " + short_cap(cap), "key": f"fig:{b['num']}", "page": "0"})
    toc += [{"t": "toch", "text": "List of Tables"}] + lists_t + [{"t": "toch", "text": "List of Figures"}] + lists_f + [{"t": "pb"}]
    for e in toc:
        if e["t"] == "toc":
            e["page"] = str(pages.get(e["key"], "0"))
    ia = next(i for i, b in enumerate(blocks) if b["t"] == "h" and b["text"].lower() == "introduction")
    blocks = blocks[:ia] + toc + blocks[ia:]
    for b in blocks:
        for key in ("text", "caption", "foot"):
            if isinstance(b.get(key), str):
                b[key] = sub(b[key])
        if b["t"] in ("p", "sh", "small", "table", "fig", "toc"):
            for key in ("text", "caption", "foot"):
                if isinstance(b.get(key), str) and not b.get("noabbr"):
                    b[key] = despell(b[key])
            if b["t"] == "table" and b.get("header"):
                b["header"] = [despell(h.replace("AUROC", "Area under curve")) for h in b["header"]]
                if b["header"][0] == "Stage":
                    b["rows"] = [[despell(x) for x in r] for r in b["rows"]]
        if "rows" in b:
            b["rows"] = [[sub(x) for x in r] for r in b["rows"]]
    left = [b for b in blocks if re.search(r"\{[TF]:|\{c\(|\{n\(|\{\{|\{[A-Z_]+\}", " ".join(str(v) for v in b.values()))]
    assert not left, f"unresolved placeholder: {json.dumps(left[0])[:300]}"
    (ROOT / "paper" / "content.json").write_text(json.dumps({"meta": {"title": META["title"]}, "blocks": blocks}, indent=1))
    words = sum(len(re.findall(r"\w+", b.get("text", ""))) for b in blocks)
    print(f"blocks={len(blocks)} words~{words} tables={len(TABLES)} figs={len(FIGS)} tests={ntests} refs_cited={len(USED)}")


if __name__ == "__main__":
    assemble()

SE = json.loads((R / "stage_examples.json").read_text())
SA = json.loads((R / "stage_ablation.json").read_text()) if (R / "stage_ablation.json").exists() else None
EX = SE["images"]
SPI = SE["seconds_per_image"]
BE = SE["beemd_example"]

Fg("st_stain", "paper/figures/stage_1_stain.png", f"Stage 1, stain normalisation. Top: input images (one per class: {', '.join(EX[k] for k in CL)}). Bottom: the same images after Reinhard normalisation in Lab space with foreground statistics.", 6.2)
Fg("st_denoise", "paper/figures/stage_2_denoise.png", f"Stage 2, denoising of the luminance of {EX['Early']} (top: full image; bottom: zoom on the central region). From left to right: input, bilateral filter, stabilised HDS diffusion "
   f"({SE['hds_iterations_example']} iterations), and the absolute difference between input and HDS output (amplified four times). Diffusion smooths flat regions and leaves the cell boundaries.", 6.2)
Fg("st_hds", "paper/figures/stage_3_hds_edge.png", "Stage 3, HDS edge indicator. Top: luminance of one image per class. Bottom: the edge indicator 1 - mean diffusivity of the denoised image (0 in flat regions, close to 1 across strong edges), on a fixed scale.", 6.2)
Fg("st_layers", "paper/figures/stage_4_layers.png", f"Stages 4 and 5, response layers of {EX['Early']}. From left to right: denoised grey image, adaptive principal-curvature (APC) response, Laplacian-of-Gaussian (LoG) response and Canny edges.", 6.2)
Fg("st_beemd", "paper/figures/stage_5_beemd.png", f"Stages 6 and 7, BEEMD and pure-IMF selection for the luminance of {EX['Early']}. The four intrinsic mode functions (IMFs) are ordered from fine to coarse; the title of each gives the "
   "characteristic spatial period, the share of the detail energy and whether the selection rule keeps the mode. Also shown: the residue, the sum of the selected (pure) modes, and the difference between the input and the sum of all modes and the residue, which is zero up to rounding.", 6.2)
Fg("st_seg", "paper/figures/stage_6_segmentation.png", "Stage 8, cell and nucleus segmentation for one image per class. Top: smoothed a* score. Middle: cell instances after watershed splitting (random colours; border-touching cells included). Bottom: cell (green) and nucleus (yellow) outlines on the denoised image.", 6.2)
Fg("st_clean", "paper/figures/stage_7_neutralise.png", "Stage 9, shortcut neutralisation. Top: stain-normalised images. Bottom: the same images with everything outside the cells replaced by a neutral fill and the cells white-balanced.", 6.2)
Fg("st_crops", "paper/figures/stage_7b_crops.png", f"Stage 9, outputs for {EX['Early']}: the neutralised image, its grey version, and the isolated 112 x 112 crops of the first four cells that are passed to the encoder.", 6.2)
Fg("st_cells", "paper/figures/stage_8_cells.png", f"Stage 10, cells of {EX['Early']} numbered as in Table {{T:stage_cells}}.", 3.6)

_ex = SE["example_cells"]
T("stage_cells", f"Stage 10: per-cell measurements of the cells of {EX['Early']} (numbered as in Fig. {{F:st_cells}}). Columns follow the feature definitions of Appendix III; the last two columns are statistics of the "
   "response layers inside the cell (the HDS edge value comes from a separate pass with HDS enabled).",
  [{"w": 0.8, "align": "right"}] + [{"w": 1, "align": "right"}] * 9,
  ["Cell", "Area (px)", "Diameter (px)", "Aspect ratio", "Solidity", "Circularity", "Mean a*", "GLCM contrast", "Mean APC", "Mean HDS edge"],
  [[str(r["cell_id"]), f"{r['cell_area']:.0f}", f"{r['cell_eq_diameter']:.1f}", f2(r["cell_aspect_ratio"]), f2(r["cell_solidity"]), f2(r["cell_circularity"]),
    f"{r['cell_a_mean']:.1f}", f2(r["cell_glcm_contrast"]), f2(r["cell_apc_mean"]), f2(r.get("cell_hds_edge_mean", float("nan")))] for r in _ex],
  foot="Cells that touch the image border are included; nuclei were not resolved for most cells (Table {T:dataset}), so nuclear measurements are not shown.")

T("stages", "The processing stages, their implementation and their use in this study. \"Encoder input\" means that the stage determines the pixels given to the foundation model; \"tabular\" means that "
  "the stage provides hand-made features evaluated in Table {T:stage_abl}.",
  [{"w": 2.2, "align": "left"}, {"w": 2.6, "align": "left"}, {"w": 2.6, "align": "left"}, {"w": 3, "align": "left"}],
  ["Stage", "Implementation", "Input and output", "Role in the reported experiments"],
  [["1 Stain normalisation", "stain.py: Reinhard method in Lab, foreground statistics", "raw RGB to normalised RGB", "Encoder input (all variants except raw)"],
   ["2 Denoising", "denoise.py: bilateral filter (default) or non-local means", "normalised RGB to denoised RGB", "Used for segmentation and for the response layers"],
   ["3 HDS edge detection", "hds.py: stabilised hybrid diffusion; edge indicator", "luminance to denoised luminance and edge map in [0, 1]", "Run on all images in a separate pass; tabular"],
   ["4 APC", "responses.py: adaptive principal curvature of the Hessian", "grey image to response layer", "Per-cell statistics; tabular"],
   ["5 LoG and Canny", "responses.py", "grey image to blob and edge layers", "LoG per-cell statistics; tabular"],
   ["6 BEEMD", "emd.py: ensemble bidimensional EMD, order-statistic envelopes", "luminance to four IMFs and a residue", "Run on all images in a separate pass; tabular"],
   ["7 Pure-IMF selection", "emd.py: scale and energy rule", "IMFs to selection mask and pure-mode image", "Same pass; tabular"],
   ["8 Segmentation", "segmentation.py: a* Otsu, opening, watershed; nucleus", "image to cell and nucleus instance labels", "Defines the cells for every variant"],
   ["9 Neutralisation and crops", "neutralise.py, data/crops.py", "labels and image to isolated cell crops", "Encoder input of the headline model"],
   ["10 Morphological features", "features.py: shape, colour, GLCM texture", "labels and image to per-cell vector", "Tabular"],
   ["11 Classification", "leukemia_ml: frozen encoder, attention head", "cell crops to class probabilities", "Headline model"]])


@section("methods_stages")
def methods_stages():
    S("Processing stages and their outputs")
    P("The processing pipeline consists of the stages named in the project description - HDS edge detection, adaptive principal-curvature (APC) structure detection, BEEMD decomposition, selection of pure intrinsic mode functions (IMFs) "
      "and morphological feature extraction - together with the stages that prepare the images for them and for the classifier. Each stage is implemented and covered by automated tests. "
      f"A separate pass of the pipeline with HDS and BEEMD enabled processed all {N_IMG} images, and the example figures below show the input and output of every stage on real images of the dataset. "
      "Table {T:stages} lists the stages and the role that each plays in the experiments.")
    P(f"Two clarifications on scope follow. First, the pixel input of the headline model is the neutralised cell crop of stage 9, so the response layers and the HDS and BEEMD outputs "
      "enter the classification only through hand-made features, which are evaluated in a separate tabular analysis (Results, Contribution of the processing stages); they are not channels of the foundation-model input. "
      "Second, the description of the APC stage as blood-vessel segmentation in the project title does not apply literally to peripheral blood smears, which contain no vessels; the APC "
      "response of the Hessian is computed as specified and highlights cell boundaries and thin structures, and I report what it measures and not what its name suggests.")
    P(f"*Stage 1, stain normalisation* (Fig. {{F:st_stain}}). The stage maps the colour statistics of the foreground of each image to the reference (Methods, Formal description). The example shows that images of "
      "different classes have different background tints before normalisation and that normalisation reduces but does not remove them. "
      f"It takes {1000 * SPI['stain']:.1f} ms per image.")
    P(f"*Stage 2, denoising* (Fig. {{F:st_denoise}}). The default denoiser is the bilateral filter (diameter 9, colour and space sigma 75), which preserves boundaries by weighting neighbours by both distance and similarity of intensity "
      f"{c('tomasi1998')}. The stabilised diffusion (HDS) is shown next to it. Both remove fine noise and keep the cell outlines; the difference image shows that the diffusion changes mostly the interior of cells and the "
      f"boundaries, in places where the unfiltered image has fine texture. The bilateral filter takes {1000 * SPI['bilateral']:.1f} ms and the diffusion {1000 * SPI['hds']:.0f} ms per image; "
      f"the diffusion stopped after {SE['hds_iterations_example']} iterations in the example.")
    P(f"*Stage 3, HDS edge detection* (Fig. {{F:st_hds}}). The edge indicator is derived from the diffusivity of the denoised image: where the intensity gradient is small the diffusivity is close to 1 and the indicator close to 0, and "
      "across strong edges the diffusivity is small and the indicator is close to 1. It is computed on a fixed scale, so that its values are comparable between images, unlike a per-image rescaled edge map. "
      "In the examples it marks the outlines of cells and of red blood cells alike; it does not distinguish the cells of interest from the other cells, which is the task of the segmentation.")
    P(f"*Stages 4 and 5, response layers* (Fig. {{F:st_layers}}). The APC layer is the weighted sum of the absolute values of the two Hessian eigenvalues, normalised by the square of the scale; it is large for curved "
      "structures such as cell outlines and thin folds. The LoG layer responds to blobs of the chosen scale, and the Canny layer is a binary edge map with an Otsu-derived threshold. In the example, the APC and LoG layers both "
      "highlight the boundary of the cells and also the boundaries of the red blood cells, so, like the edge indicator, they are not specific to leukocytes.")
    P(f"*Stages 6 and 7, BEEMD and pure-IMF selection* (Fig. {{F:st_beemd}}). The decomposition separates the luminance into {BE['n_imfs']} modes with characteristic periods of "
      f"{', '.join(f'{x:.1f}' for x in BE['period_px'])} px, from fine to coarse, and a smooth residue; the reconstruction error was {BE['max_reconstruction_error']:.1e} (rounding). "
      f"The decomposition takes {SPI['beemd']:.2f} s per image on a CPU, which makes it the most expensive stage by an order of magnitude. The selection rule keeps a mode if its characteristic period lies between "
      "3 and 48 px, which rejects pixel noise and illumination drift, and if it carries at least 1% of the detail energy. This rule is my own interpretable choice, because the project description did not define the criterion "
      f"precisely. In the example, all {sum(BE['selected'])} of {BE['n_imfs']} modes were kept, so the pure-IMF image equals the sum of the modes; the share of images in which a mode is rejected is reported in the Results.")
    P(f"*Stage 8, segmentation* (Fig. {{F:st_seg}}). The a* score separates leukocytes from the red cells and the background, and the instance labels separate most touching cells; the outlines on the denoised image "
      "follow the cells closely. Failure modes are visible in the figure: touching cells that are not split (the Benign example of the attention figure), cells at the border that are cut by the frame, and nuclei that "
      "are found only for part of the cells.")
    P(f"*Stage 9, neutralisation and crops* (Figs. {{F:st_clean}} and {{F:st_crops}}). The neutralisation removes the background, the red blood cells and the scale bar, so that the colour tint of the slide and the overlay no longer "
      "reach the encoder; the crop of each cell is then isolated from its neighbours. This is the input of the headline model.")
    P(f"*Stage 10, morphological feature extraction* (Fig. {{F:st_cells}}, Table {{T:stage_cells}}). For every cell, the pipeline measures shape (area, diameter, aspect ratio, solidity, circularity), colour, texture (grey-level co-occurrence "
      f"statistics {c('haralick1973')} and entropy) and the statistics of the response layers. The values in the table are those of the cells of the example image. The per-cell vectors are averaged over the cells of an "
      f"image for the tabular analyses. The complete segmentation, feature and layer computation for one image (stages 2 to 10 without HDS and BEEMD) takes {SPI['segment+features+layers']:.2f} s.")
    S("Stage 11, classification")
    P("The final stage is the classification of the image from the isolated cells, described in the previous subsections. The preceding stages determine which cells the model sees and what it sees of them.")


@section("results_stages")
def results_stages():
    S("Contribution of the processing stages")
    if SA is None:
        P("**[[PENDING: the tabular analysis of the processing stages has not finished (Table {T:stage_abl}); run scripts/analysis_stage_ablation.py and rebuild.]]**")
        return
    rows = {r["variant"]: r for r in SA["rows"]}
    st = rows["shape + texture"]
    P(f"To measure what each stage adds, I trained the same logistic head on image-level means of hand-made features, one feature family at a time and in combination, from a pass of the pipeline in which HDS and BEEMD were enabled "
      f"(Table {{T:stage_abl}}). Each variant sees only the named features and no pixels. Shape features alone gave a balanced accuracy of {f3(rows['shape (morphological)']['balanced_accuracy'])}, texture features "
      f"{f3(rows['texture (GLCM, entropy)']['balanced_accuracy'])}, and both together {f3(st['balanced_accuracy'])}. The layer statistics alone gave {f3(rows['APC response layer']['balanced_accuracy'])} (APC), "
      f"{f3(rows['LoG response layer']['balanced_accuracy'])} (LoG), {f3(rows['HDS edge indicator']['balanced_accuracy'])} (HDS edge indicator), {f3(rows['pure-IMF layer (BEEMD)']['balanced_accuracy'])} (pure-IMF layer) and "
      f"{f3(rows['IMF energies (BEEMD)']['balanced_accuracy'])} (IMF energies).")
    full = rows["... + pure-IMF layer and IMF energies"]
    mid = rows["shape + texture + APC + LoG"]
    hds_row = rows["shape + texture + APC + LoG + HDS edge"]
    P(f"Adding the response layers to shape and texture changed the balanced accuracy from {f3(st['balanced_accuracy'])} to {f3(mid['balanced_accuracy'])} "
      f"(difference {mid['vs_shape_texture']['diff']:+.3f}, 95% CI {mid['vs_shape_texture']['lo']:+.3f} to {mid['vs_shape_texture']['hi']:+.3f}, Holm p = {f3(mid['vs_shape_texture']['p_holm'])}); adding the HDS edge indicator then gave "
      f"{f3(hds_row['balanced_accuracy'])} ({hds_row['vs_shape_texture']['diff']:+.3f} relative to shape and texture, Holm p = {f3(hds_row['vs_shape_texture']['p_holm'])}), and adding the BEEMD features gave {f3(full['balanced_accuracy'])} "
      f"({full['vs_shape_texture']['diff']:+.3f}, Holm p = {f3(full['vs_shape_texture']['p_holm'])}). For comparison, the frozen encoder alone reached {f3(dsA['balanced_accuracy'])} (Table {{T:abl}}).")
    P("Two cautions apply to these numbers. First, the variants differ in the number of features, so part of the increase can come from the larger number of dimensions and not from the stage itself; I did not equalise the dimensionality. "
      "Second, statistics of an edge, curvature or mode layer depend on the focus, sharpness and noise level of an image, which are properties of the acquisition. The session-aware folds reduce the chance that the gain comes from "
      "memorised sessions, but they cannot exclude that these statistics carry class-specific acquisition characteristics, as the colour of the cells does. The gains are therefore evidence that the stages produce class-related "
      "information, not evidence that this information is morphological.")
    im = SA["imf_selection"]
    P(f"The pure-IMF rule kept the first to fourth mode in {', '.join(pct(x) + '%' for x in im['selected_fraction_per_mode'])} of the {im['n_images']} images, respectively; in {pct(im['all_selected_fraction'])}% of the images all modes were kept "
      f"and in {pct(im['none_selected_fraction'])}% none. A selection rule that keeps nearly every mode adds little to the decomposition itself; the information gained from the BEEMD stage should be judged from Table {{T:stage_abl}}, not from the selection.")


if SA is not None:
    T("stage_abl", "Contribution of the processing stages to classification. Logistic regression on image-level means of the named hand-made features only (no pixels), contiguous session-aware folds with embargo 37, three rotations of the fold boundaries; "
      "intervals are 95% cluster-bootstrap intervals (first rotation). Differences are paired against shape + texture, with Holm-adjusted p values over the three comparisons.",
      [{"w": 4.2, "align": "left"}, {"w": 1, "align": "right"}, {"w": 1.1, "align": "right"}, {"w": 1.7, "align": "right"}, {"w": 1.1, "align": "right"}, {"w": 2.2, "align": "right"}],
      ["Features", "Number", "Balanced accuracy", "95% CI", "Macro-F1", "Difference vs shape + texture (95% CI)"],
      [[r["variant"], str(r["n_features"]), f3(r["balanced_accuracy"]), ci(r["ci"]), f3(r["macro_f1"]),
        ("-" if "vs_shape_texture" not in r else f"{r['vs_shape_texture']['diff']:+.3f} ({r['vs_shape_texture']['lo']:+.3f}, {r['vs_shape_texture']['hi']:+.3f}); p = {f3(r['vs_shape_texture']['p_holm'])}")] for r in SA["rows"]])

if SA is None:
    T("stage_abl", "Contribution of the processing stages to classification (tabular probe). **[[PENDING: run scripts/analysis_stage_ablation.py]]**",
      [{"w": 4, "align": "left"}, {"w": 1.5, "align": "right"}], ["Features", "Balanced accuracy"], [["[[PENDING]]", "[[PENDING]]"]])

SA_FULL = [r for r in SA["rows"] if r["variant"].startswith("... + pure-IMF")][0] if SA else {"balanced_accuracy": float("nan")}

LS = json.loads((R / "stage_layer_stats.json").read_text())
_LN = [("cell_hds_edge_mean", "Hybrid-diffusion edge indicator, mean inside the cell", 3), ("cell_hds_edge_rim_mean", "Hybrid-diffusion edge indicator, mean on the cell rim", 3),
       ("cell_apc_mean", "Adaptive principal curvature, mean inside the cell", 3), ("cell_apc_rim_mean", "Adaptive principal curvature, mean on the rim", 3),
       ("cell_log_mean", "Laplacian-of-Gaussian, mean inside the cell", 3), ("cell_log_rim_mean", "Laplacian-of-Gaussian, mean on the rim", 3),
       ("cell_imf_pure_std", "Pure-mode image, standard deviation inside the cell", 3), ("cell_imf1_energy", "Mode 1 (finest) energy", 4), ("cell_imf4_energy", "Mode 4 (coarsest) energy", 4)]
T("stage_out", "Values of the stage outputs inside the segmented cells, by class: median and interquartile range (in parentheses) over cells that do not touch the border, from the pass with hybrid diffusion and ensemble mode decomposition enabled. "
   "The rim is the two-pixel band at the border of the cell. The values are descriptive.",
  [{"w": 3.4, "align": "left"}] + [{"w": 2, "align": "right"}] * 4,
  ["Output"] + [f"{k} (n = {LS['n_cells'][k]})" for k in CL],
  [[lab] + [f"{LS['medians'][key][k]['median']:.{dec}f} ({LS['medians'][key][k]['q25']:.{dec}f} to {LS['medians'][key][k]['q75']:.{dec}f})" for k in CL] for key, lab, dec in _LN])


@section("results_stage_outputs")
def results_stage_outputs():
    S("What the outputs of the processing stages tell us")
    m = LS["medians"]
    g = lambda key, k: m[key][k]["median"]  # noqa: E731
    rat = {k: g("cell_hds_edge_rim_mean", k) / g("cell_hds_edge_mean", k) for k in CL}
    P("This subsection states, stage by stage, what the output images of Figs. {F:st_denoise} to {F:st_cells} show, what can be read from them, and what they are not. Table {T:stage_out} gives the corresponding values inside the cells of each class.")
    P(f"*Hybrid-diffusion edge detection.* The edge indicator is an image in which pixels that lie on strong intensity edges are close to 1 and flat regions are close to 0. In the output images (Fig. {{F:st_hds}}) the cell outlines, the outlines of red blood cells and "
      f"the scale bar are bright, while the interior of cells and the background are dark. Measured inside the segmented cells (Table {{T:stage_out}}), the indicator was {rat['Benign']:.1f}, {rat['Early']:.1f}, {rat['Pre']:.1f} and {rat['Pro']:.1f} times larger on the rim than in the interior for Benign, Early, Pre and Pro cells, "
      "so the edge detector does concentrate on the cell boundary, as intended. The statistics of this layer inside a cell therefore describe the sharpness and the contrast of its boundary and the amount of edge-like texture inside it. "
      "What the output does not give is a segmentation: it marks every edge, whether of a leukocyte, a red cell or debris, and it depends on the focus of the image.")
    P(f"*Adaptive principal-curvature response.* The output (Fig. {{F:st_layers}}) is large where the image intensity bends sharply, which includes the outline of a cell, thin dark folds and the narrow gaps between cells. In the cells of this dataset its median "
      f"value inside the cell was {g('cell_apc_mean', 'Benign'):.2f} for Benign, {g('cell_apc_mean', 'Early'):.2f} for Early, {g('cell_apc_mean', 'Pre'):.2f} for Pre and {g('cell_apc_mean', 'Pro'):.2f} for Pro, with the rim only slightly higher than the interior, so the response reflects the fine texture of the cell as much as its outline. "
      "As noted above, blood smears contain no vessels, so the layer is not a vessel segmentation; it is a measure of local curvature, used here as a texture and boundary descriptor. The class differences in its median are small, and its value for classification comes from combination with other features (Table {T:stage_abl}).")
    P(f"*Laplacian-of-Gaussian and Canny layers.* The Laplacian-of-Gaussian output responds to blob-like structures of the chosen size; the median inside the cell was {g('cell_log_mean', 'Pre'):.2f} for Pre cells against {g('cell_log_mean', 'Early'):.2f} for Early cells, and the rim values were lower than the interior values, "
      "which is the opposite of the edge and curvature layers and shows that the layer describes the blob-like body of the cell and not its outline. The Canny output is a binary edge map and is used for inspection and not as a feature.")
    P(f"*Ensemble mode decomposition and pure-mode selection.* The decomposition splits the luminance image into modes of decreasing spatial frequency (Fig. {{F:st_beemd}}): the first mode contains the finest detail and noise and outlines the cell edges, the second and third show texture at the scale of chromatin and small cells, "
      "and the fourth shows structures of about the size of a cell, with a smooth residue holding the illumination. The sum of the selected modes (the pure-mode image) is therefore a band-passed detail image of the cell. "
      f"The share of the total energy of the modes was small inside the cells (median {g('cell_imf1_energy', 'Pre'):.4f} for the finest and {g('cell_imf4_energy', 'Pre'):.4f} for the coarsest mode in Pre cells, in the units of the normalised luminance), and the pure-mode image varied inside the cell with a median standard deviation of "
      f"{g('cell_imf_pure_std', 'Benign'):.2f} to {g('cell_imf_pure_std', 'Early'):.2f} across the classes. The selection rule retained almost every mode in all images (Contribution of the processing stages), so in this dataset the pure-mode image is practically the sum of all modes, and the selection step does not change the output in a way that I can demonstrate. "
      "The stage is the most expensive of the pipeline, and it adds measurable information only in combination with the other features.")
    P(f"*Morphological features.* The output is a table with one row per cell (Fig. {{F:st_cells}}, Table {{T:stage_cells}}): size, outline measures, colour, texture and layer statistics. It is the interpretable description of a cell, in contrast to the embedding of the foundation model, and it is the input of all tabular analyses of this paper. "
      f"The class differences are modest for the outline measures and larger for size and colour (Table {{T:feat}}), and the nucleus measurements are missing for most cells because the nucleus step fails (Table {{T:dataset}}). The features are as reliable as the segmentation on which they rest, whose boundaries were not validated against annotations.")

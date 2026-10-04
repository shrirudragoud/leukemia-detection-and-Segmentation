SE = json.loads((R / "stage_examples.json").read_text())
SA = json.loads((R / "stage_ablation.json").read_text()) if (R / "stage_ablation.json").exists() else None
EX = SE["images"]
SPI = SE["seconds_per_image"]
BE = SE["beemd_example"]

Fg("st_stain", "paper/figures/stage_1_stain.png", f"Stage 1, stain normalisation. Top: input images (one per class: {', '.join(EX[k] for k in CL)}). Bottom: the same images after Reinhard normalisation in Lab space with foreground statistics.", 6.2)
Fg("st_denoise", "paper/figures/stage_2_denoise.png", f"Stage 2, denoising of the luminance of {EX['Early']} (top: full image; bottom: zoom on the central region). From left to right: input, bilateral filter, stabilised HDS diffusion "
   f"({SE['hds_iterations_example']} iterations), and the absolute difference between input and HDS output (amplified four times). Diffusion smooths flat regions and leaves the cell boundaries.", 6.2)
Fg("st_hds", "paper/figures/stage_3_hds_edge.png", "Stage 3, HDS edge indicator. Top: luminance of one image per class. Bottom: the edge indicator 1 - mean diffusivity of the denoised image (0 in flat regions, close to 1 across strong edges), on a fixed scale.", 6.2)
Fg("st_layers", "paper/figures/stage_4_layers.png", f"Stage 4, response layers of {EX['Early']}. From left to right: denoised grey image, adaptive principal-curvature (APC) response, Laplacian-of-Gaussian (LoG) response and Canny edges.", 6.2)
Fg("st_beemd", "paper/figures/stage_5_beemd.png", f"Stage 5, BEEMD and pure-IMF selection for the luminance of {EX['Early']}. The four intrinsic mode functions (IMFs) are ordered from fine to coarse; the title of each gives the "
   "characteristic spatial period, the share of the detail energy and whether the selection rule keeps the mode. Also shown: the residue, the sum of the selected (pure) modes, and the difference between the input and the sum of all modes and the residue, which is zero up to rounding.", 6.2)
Fg("st_seg", "paper/figures/stage_6_segmentation.png", "Stage 6, cell and nucleus segmentation for one image per class. Top: smoothed a* score. Middle: cell instances after watershed splitting (random colours; border-touching cells included). Bottom: cell (green) and nucleus (yellow) outlines on the denoised image.", 6.2)
Fg("st_clean", "paper/figures/stage_7_neutralise.png", "Stage 7, shortcut neutralisation. Top: stain-normalised images. Bottom: the same images with everything outside the cells replaced by a neutral fill and the cells white-balanced.", 6.2)
Fg("st_crops", "paper/figures/stage_7b_crops.png", f"Stage 7, outputs for {EX['Early']}: the neutralised image, its grey version, and the isolated 112 x 112 crops of the first four cells that are passed to the encoder.", 6.2)
Fg("st_cells", "paper/figures/stage_8_cells.png", f"Stage 8, cells of {EX['Early']} numbered as in Table {{T:stage_cells}}.", 3.6)

_ex = SE["example_cells"]
T("stage_cells", f"Stage 8: per-cell measurements of the cells of {EX['Early']} (numbered as in Fig. {{F:st_cells}}). Columns follow the feature definitions of Appendix III; the last four columns are statistics of the "
   "response layers inside the cell.",
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
    P("An earlier version of this work was reviewed, and one review observed that the code shared with it performed deep-learning classification only and did not implement the stages named in the "
      "project description: HDS edge detection, APC (adaptive principal curvature) structure detection, BEEMD decomposition, selection of pure intrinsic mode functions (IMFs), and "
      "morphological feature extraction. The present version implements every one of these stages, tests each of them with automated tests, runs them on all "
      f"{N_IMG} images and reports their input and output. Table {{T:stages}} lists the stages and says which role each plays in the experiments; the following paragraphs describe each stage, with an example "
      "of its input and output on real images of the dataset.")
    P(f"Two statements about scope are needed for accuracy. First, the pixel input of the headline model is the neutralised cell crop of stage 9, so the response layers and the HDS and BEEMD outputs "
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

@section("methods_extra")
def methods_extra():
    S("Formal description of the preprocessing")
    P("This section gives the operations of the Materials and Methods in formal terms so that they can be reproduced independently of the code. Images are "
      "denoted by I, the CIE Lab representation by (L, a, b) with L in [0, 100] and a and b signed, and the foreground set of an image by F.")
    P("*Reinhard normalisation.* For each image, the set of tissue pixels V contains those with L >= 8 (excluding near-black annotation overlays such as the scale bar). "
      "The foreground F is the subset of V whose lightness is below the Otsu threshold of L computed over V. If V covers less than half of the image, the lightness has a "
      "standard deviation below 1, or F covers less than the minimum fraction (0.02) of the image, the image is left unchanged and flagged. Otherwise the per-channel mean "
      "m(I) and standard deviation s(I) over F are computed. With reference statistics m* and s* (the mean of the per-image statistics over a sample of 20 images), each channel "
      "ch is mapped by")
    BODY.append({"t": "eq", "lines": ["ch′ = (ch − m_ch(I)) × g_ch + m*_ch,    g_ch = clip( s*_ch / max(s_ch(I), 10⁻³), 0.25, 4.0 )"]})
    P("and L' is clipped to [0, 100]. The gain g is clipped to the interval [0.25, 4] so that an unusual image cannot be stretched without bound.")
    P("*Stabilised diffusion.* Let f be the observed single-channel image and u the iterate, with u_0 = f. For the four neighbours k (north, south, west and east) let d_k be the difference "
      "between the neighbour and the central pixel. The diffusivity is a convex combination of a Perona-Malik term and a Charbonnier total-variation term,")
    BODY.append({"t": "eq", "lines": ["c(s) = w / (1 + (s/h)²) + (1 − w) · β / √(s² + β²),     0 < c(s) ≤ 1",
                                      "u_{t+1} = u_t + Δt · ( Σ_k c(|d_k|) · d_k  −  λ · (u_t − f) )"]})
    P(f"with w = {HDS['hds_config']['hybrid_weight']}, h = {HDS['hds_config']['h']}, beta = {HDS['hds_config']['beta']}, lambda = {HDS['hds_config']['lam']}, "
      f"dt = {HDS['hds_config']['dt']} and at most {HDS['hds_config']['iterations']} iterations; iteration stops when the mean absolute update falls below "
      f"{HDS['hds_config']['tol']}. Because c is at most 1 and dt is below 1/(4 + lambda) = 0.23, each update is a convex combination of the neighbouring values and the observed value, so "
      "the iterate cannot leave the range of the input. The unmodified scheme of the original repository has a conduction coefficient with a term of size 1/epsilon, which "
      "violates the stability bound and explains the loss of signal-to-noise ratio in the benchmark.")
    P("*Foreground, nucleus and instance segmentation.* Let a_s be the a* channel smoothed with a Gaussian of standard deviation 1 pixel. The cell mask is a_s > max(Otsu(a_s), 8). "
      "The mask is opened with a disk of radius 1, components smaller than 120 pixels are removed, and holes are filled. Each connected component is split by a "
      "marker-controlled watershed on the negative smoothed distance transform, with markers at the h-maxima (h = 1.5) of the distance transform; a split is accepted only "
      "if every resulting piece has at least 120 pixels. Within each cell, the nucleus is sought only in the interior (the cell eroded by 2 pixels): Otsu's threshold is computed on the "
      "smoothed lightness of the interior, and a nucleus is accepted only if the mean lightness of the two classes differs by at least 8 units, the nucleus "
      "has at least 30 pixels and at least 5% of the cell area. When any condition fails, no nucleus is reported for that cell.")
    P("*Neutralisation.* Let K be the cell mask dilated by the configured number of pixels. The background colour b is the median Lab value of the brighter half (by lightness) of the "
      "pixels outside K, if at least a minimum number of such pixels exist. The lightness of every pixel is multiplied by a gain, clipped to a fixed interval, that brings the background lightness to "
      "the neutral fill lightness, and the median a* and b* of the background are subtracted from the a* and b* channels. Pixels outside K are then set to the neutral colour (fill "
      "lightness, a = b = 0). The grayscale variant keeps only L/100.")
    P("*Cell crops.* For every cell, a square window is cut around the bounding box of the cell, enlarged by a margin of 25% on each side. Where the window leaves the image, it "
      "is padded with the neutral fill colour. When the isolation option is on, all pixels except those of this cell, dilated by 2 pixels, are replaced by the fill colour, so that neighbouring cells do not appear "
      "in the crop. The window is resized to 128 x 128 pixels for storage and to 112 x 112 pixels for the encoder (a multiple of the 14-pixel patch size of the vision transformer).")
    S("Bag construction and the attention head")
    P("Each image is a *bag* of cells. During training, at most 24 cells per image (16 in the GPU configurations) are drawn at random; at evaluation, up to 64 cells are taken in order of their label "
      "number. For the cells h_1, ..., h_K of a bag, with embeddings of dimension D (384 for DinoBloom-S), the head computes z_k = GELU(W h_k + b) with 256 hidden units and dropout 0.2, and the "
      "attention weights")
    BODY.append({"t": "eq", "lines": ["a_k = exp( wᵀ [ tanh(V z_k) ⊙ σ(U z_k) ] ) / Σ_j exp( wᵀ [ tanh(V z_j) ⊙ σ(U z_j) ] )",
                                      "bag = Σ_k a_k z_k ;   logits = C · bag"]})
    P(f"where the softmax runs over the valid cells of the bag only and the attention dimension is 128 {c('ilse2018')}. Because the weights sum to 1, the bag representation does not grow with the "
      "number of cells, which matters because the number of cells per image differs strongly between classes (Table {T:dataset}). The head has a small number of trainable parameters compared with the encoder, which is frozen.")
    S("Training objective and optimisation")
    P(f"The loss is the cross-entropy with label smoothing of 0.1 and with class weights proportional to the inverse square root of the class frequency of the training images. "
      f"Optimisation uses AdamW {c('loshchilov2019')} (an improved variant of Adam {c('kingma2015')}) with weight decay 0.05, gradient clipping at a norm of 1, a linear warm-up over the first 10% "
      "of the steps and a cosine decay of the learning rate over the 30-epoch budget (Fig. {F:training}c; training stopped earlier in every fold). For the frozen encoder, embeddings of all cells are computed once and cached, so no augmentation is applied; "
      "the augmentation modules (flips, 90-degree rotations, scale jitter, stain-colour jitter restricted to non-fill pixels, random grayscale and blur) are used only when the encoder is adapted. "
      "Training stops when the validation macro-F1 has not improved for 8 epochs, and the weights of the best epoch are used. The temperature of the final softmax is fitted by "
      "minimising the validation negative log-likelihood (Methods, Evaluation measures).")
    S("Fold construction in detail")
    P("Let each image i have a class y_i and a capture number n_i, the number in its file name. For each class, the images are sorted by n and cut into five consecutive segments of nearly equal size. Fold k uses segment k of "
      "every class as the test set and the segment two positions later (modulo five) as the validation set. The training set contains the remaining images, from which every image of the same class "
      "whose capture number is within 37 of any test or validation image of that class is removed (the embargo). Consecutive segments therefore share at most two boundaries per class and fold, and "
      "images in a boundary region are excluded. The random-block scheme used for comparison draws 37-image blocks (merging duplicates) and assigns whole blocks to folds with class stratification.")
    P("For the purge analysis, the training images that lie within a distance e of any test image of the same class are removed from a random-block training set, for e in {0, 37, 75, 150}. The control removes, "
      "for each class, the same number of training images chosen uniformly at random. For the gap analysis, the embargo of the contiguous scheme is increased to e in {0, 37, 75, 150, 225, 300}, and the control again removes the same number of images per class at random.")
    S("Evaluation measures")
    P(f"Balanced accuracy is the mean of the per-class recalls and is the primary measure because the classes are unequal in size {c('sokolova2009')}. Macro-F1 is the unweighted mean over classes of the harmonic "
      "mean of precision and recall. The AUROC is the macro average of one-vs-rest areas under the receiver operating characteristic computed from the rank statistic. The expected calibration "
      "error (ECE) is the weighted mean absolute difference between accuracy and mean confidence over 15 equal-width confidence bins. Sensitivity and specificity for the benign-versus-malignant "
      "decision treat all predictions of Early, Pre or Pro as positive. All measures are computed from the out-of-fold predictions of the five folds pooled, and also within each fold.")
    P(f"Intervals use the cluster bootstrap over the {CV['n_groups']} groups with 1000 resamples and the 2.5 and 97.5 percentiles; a paired comparison resamples the same groups for both variants. "
      "The p value of a paired comparison is twice the smaller tail proportion of the bootstrap difference and cannot be smaller than the reciprocal of the number of resamples; Holm's correction "
      "is applied over the family of comparisons within each table.")
    S("Hyperparameters of the planned fine-tuning experiments")
    P("Eight fine-tuning configurations were prepared. They share the data, fold and optimisation settings and differ in the encoder, the adaptation method or the input, as listed in Table A13 "
      "(Appendix VIII). The adaptation uses low-rank updates of rank 8 and scaling 16 on the query-key-value and output projections of every attention block of the vision transformers "
      f"{c('hu2022')}; the full fine-tuning configurations update all weights of the convolutional network with a smaller learning rate than the head.")
    S("Computational environment")
    P(f"All reported analyses were run on a CPU. Preprocessing of the {N_IMG} images took {DS['run_seconds'] / 60:.1f} minutes. Cross-validation of the headline model took {secs / 60:.0f} minutes for the "
      "five folds (the encoder embeddings are computed once and cached). The analyses were run in an isolated container without a graphics processor; the corresponding code paths for graphics processors were "
      "tested only for correctness of the logic on a CPU, and the GPU configurations have not been run.")

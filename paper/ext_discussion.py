@section("discussion_extra")
def discussion_extra():
    S("Interpretation of the headline result")
    P(f"The balanced accuracy of {f3(ba)} with an interval of {ci(bal)} is a measurement of how well four classes of this dataset can be separated by a frozen encoder and a small head when "
      "the test images come from capture ranges that are not adjacent to the training images. It is not an estimate of accuracy for patients in general. The interval is the sampling uncertainty over "
      f"the {CV['n_groups']} groups, and it does not include the uncertainty that comes from the unknown relation between groups and patients, from the confounding of class and session, or from the choice "
      "of the dataset. Where this paper uses the word accuracy without qualification, it refers to this in-dataset quantity.")
    P(f"Two features of the result are worth stating. First, the frozen encoder reached a high accuracy without any adaptation to the dataset, with a head that has about 166,000 trainable parameters; training loss and validation scores plateaued within a few epochs (Appendix VII), and the per-fold balanced accuracy had a standard deviation of "
      f"{f3(CV['per_fold_sd']['balanced_accuracy'])} (range {f3(min(p['test']['balanced_accuracy'] for p in pf))}-{f3(max(p['test']['balanced_accuracy'] for p in pf))}). Second, the high accuracy extends to the hand-made features only partially ({f3(dsE['balanced_accuracy'])}), which is consistent with the encoder containing "
      "information beyond shape, texture and curvature as I measured them. What this information is cannot be determined from the present experiments. It may include fine nuclear and cytoplasmic texture "
      "that is relevant to the disease, and it may include colour and focus characteristics of the sessions.")
    S("Relation to other work")
    P(f"Several studies on public leukemia datasets report evaluation on random image-level splits (I did not audit the literature systematically). The protocol of the present study differs, so its numbers are not "
      f"comparable with theirs and should not be ranked against them. The concern that random splits inflate results is shared by the recent benchmark of {n('albzour2026')}, and the broader literature on leakage "
      f"{c('kapoor2023', 'varoquaux2022')} describes the same failure in other fields. The contribution of the present analysis is a measurement for this dataset: the size of the overestimate from random block folds, and the "
      "proximity mechanism behind it, are quantified instead of assumed.")
    S("Consequences for users of the dataset")
    P("The finding that the background alone predicts the class has practical consequences. A classifier trained on the original images and evaluated on a random split can reach a high accuracy by "
      "recognising the session; such a classifier may fail on images from a laboratory with different staining or imaging; this was not tested here. I therefore recommend that studies "
      "using this dataset (1) report accuracy under a session-aware protocol, (2) report the accuracy that a classifier reaches from the background alone, (3) mask or neutralise the background if the goal "
      "is to measure cell morphology, and (4) test on an external dataset. The code released with this paper implements the first three, and the fourth requires data that were not part of this study.")
    S("Threats to validity")
    P("*Internal validity.* The groups are defined by capture number, which I showed to be informative about similarity, but which may not coincide with patients. If images of a single patient are spread over "
      "several capture ranges of the same class, leakage remains even in the contiguous design. The embargo reduces but cannot remove this possibility. The hyperparameters of the head were fixed before "
      "the evaluation and were not tuned on the test folds; early stopping and temperature scaling used the validation stretch only. The purge and gap analyses are single runs per setting and the "
      "differences they show are large relative to the differences between the random controls, but I did not quantify their sampling variation.")
    P("*Construct validity.* The labels correspond to stages of precursor B-cell leukemia as defined by the dataset authors, and the benign class consists of hematogones. A classifier that separates these classes "
      "does not necessarily detect leukemia in a general sample, which contains many other cell types and abnormalities. The four-class task is also a coarse description of a continuum.")
    P("*External validity.* One dataset from one laboratory was used. The acquisition cues identified here may not exist in other datasets, and the accuracy obtained for a different dataset may differ in either "
      "direction. I did not test the models on any external data.")
    P("*Statistical validity.* Group-level bootstrap intervals assume that groups are independent and exchangeable, and with 90 groups the intervals are themselves uncertain. The Holm correction was applied within each table, "
      "not across all analyses of the paper. Single-seed deterministic comparisons (the head with a fixed seed) do not reflect the variation of training with different initialisations.")
    S("Response to the review of the earlier version")
    P(f"The review of the earlier version observed that the shared code classified images with deep learning only. In this version every named stage runs on all {N_IMG} images and is shown on real images (Figs. {{F:st_stain}} to {{F:st_cells}}). "
      f"Their measured value is modest and uneven: the hand-made features of the stages together reached {f3(SA_FULL['balanced_accuracy'])}, well below the frozen encoder ({f3(dsA['balanced_accuracy'])}), and the "
      "pure-IMF selection rule kept almost every mode, so it does little. I report this as it is: the stages are implemented, verified and informative, but the foundation-model features carry most of the classification performance. "
      "The stages are most useful as interpretable descriptors and as a way to inspect what the images contain.")
    S("Ethical and clinical considerations")
    P("The images are from a public dataset released for research. I did not link them to any identifying information and I did not attempt to. The software is a research prototype, not a medical device, and "
      "nothing in this paper supports its use for diagnosis. A tool intended for clinical use would need prospective validation on the intended population, evaluation of subgroup performance and "
      "regulatory approval, none of which is addressed here.")
    S("Fine-tuning")
    if CUT:
        P("Fine-tuning was not performed, so this study cannot say whether adapting the encoder would help. Two outcomes would be informative in different ways: an improvement under the session-aware protocol would show that the "
          "frozen features were limiting, whereas an improvement only when the background is visible would point to the shortcut. Configurations that allow these two readings to be told apart were prepared (Appendix VIII) but not run.")
    else:
        P("Fine-tuning changes the question of the study because it lets the model adapt to the dataset. Two outcomes are possible and they would be informative in different ways. If adaptation of the encoder "
          "improves the accuracy under the session-aware protocol, the frozen features were limiting and the dataset contains additional morphological information. If it improves the accuracy mainly when the background is "
          "visible (g08) and not when it is neutralised (g01), the gain would be attributable to the shortcut. The experiments were designed so that these two readings can be told apart.")
        if N_SLOTS:
            P("**[[GPU: after the runs finish, replace this section by a discussion of which of the two readings the results support; cite Table {T:ft} and Fig. {F:ft_bars}. Delete this paragraph if no run is performed.]]**")
    S("Future work")
    P("The most valuable next step is evaluation on an external dataset acquired in a different laboratory, with the same preprocessing and without any adaptation to it. Public datasets with "
      "labelled leukemic and normal cells exist, and some of them provide patient identifiers that would also allow a validation of the capture-order proxy. A second step is the segmentation of nuclei, "
      "which failed for most cells here; a segmentation network trained on annotated nuclei would allow reliable nuclear measures. A third step is a prospective study of how the confidence of the "
      "classifier relates to the disagreement between observers. Finally, an analysis of the encoder with controlled perturbations of colour and background would complement the correlational audit by a causal one.")
    S("Conclusion")
    P("A reproducible, tested pipeline and a leakage-aware evaluation show that a frozen hematology foundation model separates the four classes of the studied dataset with a balanced accuracy of "
      f"{f3(ba)}, that a large part of the class information is accessible without any cell, and that random block evaluation overestimates accuracy through proximity in capture order. The numbers "
      "reported here are an optimistic, unvalidated estimate for new patients and laboratories. The practical message for the field is that the evaluation protocol and a shortcut audit must be part of every report of accuracy "
      "on datasets that lack patient identifiers.")

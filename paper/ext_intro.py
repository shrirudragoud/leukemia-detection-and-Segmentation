@section("intro_extra")
def intro_extra():
    S("Clinical background")
    P(f"Acute lymphoblastic leukemia is a malignancy of lymphoid progenitor cells (lymphoblasts) that accumulate in the bone marrow, suppress normal "
      f"blood formation and enter the circulation. It is the most common cancer of children {c('hunger2015')}. Current classification systems define the disease "
      f"by the lineage of the malignant cell (B-cell or T-cell precursor) and by genetic subtype {c('arber2016')}. Treatment depends strongly on the subtype and on "
      "the response to the first phase of therapy, so rapid and accurate recognition of the disease at presentation matters for the patient.")
    P("The diagnostic work-up combines several investigations. A complete blood count indicates abnormal numbers of blood cells. The peripheral blood smear, a "
      "drop of blood spread on a glass slide and stained with a Romanowsky-type stain, allows a hematologist to inspect the morphology of individual "
      "white blood cells under the microscope. Confirmation and subtyping rely on bone-marrow examination, flow cytometry for cell-surface markers, "
      "and cytogenetic and molecular tests. The smear is thus a screening step, not the final diagnosis, and image-based tools for the smear aim to support "
      "that screening step, for example by flagging smears that contain suspicious cells.")
    P("In a smear, lymphoblasts differ from mature lymphocytes and from normal immature cells in size, in the ratio of nucleus to cytoplasm, in the "
      "texture of the chromatin and in the presence of nucleoli. These differences are graded and overlap between classes, which is one reason why agreement "
      "between observers is imperfect and why an automated and reproducible description of cell appearance is attractive.")
    S("The dataset and its four classes")
    P(f"The dataset used in this study {c('aria2021', 'ghaderzadeh2022')} groups smear images into four classes. The Benign class contains images of hematogones, "
      "which are normal immature B-lymphocyte precursors that resemble lymphoblasts and are a classical source of confusion. The three malignant classes correspond to "
      "stages of precursor B-cell lymphoblasts, called Early (early pre-B), Pre (pre-B) and Pro (pro-B) in the dataset documentation. The images were "
      "acquired from patients and labelled by experts, according to the dataset description, using flow cytometry as the reference standard. "
      "I rely on that documentation and could not verify the labelling process independently.")
    P("Two practical properties of the dataset shape the analysis in this paper. First, the images are small (224 x 224 pixels) and each contains a handful of cells "
      "(on average about 11 per image), so the context available to a model is limited. Second, the dataset provides no table linking the images to patients, slides or "
      "acquisition sessions. Both properties are common among public datasets, and the second is the reason why evaluation design receives so much attention here.")
    S("Computer-aided analysis of blood smears")
    P(f"Early systems for leukemia detection used the classical pipeline of image analysis: colour-space conversion, thresholding or clustering to find "
      f"white blood cells, morphological operations and watershed transforms to separate touching cells {c('meyer1990')}, and hand-designed descriptors "
      f"of shape, colour and texture that were fed to a conventional classifier {c('labati2011')}. The Haralick texture features {c('haralick1973')} and measures "
      "such as the nucleus-to-cytoplasm ratio were typical. The strength of this approach is interpretability; its weakness is the dependence of every "
      "step on thresholds that must be tuned to a particular staining and imaging set-up.")
    P(f"Deep convolutional networks {c('krizhevsky2012', 'he2016')} replaced hand-designed descriptors by features learned from labelled images, and "
      f"efficient architectures {c('tan2019')} reduced the cost of training. Encoder-decoder networks {c('ronneberger2015')} and generalist cell segmentation "
      f"tools {c('stringer2021', 'schmidt2018')} addressed the segmentation step. Vision transformers {c('vaswani2017', 'dosovitskiy2021')} added global "
      "self-attention over image patches. For blood smears, large labelled collections such as the single-cell dataset of "
      f"{n('matek2019')} made it possible to train and test networks for the recognition of cell types and blasts.")
    P(f"Because labelled medical images are scarce, transfer learning is central: a network trained on a large source dataset is adapted to the target task "
      f"{c('pan2010')}. How well features transfer depends on the architecture and the source data {c('kornblith2019')}. Self-supervised pre-training "
      f"{c('caron2021', 'oquab2024')} produces general features without labels, and domain-specific pre-training on hematology images "
      f"{c('koch2024')} aims to capture the appearance of blood cells in particular. The simplest and most reproducible way to use such a model is to freeze it, "
      "compute an embedding for each image or cell once, and train only a small classifier on top.")
    S("Stain variation and normalisation")
    P(f"Smears are stained by hand or by automatic stainers, and the resulting colour depends on the stain batch, the staining time, the thickness of the smear "
      f"and the light source and camera. Differences in colour between laboratories can degrade the accuracy of classifiers that were trained elsewhere. "
      f"Colour normalisation methods map the colour distribution of an image to that of a reference. The method of {n('reinhard2001')} matches the mean and "
      f"standard deviation of each channel of a decorrelated colour space; methods based on stain separation {c('macenko2009', 'vahadane2016')} estimate the stain "
      f"vectors of each image. Augmentation that perturbs stain colour is an alternative or a complement, and {n('tellez2019')} reported that colour augmentation "
      "can matter more than normalisation for the robustness of classifiers in computational pathology.")
    S("Validity of evaluation in medical machine learning")
    P(f"A growing literature documents that reported performance in medical machine learning is often inflated by weaknesses of study design rather than by the models. "
      f"{n('varoquaux2022')} reviewed such failures in medical imaging, including small and biased datasets and evaluation on data that overlap with the training data. "
      f"{n('roberts2021')} found that most machine-learning studies of COVID-19 imaging were at high risk of bias and had methodological flaws that prevented clinical use. "
      f"{n('kapoor2023')} described leakage as a cause of failures of reproducibility across scientific fields and gave a taxonomy of leakage types, including the "
      "case in which the test set is not independent of the training set.")
    P(f"For image data, the independence of test and training sets requires that images from the same patient (or the same slide, or the same session) are never split "
      f"across them. When the dataset carries no patient identifier, a common practice is a random split of images, which does not satisfy this requirement. Related to this "
      f"is shortcut learning {c('geirhos2020')}: a model may rely on features that correlate with the label in the dataset, such as the source of the images, but that "
      f"have no causal relation to the disease. {n('degrave2021')} showed that classifiers of chest radiographs can learn such shortcuts, and that explanation methods can mislead "
      f"about it {c('adebayo2018')}. Blocked cross-validation is a recognised remedy where observations are dependent {c('roberts2017')}, and estimates of generalisation "
      f"error from cross-validation are themselves uncertain {c('kohavi1995', 'nadeau2003')}.")
    S("Contributions")
    P("This paper makes five contributions, all of which are supported by result files released with the code. (1) A documented and tested preprocessing pipeline for smear "
      "images with quantitative checks, including a stabilised diffusion denoiser and a procedure that isolates cells from their background. (2) A shortcut audit that measures "
      "how much class information is present in the background, the scale bar and the colour of cells. (3) A demonstration, with a purge analysis, that proximity in capture order "
      "carries class-relevant information and that random block folds overestimate accuracy. (4) A reproducible baseline with a frozen hematology foundation model, with "
      "intervals computed over groups and calibrated probabilities. (5) A protocol, implemented in code, for fine-tuning and encoder comparison under the same folds.")
    P("The remainder of the paper is organised as follows. The Materials and Methods give the data, the preprocessing, the shortcut audit, the evaluation design, the models "
      "and the statistical procedures. The Results report the measurements in the order in which they were made. The Discussion interprets the findings, states "
      "the limitations and proposes next steps. The appendices give per-fold results, definitions, the configurations and listings of the core algorithms, so that every reported "
      "number can be traced and every step repeated.")

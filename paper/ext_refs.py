ROLES = {
 "adebayo2018": "Sanity checks for saliency maps; basis of the model-randomisation test", "albzour2026": "Leakage-aware benchmark for leukemia classification (related concern)",
 "arber2016": "Classification of acute leukemia (clinical background)", "aria2021": "The public dataset analysed in this study", "bradski2000": "OpenCV library used in the pipeline",
 "caron2021": "Self-supervised vision transformers (background to DINOv2)", "degrave2021": "Shortcut learning in radiographic COVID-19 classifiers",
 "dosovitskiy2021": "Vision transformer architecture", "efron1979": "Bootstrap resampling", "frangi1998": "Multiscale vesselness filter (basis of the curvature layer)",
 "geirhos2020": "Concept of shortcut learning", "ghaderzadeh2022": "Description of the dataset and its classes", "guo2017": "Temperature scaling for calibration",
 "haralick1973": "Grey-level co-occurrence texture features", "he2016": "Residual convolutional network (ResNet)", "holm1979": "Holm adjustment for multiple comparisons",
 "hu2022": "Low-rank adaptation of network weights", "hunger2015": "Clinical background of childhood acute lymphoblastic leukemia", "ilse2018": "Gated attention pooling over instances",
 "kapoor2023": "Data leakage and reproducibility in machine-learning-based science", "kingma2015": "Adam optimiser (basis of AdamW)", "koch2024": "DinoBloom hematology foundation model (the encoder used)",
 "kohavi1995": "Estimating accuracy by cross-validation and bootstrap", "kornblith2019": "Transferability of pretrained image models", "kouzehkanan2022": "Raabin-WBC dataset (proposed external data)",
 "krizhevsky2012": "Deep convolutional networks for image classification", "labati2011": "Classical image analysis of leukemia smears (ALL-IDB)", "loshchilov2019": "AdamW optimiser",
 "macenko2009": "Stain-separation colour normalisation", "matek2019": "Single-cell blast recognition (myeloid) with convolutional networks", "meyer1990": "Watershed segmentation of touching cells",
 "mourya2019": "C-NMC challenge dataset (proposed external data)", "nadeau2003": "Corrected resampled t test for cross-validation comparisons", "oquab2024": "DINOv2 self-supervised vision transformer",
 "otsu1979": "Otsu threshold selection", "pan2010": "Survey of transfer learning", "paszke2019": "PyTorch deep-learning library", "pedregosa2011": "scikit-learn library (random forest, logistic regression)",
 "perona1990": "Anisotropic diffusion (basis of the hybrid diffusion)", "reinhard2001": "Colour normalisation by matching channel statistics", "roberts2017": "Blocked cross-validation for dependent data",
 "roberts2021": "Methodological flaws in COVID-19 imaging machine learning", "ronneberger2015": "U-Net segmentation network (background)", "schmidt2018": "StarDist cell detection (background)",
 "selvaraju2017": "Grad-CAM saliency maps", "sokolova2009": "Classification performance measures (balanced accuracy)", "stringer2021": "Cellpose segmentation (background)",
 "tan2019": "EfficientNet architecture", "tellez2019": "Effect of stain normalisation and augmentation on classifiers", "tomasi1998": "Bilateral filtering",
 "vahadane2016": "Structure-preserving stain normalisation", "vanderwalt2014": "scikit-image library", "varoquaux2022": "Methodological failures in medical-imaging machine learning",
 "vaswani2017": "Transformer architecture and attention", "wightman2019": "timm library of image models"}
IDS = {"wightman2019": "doi:10.5281/zenodo.4414861", "aria2021": "doi:10.34740/KAGGLE/DSV/2175623", "mourya2019": "doi:10.7937/TCIA.2019.DC64I46R", "dosovitskiy2021": "arXiv:2010.11929", "hu2022": "arXiv:2106.09685",
       "loshchilov2019": "arXiv:1711.05101", "oquab2024": "arXiv:2304.07193", "kingma2015": "arXiv:1412.6980", "albzour2026": "arXiv:2606.24944", "kohavi1995": "IJCAI 1995 proceedings", "holm1979": "JSTOR 4615733",
       "bradski2000": "Dr. Dobb's Journal 25(11)", "hunger2015": "doi:10.1056/NEJMra1400972", "pan2010": "doi:10.1109/TKDE.2009.191", "roberts2021": "doi:10.1038/s42256-021-00307-0", "meyer1990": "doi:10.1016/1047-3203(90)90014-M"}
VIA = {"hunger2015": "Crossref", "pan2010": "Crossref", "roberts2021": "Crossref (authors shortened)", "meyer1990": "Crossref", "aria2021": "Kaggle dataset page", "wightman2019": "Zenodo DOI", "mourya2019": "TCIA DOI (search result)",
       "kohavi1995": "IJCAI PDF (7 pages)", "dosovitskiy2021": "arXiv", "hu2022": "arXiv", "loshchilov2019": "arXiv", "oquab2024": "arXiv", "kingma2015": "arXiv", "albzour2026": "arXiv", "holm1979": "not verified online",
       "bradski2000": "not verified online"}
_CR = json.loads((R / "ref_check.json").read_text())
_MT = json.loads((R / "ref_check_meta.json").read_text())
_CP = json.loads((ROOT / "paper" / "cite_pages.json").read_text()) if (ROOT / "paper" / "cite_pages.json").exists() else {}


def _pages(pgs):
    pgs = sorted(set(pgs))
    out, i = [], 0
    while i < len(pgs):
        j = i
        while j + 1 < len(pgs) and pgs[j + 1] == pgs[j] + 1:
            j += 1
        out.append(str(pgs[i]) if i == j else f"{pgs[i]}-{pgs[j]}")
        i = j + 1
    return ", ".join(out) or "0"


@section("appendix_refs")
def appendix_refs():
    keys = sorted([k for k in REFS if k in USED], key=lambda k: REFS[k][1].lower())
    PB()
    H("Appendix XIV. Citation index")
    P("The citations follow the name-year style of the guide: one or two authors are named, three or more are given as the first author and \"et al.\", several citations in one parenthesis are separated by semicolons, and the "
      "full reference is in Literature Cited, in alphabetical order with a hanging indent. Table A17 lists every reference that is cited, what it is cited for in this paper, and the pages of the paper on which it is cited. "
      f"The {len(REFS) - len(keys)} sources that were consulted but are not cited are not listed in Literature Cited.", noindent=True)
    BODY.append({"t": "table", "num": "A17", "caption": "Citation index: reference, use in this paper and pages on which it is cited.",
                 "cols": [{"w": 2.6, "align": "left"}, {"w": 4.8, "align": "left"}, {"w": 1.6, "align": "left"}], "header": ["Reference", "Cited for", "Pages"],
                 "rows": [[REFS[k][0], ROLES.get(k, ""), _pages(_CP.get(k, []))] for k in keys], "foot": None})
    PB()
    H("Appendix XV. Reference verification record")
    P("Every reference was compared with an external record on 4 October 2026: author list, year, title, venue, volume and pages. Table A18 gives the identifier of the record and the source that was used. "
      "\"Crossref\" means the DOI registration agency's record; \"proceedings page\" means the publisher page of the conference; \"arXiv\" means the preprint page. Entries marked \"not verified online\" are standard citations whose "
      "metadata could not be reached from the computing environment and should be checked against the original source.", noindent=True)
    rows = []
    for k in keys:
        ident = IDS.get(k) or (f"doi:{_CR[k]['doi']}" if _CR.get(k, {}).get("found") and _CR[k].get("doi") else "-")
        via = VIA.get(k) or ("Crossref" if _CR.get(k, {}).get("found") else (f"proceedings page ({_MT[k]['url'].split('/')[2]})" if k in _MT and _MT[k].get("title") else "-"))
        rows.append([REFS[k][0], ident, via])
    BODY.append({"t": "table", "num": "A18", "caption": "Reference verification record.", "cols": [{"w": 2.6, "align": "left"}, {"w": 3.8, "align": "left"}, {"w": 2.6, "align": "left"}],
                 "header": ["Reference", "Identifier", "Verified against"], "rows": rows, "foot": None})

# Reference verification (programmatic, with sources)

Checked on 2026-10-04 against Crossref (DOI records), publisher/proceedings pages (PMLR, NeurIPS, JMLR, CVF), arXiv and Zenodo/Kaggle pages. Scripts: `scripts/verify_refs.py`, `scripts/verify_refs_meta.py`; raw results: `docs/results/ref_check*.json`.

## Confirmed (60 entries)
- adebayo2018: publisher/proceedings page (proceedings.neurips.cc): title and authors match
- albzour2026: publisher/proceedings page (arxiv.org): title and authors match
- arber2016: Crossref DOI 10.1182/blood-2016-03-643544: title and authors match
- aria2021: Kaggle dataset page (DOI 10.34740/KAGGLE/DSV/2175623; publication citation matches)
- benavoli2017: publisher/proceedings page (jmlr.org): title and authors match
- bhuiyan2008: Crossref DOI 10.1155/2008/728356: title and authors match
- caron2021: publisher/proceedings page (openaccess.thecvf.com): title and authors match
- degrave2021: Crossref DOI 10.1101/2020.09.13.20193565: title and authors match
- demsar2006: publisher/proceedings page (www.jmlr.org): title and authors match
- dosovitskiy2021: arXiv 2010.11929 (12 authors match)
- efron1979: Crossref DOI 10.1214/aos/1176344552: title and authors match
- efron1993: Crossref DOI 10.1201/9780429246593: title and authors match
- field2007: Crossref DOI 10.1111/j.1467-9868.2007.00593.x: title and authors match
- frangi1998: Crossref DOI 10.1007/bfb0056195: title and authors match
- geirhos2020: Crossref DOI 10.1038/s42256-020-00257-z: title and authors match
- ghaderzadeh2022: Crossref DOI 10.1002/int.22753: title and authors match
- guo2017: publisher/proceedings page (proceedings.mlr.press): title and authors match
- haralick1973: Crossref DOI 10.1109/tsmc.1973.4309314: title and authors match
- he2016: publisher/proceedings page (openaccess.thecvf.com): title and authors match
- hu2022: arXiv 2106.09685 (8 authors match)
- huang1998: Crossref DOI 10.1098/rspa.1998.0193: title and authors match
- hunger2015: Crossref DOI 10.1056/NEJMra1400972
- ilse2018: publisher/proceedings page (proceedings.mlr.press): title and authors match
- kapoor2023: Crossref DOI 10.1016/j.patter.2023.100804: title and authors match
- kingma2015: publisher/proceedings page (arxiv.org): title and authors match
- koch2024: publisher/proceedings page (arxiv.org): title and authors match
- kohavi1995: IJCAI PDF: 7 pages, 1137-1143 consistent
- kornblith2019: publisher/proceedings page (openaccess.thecvf.com): title and authors match
- kouzehkanan2022: Crossref DOI 10.1038/s41598-021-04426-x: title and authors match
- krizhevsky2012: publisher/proceedings page (proceedings.neurips.cc): title and authors match
- labati2011: Crossref DOI 10.1109/icip.2011.6115881: title and authors match
- loshchilov2019: arXiv 1711.05101
- macenko2009: Crossref DOI 10.1109/isbi.2009.5193250: title and authors match
- matek2019: Crossref DOI 10.1038/s42256-019-0101-9: title and authors match
- meyer1990: Crossref DOI 10.1016/1047-3203(90)90014-M
- mourya2019: TCIA DOI 10.7937/tcia.2019.dc64i46r (seen in earlier search; Crossref title query did not return it)
- nadeau2003: Crossref DOI 10.1023/a:1024068626366: title and authors match
- oquab2024: arXiv 2304.07193 (26 authors match)
- otsu1979: Crossref DOI 10.1109/tsmc.1979.4310076: title and authors match
- pan2010: Crossref DOI 10.1109/TKDE.2009.191
- paszke2019: publisher/proceedings page (proceedings.neurips.cc): title and authors match
- pedregosa2011: publisher/proceedings page (jmlr.org): title and authors match
- perona1990: Crossref DOI 10.1109/34.56205: title and authors match
- reinhard2001: Crossref DOI 10.1109/38.946629: title and authors match
- roberts2017: Crossref DOI 10.1111/ecog.02881: title and authors match
- roberts2021: Crossref DOI 10.1038/s42256-021-00307-0 (54 authors; list shortened with "and others")
- ronneberger2015: Crossref DOI 10.1007/978-3-319-24574-4_28: title and authors match
- schmidt2018: Crossref DOI 10.1007/978-3-030-00934-2_30: title and authors match
- selvaraju2017: Crossref DOI 10.1109/iccv.2017.74: title and authors match
- sokolova2009: Crossref DOI 10.1016/j.ipm.2009.03.002: title and authors match
- stringer2021: Crossref DOI 10.1101/2020.02.02.931238: title and authors match
- tan2019: publisher/proceedings page (proceedings.mlr.press): title and authors match
- tellez2019: Crossref DOI 10.1016/j.media.2019.101544: title and authors match
- tomasi1998: Crossref DOI 10.1109/iccv.1998.710815: title and authors match
- vahadane2016: Crossref DOI 10.1109/tmi.2016.2529665: title and authors match
- vanderwalt2014: Crossref DOI 10.7287/peerj.preprints.336v1: title and authors match
- varoquaux2022: Crossref DOI 10.1038/s41746-022-00592-y: title and authors match
- vaswani2017: publisher/proceedings page (proceedings.neurips.cc): title and authors match
- wightman2019: Zenodo DOI 10.5281/zenodo.4414861
- wu2009: Crossref DOI 10.1142/s1793536909000047: title and authors match

## Corrections applied after the check
- koch2024: added MICCAI 2024 proceedings title and pages 520-530 (Crossref).
- caron2021: pages changed to 9630-9640 (IEEE/Crossref; the CVF open-access copy prints 9650-9660).
- vanderwalt2014: added "and the scikit-image contributors" (Crossref author list).
- haralick1973: volume given as SMC-3 (Crossref).
- labati2011 and bhuiyan2008 titles (earlier pass).

## Not verifiable online here (check by hand)
- bradski2000
- holm1979
- holm1979 (Scand J Stat 6(2):65-70, JSTOR) and bradski2000 (Dr Dobb's J Softw Tools 25(11):120-125): no open metadata source reachable; both are standard citations.
- roberts2021 lists the first 10 of 54 authors followed by "and others" (the guide asks for all authors; complete from the DOI page if required).
- Entries cited as arXiv preprints or proceedings without DOI follow the source page; page numbers for proceedings follow the publisher version.

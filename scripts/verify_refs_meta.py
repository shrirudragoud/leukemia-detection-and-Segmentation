"""Second reference check: read citation_* meta tags (Highwire/Google Scholar) from the publisher / proceedings / archive page of each entry that Crossref does not cover.
-> docs/results/ref_check_meta.json"""
import json
import re
import urllib.request
from html import unescape

URLS = {
    "guo2017": "https://proceedings.mlr.press/v70/guo17a.html", "ilse2018": "https://proceedings.mlr.press/v80/ilse18a.html",
    "tan2019": "https://proceedings.mlr.press/v97/tan19a.html",
    "vaswani2017": "https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
    "krizhevsky2012": "https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html",
    "paszke2019": "https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html",
    "adebayo2018": "https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html",
    "demsar2006": "https://www.jmlr.org/papers/v7/demsar06a.html", "benavoli2017": "https://jmlr.org/papers/v18/16-305.html",
    "pedregosa2011": "https://jmlr.org/papers/v12/pedregosa11a.html",
    "dosovitskiy2021": "https://openreview.net/forum?id=YicbFdNTTy", "hu2022": "https://openreview.net/forum?id=nZeVKeeFYf9",
    "loshchilov2019": "https://openreview.net/forum?id=Bkg6RiCqY7", "oquab2024": "https://openreview.net/forum?id=a68SUt6zFt",
    "kornblith2019": "https://openaccess.thecvf.com/content_CVPR_2019/html/Kornblith_Do_Better_ImageNet_Models_Transfer_Better_CVPR_2019_paper.html",
    "kingma2015": "https://arxiv.org/abs/1412.6980", "albzour2026": "https://arxiv.org/abs/2606.24944", "koch2024": "https://arxiv.org/abs/2404.05022",
    "kohavi1995": "https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf", "he2016": "https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
    "caron2021": "https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html",
    "zenodo_dinobloom": "https://zenodo.org/records/10908163", "aria2021": "https://www.kaggle.com/datasets/mehradaria/leukemia",
}
out = {}
for k, u in URLS.items():
    try:
        html = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 refcheck"}), timeout=25).read()[:400000].decode("utf8", "ignore")
    except Exception as e:
        out[k] = {"url": u, "error": str(e)[:80]}
        print(k, "ERROR", str(e)[:60]); continue
    tags = {}
    for m in re.finditer(r'<meta\s+name="(citation_[a-z_]+|og:title|DC\.[A-Za-z.]+)"\s+content="([^"]*)"', html):
        tags.setdefault(m.group(1), []).append(unescape(m.group(2)))
    for m in re.finditer(r'<meta\s+(?:name|property)="(og:title|og:description)"\s+content="([^"]*)"', html):
        tags.setdefault(m.group(1), []).append(unescape(m.group(2)))
    lic = re.findall(r"(Creative Commons[^<]{0,60}|CC[- ]BY[^<]{0,20})", html)[:2]
    out[k] = {"url": u, "title": tags.get("citation_title", tags.get("og:title", [""]))[0][:110], "authors": "; ".join(tags.get("citation_author", []))[:260],
              "date": (tags.get("citation_publication_date") or tags.get("citation_date") or [""])[0], "venue": (tags.get("citation_conference_title") or tags.get("citation_journal_title") or [""])[0],
              "volume": (tags.get("citation_volume") or [""])[0], "pages": f"{(tags.get('citation_firstpage') or [''])[0]}-{(tags.get('citation_lastpage') or [''])[0]}", "license": lic}
    print(k, "|", out[k]["title"][:50], "|", out[k]["date"], out[k]["venue"][:40], out[k]["volume"], out[k]["pages"], "|", out[k]["authors"][:80], out[k]["license"])
json.dump(out, open("docs/results/ref_check_meta.json", "w"), indent=1)

"""Check every entry of paper/refs.py against Crossref (title match, authors, year, container, volume, issue, pages).
Output: docs/results/ref_check.json and a printed table. No entry is changed automatically."""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

sys.path.insert(0, "paper")
from refs import REFS  # noqa: E402

HDR = {"User-Agent": "leukemia-paper-refcheck/1.0 (mailto:shridharrudragoud6@gmail.com)"}


def get(url):
    for _ in range(3):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=20).read())
        except Exception:
            time.sleep(1.5)
    return None


norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())  # noqa: E731
out = {}
for key, (label, cite) in sorted(REFS.items()):
    m = re.match(r"^(.*?)\. (\d{4})\. (.*)$", cite)
    authors, year, rest = m.groups()
    title = rest.split(". ")[0]
    first = authors.split(",")[0].split(" ")[0]
    q = urllib.parse.urlencode({"query.bibliographic": f"{title} {first}", "rows": 3, "select": "DOI,title,author,issued,container-title,volume,issue,page,type,article-number"})
    d = get("https://api.crossref.org/works?" + q)
    best = None
    for it in (d or {}).get("message", {}).get("items", []):
        t = (it.get("title") or [""])[0]
        r = SequenceMatcher(None, norm(t), norm(title)).ratio()
        if best is None or r > best[0]:
            best = (r, it)
    rec = {"label": label, "mine": cite, "found": False}
    if best and best[0] > 0.85:
        r, it = best
        a = it.get("author", [])
        rec.update(found=True, title_ratio=round(r, 3), doi=it.get("DOI"), n_authors=len(a),
                   authors=", ".join(f"{x.get('family', '')} {''.join(p[0] for p in x.get('given', '').replace('-', ' ').split())}" for x in a),
                   year=(it.get("issued", {}).get("date-parts") or [[None]])[0][0], container=(it.get("container-title") or [""])[0],
                   volume=it.get("volume"), issue=it.get("issue"), pages=it.get("page") or it.get("article-number"), type=it.get("type"))
        rec["year_ok"] = str(rec["year"]) == year
        rec["pages_in_mine"] = bool(rec["pages"]) and str(rec["pages"]).replace("--", "-").split("-")[0] in cite
        rec["volume_in_mine"] = bool(rec["volume"]) and str(rec["volume"]) in cite
        rec["author_count_mine"] = len(authors.split(", "))
    out[key] = rec
    print(f"{key:18s} {'FOUND' if rec['found'] else 'none ':5s} " + (f"ratio={rec['title_ratio']} year_ok={rec['year_ok']} vol_ok={rec['volume_in_mine']} pages_ok={rec['pages_in_mine']} authors {rec['author_count_mine']}/{rec['n_authors']}" if rec["found"] else ""), flush=True)
    time.sleep(0.3)
json.dump(out, open("docs/results/ref_check.json", "w"), indent=1)

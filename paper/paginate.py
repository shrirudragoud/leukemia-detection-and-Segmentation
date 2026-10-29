"""Measure the page of every heading / table / figure in paper/paper.pdf and write paper/toc_pages.json (used by build_paper.py for the contents pages)."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
blocks = json.loads((ROOT / "paper" / "content.json").read_text())["blocks"]
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(ROOT / "paper/paper.pdf")], capture_output=True, text=True).stdout).group(1))
pages = [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", str(ROOT / "paper/paper.pdf"), "-"], capture_output=True, text=True).stdout.split("\n") for i in range(1, n + 1)]
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[*^~]", "", t)).strip()  # noqa: E731
# first page whose line is exactly INTRODUCTION = first page after the contents
start = next(i for i, L in enumerate(pages) if any(norm(x) == "INTRODUCTION" for x in L)) + 1
out = {}


def find(pred):
    for i in range(start - 1, n):
        if any(pred(norm(x)) for x in pages[i]):
            return i + 1
    return None


for b in blocks:
    if b["t"] == "h" and b["text"].lower() != "abstract":
        k = "h:" + b["text"][:60]
        pre = norm(b["text"].upper())[:28]
        out[k] = find(lambda x, pre=pre: x.startswith(pre) and x == x.upper())
    elif b["t"] == "sh":
        pre = norm(b["text"])[:30]
        full = norm(b["text"])
        out["sh:" + b["text"][:60]] = find(lambda x, full=full: x == full or (len(x) >= 25 and full.startswith(x)))
    elif b["t"] == "table":
        out[f"table:{b['num']}"] = find(lambda x, nm=b["num"]: x.startswith(f"Table {nm}. "))
    elif b["t"] == "fig":
        out[f"fig:{b['num']}"] = find(lambda x, nm=b["num"]: x.startswith(f"Figure {nm}. "))
missing = [k for k, v in out.items() if v is None]
(ROOT / "paper" / "toc_pages.json").write_text(json.dumps({k: v for k, v in out.items() if v}, indent=0))
print("pages", n, "entries", len(out), "missing", missing[:6])


# ---- pages on which each reference is cited (main text only, before Literature Cited)
sys.path.insert(0, str(ROOT / "paper"))
from refs import REFS  # noqa: E402

lit = next((i for i in range(start - 1, n) if any(norm(x) == "LITERATURE CITED" for x in pages[i])), n - 1) + 1
def _body(i):
    ls = [norm(x) for x in pages[i] if norm(x)]
    return " ".join(ls[:-1] if ls and ls[-1].isdigit() else ls)


_t = {i + 1: _body(i) for i in range(start - 1, lit + 1)}
flat = {pg: t + " " + " ".join(_t.get(pg + 1, "").split()[:8]) for pg, t in _t.items() if pg <= lit}
cp = {}
for k, (label, _c) in REFS.items():
    m = re.match(r"(.*) (\d{4})$", label)
    forms = [label, f"{m.group(1)} ({m.group(2)})"]
    hit = [pg for pg, t in flat.items() if any(f in t for f in forms)]
    # citations inside a multi-citation parenthesis: "(A 2001; B 2002)" -> the label still appears verbatim
    cp[k] = hit
(ROOT / "paper" / "cite_pages.json").write_text(json.dumps(cp))
print("citation pages found for", sum(1 for v in cp.values() if v), "references")

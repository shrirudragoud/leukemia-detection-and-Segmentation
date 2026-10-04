"""Build the manuscript and check it for unfinished parts.

  python scripts/finalize_paper.py --build     regenerate figures' data-driven content, content.json, paper.docx and paper.pdf
  python scripts/finalize_paper.py --check     list every placeholder that is still in the manuscript (exit 1 if any)

Placeholders: '[[GPU' (a result or paragraph that needs GPU runs), 'PLACEHOLDER' (a figure to replace), 'FILL IN' (author details).
When results_for_paper/<run>/cv_summary.json exist, the fine-tuning table fills itself on --build; the bracketed paragraphs must
be rewritten by hand (or by the assistant) and are then deleted. Nothing numeric is ever typed into the text."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAT = re.compile(r"\[\[GPU|PLACEHOLDER|SAMPLE IMAGE|PENDING|FILL IN")


def build():
    subprocess.run([sys.executable, "scripts/make_figures.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/make_ft_figures.py", "--sample"], cwd=ROOT, check=True)          # SAMPLE placeholders (synthetic)
    if any((ROOT / "results_for_paper").glob("g0*/cv_summary.json")):
        subprocess.run([sys.executable, "scripts/make_ft_figures.py"], cwd=ROOT, check=True)                 # REAL figures replace the samples
    (ROOT / "paper" / "toc_pages.json").unlink(missing_ok=True)
    for _ in range(3):            # build, measure pages, rebuild until the contents pages are stable
        subprocess.run([sys.executable, "paper/build_paper.py"], cwd=ROOT, check=True)
        subprocess.run(["node", "paper/render_docx.js", "paper/content.json", "paper/paper.docx"], cwd=ROOT, check=True)
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", "paper", "paper/paper.docx"], cwd=ROOT, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rd = lambda n: (ROOT / "paper" / n).read_text() if (ROOT / "paper" / n).exists() else ""  # noqa: E731
        before = rd("toc_pages.json") + rd("cite_pages.json")
        subprocess.run([sys.executable, "paper/paginate.py"], cwd=ROOT, check=True)
        if before == rd("toc_pages.json") + rd("cite_pages.json"):
            break


def check():
    blocks = json.loads((ROOT / "paper" / "content.json").read_text())["blocks"]
    hits = []
    for b in blocks:
        txt = " ".join(str(v) for v in b.values() if isinstance(v, str)) + " ".join(" ".join(r) for r in b.get("rows", []))
        m = PAT.findall(txt)
        if m:
            hits.append((len(m), (b.get("caption") or b.get("text") or "")[:110]))
    for n, t in hits:
        print(f"{n:3d} x  {t}")
    print(f"{len(hits)} blocks with unfinished parts")
    return 1 if hits else 0


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    sys.exit(check() if "--check" in sys.argv else 0)

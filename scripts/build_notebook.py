"""Convert scripts/notebook_source.py (percent-format cells) into the Colab notebook.

Usage:  python scripts/build_notebook.py   -> writes Ecoli_Ciprofloxacin_Resistance.ipynb
Then execute it with:
  jupyter nbconvert --to notebook --execute --inplace Ecoli_Ciprofloxacin_Resistance.ipynb
"""
import os
import re
import nbformat

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "scripts", "notebook_source.py")
OUT = os.path.join(ROOT, "Ecoli_Ciprofloxacin_Resistance.ipynb")

text = open(SRC, encoding="utf-8").read()
chunks = re.split(r"^# %%(.*)$", text, flags=re.M)
cells = []
for header, body in zip(chunks[1::2], chunks[2::2]):
    body = body.strip("\n")
    if "[markdown]" in header:
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(body))

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nb.metadata["colab"] = {"provenance": []}
nbformat.write(nb, OUT)
print(f"wrote {OUT} with {len(cells)} cells")

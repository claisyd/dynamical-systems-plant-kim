#!/usr/bin/env python3
"""Attach figures into Notebook 07 markdown cells for durable HTML export."""
from __future__ import annotations

import base64
from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "07_two_parameter_bifurcation.ipynb"
FIGDIR = ROOT / "outputs" / "figures" / "two_parameter_bifurcation"

CAPTIONS = {
    "07_alphaIP_bifurcation_map.png": (
        r"$(\alpha_I,\alpha_P)$ bifurcation map",
        r"""
Biological square outlined. Red: Hopf curve (H_bio and H_ext seeds on one physical branch).
Blue: LP curve from LP_bio. ZH markers only where PyCont detected them (exterior).
""",
    ),
    "07_alphaIP_biological_square.png": (
        r"Biological square focus",
        r"""
Hopf arc through $H_{\mathrm{bio}}$ stays at $\alpha_P\lesssim 0.27$ and reaches only to
$\alpha_I\approx 0.57$ near $\alpha_P=0$. LP curve crosses $\alpha_P=1$ at $LP_{\mathrm{bio}}$.
Control $(1,1)$ lies to the right of the LP curve. Do not claim BT from visual H–LP crossings.
""",
    ),
    "07_gI_Iep_transformed_map.png": (
        r"Transformed $(g_I,I_{\mathrm{ep}})$ map",
        r"""
Linear image of the same curves: $g_I=G_{I0}\alpha_I$, $I_{\mathrm{ep}}=I_{EP0}\alpha_P$ with
$I_{EP0}<0$ so $\alpha_P\uparrow$ makes $I_{\mathrm{ep}}$ more negative. No second continuation.
""",
    ),
}


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    # Drop previous figure attachment cells
    nb.cells = [
        c
        for c in nb.cells
        if not (
            c.cell_type == "markdown"
            and c.source.lstrip().startswith("**Figure —")
        )
    ]

    # Insert after the section-12 / section-14 display code cells by filename mention
    inserts = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for fname, (title, body) in CAPTIONS.items():
            if fname in c.source:
                inserts.append((i + 1, fname, title, body.strip()))

    # Apply from the end so indices stay valid
    for i, fname, title, body in sorted(inserts, key=lambda t: -t[0]):
        png = FIGDIR / fname
        if not png.exists():
            print("missing", png)
            continue
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        md = (
            f"**Figure — {title}**\n\n"
            f"![{fname}](attachment:{fname})\n\n"
            f"{body}\n"
        )
        cell = new_markdown_cell(md)
        cell["attachments"] = {
            fname: {"image/png": b64}
        }
        nb.cells.insert(i, cell)
        print("attached", fname, "after cell", i - 1)

    nbformat.write(nb, NB_PATH)
    print("Updated", NB_PATH)


if __name__ == "__main__":
    main()

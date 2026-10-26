#!/usr/bin/env python3
"""Attach cleaned report figures (``ReportFigs/cleaned/``) into their notebooks.

Idempotent: removes previous ``**Cleaned report figure —`` markdown cells, then
inserts a fresh caption cell (PNG as attachment) directly after the code cell
that saves each cleaned figure. Does not execute or modify code cells.

Usage: python scripts/attach_cleaned_report_figs.py [NOTEBOOK_STEM ...]
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "ReportFigs" / "cleaned"
PREFIX = "**Cleaned report figure —"

# notebook stem -> list of (cleaned filename, title, caption body)
FIGURES = {
    "03_drug_endpoints": [
        (
            "baseline_drug_dynamics.png",
            "simulated membrane-potential dynamics under drug interventions",
            r"""
Report copy of the overview figure above: identical simulations, time windows, axes and colours;
only the titles changed. Dashed vertical lines are the transient cutoffs used for regime
classification (10, 20, 15 and 40 s respectively). Saved to `ReportFigs/cleaned/`.
""",
        ),
    ],
    "05_ttx_bifurcation": [
        (
            "ttx_alphaI_continuation.png",
            r"TTX equilibrium continuation in $(\alpha_I,V)$",
            r"""
Report copy of the report-quality $\alpha_I$ EP-C diagram above: identical branch, stability
(solid stable / dashed unstable from stored eigenvalues), special points, shading and limits;
only title, legend and point labels are typeset. The orange dashed line marks $LP_{\mathrm{bio}}$
($\alpha_I\approx 0.614$); $H_{\mathrm{ext}}$ ($\alpha_I\approx 1.89$) lies outside the biological
range. Saved to `ReportFigs/cleaned/`.
""",
        ),
    ],
    "06_ouabain_bifurcation": [
        (
            "ouabain_alphaP_continuation.png",
            r"ouabain equilibrium continuation in $(\alpha_P,V)$",
            r"""
Report copy of the report-quality $\alpha_P$ EP-C diagram above: identical branch, stability,
special points, shading and limits; only title, legend and point labels are typeset.
$H_{\mathrm{ext}}$ ($\alpha_P\approx -2.01$) and $LP_{\mathrm{ext}}$ ($\alpha_P\approx 3.22$) are both
outside the biological range. Saved to `ReportFigs/cleaned/`.
""",
        ),
        (
            "ouabain_complete_ttx_continuation.png",
            "ouabain continuation under complete TTX",
            r"""
Report copy of the complete-TTX ($\alpha_I=0$) LC report figure above: identical equilibrium
branch, Hopf points ($H_{\mathrm{bio}}$ at $\alpha_P\approx 0.268$, $H_{\mathrm{ext}}$ at
$\alpha_P\approx 1.48$), AUTO limit-cycle $V_{\max}$/$V_{\min}$ envelope, shading and limits; only
title, legend and point labels are typeset. Saved to `ReportFigs/cleaned/`.
""",
        ),
    ],
    "07_two_parameter_bifurcation": [
        (
            "alphaI_alphaP_bifurcation_map.png",
            r"two-parameter bifurcation map in $(\alpha_I,\alpha_P)$",
            r"""
Report copy of `07_alphaIP_bifurcation_map.png`, redrawn from the saved continuation CSVs
(`hopf_curve_dedup.csv`, `lp_curve_dedup.csv`, `codim2_special_points.csv` and the NB05/NB06
special points) with the same geometry, limits and styles; only title, legend and point labels
are typeset. Each saved forward/backward continuation segment is drawn separately, as in the
original figure. The second detected ZH point, at $(\alpha_I,\alpha_P)\approx(1.12, 3.97)$, lies
outside the plotted window, as in the original. Saved to `ReportFigs/cleaned/`.
""",
        ),
    ],
}


def caption_cell(filename: str, title: str, body: str):
    path = CLEAN_DIR / filename
    src = f"{PREFIX} {title}.**\n\n{body.strip()}\n\n![{title}](attachment:{filename})\n"
    cell = new_markdown_cell(src)
    cell["attachments"] = {
        filename: {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")}
    }
    return cell


def process(stem: str) -> None:
    nb_path = ROOT / "notebooks" / f"{stem}.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    nb.cells = [
        c
        for c in nb.cells
        if not (c.cell_type == "markdown" and c.source.lstrip().startswith(PREFIX))
    ]
    new_cells = []
    for cell in nb.cells:
        new_cells.append(cell)
        if cell.cell_type != "code":
            continue
        for filename, title, body in FIGURES[stem]:
            if f"'{filename}'" in cell.source or f'"{filename}"' in cell.source:
                if not (CLEAN_DIR / filename).is_file():
                    print("skip (not generated yet):", filename)
                    continue
                new_cells.append(caption_cell(filename, title, body))
                print("attached", filename, "in", nb_path.name)
    nb.cells = new_cells
    nbformat.write(nb, nb_path)


def main() -> None:
    stems = sys.argv[1:] or list(FIGURES)
    for stem in stems:
        process(stem)


if __name__ == "__main__":
    main()

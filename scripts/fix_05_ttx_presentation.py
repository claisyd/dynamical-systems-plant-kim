#!/usr/bin/env python3
"""Attach figures + short captions into Notebook 05 markdown cells.

Idempotent: removes previous ``**Figure —`` markdown cells, then re-inserts
fresh attachments from ``outputs/figures/ttx_bifurcation/``.

Does not recompute continuation; preserves existing code outputs.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "05_ttx_bifurcation.ipynb"
FIGDIR = ROOT / "outputs" / "figures" / "ttx_bifurcation"


def fix_math(text: str) -> str:
    """Convert LaTeX \\( \\) / \\[ \\] delimiters to Jupyter $ / $$."""
    text = re.sub(r"\\\[(.*?)\\\]", lambda m: "$$\n" + m.group(1).strip() + "\n$$", text, flags=re.S)
    text = re.sub(r"\\\((.*?)\\\)", lambda m: "$" + m.group(1) + "$", text, flags=re.S)
    text = text.replace("\x07lpha", r"\alpha")
    return text


def attach_png(cell, filename: str) -> None:
    path = FIGDIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing figure: {path}")
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    cell.setdefault("attachments", {})
    cell["attachments"][filename] = {"image/png": b64}


def figure_md(filename: str, title: str, body: str) -> nbformat.NotebookNode:
    src = (
        f"**Figure — {title}.**\n\n"
        f"{body.strip()}\n\n"
        f"![{title}](attachment:{filename})\n"
    )
    cell = new_markdown_cell(src)
    attach_png(cell, filename)
    return cell


# Insert AFTER the code cell whose source contains this marker string.
# Order matters when several markers share one code cell: list them in the
# desired display order.
FIGURES = [
    (
        'fig.savefig(FIGDIR / "05_ttx_gI_epc.png"',
        figure_md(
            "05_ttx_gI_epc.png",
            r"PyDSTool $g_I$ EP-C diagnostic",
            r"""
Raw PyCont display of the bounded equilibrium continuation in $(g_I, V)$.
Special-point markers are as reported by PyDSTool (software labels).
""",
        ),
    ),
    (
        'fig.savefig(FIGDIR / "05_ttx_gI_epc_report.png"',
        figure_md(
            "05_ttx_gI_epc_report.png",
            r"Report-quality $g_I$ EP-C diagram (unique specials)",
            r"""
Folded equilibrium branch with **unique** report labels: $H_{\mathrm{pos}}$, $H_{\mathrm{neg}}$,
$LP_{\mathrm{hi}}$, $LP_{\mathrm{bio}}$. The shaded band $[0,4]$ is the image of biological
$\alpha_I\in[0,1]$. Negative $g_I$ and $g_I>4$ are mathematical, not TTX doses.
""",
        ),
    ),
    (
        'fig.savefig(FIGDIR / "05_ttx_gI_lcc_report.png"',
        figure_md(
            "05_ttx_gI_lcc_report.png",
            r"$g_I$ unique LC envelopes (split panels)",
            r"""
Left: periodic envelope from the unique positive Hopf. Right: envelope from the unique
negative Hopf. Native PyDSTool LC `display` overlays were unreliable here; these plots use
extracted AUTO $V_{\max}$ / $V_{\min}$.
""",
        ),
    ),
    (
        'fig.savefig(FIGDIR / "05_ttx_gI_lcc.png"',
        figure_md(
            "05_ttx_gI_lcc.png",
            r"$g_I$ equilibria with unique LC envelopes (combined)",
            r"""
Combined view of the equilibrium branch and the two unique limit-cycle envelopes
($H_{\mathrm{pos}}$ and $H_{\mathrm{neg}}$), each shown only once after forward/backward deduplication.
""",
        ),
    ),
    (
        'fig.savefig(FIGDIR / "05_ttx_gI_period.png"',
        figure_md(
            "05_ttx_gI_period.png",
            r"Limit-cycle period vs $g_I$ (separate panels)",
            r"""
Period $T$ in **milliseconds** (Plant–Kim model time) for each unique LC branch on its own axis,
so the $\sim 10^2\,\mathrm{ms}$ positive-Hopf branch is not visually erased by the large-period
negative-Hopf branch. Large periods alone are **not** proof of HOM/SNIC.
""",
        ),
    ),
    (
        'fig.savefig(FIGDIR / "05_ttx_gI_period_log.png"',
        figure_md(
            "05_ttx_gI_period_log.png",
            r"Limit-cycle period vs $g_I$ (log scale)",
            r"""
Same period data on a shared log-$T$ axis for comparison across scales. Time unit: **ms**.
""",
        ),
    ),
    (
        'fig.savefig(FIGDIR / "05_ttx_alphaI_epc.png"',
        figure_md(
            "05_ttx_alphaI_epc.png",
            r"PyDSTool $\alpha_I$ EP-C diagnostic",
            r"""
Raw PyCont equilibrium continuation in $(\alpha_I, V)$ with $\alpha_P=1$.
The untreated-seeded branch folds near $\alpha_I\approx 0.614$ and does not reach $\alpha_I=0$
on this connected component.
""",
        ),
    ),
    (
        'fig.savefig(FIGDIR / "05_ttx_alphaI_epc_report.png"',
        figure_md(
            "05_ttx_alphaI_epc_report.png",
            r"Report-quality $\alpha_I$ EP-C diagram",
            r"""
Biological TTX interval shaded as $\alpha_I\in[0,1]$, with markers at control ($\alpha_I=1$) and
complete block ($\alpha_I=0$). Inside $[0,1]$: **no Hopf**, one fold $LP_{\mathrm{bio}}$ near
$\alpha_I\approx 0.614$. The Hopf near $\alpha_I\approx 1.89$ is exterior / mathematical only.
""",
        ),
    ),
    (
        'fig.savefig(FIGDIR / "05_ttx_alphaI_lcc.png"',
        figure_md(
            "05_ttx_alphaI_lcc.png",
            r"$\alpha_I$ exterior LC branch (mathematical cross-check)",
            r"""
Limit-cycle envelope from the exterior Hopf ($H_{\mathrm{ext}}$, $\alpha_I\approx 1.89$), equivalent
under $g_I=4\alpha_I$ to the positive-$g_I$ LC family. Useful for validation; **not** part of
ordinary biological TTX blockade $\alpha_I:1\to 0$.
""",
        ),
    ),
]


def main() -> None:
    missing = [fn for (_, cell) in FIGURES for fn in cell.get("attachments", {})]
    # attachments already loaded in figure_md; re-check files exist
    for marker, _ in FIGURES:
        # extract filename from marker
        pass
    for path in FIGDIR.glob("05_ttx_*.png"):
        print("available", path.name, path.stat().st_size, "bytes")

    nb = nbformat.read(NB_PATH, as_version=4)

    for cell in nb.cells:
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if cell.cell_type == "markdown":
            cell.source = fix_math(src)

    # Remove previously inserted figure markdown cells (idempotent)
    cleaned = []
    for cell in nb.cells:
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if cell.cell_type == "markdown" and src.lstrip().startswith("**Figure —"):
            continue
        cleaned.append(cell)
    nb.cells = cleaned

    # Insert figure cells after matching savefig code cells
    new_cells = []
    for cell in nb.cells:
        new_cells.append(cell)
        if cell.cell_type != "code":
            continue
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        for marker, fig_cell in FIGURES:
            if marker in src:
                # Rebuild attachment from disk in case figures were regenerated
                filename = next(iter(fig_cell["attachments"].keys()))
                attach_png(fig_cell, filename)
                fig_cell.source = fix_math(
                    fig_cell.source if isinstance(fig_cell.source, str) else "".join(fig_cell.source)
                )
                new_cells.append(fig_cell)
                print("attached", filename, "after marker", marker[:48], "...")
    nb.cells = new_cells

    nbformat.write(nb, NB_PATH)
    print("updated", NB_PATH, "cells=", len(nb.cells))


if __name__ == "__main__":
    main()

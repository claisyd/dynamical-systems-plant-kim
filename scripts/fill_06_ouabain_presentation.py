#!/usr/bin/env python3
"""Attach figures + short captions into Notebook 06 markdown cells.

Idempotent: removes previous ``**Figure —`` markdown cells, then re-inserts
fresh attachments from ``outputs/figures/ouabain_bifurcation/``.

Walks the notebook for ``fig.savefig(FIGDIR / "…"`` markers and attaches the
matching PNG when present on disk. Captions are looked up by filename; unknown
figures get a short generic caption.

Does not recompute continuation; preserves existing code outputs.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "06_ouabain_bifurcation.ipynb"
FIGDIR = ROOT / "outputs" / "figures" / "ouabain_bifurcation"

SAVEFIG_RE = re.compile(
    r'fig\.savefig\(\s*FIGDIR\s*/\s*["\']([^"\']+\.png)["\']',
    re.MULTILINE,
)

# Short captions keyed by basename. Keep in sync with build_06 figure names.
CAPTIONS = {
    "06_ouabain_Iep_epc.png": (
        r"PyDSTool $I_{\mathrm{ep}}$ EP-C diagnostic",
        r"""
Raw PyCont display of the bounded equilibrium continuation in $(I_{\mathrm{ep}}, V)$.
Special-point markers are as reported by PyDSTool (software labels).
""",
    ),
    "06_ouabain_Iep_epc_report.png": (
        r"Report-quality $I_{\mathrm{ep}}$ EP-C diagram (unique specials)",
        r"""
Equilibrium branch with unique report labels ($H_{\mathrm{bio}}$ / $H_{\mathrm{ext}}$,
$LP_{\mathrm{bio}}$ / $LP_{\mathrm{ext}}$). Shaded band $[-0.22,0]$ is the biological image of
$\alpha_P\in[0,1]$. Values outside this interval are mathematical, not ouabain doses.
""",
    ),
    "06_ouabain_Iep_lcc_report.png": (
        r"$I_{\mathrm{ep}}$ unique LC envelopes (split panels)",
        r"""
Periodic envelopes seeded once per unique Hopf. Native PyDSTool LC `display` overlays
were unreliable in related work; these plots use extracted AUTO $V_{\max}$ / $V_{\min}$.
Near-Hopf LPC markers are omitted when coincident with the parent Hopf.
""",
    ),
    "06_ouabain_Iep_lcc.png": (
        r"$I_{\mathrm{ep}}$ equilibria with unique LC envelopes (combined)",
        r"""
Combined view of the equilibrium branch and unique limit-cycle envelopes after
forward/backward overlap deduplication.
""",
    ),
    "06_ouabain_Iep_period.png": (
        r"Limit-cycle period vs $I_{\mathrm{ep}}$ (separate panels)",
        r"""
Period $T$ in **milliseconds** (Plant–Kim model time) for each unique LC branch.
Large periods alone are **not** proof of HOM/SNIC.
""",
    ),
    "06_ouabain_Iep_period_log.png": (
        r"Limit-cycle period vs $I_{\mathrm{ep}}$ (log scale)",
        r"""
Same period data on a shared log-$T$ axis for comparison across scales. Time unit: **ms**.
""",
    ),
    "06_ouabain_alphaP_epc.png": (
        r"PyDSTool $\alpha_P$ EP-C diagnostic",
        r"""
Raw PyCont equilibrium continuation in $(\alpha_P, V)$ with $\alpha_I=1$.
If the untreated-seeded branch folds before $\alpha_P=0$, that is stated in the notebook text.
""",
    ),
    "06_ouabain_alphaP_epc_report.png": (
        r"Report-quality $\alpha_P$ EP-C diagram",
        r"""
Biological ouabain interval shaded as $\alpha_P\in[0,1]$, with markers at control ($\alpha_P=1$)
and complete block ($\alpha_P=0$). Detected unique specials only — no forced Hopf/LP count.
""",
    ),
    "06_ouabain_alphaP_lcc.png": (
        r"$\alpha_P$ LC envelopes (unique Hopfs)",
        r"""
Limit-cycle envelopes from unique $\alpha_P$ Hopfs (biological or exterior). Exterior
branches are useful for $I_{\mathrm{ep}}\leftrightarrow\alpha_P$ cross-checking.
""",
    ),
    "06_ouabain_alphaP_lcc_report.png": (
        r"$\alpha_P$ unique LC envelopes (report)",
        r"""
Report-quality overlay of unique LC $V_{\max}$ / $V_{\min}$ on the $\alpha_P$ equilibrium branch.
""",
    ),
    "06_ouabain_alphaP_period.png": (
        r"Limit-cycle period vs $\alpha_P$",
        r"""
Period $T$ (ms) for each successful unique $\alpha_P$ LC branch.
""",
    ),
    "06_ouabain_alphaP_period_log.png": (
        r"Limit-cycle period vs $\alpha_P$ (log scale)",
        r"""
Shared log-$T$ axis for $\alpha_P$ LC periods. Time unit: **ms**.
""",
    ),
    "06_ouabain_ttx0_alphaP_epc.png": (
        r"Complete-TTX ($g_I=0$) $\alpha_P$ EP-C diagnostic",
        r"""
Raw PyCont continuation under complete TTX with freepar $\alpha_P$ (7D, $g_I=0$).
""",
    ),
    "06_ouabain_ttx0_alphaP_epc_report.png": (
        r"Complete-TTX ouabain EP-C (report)",
        r"""
Unique specials on the complete-TTX slice $\alpha_I=0$, $\alpha_P:1\to 0$. Biological
$[0,1]$ shaded. Seed may differ from untreated $V^*\approx -18.3$ mV.
""",
    ),
    "06_ouabain_ttx0_alphaP_lcc.png": (
        r"Complete-TTX $\alpha_P$ LC (if any)",
        r"""
LC envelopes only if a unique Hopf appears under $g_I=0$. Otherwise EP-C + NB03 time
domain carry the interpretation.
""",
    ),
    "06_ouabain_ttx0_alphaP_lcc_report.png": (
        r"Complete-TTX unique LC envelopes (report)",
        r"""
Report-quality LC overlay for the complete-TTX ouabain slice when AUTO succeeds.
""",
    ),
    "06_ouabain_ttx0_alphaP_period.png": (
        r"Complete-TTX LC period vs $\alpha_P$",
        r"""
Period $T$ (ms) for successful complete-TTX LC branches.
""",
    ),
    "06_ouabain_ttx0_alphaP_period_log.png": (
        r"Complete-TTX LC period vs $\alpha_P$ (log)",
        r"""
Log-scale period comparison for complete-TTX LC branches. Time unit: **ms**.
""",
    ),
}


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


def caption_for(filename: str) -> tuple[str, str]:
    if filename in CAPTIONS:
        return CAPTIONS[filename]
    stem = filename.replace(".png", "").replace("_", " ")
    return (
        stem,
        f"Figure `{filename}` from `outputs/figures/ouabain_bifurcation/`.",
    )


def collect_savefig_order(nb) -> list[tuple[int, str]]:
    """Return (cell_index, filename) in notebook order for savefig calls."""
    hits = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        for m in SAVEFIG_RE.finditer(src):
            hits.append((i, m.group(1)))
    return hits


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    available = sorted(FIGDIR.glob("06_ouabain_*.png"))
    for path in available:
        print("available", path.name, path.stat().st_size, "bytes")
    if not available:
        print(
            "WARNING: no PNGs yet under",
            FIGDIR,
            "— run the notebook first; this script will skip missing files.",
        )

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

    # Map each code cell index -> list of figure filenames saved in that cell
    by_cell: dict[int, list[str]] = {}
    for idx, filename in collect_savefig_order(nb):
        by_cell.setdefault(idx, []).append(filename)

    new_cells = []
    attached = []
    skipped = []
    for i, cell in enumerate(nb.cells):
        new_cells.append(cell)
        for filename in by_cell.get(i, []):
            path = FIGDIR / filename
            if not path.is_file():
                skipped.append(filename)
                print("skip (missing on disk):", filename)
                continue
            title, body = caption_for(filename)
            fig_cell = figure_md(filename, title, body)
            fig_cell.source = fix_math(
                fig_cell.source if isinstance(fig_cell.source, str) else "".join(fig_cell.source)
            )
            new_cells.append(fig_cell)
            attached.append(filename)
            print("attached", filename)

    # Also attach any on-disk figures not referenced by savefig (orphan report assets)
    referenced = set(attached) | set(skipped)
    for path in available:
        if path.name in referenced:
            continue
        title, body = caption_for(path.name)
        fig_cell = figure_md(path.name, title, body)
        fig_cell.source = fix_math(
            fig_cell.source if isinstance(fig_cell.source, str) else "".join(fig_cell.source)
        )
        # Append orphans before the final summary if possible
        new_cells.append(fig_cell)
        attached.append(path.name)
        print("attached orphan", path.name)

    nb.cells = new_cells
    nbformat.write(nb, NB_PATH)
    print("updated", NB_PATH, "cells=", len(nb.cells))
    print("attached_n=", len(attached), "skipped_n=", len(skipped))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build notebooks/07_two_parameter_bifurcation.ipynb (Section 3.7).

Heavy continuation is performed by scripts/run_07_two_parameter_bifurcation.py
(Docker). This notebook loads validated outputs, documents methodology, and
renders report figures. Re-running the engine is optional and expensive.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "07_two_parameter_bifurcation.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3.9 (PyDSTool)",
    "language": "python",
    "name": "pydstool",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.9.20"}

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n") + "\n"))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n") + "\n"))


md(
    r"""
# 07 — Two-parameter bifurcation analysis (checklist 3.7)

Convert validated one-parameter special points from Notebooks 05 (TTX) and 06
(ouabain / complete-TTX) into **bifurcation curves** in the biological plane

$$
(\alpha_I,\alpha_P)\in[0,1]^2.
$$

- Primary continuation plane: $(\alpha_I,\alpha_P)$
- PyCont types: **`H-C2`** (Hopf) and **`LP-C`** (equilibrium fold)
- Secondary $(g_I,I_{\mathrm{ep}})$ figure: **linear transform only** (no second continuation)
- Codimension-2 labels reported only when the installed software detects them

**Out of scope:** individual Jacobian / first-Lyapunov coefficient project
(`Notes/MATH3888_individual_component_jacobian_hopf_plan.md`); automatic
two-parameter LPC / HOM campaigns.

**Environment:** continuation engine
`scripts/run_07_two_parameter_bifurcation.py` is intended for Docker
(`curuy/math3888:latest`). This notebook loads saved CSVs/figures; set
`RERUN_CONTINUATION = True` only if you intentionally want a full recompute.
"""
)

md(
    r"""
## 1. Aim and relation to one-parameter slices

Notebook 05 found a biological equilibrium fold **LP_bio** on the TTX-only
slice $\alpha_P=1$ at $\alpha_I\approx 0.614$, and no biological Hopf on
$\alpha_I\in[0,1]$.

Notebook 06 found, on the complete-TTX slice $\alpha_I=0$, two Hopfs on one
slow-orbit skeleton: **H_bio** at $\alpha_P\approx 0.268$ and **H_ext** at
$\alpha_P\approx 1.48$. Ouabain alone ($\alpha_I=1$) had **no** biological
Hopf/LP on the untreated-connected EP-C branch.

This notebook asks how those landmarks organise as curves in
$(\alpha_I,\alpha_P)$, and whether they meet at software-detected
codimension-2 points.
"""
)

md("## 2. Imports / environment")

code(
    r"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path.cwd().resolve()
if REPO_ROOT.name == "notebooks":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plant_kim_analysis import (
    G_I0,
    I_EP0,
    build_alpha_IP_vode_system,
    pars_from_alpha,
)
from src.plant_kim_model import STATE_VARS

FIGDIR = REPO_ROOT / "outputs" / "figures" / "two_parameter_bifurcation"
DATADIR = REPO_ROOT / "outputs" / "data" / "two_parameter_bifurcation"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATADIR.mkdir(parents=True, exist_ok=True)

TTX_CSV = REPO_ROOT / "outputs" / "data" / "ttx_bifurcation" / "alphaI_epc_special_points.csv"
TTX0_CSV = (
    REPO_ROOT / "outputs" / "data" / "ouabain_bifurcation" / "ttx0_alphaP_epc_special_points.csv"
)

# Set True only to recompute H-C2 / LP-C in Docker (tens of minutes).
RERUN_CONTINUATION = False

print("REPO_ROOT:", REPO_ROOT)
print("G_I0 =", G_I0, "I_EP0 =", I_EP0)
print("FIGDIR:", FIGDIR)
print("DATADIR:", DATADIR)
print("RERUN_CONTINUATION:", RERUN_CONTINUATION)
"""
)

md(
    r"""
## 3. Verify $\alpha_I$ / $\alpha_P$ mappings

Analysis-level scalings (canonical ODEs unchanged):

$$
g_I = \alpha_I\, G_{I0},\qquad I_{\mathrm{ep}} = \alpha_P\, I_{EP0},
$$

with $G_{I0}=4$ and $I_{EP0}=-0.22$. Joint ContClass helper
`build_alpha_IP_vode_system` freezes baseline $g_I$ and $I_{\mathrm{ep}}$ and
places both $\alpha$ factors in the voltage equation.
"""
)

code(
    r"""
assert abs(G_I0 - 4.0) < 1e-15
assert abs(I_EP0 - (-0.22)) < 1e-15
assert abs(float(pars_from_alpha(1, 1)["g_I"]) - G_I0) < 1e-15
assert abs(float(pars_from_alpha(0, 1)["g_I"]) - 0.0) < 1e-15
assert abs(float(pars_from_alpha(1, 0)["I_ep"]) - 0.0) < 1e-15
assert abs(float(pars_from_alpha(1, 0.5)["I_ep"]) - 0.5 * I_EP0) < 1e-15

ode = build_alpha_IP_vode_system(alpha_I=0.0, alpha_P=0.268)
assert abs(float(ode.pars["g_I"]) - G_I0) < 1e-15
assert abs(float(ode.pars["I_ep"]) - I_EP0) < 1e-15
assert "alpha_I" in ode.pars and "alpha_P" in ode.pars
# Effective V equation uses alpha_I*g_I and alpha_P*I_ep
assert "alpha_I*g_I" in ode.funcspec.varspecs["V"]
assert "alpha_P*I_ep" in ode.funcspec.varspecs["V"]
print("Parameterisation / joint ContClass helper: PASS")
print("Biological square [0,1]^2 maps to g_I in [0,4], I_ep in [-0.22, 0]")
"""
)

md(
    r"""
## 4. Load validated one-parameter seed points

Seeds are read from Notebook 05/06 CSVs (not hard-coded).
"""
)

code(
    r"""
def load_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_by_label(rows, label):
    for r in rows:
        if r.get("label") == label or r.get("report_label") == label:
            return r
    raise KeyError(label)


ttx_rows = load_csv(TTX_CSV)
ttx0_rows = load_csv(TTX0_CSV)
lp_bio = row_by_label(ttx_rows, "LP_bio")
h_bio = row_by_label(ttx0_rows, "H_bio")
h_ext = row_by_label(ttx0_rows, "H_ext")

print("LP_bio (NB05, alpha_P=1): alpha_I =", lp_bio["alpha_I"], "V =", lp_bio["V"])
print("H_bio (NB06 TTX0): alpha_P =", h_bio["alpha_P"], "V =", h_bio["V"])
print("H_ext (NB06 TTX0): alpha_P =", h_ext["alpha_P"], "V =", h_ext["V"])
"""
)

md(
    r"""
## 5. Two-parameter continuation methodology

Installed PyCont types (Lab 7 / `ContClass.curve_list`):

| Purpose | `type` | `freepars` | Codim-2 if requested |
|---|---|---|---|
| Hopf curve | **`H-C2`** | `['alpha_I','alpha_P']` | BT, ZH, GH (`DH` not for H-C2) |
| LP / fold curve | **`LP-C`** | `['alpha_I','alpha_P']` | BT, ZH, CP |

Workflow used by the engine:

1. Build joint $\alpha$ VODE system.
2. Short `EP-C` on $\alpha_I=0$ (recover H1/H2) and on $\alpha_P=1$ (recover LP1).
3. `H-C2` from each Hopf, forward and backward.
4. `LP-C` from LP_bio, forward and backward.
5. Deduplicate overlapping segments / special points.
6. Cross-validate against NB05/06 anchors before interpretation.

Optional full recompute:
"""
)

code(
    r"""
if RERUN_CONTINUATION:
    import runpy
    print("Re-running scripts/run_07_two_parameter_bifurcation.py ...")
    runpy.run_path(str(REPO_ROOT / "scripts" / "run_07_two_parameter_bifurcation.py"))
else:
    required = [
        "hopf_curve_dedup.csv",
        "lp_curve_dedup.csv",
        "slice_validation.csv",
        "codim2_special_points.csv",
        "stability_sample_points.csv",
    ]
    missing = [n for n in required if not (DATADIR / n).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing two-parameter outputs: {}. Run "
            "`python scripts/run_07_two_parameter_bifurcation.py` in Docker, "
            "or set RERUN_CONTINUATION=True.".format(missing)
        )
    print("Using saved continuation outputs in", DATADIR)
"""
)

md(
    r"""
## 6–8. Hopf continuation, deduplication, and comparison

Both complete-TTX Hopfs were continued with `H-C2`. The two seeds meet outside
the biological square near $(\alpha_I,\alpha_P)\approx(-2.91,\,1.33)$
(sample distance $\approx 0.004$), so they belong to **one physical Hopf
curve**. Inside $[0,1]^2$ the biologically relevant arc is the branch through
**H_bio**; the **H_ext** arc lies at $\alpha_P>1$ (mathematical exterior).
"""
)

code(
    r"""
hopf = load_csv(DATADIR / "hopf_curve_dedup.csv")
hopf_note = (DATADIR / "hopf_dedup_note.txt").read_text()
print(hopf_note)
ai = np.array([float(r["alpha_I"]) for r in hopf])
ap = np.array([float(r["alpha_P"]) for r in hopf])
print("Hopf points:", len(hopf))
print("alpha_I range:", ai.min(), ai.max())
print("alpha_P range:", ap.min(), ap.max())
in_sq = ((ai >= 0) & (ai <= 1) & (ap >= 0) & (ap <= 1)).sum()
print("points in biological square:", int(in_sq))
"""
)

md(
    r"""
## 9. LP continuation from TTX LP_bio

`LP-C` from LP_bio traces a fold curve that crosses $\alpha_P=1$ at the
validated TTX fold and continues through the biological square toward lower
$\alpha_P$ (exiting near $\alpha_P=0$).
"""
)

code(
    r"""
lp = load_csv(DATADIR / "lp_curve_dedup.csv")
ai = np.array([float(r["alpha_I"]) for r in lp])
ap = np.array([float(r["alpha_P"]) for r in lp])
print("LP points:", len(lp))
print("alpha_I range:", ai.min(), ai.max())
print("alpha_P range:", ap.min(), ap.max())
in_sq = ((ai >= 0) & (ai <= 1) & (ap >= 0) & (ap <= 1)).sum()
print("points in biological square:", int(in_sq))
"""
)

md(
    r"""
## 10. Codimension-2 special-point summary

On **H-C2**, LocBifPoints included BT, ZH, GH. On **LP-C**, BT, ZH, CP.

**Detected in this run:** two unique **ZH** (zero-Hopf / fold–Hopf) points on
the LP curve, both **outside** $[0,1]^2$:

- near $(\alpha_I,\alpha_P)\approx(1.12,\,3.97)$
- near $(\alpha_I,\alpha_P)\approx(0.345,\,-0.184)$

**Not detected:** GH, BT, CP.

Do **not** invent GH/BT/CP from visual curve crossings inside the square.
A nearby H–LP crossing without a software label remains an observation, not a
claimed Bogdanov–Takens point.
"""
)

code(
    r"""
codim2 = load_csv(DATADIR / "codim2_special_points.csv")
print("Unique software-detected codim-2 points:")
if not codim2:
    print("  (none)")
for r in codim2:
    print(
        "  {type}  alpha_I={alpha_I}  alpha_P={alpha_P}  V={V}  "
        "parent={parent_curve}  bio_square={in_biological_square}".format(**r)
    )
"""
)

md(
    r"""
### Conditional interpretation (only for detected types)

**ZH (zero-Hopf):** local interaction of a fold and a Hopf (pure imaginary pair
with a zero eigenvalue). Here both ZH points are exterior to the biological
square, so they organise mathematical skeleton rather than ordinary TTX/ouabain
doses.

**GH / BT / CP:** not labelled by PyCont in this campaign. No LPC follow-up is
forced. Individual Jacobian / Lyapunov work remains a separate optional project.
"""
)

md("## 11. Slice cross-validation against Notebooks 05/06")

code(
    r"""
val = load_csv(DATADIR / "slice_validation.csv")
print("curve | slice | target | status | details")
for r in val:
    extra = []
    for k in ("diff_alpha", "diff_V", "twoD_alpha_I", "twoD_alpha_P", "n_bio_alpha_P", "n_bio_alpha_I", "note"):
        if k in r and r[k] != "":
            extra.append(f"{k}={r[k]}")
    print(
        f"{r.get('curve')} | {r.get('slice')} | {r.get('target')} | {r.get('status')} | "
        + "; ".join(extra)
    )

gate = [
    r for r in val
    if r.get("target") in ("LP_bio", "H_bio") and r.get("status") != "PASS"
]
if gate:
    raise RuntimeError("Validation gate FAILED for LP_bio/H_bio — do not interpret map")
print("\nValidation gate (LP_bio, H_bio): PASS")
"""
)

md(
    r"""
## 12. Biological $(\alpha_I,\alpha_P)$ bifurcation map

Report-quality figures written by the continuation engine / post-process:
"""
)

code(
    r"""
from IPython.display import Image, display, Markdown

for name, caption in [
    (
        "07_alphaIP_bifurcation_map.png",
        r"**Figure — $(\alpha_I,\alpha_P)$ map (modest exterior).** "
        r"Biological square outlined. Red: Hopf curve (H_bio/H_ext seeds, same physical branch). "
        r"Blue: LP curve from LP_bio. ZH markers only where software-detected.",
    ),
    (
        "07_alphaIP_biological_square.png",
        r"**Figure — biological square focus.** "
        r"Hopf arc through H_bio stays at $\alpha_P\lesssim 0.27$ and reaches only to "
        r"$\alpha_I\approx 0.57$ before leaving near $\alpha_P=0$. "
        r"LP curve crosses $\alpha_P=1$ at LP_bio and traverses the square. "
        r"Control $(1,1)$ lies to the right of the LP curve.",
    ),
]:
    path = FIGDIR / name
    display(Markdown(caption))
    display(Image(filename=str(path)))
"""
)

md(
    r"""
## 13. Representative equilibrium-stability regions

Sparse Jacobian checks (not a dense grid; not bursting/beating labels):
"""
)

code(
    r"""
stab = load_csv(DATADIR / "stability_sample_points.csv")
print(f"{'name':22s} {'aI':>5s} {'aP':>5s} {'V*':>10s} {'stab':>12s} {'maxRe':>12s}")
for r in stab:
    V = float(r["V_star"]) if r.get("V_star") not in (None, "") else float("nan")
    mr = float(r["max_real_part"]) if r.get("max_real_part") not in (None, "") else float("nan")
    print(
        f"{r['name'][:22]:22s} {float(r['alpha_I']):5.2f} {float(r['alpha_P']):5.2f} "
        f"{V:10.3f} {r.get('stability','?'):>12s} {mr:12.4g}"
    )

print(
    "\nNotes: untreated/control has multiple accepted equilibria; the known "
    "bursting-related root near V*≈-18.3 mV is unstable. Combined corner (0,0) "
    "is stable — consistent with NB03. On alpha_I=0, alpha_P=0.15 (below H_bio) "
    "is stable while alpha_P=0.5 (between H_bio and H_ext) is unstable."
)
"""
)

md(
    r"""
## 14. Transformed $(g_I,I_{\mathrm{ep}})$ map

Same continuation data under

$$
g_I = 4\,\alpha_I,\qquad I_{\mathrm{ep}} = -0.22\,\alpha_P.
$$

Because $I_{EP0}<0$, increasing $\alpha_P$ makes $I_{\mathrm{ep}}$ **more negative**.
"""
)

code(
    r"""
path = FIGDIR / "07_gI_Iep_transformed_map.png"
display(Markdown(
    r"**Figure — physical-parameter view.** Linear image of the biological square "
    r"is $g_I\in[0,4]$, $I_{\mathrm{ep}}\in[-0.22,0]$. No new continuation was run."
))
display(Image(filename=str(path)))
"""
)

md(
    r"""
## 15. Interpretation for TTX / ouabain / combined drugs

1. **TTX fold as pump changes.** LP_bio continues as an LP curve through the
   biological square: the fold persists as $\alpha_P$ decreases, shifting
   mildly in $\alpha_I$. Combined blockade can cross this fold boundary.

2. **Complete-TTX Hopf as TTX is relaxed.** From H_bio, the Hopf curve moves to
   larger $\alpha_I$ while $\alpha_P$ falls, leaving the square near
   $\alpha_P\approx 0$ around $\alpha_I\approx 0.57$. It does **not** reach the
   ouabain-alone edge $\alpha_I=1$ inside $[0,1]$ — consistent with NB06 having
   no biological Hopf on that slice.

3. **Why Hopf at $\alpha_I=0$ but not at $\alpha_I=1$.** Geometry: the Hopf
   curve intersects $\alpha_I=0$ twice (H_bio, H_ext) but only H_bio is in
   $[0,1]$; the same curve never crosses $\alpha_I=1$ with $\alpha_P\in[0,1]$.

4. **Combined-drug path.** Decreasing $\alpha_P$ at $\alpha_I=0$ crosses H_bio
   (unstable→stable), matching NB03/NB06. Paths with partial TTX may instead
   meet the LP curve depending on dose combination.

5. **Codim-2.** Only exterior ZH points were labelled; no GH/BT/CP inside the
   drug square. Visual H–LP proximity inside the square is **not** claimed as BT.
"""
)

md(
    r"""
## 16. Numerical limitations

- `H-C2` / `LP-C` are not AUTO-gated; still run in Docker for stack parity with NB05/06.
- Forward/backward and multi-seed continuations can retrace the same branch; dedup is required.
- `MX` is numerical termination, not a bifurcation.
- Period / LC structure is **not** re-continued here; GH→LPC would be a focused follow-up if GH appeared.
- Stability samples use accepted current-balance roots; multiple equilibria can coexist near folds.
"""
)

md(
    r"""
## 17. Summary and saved outputs

| Output | Path |
|---|---|
| Hopf raw / dedup | `outputs/data/two_parameter_bifurcation/hopf_*.csv` |
| LP raw / dedup | `outputs/data/two_parameter_bifurcation/lp_*.csv` |
| Codim-2 | `.../codim2_special_points.csv` |
| Slice validation | `.../slice_validation.csv` |
| Stability samples | `.../stability_sample_points.csv` |
| Figures | `outputs/figures/two_parameter_bifurcation/` |
| Engine | `scripts/run_07_two_parameter_bifurcation.py` |

**Verdict.** One physical Hopf curve (H_bio↔H_ext) and one LP curve (from LP_bio)
organise the biological square; slice anchors reproduce NB05/06 to continuation
tolerance; software-detected codim-2 points are exterior ZH only.
"""
)

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("Wrote", OUT)

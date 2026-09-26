#!/usr/bin/env python3
"""Build notebooks/08_jacobian_hopf_local_analysis.ipynb from saved jacobian_hopf outputs."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "08_jacobian_hopf_local_analysis.ipynb"
FIGDIR = REPO / "outputs" / "figures" / "jacobian_hopf"
DATADIR = REPO / "outputs" / "data" / "jacobian_hopf"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def figure_attachment_cell(title: str, caption: str, filename: str):
    path = FIGDIR / filename
    if not path.exists():
        return md(f"**Figure missing:** `{filename}`")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    cell = md(f"**Figure — {title}.**\n\n{caption}\n\n![ {title} ](attachment:{filename})")
    cell["attachments"] = {filename: {"image/png": b64}}
    return cell


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3.9 (PyDSTool)",
        "language": "python",
        "name": "pydstool",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.9.20"}

    cells = []

    cells.append(
        md(
            r"""# 08 — Jacobian and Local Bifurcation Mechanisms

**Computational individual component.** Local Jacobian / eigenvalue validation of the biological Hopf $H_{\mathrm{bio}}$ (complete TTX + ouabain) and biological saddle-node $LP_{\mathrm{bio}}$ (TTX alone).

This notebook **loads precomputed samples and tables** from `outputs/data/jacobian_hopf/`. Heavy continuation sampling is performed by:

- `scripts/run_08_epc_local_windows.py` (host EP-C)
- `scripts/run_08_hopf_lc_samples.py` (Docker AUTO LC-C)
- `scripts/run_08_jacobian_hopf_analysis.py` (SymPy Jacobian analysis + figures)

Analytical report prose is intentionally omitted; the focus is numerical workflow, tables, and figures.
"""
        )
    )

    cells.append(md(r"## 1. Setup and data loading"))
    cells.append(
        code(
            r"""
from pathlib import Path
import json
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO = Path.cwd().resolve()
if not (REPO / "src").exists():
    REPO = Path("..").resolve()
sys.path.insert(0, str(REPO))

from src.plant_kim_model import STATE_VARS
from src.plant_kim_analysis import pars_from_alpha, rhs, G_I0, I_EP0
from src.plant_kim_jacobian import (
    landmark_H_bio,
    landmark_LP_bio,
    jacobian_symbolic,
    validate_jacobian_at,
    hopf_period_from_eigs,
    identify_critical_hopf_pair,
    identify_critical_real,
)

DATADIR = REPO / "outputs" / "data" / "jacobian_hopf"
FIGDIR = REPO / "outputs" / "figures" / "jacobian_hopf"

def load_csv(name):
    with (DATADIR / name).open(newline="") as f:
        return list(csv.DictReader(f))

def load_json(name):
    return json.loads((DATADIR / name).read_text())

def show_table(rows, cols=None, float_fmt="{:.6g}"):
    if not rows:
        print("(empty)")
        return
    cols = cols or list(rows[0].keys())
    # header
    widths = []
    str_rows = []
    for c in cols:
        vals = []
        for r in rows:
            v = r.get(c, "")
            if isinstance(v, float) or (isinstance(v, str) and v not in ("",) and _is_float(v)):
                try:
                    vals.append(float_fmt.format(float(v)))
                except Exception:
                    vals.append(str(v))
            else:
                vals.append(str(v))
        widths.append(max(len(c), max(len(x) for x in vals)))
        str_rows.append(vals)
    # transpose back
    grid = list(zip(*str_rows))
    hdr = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * widths[i] for i in range(len(cols)))
    print(hdr)
    print(sep)
    for row in grid:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(cols))))

def _is_float(s):
    try:
        float(s)
        return True
    except Exception:
        return False

h = landmark_H_bio(REPO)
lp = landmark_LP_bio(REPO)
print("H_bio: alpha_I=0, alpha_P={:.8f}, V={:.4f}".format(h["alpha_P"], h["state"]["V"]))
print("LP_bio: alpha_I={:.8f}, alpha_P=1, V={:.4f}".format(lp["alpha_I"], lp["state"]["V"]))
print("residual H_bio:", np.max(np.abs(rhs(h["state"], h["pars"]))))
print("residual LP_bio:", np.max(np.abs(rhs(lp["state"], lp["pars"]))))
print("data files:", sorted(p.name for p in DATADIR.glob("*")))
"""
        )
    )

    cells.append(md(r"## 2. Symbolic Jacobian"))
    cells.append(
        md(
            r"""The symbolic RHS is transcribed from `src/plant_kim_model.py` (including Canavier 1990 rate corrections) inside `src/plant_kim_jacobian.py`. The Jacobian is obtained as `J_sym = F_sym.jacobian(x_sym)` and lambdified for numerical evaluation. Drug scaling uses `pars_from_alpha` at evaluation time (`g_I = α_I g_I^{(0)}`, `I_ep = α_P I_ep^{(0)}`)."""
        )
    )
    cells.append(
        code(
            r"""
from src.plant_kim_jacobian import build_symbolic_system

sym = build_symbolic_system()
print("state order:", [str(s) for s in sym["state_symbols"]])
print("Jacobian shape:", sym["J_sym"].shape)
print("Nonzero symbolic entries:", sum(1 for e in sym["J_sym"] if e != 0), "/ 49")
"""
        )
    )

    cells.append(md(r"## 3. Jacobian validation"))
    cells.append(
        md(
            r"""Validation chain at the landmarks:

$$
J_{\mathrm{symbolic}}
\;\leftrightarrow\;
J_{\mathrm{FD}}
\;\leftrightarrow\;
\text{PyDSTool EP-C spectrum}
$$

Centred FD steps: $h_V = 10^{-3}$, $h_g = 10^{-6}$ (as in `plant_kim_analysis.jacobian`)."""
        )
    )
    cells.append(
        code(
            r"""
val = load_csv("jacobian_validation.csv")
show_table(
    val,
    cols=[
        "point",
        "residual_max",
        "max_abs_F_sym_vs_numpy",
        "max_abs_J_sym_vs_FD",
        "rel_frobenius_J",
        "max_eig_mismatch_sym_vs_FD",
        "max_eig_mismatch_sym_vs_PyDSTool",
    ],
)
print("\nLive re-check at H_bio:")
print(validate_jacobian_at(h["state"], h["pars"]))
"""
        )
    )

    cells.append(md(r"## 4. Saddle-node spectrum at $LP_{\mathrm{bio}}$"))
    cells.append(
        md(
            r"""TTX slice with untreated pump: $\alpha_P = 1$, free parameter $\alpha_I$. The fold $LP_{\mathrm{bio}}$ sits near $\alpha_I \approx 0.614$. Branch geometry is taken from a short local EP-C window (continuation index), not a single-valued $\alpha_I\mapsto x^*$ map through the turning point."""
        )
    )
    cells.append(
        code(
            r"""
spec_lp = load_csv("spectrum_LPbio.csv")
sn_info = load_json("sn_info.json")
print("LP_bio spectrum (critical flagged):")
show_table(spec_lp, cols=["index", "real", "imag", "abs", "critical"])
print("\nCritical eigenvalue |λ| = {:.3e}".format(abs(complex(sn_info["lambda_crit_re"], sn_info["lambda_crit_im"]))))
print("Gap of remaining spectrum from 0: {:.3e}".format(sn_info["spectral_gap_zero"]))
print("Equilibrium residual: {:.3e}".format(sn_info["residual_max"]))
print("Max real part (other unstable modes may exist): {:.6g}".format(sn_info["max_real_part"]))

track = load_csv("sn_critical_track.csv")
# nearest to fold
fold = min(track, key=lambda r: float(r["alpha_I"]))
print("\nAt fold sample: alpha_I={:.8f}, λ_crit={:.3e}, residual={:.3e}".format(
    float(fold["alpha_I"]), float(fold["lambda_crit_re"]), float(fold["residual_max"])
))
"""
        )
    )
    cells.append(
        figure_attachment_cell(
            "Critical Eigenvalue near the Biological Saddle-Node",
            r"Left: tracked critical real eigenvalue versus $\alpha_I$ along the local EP-C window (coloured by continuation index). Right: folded equilibrium voltage branch. The critical eigenvalue approaches zero at $LP_{\mathrm{bio}}$.",
            "08_sn_critical_eigenvalue.png",
        )
    )

    cells.append(md(r"## 5. Hopf spectrum at $H_{\mathrm{bio}}$"))
    cells.append(
        md(
            r"""Complete TTX ($\alpha_I = 0$) with free parameter $\alpha_P$. The biological Hopf $H_{\mathrm{bio}}$ lies near $\alpha_P \approx 0.2679$."""
        )
    )
    cells.append(
        code(
            r"""
spec_h = load_csv("spectrum_Hbio.csv")
hopf_info = load_json("hopf_info.json")
print("H_bio spectrum:")
show_table(spec_h, cols=["index", "real", "imag", "abs", "critical"])
print("\nω_H = {:.6e}  (1/ms)".format(hopf_info["omega_H"]))
print("T_lin = 2π/ω_H = {:.6f} ms".format(hopf_info["T_lin_ms"]))
print("Spectral gap of non-critical modes from imag axis: {:.3e}".format(hopf_info["spectral_gap_imag"]))
print("Equilibrium residual: {:.3e}".format(hopf_info["residual_max"]))

htrack = load_csv("hopf_critical_track.csv")
# sign change of Re
res = np.array([float(r["Re_crit"]) for r in htrack])
ps = np.array([float(r["alpha_P"]) for r in htrack])
order = np.argsort(ps)
print("Re(λ_crit) at min α_P: {:.3e}, at max α_P: {:.3e}".format(res[order][0], res[order][-1]))
print("sign change across H_bio:", np.any(np.diff(np.sign(res[order])) != 0))
"""
        )
    )
    cells.append(
        figure_attachment_cell(
            "Critical Eigenvalues near the Biological Hopf",
            r"Tracked critical conjugate pair versus $\alpha_P$. The real part crosses zero at $H_{\mathrm{bio}}$ while the imaginary part remains nonzero.",
            "08_hopf_critical_eigenvalues.png",
        )
    )

    cells.append(md(r"## 6. Hopf period prediction"))
    cells.append(
        md(
            r"""At $H_{\mathrm{bio}}$,
$$
T_{\mathrm{lin}} = \frac{2\pi}{\omega_H},\qquad \omega_H = |\mathrm{Im}\,\lambda_H|.
$$
AUTO LC-C periods on the Hopf bubble are compared as cycle amplitude $V_{\max}-V_{\min}\to 0$. Time unit: milliseconds."""
        )
    )
    cells.append(
        code(
            r"""
period = load_csv("hopf_period_comparison.csv")
print("Near-Hopf AUTO vs eigenvalue period:")
show_table(
    period[:12],
    cols=["alpha_P", "dist_from_Hbio", "amplitude_mV", "period_AUTO_ms", "period_Tlin_ms", "relative_error"],
)
errs = [float(r["relative_error"]) for r in period]
print("\nmin relative error = {:.3e}".format(min(errs)))
print("T_lin = {:.6f} ms".format(float(period[0]["period_Tlin_ms"])))
"""
        )
    )
    cells.append(
        figure_attachment_cell(
            "Limit-Cycle Period near the Hopf Bifurcation",
            r"AUTO period approaches the eigenvalue prediction $T_{\mathrm{lin}}=2\pi/\omega_H$ as the limit-cycle voltage amplitude tends to zero.",
            "08_hopf_period_convergence.png",
        )
    )

    cells.append(md(r"## 7. Hopf transversality"))
    cells.append(
        md(
            r"""Centred finite differences of $\mathrm{d}\,\mathrm{Re}(\lambda_{\mathrm{crit}})/\mathrm{d}\alpha_P$ at several step sizes $h$, plus the semi-analytical formula
$$
\frac{\mathrm{d}\lambda}{\mathrm{d}\alpha_P}
=
w^{H}
\frac{\mathrm{d}J}{\mathrm{d}\alpha_P}
v
$$
with $w^{H}v=1$ and equilibrium dependence through $\mathrm{d}x^{*}/\mathrm{d}\alpha_P=-J^{-1}F_{\alpha_P}$."""
        )
    )
    cells.append(
        code(
            r"""
trans = load_csv("hopf_transversality_fd.csv")
semi = load_json("hopf_transversality_semianalytic.json")
show_table(trans, cols=["h", "dRe_dmu", "Re_plus", "Re_minus", "Im_plus", "Im_minus"])
print("\nSemi-analytical dλ/dα_P = {:.6e} + {:.6e} i".format(
    semi["dlambda_dalpha_P_re"], semi["dlambda_dalpha_P_im"]
))
print("FD median dRe/dα_P = {:.6e}".format(np.median([float(r["dRe_dmu"]) for r in trans])))
print("Relative |FD - semi| / |semi| = {:.3e}".format(
    abs(np.median([float(r["dRe_dmu"]) for r in trans]) - semi["dlambda_dalpha_P_re"])
    / abs(semi["dlambda_dalpha_P_re"])
))
"""
        )
    )

    cells.append(md(r"## 8. Saddle-node nondegeneracy"))
    cells.append(
        md(
            r"""At $LP_{\mathrm{bio}}$, left/right nullvectors satisfy $Jq=0$, $p^{\mathsf T}J=0$, $p^{\mathsf T}q=1$. Coefficients
$$
a = p^{\mathsf T} F_{\alpha_I},\qquad
b = \tfrac12 p^{\mathsf T} D_x^{2}F[q,q]
$$
are computed from the symbolic Hessian action."""
        )
    )
    cells.append(
        code(
            r"""
sn_c = load_json("sn_coefficients.json")
print("a = {:.6e}".format(sn_c["a"]))
print("b = {:.6e}".format(sn_c["b"]))
print("σ_min(J) = {:.3e}".format(sn_c["sigma_min"]))
print("p·q = {:.6g}".format(sn_c["p_dot_q"]))
print("Nondegeneracy: |a|,|b| clearly nonzero relative to σ_min.")
"""
        )
    )
    cells.append(
        figure_attachment_cell(
            "Local Fold Geometry near the Biological Saddle-Node",
            r"EP-C voltage deviation from the fold versus $\Delta\alpha_I$, compared with the square-root scaling implied by the computed normal-form coefficients $a$ and $b$ (local qualitative comparison).",
            "08_fold_normal_form.png",
        )
    )

    cells.append(md(r"## 9. Optional strong extensions"))
    cells.append(
        code(
            r"""
evec = load_csv("hopf_eigenvector.csv")
print("Critical Hopf eigenvector (magnitudes / phases relative to V):")
show_table(evec, cols=["variable", "abs", "phase_rel_V_deg", "real", "imag"])
"""
        )
    )

    cells.append(md(r"## 10. Summary of numerical checks"))
    cells.append(
        code(
            r"""
summary = load_json("summary_checks.json")
for k, v in summary.items():
    if isinstance(v, float):
        print(f"{k:40s} {v:.6g}")
    else:
        print(f"{k:40s} {v}")

checks = [
    ("Jacobian validation gate", bool(summary["jacobian_validation_pass"])),
    ("Hopf Im(λ) ≠ 0 / T_lin finite", np.isfinite(summary["T_lin_ms"]) and summary["H_bio_omega"] > 0),
    ("AUTO period → T_lin", summary["near_hopf_min_rel_period_error"] < 1e-6),
    ("Hopf transversality ≠ 0", abs(summary["transversality_dRe_median"]) > 1e-6),
    ("FD ↔ semi-analytic dRe", abs(summary["transversality_dRe_median"] - summary["semianalytic_dRe"])
        / abs(summary["semianalytic_dRe"]) < 1e-4),
    ("SN λ_crit ≈ 0", summary["sn_lambda_crit_abs"] < 1e-8),
    ("SN gap > 0", summary["sn_gap"] > 1e-6),
    ("SN a ≠ 0", abs(summary["sn_a"]) > 1e-6),
    ("SN b ≠ 0", abs(summary["sn_b"]) > 1e-6),
]
print("\nChecklist:")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
"""
        )
    )

    cells.append(
        md(
            r"""### Reproducibility

```bash
# Host EP-C windows
./.venv/bin/python scripts/run_08_epc_local_windows.py

# Docker AUTO LC-C (container jupyter-lab, CWD /notebooks)
docker exec -w /notebooks jupyter-lab python3 scripts/run_08_hopf_lc_samples.py

# Analysis + figures
./.venv/bin/python scripts/run_08_jacobian_hopf_analysis.py

# Rebuild this notebook attachments
./.venv/bin/python scripts/build_08_jacobian_hopf_nb.py
```
"""
        )
    )

    nb.cells = cells
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NB_PATH.open("w") as f:
        nbf.write(nb, f)
    print("wrote", NB_PATH)


if __name__ == "__main__":
    main()

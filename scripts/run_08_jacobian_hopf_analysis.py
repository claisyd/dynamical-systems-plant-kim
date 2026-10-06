#!/usr/bin/env python3
"""Notebook 08 analysis engine: Jacobian validation, spectra, period, transversality.

Reads local EP-C / LC-C samples from outputs/data/jacobian_hopf/ and writes
report-quality tables and figures. Host .venv (SymPy required).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.plant_kim_analysis import G_I0, I_EP0, jacobian as jacobian_fd, pars_from_alpha, rhs
from src.plant_kim_jacobian import (
    format_spectrum_table,
    hopf_eigenvalue_derivative,
    hopf_period_from_eigs,
    hopf_transversality_fd,
    identify_critical_hopf_pair,
    identify_critical_real,
    jacobian_symbolic,
    landmark_H_bio,
    landmark_LP_bio,
    match_spectra,
    nullvectors_zero,
    sn_coefficients,
    spectral_gap_excluding,
    track_eigenvalues,
    validate_jacobian_at,
)
from src.plant_kim_model import STATE_VARS

DATADIR = REPO_ROOT / "outputs" / "data" / "jacobian_hopf"
FIGDIR = REPO_ROOT / "outputs" / "figures" / "jacobian_hopf"
DATADIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: Sequence[Dict], fieldnames: Optional[Sequence[str]] = None):
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, float):
                    out[k] = f"{v:.12g}" if np.isfinite(v) else ""
                elif isinstance(v, (bool, int)):
                    out[k] = str(v)
                elif v is None:
                    out[k] = ""
                else:
                    out[k] = v
            w.writerow(out)


def save_json(path: Path, obj):
    def conv(o):
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, complex):
            return {"re": float(o.real), "im": float(o.imag)}
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o

    path.write_text(json.dumps(conv(obj), indent=2))


def parse_pydstool_eigs(row: Dict[str, str]) -> Optional[np.ndarray]:
    n = int(float(row.get("n_evals") or 0))
    if n <= 0:
        return None
    ev = []
    for j in range(n):
        re = float(row[f"eig{j}_re"])
        im = float(row[f"eig{j}_im"])
        ev.append(re + 1j * im)
    return np.asarray(ev, dtype=complex)


def state_from_epc_row(row: Dict[str, str]) -> Dict[str, float]:
    return {k: float(row[k]) for k in STATE_VARS}


def epc_sort_by_arclength(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Order by continuation index (preserves fold geometry)."""
    return sorted(rows, key=lambda r: int(r["index"]))


# ---------------------------------------------------------------------------
# 1–3. Validation
# ---------------------------------------------------------------------------


def run_validation(h, lp) -> List[Dict]:
    rows_out = []
    for lm, tag in ((h, "H_bio"), (lp, "LP_bio")):
        # nearest EP-C point for PyDSTool spectrum
        epc_path = DATADIR / ("epc_Hbio_local.csv" if tag == "H_bio" else "epc_LPbio_local.csv")
        epc = load_csv(epc_path)
        pkey = "alpha_P" if tag == "H_bio" else "alpha_I"
        target = lm["alpha_P"] if tag == "H_bio" else lm["alpha_I"]
        nearest = min(epc, key=lambda r: abs(float(r["param"]) - target))
        pyd = parse_pydstool_eigs(nearest)

        v = validate_jacobian_at(lm["state"], lm["pars"], pydstool_evals=pyd)
        # also compare eigvals of symbolic vs pydstool explicitly
        eigs_s = np.linalg.eigvals(jacobian_symbolic(lm["state"], lm["pars"]))
        eigs_fd = np.linalg.eigvals(jacobian_fd(lm["state"], lm["pars"]))
        _, _, eig_mis_fd = match_spectra(eigs_s, eigs_fd)
        eig_mis_p = float("nan")
        if pyd is not None:
            _, _, eig_mis_p = match_spectra(eigs_s, pyd)
        rows_out.append(
            {
                "point": tag,
                "alpha_I": lm["alpha_I"],
                "alpha_P": lm["alpha_P"],
                "residual_max": v.residual_max,
                "max_abs_F_sym_vs_numpy": v.max_abs_F_diff,
                "max_abs_J_sym_vs_FD": v.max_abs_J_diff,
                "rel_frobenius_J": v.rel_frobenius_J,
                "max_eig_mismatch_sym_vs_FD": eig_mis_fd,
                "max_eig_mismatch_sym_vs_PyDSTool": eig_mis_p,
                "FD_h_V": 1e-3,
                "FD_h_g": 1e-6,
            }
        )
        # random-state spot check
    rng = np.random.default_rng(1)
    pars = pars_from_alpha(0.4, 0.7)
    maxF = maxJ = 0.0
    for _ in range(30):
        x = np.array([-55 + 30 * rng.random(), *rng.random(6)])
        from src.plant_kim_jacobian import rhs_symbolic

        maxF = max(maxF, float(np.max(np.abs(rhs_symbolic(x, pars) - rhs(x, pars)))))
        maxJ = max(
            maxJ,
            float(
                np.max(
                    np.abs(
                        jacobian_symbolic(x, pars) - jacobian_fd(x, pars)
                    )
                )
            ),
        )
    rows_out.append(
        {
            "point": "random_states_n30",
            "alpha_I": 0.4,
            "alpha_P": 0.7,
            "residual_max": float("nan"),
            "max_abs_F_sym_vs_numpy": maxF,
            "max_abs_J_sym_vs_FD": maxJ,
            "rel_frobenius_J": float("nan"),
            "max_eig_mismatch_sym_vs_FD": float("nan"),
            "max_eig_mismatch_sym_vs_PyDSTool": float("nan"),
            "FD_h_V": 1e-3,
            "FD_h_g": 1e-6,
        }
    )
    save_csv(DATADIR / "jacobian_validation.csv", rows_out)
    return rows_out


# ---------------------------------------------------------------------------
# Spectrum tracking helpers
# ---------------------------------------------------------------------------


def track_along_epc(epc_rows: List[Dict[str, str]], freepar: str) -> Dict:
    ordered = epc_sort_by_arclength(epc_rows)
    eigs_list = []
    params = []
    Vs = []
    residuals = []
    max_re_pyd = []
    states = []
    for r in ordered:
        a_I, a_P = float(r["alpha_I"]), float(r["alpha_P"])
        st = state_from_epc_row(r)
        pars = pars_from_alpha(a_I, a_P)
        J = jacobian_symbolic(st, pars)
        eigs_list.append(np.linalg.eigvals(J))
        params.append(float(r["param"]))
        Vs.append(float(r["V"]))
        residuals.append(float(r["residual_max"]))
        max_re_pyd.append(float(r.get("max_real_pydstool") or "nan"))
        states.append(st)
    tracked = track_eigenvalues(eigs_list)
    return {
        "param": np.asarray(params),
        "V": np.asarray(Vs),
        "residual": np.asarray(residuals),
        "max_re_pyd": np.asarray(max_re_pyd),
        "tracked": tracked,
        "states": states,
        "freepar": freepar,
        "alpha_I": np.asarray([float(r["alpha_I"]) for r in ordered]),
        "alpha_P": np.asarray([float(r["alpha_P"]) for r in ordered]),
        "index": np.asarray([int(r["index"]) for r in ordered]),
    }


def spectrum_at_landmark(lm, kind: str) -> Tuple[List[Dict], Dict]:
    eigs = np.linalg.eigvals(jacobian_symbolic(lm["state"], lm["pars"]))
    if kind == "hopf":
        i_pos, i_neg = identify_critical_hopf_pair(eigs)
        crit = [i_pos, i_neg]
        gap = spectral_gap_excluding(eigs, crit, mode="imag")
        info = hopf_period_from_eigs(eigs)
        info["spectral_gap_imag"] = gap
        info["residual_max"] = float(np.max(np.abs(rhs(lm["state"], lm["pars"]))))
    else:
        i0 = identify_critical_real(eigs)
        crit = [i0]
        gap = spectral_gap_excluding(eigs, crit, mode="zero")
        info = {
            "lambda_crit": complex(eigs[i0]),
            "spectral_gap_zero": gap,
            "residual_max": float(np.max(np.abs(rhs(lm["state"], lm["pars"])))),
            "max_real_part": float(np.max(np.real(eigs))),
        }
    table = format_spectrum_table(eigs, crit)
    return table, info


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_hopf_critical(track: Dict, alpha_H: float, path: Path):
    eigs0 = track["tracked"][np.argmin(np.abs(track["param"] - alpha_H))]
    i_pos, i_neg = identify_critical_hopf_pair(eigs0)
    # re-identify along track using fixed indices after matching from centre
    # Re-track: pick indices at Hopf and follow those columns
    re = np.real(track["tracked"][:, i_pos])
    im = np.imag(track["tracked"][:, i_pos])
    # Also verify partner
    re2 = np.real(track["tracked"][:, i_neg])
    p = track["param"]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    axes[0].axhline(0.0, color="0.4", lw=0.8)
    axes[0].axvline(alpha_H, color="C3", ls="--", lw=1.0, label=r"$H_{\mathrm{bio}}$")
    axes[0].plot(p, re, "C0-", lw=1.8, label=r"$\mathrm{Re}\,\lambda_{\mathrm{crit}}$")
    axes[0].plot(p, re2, "C0--", lw=1.0, alpha=0.7)
    axes[0].set_ylabel(r"$\mathrm{Re}\,\lambda$")
    axes[0].set_title("Critical Eigenvalues near the Biological Hopf")
    axes[0].legend(loc="best")

    axes[1].axvline(alpha_H, color="C3", ls="--", lw=1.0)
    axes[1].plot(p, im, "C1-", lw=1.8, label=r"$\mathrm{Im}\,\lambda_{\mathrm{crit}}$")
    axes[1].plot(p, -im, "C1--", lw=1.0, alpha=0.7)
    axes[1].set_xlabel(r"$\alpha_P$")
    axes[1].set_ylabel(r"$\mathrm{Im}\,\lambda$")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def fig_sn_critical(track: Dict, alpha_LP: float, path: Path):
    # Identify critical real column at the fold (min alpha_I for left fold)
    i_fold = int(np.argmin(track["param"]))
    eigs_f = track["tracked"][i_fold]
    i_crit = identify_critical_real(eigs_f)
    lam = np.real(track["tracked"][:, i_crit])
    p = track["param"]
    V = track["V"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].axhline(0.0, color="0.4", lw=0.8)
    axes[0].axvline(alpha_LP, color="C1", ls="--", lw=1.0, label=r"$LP_{\mathrm{bio}}$")
    # colour by branch index order
    sc = axes[0].scatter(p, lam, c=track["index"], cmap="viridis", s=12, zorder=3)
    axes[0].plot(p, lam, color="0.7", lw=0.8, zorder=1)
    axes[0].set_xlabel(r"$\alpha_I$")
    axes[0].set_ylabel(r"$\lambda_{\mathrm{crit}}$")
    axes[0].set_title("Critical Eigenvalue near the Biological Saddle-Node")
    axes[0].legend(loc="best")
    fig.colorbar(sc, ax=axes[0], label="continuation index")

    axes[1].axvline(alpha_LP, color="C1", ls="--", lw=1.0)
    axes[1].plot(p, V, "C0-", lw=1.5)
    axes[1].set_xlabel(r"$\alpha_I$")
    axes[1].set_ylabel(r"$V^*$ (mV)")
    axes[1].set_title("Equilibrium Branch near the Fold")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def fig_period_convergence(lc_rows: List[Dict], T_lin: float, alpha_H: float, path: Path):
    a = np.array([float(r["alpha_P"]) for r in lc_rows])
    T = np.array([float(r["period_ms"]) for r in lc_rows])
    A = np.array([float(r["amplitude_mV"]) for r in lc_rows])
    # near H_bio end: alpha_P close to H and prefer small amplitude; drop exact zeros duplicates
    mask = np.isfinite(A) & np.isfinite(T) & (a < 0.6)
    a, T, A = a[mask], T[mask], A[mask]
    # unique by amplitude bin
    order = np.argsort(A)
    a, T, A = a[order], T[order], A[order]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].axhline(T_lin, color="C3", ls="--", lw=1.4, label=rf"$T_{{\mathrm{{lin}}}}={T_lin:.2f}$ ms")
    axes[0].plot(A, T, "o-", ms=4, lw=1.2, color="C0")
    axes[0].set_xlabel(r"cycle amplitude $V_{\max}-V_{\min}$ (mV)")
    axes[0].set_ylabel(r"period (ms)")
    axes[0].set_title("Limit-Cycle Period near the Hopf Bifurcation")
    axes[0].legend(loc="best")

    rel = np.abs(T - T_lin) / T_lin
    axes[1].semilogy(A, np.maximum(rel, 1e-16), "o-", ms=4, lw=1.2, color="C2")
    axes[1].set_xlabel(r"cycle amplitude (mV)")
    axes[1].set_ylabel(r"relative period error")
    axes[1].set_title("Period Error versus Amplitude")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return a, T, A, rel


def fig_fold_normal_form(
    track: Dict,
    alpha_LP: float,
    a_coef: float,
    b_coef: float,
    path: Path,
    q_V: float,
):
    """Compare local V-shift vs sqrt scaling if coefficients reliable.

    With nullvector normalization p^T q = 1 (and ||q||_2 = 1 from SVD),
    the reduced coordinate satisfies u = ±sqrt(-(a/b) Δα_I), while the
    voltage component of the branch is V - V_LP ≈ q_V u.
    """
    p = track["param"]
    V = track["V"]
    i_fold = int(np.argmin(p))
    V_lp = V[i_fold]
    dV = V - V_lp
    mu = p - alpha_LP
    # predicted: u^2 = - (a/b) mu; voltage = q_V * u
    ratio = -a_coef / b_coef if abs(b_coef) > 0 else float("nan")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(mu, dV, "o", ms=3, color="C0", label="EP-C $(V-V_{LP})$")
    mu_grid = np.linspace(0, max(0, mu.max()), 200)
    if ratio > 0:
        pred = float(q_V) * np.sqrt(np.maximum(ratio * mu_grid, 0.0))
        ax.plot(
            mu_grid,
            pred,
            "C3-",
            lw=1.6,
            label=r"$\pm q_V\sqrt{-(a/b)\,\Delta\alpha_I}$",
        )
        ax.plot(mu_grid, -pred, "C3-", lw=1.6)
    ax.axvline(0.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel(r"$\alpha_I - \alpha_I^{LP}$")
    ax.set_ylabel(r"$V - V_{LP}$ (mV)")
    ax.set_title("Local Fold Geometry near the Biological Saddle-Node")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    h = landmark_H_bio(REPO_ROOT)
    lp = landmark_LP_bio(REPO_ROOT)
    print("Landmarks OK:", h["alpha_P"], lp["alpha_I"])

    print("Validation ...")
    val_rows = run_validation(h, lp)
    for r in val_rows:
        print(
            f"  {r['point']}: Fdiff={r['max_abs_F_sym_vs_numpy']:.3e} "
            f"Jdiff={r['max_abs_J_sym_vs_FD']:.3e}"
        )

    # Spectra at landmarks
    print("Spectra at landmarks ...")
    hopf_table, hopf_info = spectrum_at_landmark(h, "hopf")
    sn_table, sn_info = spectrum_at_landmark(lp, "sn")
    save_csv(DATADIR / "spectrum_Hbio.csv", hopf_table)
    save_csv(DATADIR / "spectrum_LPbio.csv", sn_table)
    save_json(DATADIR / "hopf_info.json", {**hopf_info, "alpha_P": h["alpha_P"], "alpha_I": 0.0})
    save_json(
        DATADIR / "sn_info.json",
        {
            "lambda_crit_re": float(np.real(sn_info["lambda_crit"])),
            "lambda_crit_im": float(np.imag(sn_info["lambda_crit"])),
            "spectral_gap_zero": sn_info["spectral_gap_zero"],
            "residual_max": sn_info["residual_max"],
            "max_real_part": sn_info["max_real_part"],
            "alpha_I": lp["alpha_I"],
            "alpha_P": 1.0,
        },
    )
    print(f"  T_lin = {hopf_info['T_lin_ms']:.6f} ms, omega={hopf_info['omega_H']:.6e}")
    print(f"  LP lambda_crit = {sn_info['lambda_crit']}, gap={sn_info['spectral_gap_zero']:.3e}")

    # Track along branches
    print("Tracking eigenvalues ...")
    epc_h = load_csv(DATADIR / "epc_Hbio_local.csv")
    epc_lp = load_csv(DATADIR / "epc_LPbio_local.csv")
    track_h = track_along_epc(epc_h, "alpha_P")
    track_lp = track_along_epc(epc_lp, "alpha_I")

    # Export tracked critical eigenvalues
    iH = np.argmin(np.abs(track_h["param"] - h["alpha_P"]))
    ip, _ = identify_critical_hopf_pair(track_h["tracked"][iH])
    hopf_track_rows = []
    for i in range(len(track_h["param"])):
        lam = track_h["tracked"][i, ip]
        hopf_track_rows.append(
            {
                "index": int(track_h["index"][i]),
                "alpha_P": float(track_h["param"][i]),
                "V": float(track_h["V"][i]),
                "Re_crit": float(lam.real),
                "Im_crit": float(lam.imag),
                "residual_max": float(track_h["residual"][i]),
                "max_real_pydstool": float(track_h["max_re_pyd"][i]),
                "max_real_symbolic": float(np.max(np.real(track_h["tracked"][i]))),
            }
        )
    save_csv(DATADIR / "hopf_critical_track.csv", hopf_track_rows)

    iLP = int(np.argmin(track_lp["param"]))
    ic = identify_critical_real(track_lp["tracked"][iLP])
    sn_track_rows = []
    for i in range(len(track_lp["param"])):
        lam = track_lp["tracked"][i, ic]
        sn_track_rows.append(
            {
                "index": int(track_lp["index"][i]),
                "alpha_I": float(track_lp["param"][i]),
                "V": float(track_lp["V"][i]),
                "lambda_crit_re": float(lam.real),
                "lambda_crit_im": float(lam.imag),
                "residual_max": float(track_lp["residual"][i]),
                "max_real_pydstool": float(track_lp["max_re_pyd"][i]),
            }
        )
    save_csv(DATADIR / "sn_critical_track.csv", sn_track_rows)

    fig_hopf_critical(track_h, h["alpha_P"], FIGDIR / "08_hopf_critical_eigenvalues.png")
    fig_sn_critical(track_lp, lp["alpha_I"], FIGDIR / "08_sn_critical_eigenvalue.png")
    print("  figures: hopf + sn critical eigenvalues")

    # Period comparison
    print("Period comparison ...")
    lc = load_csv(DATADIR / "lcc_Hbio_samples.csv")
    T_lin = float(hopf_info["T_lin_ms"])
    a, T, A, rel = fig_period_convergence(
        lc, T_lin, h["alpha_P"], FIGDIR / "08_hopf_period_convergence.png"
    )
    # Build comparison table: subsample by amplitude
    period_rows = []
    # take unique-ish amplitudes
    seen = []
    for ai, Ti, Ai, ri in zip(a, T, A, rel):
        if Ai < 1e-12:
            continue
        if any(abs(Ai - s) < 1e-4 for s in seen):
            continue
        seen.append(Ai)
        period_rows.append(
            {
                "alpha_P": float(ai),
                "dist_from_Hbio": abs(float(ai) - h["alpha_P"]),
                "amplitude_mV": float(Ai),
                "period_AUTO_ms": float(Ti),
                "period_Tlin_ms": T_lin,
                "relative_error": float(ri),
            }
        )
        if len(period_rows) >= 20:
            break
    # also include a few larger amplitudes
    for ai, Ti, Ai, ri in zip(a, T, A, rel):
        if Ai < 0.5:
            continue
        if any(abs(Ai - r["amplitude_mV"]) < 0.05 for r in period_rows):
            continue
        period_rows.append(
            {
                "alpha_P": float(ai),
                "dist_from_Hbio": abs(float(ai) - h["alpha_P"]),
                "amplitude_mV": float(Ai),
                "period_AUTO_ms": float(Ti),
                "period_Tlin_ms": T_lin,
                "relative_error": float(ri),
            }
        )
        if len(period_rows) >= 28:
            break
    period_rows.sort(key=lambda r: r["amplitude_mV"])
    save_csv(DATADIR / "hopf_period_comparison.csv", period_rows)
    print(f"  near-Hopf relative errors: min={min(r['relative_error'] for r in period_rows):.3e}")

    # Transversality
    print("Hopf transversality ...")
    hs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    trans = hopf_transversality_fd(h["state"], 0.0, h["alpha_P"], hs, freepar="alpha_P")
    save_csv(DATADIR / "hopf_transversality_fd.csv", trans)
    for t in trans:
        print(f"  h={t['h']:.1e}  dRe/dalpha_P={t['dRe_dmu']:.6e}")

    # Semi-analytical derivative
    print("Semi-analytical dlambda/dalpha_P ...")
    dlam = hopf_eigenvalue_derivative(h["state"], h["pars"], "I_ep", I_EP0)
    print(f"  dlambda/dalpha_P = {dlam.real:.6e} + {dlam.imag:.6e} i")
    save_json(
        DATADIR / "hopf_transversality_semianalytic.json",
        {"dlambda_dalpha_P_re": dlam.real, "dlambda_dalpha_P_im": dlam.imag},
    )

    # SN coefficients
    print("Saddle-node coefficients ...")
    sn_c = sn_coefficients(lp["state"], lp["pars"], "g_I", G_I0)
    q_sn, _p_sn = nullvectors_zero(jacobian_symbolic(lp["state"], lp["pars"]))
    q_V = float(q_sn[0])  # STATE_VARS index 0 is V; existing p^T q = 1 normalization
    sn_c["q_V"] = q_V
    print(f"  a={sn_c['a']:.6e}  b={sn_c['b']:.6e}  sigma_min={sn_c['sigma_min']:.3e}")
    print(f"  q_V={q_V:.12g}  (voltage component of right nullvector)")
    save_json(DATADIR / "sn_coefficients.json", sn_c)
    save_csv(
        DATADIR / "sn_coefficients.csv",
        [
            {
                "alpha_I": lp["alpha_I"],
                "a": sn_c["a"],
                "b": sn_c["b"],
                "sigma_min": sn_c["sigma_min"],
                "p_dot_q": sn_c["p_dot_q"],
                "q_V": q_V,
            }
        ],
    )
    if abs(sn_c["a"]) > 1e-10 and abs(sn_c["b"]) > 1e-10:
        fig_fold_normal_form(
            track_lp,
            lp["alpha_I"],
            sn_c["a"],
            sn_c["b"],
            FIGDIR / "08_fold_normal_form.png",
            q_V=q_V,
        )
        print("  fold normal-form figure saved")

    # Hopf eigenvector structure
    print("Hopf eigenvector structure ...")
    J = jacobian_symbolic(h["state"], h["pars"])
    eigs = np.linalg.eigvals(J)
    ip, _ = identify_critical_hopf_pair(eigs)
    from src.plant_kim_jacobian import left_right_eigenvectors

    v, w = left_right_eigenvectors(J, eigs[ip])
    # phase relative to V (index 0)
    phase0 = np.angle(v[0])
    eigvec_rows = []
    for i, name in enumerate(STATE_VARS):
        eigvec_rows.append(
            {
                "variable": name,
                "abs": float(np.abs(v[i])),
                "phase_rad": float(np.angle(v[i])),
                "phase_rel_V_rad": float(np.angle(v[i]) - phase0),
                "phase_rel_V_deg": float(np.degrees(np.angle(v[i]) - phase0)),
                "real": float(np.real(v[i])),
                "imag": float(np.imag(v[i])),
            }
        )
    save_csv(DATADIR / "hopf_eigenvector.csv", eigvec_rows)

    # Summary checks
    summary = {
        "jacobian_validation_pass": all(
            (r["max_abs_J_sym_vs_FD"] < 1e-5 or r["point"] == "random_states_n30")
            and (r["max_abs_F_sym_vs_numpy"] < 1e-10 or r["point"] == "random_states_n30")
            for r in val_rows
            if r["point"] != "random_states_n30"
        )
        and val_rows[-1]["max_abs_J_sym_vs_FD"] < 1e-5,
        "H_bio_omega": hopf_info["omega_H"],
        "T_lin_ms": T_lin,
        "near_hopf_min_rel_period_error": min(r["relative_error"] for r in period_rows),
        "transversality_dRe_median": float(np.median([t["dRe_dmu"] for t in trans])),
        "semianalytic_dRe": float(dlam.real),
        "sn_a": sn_c["a"],
        "sn_b": sn_c["b"],
        "sn_lambda_crit_abs": abs(sn_info["lambda_crit"]),
        "sn_gap": sn_info["spectral_gap_zero"],
    }
    save_json(DATADIR / "summary_checks.json", summary)
    print("Summary:", json.dumps(summary, indent=2))
    print("done")


if __name__ == "__main__":
    main()

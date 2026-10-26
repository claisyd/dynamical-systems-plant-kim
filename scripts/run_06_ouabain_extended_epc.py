#!/usr/bin/env python3
"""Extended standalone-ouabain equilibrium continuation in alpha_P.

Recovers the hyperpolarized biological fold (LP_bio) missed by the
untreated-seeded EP-C in Notebook 06, and writes a report-ready
equilibrium-only figure without overwriting prior outputs.
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PyDSTool as dst

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.plant_kim_analysis import (  # noqa: E402
    I_EP0,
    STATE_VARS,
    analyse_equilibrium,
    build_alpha_P_vode_system,
    pars_from_alpha,
    scan_slice,
)
from src.plant_kim_model import BASELINE_PARS  # noqa: E402

DATA_DIR = REPO / "outputs" / "data" / "ouabain_bifurcation"
FIG_DIR = REPO / "ReportFigs" / "extended_ouabain"
BRANCH_NPZ = DATA_DIR / "alphaP_extended_epc_branches.npz"
SPECIALS_CSV = DATA_DIR / "alphaP_extended_epc_special_points.csv"
SPECIALS_RAW_CSV = DATA_DIR / "alphaP_extended_epc_special_points_raw.csv"
FIG_PATH = FIG_DIR / "ouabain_alphaP_extended_equilibrium.png"

# Reference targets (acceptance checks only — not hardcoded into continuation).
REF = {
    "LP_bio": {"alpha_P": 0.0941, "I_ep": -0.020706, "V": -41.499},
    "H_ext,1": {"alpha_P": -2.0125, "I_ep": 0.442745, "V": -15.364},
    "H_ext,2": {"alpha_P": 1.5041, "I_ep": -0.330897, "V": -50.056},
    "LP_ext": {"alpha_P": 3.2234, "I_ep": -0.709142, "V": -24.113},
}


def state_from_eq_info(info):
    return {k: float(info[k]) for k in STATE_VARS}


def collect_special_points(curve, kinds=("H", "LP", "B", "MX")):
    rows = []
    for kind in kinds:
        for i in range(1, 40):
            lab = f"{kind}{i}"
            pt = curve.getSpecialPoint(lab)
            if pt is None:
                break
            row = {"label": lab, "type": kind, "software_label": lab, "curve": curve.name}
            for key in pt.coordnames:
                try:
                    row[key] = float(pt[key])
                except Exception:
                    pass
            rows.append(row)
    return rows


def deduplicate_specials(rows, param_key="alpha_P", atol_param=1e-3, atol_V=1e-2):
    unique = []
    for row in rows:
        if param_key not in row or "V" not in row:
            unique.append(dict(row))
            continue
        matched = None
        for u in unique:
            if u["type"] != row["type"]:
                continue
            if param_key not in u or "V" not in u:
                continue
            if abs(u[param_key] - row[param_key]) <= atol_param and abs(u["V"] - row["V"]) <= atol_V:
                matched = u
                break
        if matched is None:
            u = dict(row)
            u["raw_labels"] = [row.get("software_label", row.get("label"))]
            unique.append(u)
        else:
            matched["raw_labels"].append(row.get("software_label", row.get("label")))
    return unique


def assign_extended_labels(unique_rows):
    """Report labels: LP_bio / LP_ext / H_ext,1 / H_ext,2 / ..."""
    hops = sorted(
        [r for r in unique_rows if r["type"] == "H" and "alpha_P" in r],
        key=lambda r: float(r["alpha_P"]),
    )
    lps = sorted(
        [r for r in unique_rows if r["type"] == "LP" and "alpha_P" in r],
        key=lambda r: float(r["alpha_P"]),
    )
    out = []
    for i, r in enumerate(hops, start=1):
        u = dict(r)
        in_bio = 0.0 - 1e-8 <= float(r["alpha_P"]) <= 1.0 + 1e-8
        if in_bio:
            u["report_label"] = "H_bio" if i == 1 and len([h for h in hops if 0 <= h["alpha_P"] <= 1]) == 1 else f"H_bio,{i}"
        else:
            u["report_label"] = f"H_ext,{i}"
        u["label"] = u["report_label"]
        u["in_biological_01"] = bool(in_bio)
        u["implied_I_ep"] = float(r["alpha_P"]) * I_EP0
        out.append(u)
    for r in lps:
        u = dict(r)
        in_bio = 0.0 - 1e-8 <= float(r["alpha_P"]) <= 1.0 + 1e-8
        u["report_label"] = "LP_bio" if in_bio else "LP_ext"
        # Number exterior LPs if several
        u["label"] = u["report_label"]
        u["in_biological_01"] = bool(in_bio)
        u["implied_I_ep"] = float(r["alpha_P"]) * I_EP0
        out.append(u)
    # Disambiguate multiple LP_ext
    ext_lps = [u for u in out if u["report_label"] == "LP_ext"]
    if len(ext_lps) > 1:
        for i, u in enumerate(sorted(ext_lps, key=lambda r: r["alpha_P"]), start=1):
            u["report_label"] = f"LP_ext,{i}"
            u["label"] = u["report_label"]
    return out


def max_real_from_point_index(sol, idx):
    try:
        evals = sol.labels[idx]["EP"]["data"]["evals"]
        return float(np.max(np.real(evals)))
    except Exception:
        return float("nan")


def branch_arrays(sol, freepar="alpha_P"):
    g = np.asarray(sol[freepar], dtype=float)
    V = np.asarray(sol["V"], dtype=float)
    stable = np.zeros(len(g), dtype=bool)
    known = False
    for i in range(len(g)):
        m = max_real_from_point_index(sol, i)
        if np.isfinite(m):
            stable[i] = m < -1e-8
            known = True
    return g, V, stable, known


def print_specials(curve, tag=""):
    print(f"Special points{tag}:")
    for kind in ("H", "LP", "MX", "B"):
        for i in range(1, 40):
            lab = f"{kind}{i}"
            pt = curve.getSpecialPoint(lab)
            if pt is None:
                break
            extras = []
            for k in ("alpha_P", "V"):
                if k in pt.coordnames:
                    extras.append(f"{k}={float(pt[k]):.6g}")
            print(f"  {lab}: " + ", ".join(extras))


def run_epc(ics, alpha_P0, name, max_points=1200, step=0.01, max_step=0.05):
    ode = build_alpha_P_vode_system(ics=ics, alpha_P=alpha_P0, name=f"PlantKim_{name}")
    PC = dst.ContClass(ode)
    init = dict(ics)
    init["alpha_P"] = float(alpha_P0)
    args = dst.args(name=name, type="EP-C")
    args.freepars = ["alpha_P"]
    args.StepSize = step
    args.MinStepSize = 1e-6
    args.MaxStepSize = max_step
    args.MaxNumPoints = max_points
    args.LocBifPoints = "all"
    args.SaveEigen = True
    args.verbosity = 0
    args.initpoint = init
    PC.newCurve(args)
    print(f"Computing {name} forward...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PC[name].forward()
    print_specials(PC[name], " after forward")
    print(f"Computing {name} backward...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PC[name].backward()
    print_specials(PC[name], " after backward")
    sol = PC[name].sol
    a = np.asarray(sol["alpha_P"], dtype=float)
    V = np.asarray(sol["V"], dtype=float)
    print(f"  n={len(a)}  alpha_P=[{a.min():.4f}, {a.max():.4f}]  V=[{V.min():.4f}, {V.max():.4f}]")
    return PC, sol, collect_special_points(PC[name])


def pick_seeds():
    """Return (depolarized, hyperpolarized, mid) accepted equilibria at alpha_P=1."""
    rows = [r for r in scan_slice([1.0], vary="alpha_P", alpha_I_fixed=1.0) if r.get("accepted")]
    rows = sorted(rows, key=lambda r: float(r["V_star"]))
    if len(rows) < 2:
        raise RuntimeError(f"expected >=2 control equilibria, got {len(rows)}")
    hyp = rows[0]  # most negative V
    dep = rows[-1]  # depolarized ~ -18
    mid = rows[1] if len(rows) >= 3 else None
    print("Control equilibria at alpha_P=1:")
    for r in rows:
        print(
            f"  V*={float(r['V_star']):.6f}  maxRe={float(r['max_real_part']):.6g}  "
            f"stability={r['stability']}"
        )
    # Sanity: depolarized control
    if abs(float(dep["V_star"]) + 18.311553) > 0.05:
        print("WARNING: depolarized control V* differs from baseline -18.3116")
    return dep, hyp, mid


def save_csv(path, rows):
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    out = []
    for row in rows:
        rr = dict(row)
        for k, v in list(rr.items()):
            if isinstance(v, (list, tuple)):
                rr[k] = ";".join(str(x) for x in v)
        out.append(rr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in out:
            w.writerow(row)
    print("saved", path, "n=", len(out))


def match_ref(specials):
    print("\n=== Comparison to reference targets ===")
    print(f"{'label':10s} {'alpha_P':>10s} {'I_ep':>10s} {'V':>10s} {'d_aP':>10s} {'d_V':>10s} {'bio':>6s}")
    for lab, target in REF.items():
        best = None
        best_score = 1e9
        for r in specials:
            if r["type"] != ("LP" if lab.startswith("LP") else "H"):
                continue
            score = abs(float(r["alpha_P"]) - target["alpha_P"]) + 0.02 * abs(float(r["V"]) - target["V"])
            if score < best_score:
                best_score = score
                best = r
        if best is None:
            print(f"{lab:10s}  NOT FOUND")
            continue
        da = float(best["alpha_P"]) - target["alpha_P"]
        dV = float(best["V"]) - target["V"]
        print(
            f"{best['label']:10s} {best['alpha_P']:10.6f} {best['implied_I_ep']:10.6f} "
            f"{best['V']:10.6f} {da:10.6f} {dV:10.6f} "
            f"{'yes' if best['in_biological_01'] else 'no':>6s}"
        )


def plot_extended(sols, specials, path):
    """Plot the primary (hyperpolarized-seeded) branch inside a focused window."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ys = [r["V"] for r in specials if r["type"] in ("H", "LP")]
    xlim = (-2.6, 3.6)
    ylim = (min(ys) - 4.0, max(ys) + 4.0) if ys else (-55.0, -5.0)

    # Prefer first sol (hyperpolarized seed); it recovers all four specials.
    sol = sols[0]
    g, V, stable, known = branch_arrays(sol)
    keep = (
        (g >= xlim[0] - 0.5)
        & (g <= xlim[1] + 0.5)
        & (V >= ylim[0] - 10.0)
        & (V <= ylim[1] + 10.0)
    )
    keep = keep | np.roll(keep, 1) | np.roll(keep, -1)
    g, V, stable = g[keep], V[keep], stable[keep]
    dg = np.abs(np.diff(g, prepend=g[0] if len(g) else 0.0))
    dV = np.abs(np.diff(V, prepend=V[0] if len(V) else 0.0))
    jump = (dg > 0.25) | (dV > 3.0)
    g = g.astype(float).copy()
    V = V.astype(float).copy()
    g[jump] = np.nan
    V[jump] = np.nan

    plotted_stable = plotted_unstable = False
    if known and (np.any(stable) or np.any(~stable)):
        if np.any(stable):
            gg, VV = g.copy(), V.copy()
            gg[~stable] = np.nan
            VV[~stable] = np.nan
            ax.plot(gg, VV, "-", color="C0", lw=1.8, label="Stable equilibrium", zorder=4)
            plotted_stable = True
        if np.any(~stable):
            gg, VV = g.copy(), V.copy()
            gg[stable] = np.nan
            VV[stable] = np.nan
            ax.plot(gg, VV, "--", color="C0", lw=1.8, label="Unstable equilibrium", zorder=4)
            plotted_unstable = True
    else:
        ax.plot(g, V, "-", color="C0", lw=1.8, label="equilibrium branch", zorder=4)

    ax.axvspan(0.0, 1.0, color="C2", alpha=0.12, label=r"Biological range $0\leq\alpha_P\leq1$")
    ax.axvline(1.0, color="C0", ls=":", lw=1.2, label=r"Control ($\alpha_P=1$)")
    ax.axvline(0.0, color="C3", ls=":", lw=1.2, label=r"Complete ouabain ($\alpha_P=0$)")

    markers = {"H": "o", "LP": "s"}
    colors = {"H": "C3", "LP": "C1"}
    label_tex = {
        "LP_bio": r"$LP_{\mathrm{bio}}$",
        "LP_ext": r"$LP_{\mathrm{ext}}$",
        "H_ext,1": r"$H_{\mathrm{ext},1}$",
        "H_ext,2": r"$H_{\mathrm{ext},2}$",
    }
    annot = {
        "LP_bio": ((-30, -18), "offset points"),
        "LP_ext": ((-18, -30), "offset points"),
        "H_ext,1": ((8, 12), "offset points"),
        "H_ext,2": ((4, 2), "offset points"),
    }
    for row in specials:
        if row["type"] not in ("H", "LP"):
            continue
        lab = row["label"]
        tex = label_tex.get(lab, lab)
        ax.plot(
            row["alpha_P"],
            row["V"],
            markers[row["type"]],
            color=colors[row["type"]],
            ms=8,
            zorder=6,
            label=tex,
        )
        xytext, tcoords = annot.get(lab, ((8, 8), "offset points"))
        ax.annotate(
            tex,
            (row["alpha_P"], row["V"]),
            textcoords=tcoords,
            xytext=xytext,
            fontsize=9,
            zorder=25,
        )

    ax.set_xlabel(r"$\alpha_P$ (1 = control, 0 = complete ouabain block)")
    ax.set_ylabel(r"$V$ (mV)")
    ax.set_title(r"Ouabain equilibrium continuation in ($\alpha_P$, $V$)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)

    preferred = [
        "Stable equilibrium",
        "Unstable equilibrium",
        r"Biological range $0\leq\alpha_P\leq1$",
        r"Control ($\alpha_P=1$)",
        r"Complete ouabain ($\alpha_P=0$)",
        r"$LP_{\mathrm{bio}}$",
        r"$LP_{\mathrm{ext}}$",
        r"$H_{\mathrm{ext},1}$",
        r"$H_{\mathrm{ext},2}$",
    ]
    handles, labels = ax.get_legend_handles_labels()
    by = dict(zip(labels, handles))
    final_labels = []
    seen = set()
    for lab in preferred + list(labels):
        if lab in by and lab not in seen:
            seen.add(lab)
            final_labels.append(lab)
    ax.legend(
        [by[lab] for lab in final_labels],
        final_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    print("I_EP0 =", I_EP0)
    print("BASELINE I_ep =", BASELINE_PARS["I_ep"])
    assert abs(I_EP0 + 0.22) < 1e-12
    assert abs(float(pars_from_alpha(1.0, 1.0)["I_ext"])) < 1e-15

    dep, hyp, mid = pick_seeds()
    sols = []
    raw_specials = []

    # 1) Hyperpolarized-seeded EP-C (primary path to LP_bio / H_ext,2)
    _, sol_hyp, sp_hyp = run_epc(
        state_from_eq_info(hyp),
        alpha_P0=1.0,
        name="EQ_alphaP_hyp",
        max_points=1500,
        step=0.008,
        max_step=0.04,
    )
    sols.append(sol_hyp)
    raw_specials.extend(sp_hyp)

    # 2) Mid-branch seed if present (may connect sheets)
    if mid is not None:
        _, sol_mid, sp_mid = run_epc(
            state_from_eq_info(mid),
            alpha_P0=1.0,
            name="EQ_alphaP_mid",
            max_points=1200,
            step=0.008,
            max_step=0.04,
        )
        sols.append(sol_mid)
        raw_specials.extend(sp_mid)

    # 3) Extended untreated/depolarized seed (recover H_ext,1 + LP_ext + possibly LP_bio)
    _, sol_dep, sp_dep = run_epc(
        state_from_eq_info(dep),
        alpha_P0=1.0,
        name="EQ_alphaP_dep_ext",
        max_points=2000,
        step=0.008,
        max_step=0.04,
    )
    sols.append(sol_dep)
    raw_specials.extend(sp_dep)

    unique = deduplicate_specials([r for r in raw_specials if r["type"] in ("H", "LP")])
    specials = assign_extended_labels(unique)

    # Attach nearby max Re from first sol that contains the point
    for row in specials:
        row["max_real_part_near"] = float("nan")
        for sol in sols:
            a = np.asarray(sol["alpha_P"], dtype=float)
            V = np.asarray(sol["V"], dtype=float)
            d = np.abs(a - row["alpha_P"]) + 0.05 * np.abs(V - row["V"])
            idx = int(np.argmin(d))
            if d[idx] < 0.05:
                row["max_real_part_near"] = max_real_from_point_index(sol, idx)
                break

    save_csv(SPECIALS_RAW_CSV, raw_specials)
    save_csv(SPECIALS_CSV, specials)

    # Persist branch samples for notebook replot without recomputation
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {}
    for i, sol in enumerate(sols):
        g, V, stable, known = branch_arrays(sol)
        payload[f"a_{i}"] = g
        payload[f"V_{i}"] = V
        payload[f"stable_{i}"] = stable.astype(np.uint8)
        payload[f"known_{i}"] = np.array([1 if known else 0], dtype=np.uint8)
    payload["n_branches"] = np.array([len(sols)], dtype=np.int32)
    np.savez_compressed(BRANCH_NPZ, **payload)
    print("saved", BRANCH_NPZ)

    print("\n=== Verified special points ===")
    print(f"{'label':10s} {'type':4s} {'alpha_P':>10s} {'I_ep':>10s} {'V':>10s} {'bio':>6s} {'maxRe':>12s}")
    for row in sorted(specials, key=lambda r: (r["type"], r["alpha_P"])):
        print(
            f"{row['label']:10s} {row['type']:4s} {row['alpha_P']:10.6f} "
            f"{row['implied_I_ep']:10.6f} {row['V']:10.6f} "
            f"{'inside' if row['in_biological_01'] else 'outside':>6s} "
            f"{row['max_real_part_near']:12.4g}"
        )

    match_ref(specials)
    plot_extended(sols, specials, FIG_PATH)

    # Require LP_bio recovery
    bio_lps = [r for r in specials if r["label"] == "LP_bio"]
    if not bio_lps:
        print("\nWARNING: LP_bio was not recovered. Figure still written; inspect branches.")
        return 1
    lp = bio_lps[0]
    if abs(lp["alpha_P"] - REF["LP_bio"]["alpha_P"]) > 0.02 or abs(lp["V"] - REF["LP_bio"]["V"]) > 1.0:
        print("\nWARNING: LP_bio far from NB04/reference candidate.")
    else:
        print("\nLP_bio recovered within acceptance tolerance of NB04/reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

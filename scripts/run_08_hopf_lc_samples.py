#!/usr/bin/env python3
"""Focused LC-C from H_bio for near-Hopf period comparison (Notebook 08).

Intended to run inside Docker (curuy/math3888) with AUTO available.
Exports dense limit-cycle samples to outputs/data/jacobian_hopf/.
"""
from __future__ import annotations

import csv
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import PyDSTool as dst

REPO_ROOT = Path(__file__).resolve().parents[1]
# Inside Docker the repo is mounted at /notebooks
if Path("/notebooks").exists() and str(REPO_ROOT).startswith("/"):
    # Prefer mounted tree when running in container via bind mount
    pass
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
cwd = Path.cwd().resolve()
if str(cwd) not in sys.path:
    sys.path.insert(0, str(cwd))

from src.plant_kim_analysis import build_alpha_P_vode_system, pars_from_alpha
from src.plant_kim_model import STATE_VARS

# Resolve data dir relative to repo (works for host path or /notebooks mount)
DATADIR = REPO_ROOT / "outputs" / "data" / "jacobian_hopf"
DATADIR.mkdir(parents=True, exist_ok=True)

TTX0_CSV = (
    REPO_ROOT / "outputs" / "data" / "ouabain_bifurcation" / "ttx0_alphaP_epc_special_points.csv"
)


def landmark_H_bio_from_csv():
    """Load H_bio without importing sympy-dependent jacobian module."""
    with TTX0_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    row = None
    for r in rows:
        if r.get("label") == "H_bio" or r.get("report_label") == "H_bio":
            row = r
            break
    if row is None:
        raise KeyError("H_bio")
    state = {k: float(row[k]) for k in STATE_VARS}
    alpha_P = float(row["alpha_P"])
    return {
        "alpha_P": alpha_P,
        "state": state,
        "pars": pars_from_alpha(0.0, alpha_P),
    }


def quiet_call(fn, *args, **kwargs):
    buf_out, buf_err = StringIO(), StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            return fn(*args, **kwargs)


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
                    out[k] = repr(v) if np.isfinite(v) else ""
                elif v is None:
                    out[k] = ""
                else:
                    out[k] = v
            w.writerow(out)


def period_of(sol, i: int) -> float:
    for key in ("_T", "period", "_p"):
        if key in sol.coordnames:
            return float(sol[key][i])
    return float("nan")


def main():
    h = landmark_H_bio_from_csv()
    ics = dict(h["state"])
    print(f"Building complete-TTX system at H_bio alpha_P={h['alpha_P']:.8f}")
    ode = build_alpha_P_vode_system(
        pars={**dict(h["pars"]), "g_I": 0.0},
        ics=ics,
        name="PK_ttx0_LC_Hbio",
        alpha_P=h["alpha_P"],
    )
    PC = dst.ContClass(ode)

    # Seed EP-C briefly to obtain a software Hopf label for LC-C
    PCeq = dst.args(name="EQ_seed", type="EP-C")
    PCeq.freepars = ["alpha_P"]
    PCeq.StepSize = 1e-3
    PCeq.MinStepSize = 1e-7
    PCeq.MaxStepSize = 5e-3
    PCeq.MaxNumPoints = 60
    PCeq.LocBifPoints = "all"
    PCeq.SaveEigen = True
    PCeq.verbosity = 0
    PCeq.initpoint = {**ics, "alpha_P": h["alpha_P"]}
    PC.newCurve(PCeq)
    quiet_call(PC["EQ_seed"].forward)
    quiet_call(PC["EQ_seed"].backward)

    hopf_lab = None
    for k in range(1, 20):
        lab = f"H{k}"
        pt = PC["EQ_seed"].getSpecialPoint(lab)
        if pt is None:
            break
        aP = float(pt["alpha_P"])
        print(f"  found {lab} at alpha_P={aP:.8f}")
        if abs(aP - h["alpha_P"]) < 5e-3:
            hopf_lab = lab
            break
    if hopf_lab is None:
        # fallback: closest H
        best = None
        for k in range(1, 20):
            lab = f"H{k}"
            pt = PC["EQ_seed"].getSpecialPoint(lab)
            if pt is None:
                break
            d = abs(float(pt["alpha_P"]) - h["alpha_P"])
            if best is None or d < best[0]:
                best = (d, lab)
        if best is None:
            raise RuntimeError("No Hopf found on seed EP-C")
        hopf_lab = best[1]
    print(f"LC-C seed: EQ_seed:{hopf_lab}")

    PCLC = dst.args(name="LC_Hbio", type="LC-C")
    PCLC.initpoint = f"EQ_seed:{hopf_lab}"
    PCLC.freepars = ["alpha_P"]
    PCLC.MaxNumPoints = 120
    PCLC.StepSize = 5e-4
    PCLC.MaxStepSize = 2e-2
    PCLC.MinStepSize = 1e-7
    PCLC.LocBifPoints = "all"
    PCLC.SaveEigen = False
    PCLC.verbosity = 0
    PC.newCurve(PCLC)
    print("Running LC-C forward ...")
    quiet_call(PC["LC_Hbio"].forward)
    print("Running LC-C backward ...")
    try:
        quiet_call(PC["LC_Hbio"].backward)
    except Exception as exc:
        print(f"  backward LC-C note: {exc}")

    sol = PC["LC_Hbio"].sol
    n = len(sol["alpha_P"])
    print(f"LC points: {n}")
    rows = []
    for i in range(n):
        aP = float(sol["alpha_P"][i])
        vmin = float(sol["V_min"][i]) if "V_min" in sol.coordnames else float("nan")
        vmax = float(sol["V_max"][i]) if "V_max" in sol.coordnames else float("nan")
        T = period_of(sol, i)
        amp = vmax - vmin if np.isfinite(vmax) and np.isfinite(vmin) else float("nan")
        rows.append(
            {
                "index": i,
                "alpha_P": aP,
                "period_ms": T,
                "V_max": vmax,
                "V_min": vmin,
                "amplitude_mV": amp,
                "dist_Hbio": abs(aP - h["alpha_P"]),
            }
        )
    save_csv(DATADIR / "lcc_Hbio_samples.csv", rows)
    print(f"saved {len(rows)} -> {DATADIR / 'lcc_Hbio_samples.csv'}")
    amps = [r["amplitude_mV"] for r in rows if np.isfinite(r["amplitude_mV"])]
    if amps:
        print(f"amplitude range [{min(amps):.6g}, {max(amps):.6g}] mV")
    print("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Short local EP-C windows around H_bio and LP_bio for Notebook 08.

Runs on the host .venv (EP-C does not require AUTO). Saves dense equilibrium
samples with PyDSTool eigenvalues under outputs/data/jacobian_hopf/.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plant_kim_analysis import (
    build_alpha_I_vode_system,
    build_alpha_P_vode_system,
    pars_from_alpha,
    rhs,
)
from src.plant_kim_jacobian import landmark_H_bio, landmark_LP_bio
from src.plant_kim_model import STATE_VARS

DATADIR = REPO_ROOT / "outputs" / "data" / "jacobian_hopf"
DATADIR.mkdir(parents=True, exist_ok=True)


def quiet_call(fn, *args, **kwargs):
    buf_out, buf_err = StringIO(), StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            return fn(*args, **kwargs)


def evals_at(sol, idx) -> Optional[np.ndarray]:
    try:
        ev = sol.labels[idx]["EP"]["data"]["evals"]
        return np.asarray(ev, dtype=complex)
    except Exception:
        return None


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
                if isinstance(v, complex):
                    out[k] = repr(v)
                elif isinstance(v, float):
                    out[k] = repr(v) if np.isfinite(v) else ""
                elif v is None:
                    out[k] = ""
                else:
                    out[k] = v
            w.writerow(out)


def extract_branch_rows(sol, freepar: str, alpha_I_fixed=None, alpha_P_fixed=None) -> List[Dict]:
    p = np.asarray(sol[freepar], dtype=float)
    n = len(p)
    rows = []
    for i in range(n):
        state = {k: float(sol[k][i]) for k in STATE_VARS}
        if freepar == "alpha_P":
            a_I, a_P = float(alpha_I_fixed), float(p[i])
        else:
            a_I, a_P = float(p[i]), float(alpha_P_fixed)
        pars = pars_from_alpha(a_I, a_P)
        res = float(np.max(np.abs(rhs(state, pars))))
        ev = evals_at(sol, i)
        row = {
            "index": i,
            "freepar": freepar,
            "param": float(p[i]),
            "alpha_I": a_I,
            "alpha_P": a_P,
            "residual_max": res,
            "max_real_pydstool": float(np.max(np.real(ev))) if ev is not None else float("nan"),
            "n_evals": int(len(ev)) if ev is not None else 0,
        }
        for k in STATE_VARS:
            row[k] = state[k]
        if ev is not None:
            for j, lam in enumerate(ev):
                row[f"eig{j}_re"] = float(np.real(lam))
                row[f"eig{j}_im"] = float(np.imag(lam))
        rows.append(row)
    return rows


def run_epc(
    ode,
    name: str,
    freepar: str,
    initpoint: Dict,
    max_points: int = 80,
    step: float = 1e-3,
    max_step: float = 5e-3,
    min_step: float = 1e-7,
):
    PC = dst.ContClass(ode)
    args = dst.args(name=name, type="EP-C")
    args.freepars = [freepar]
    args.StepSize = step
    args.MinStepSize = min_step
    args.MaxStepSize = max_step
    args.MaxNumPoints = max_points
    args.LocBifPoints = "all"
    args.SaveEigen = True
    args.verbosity = 0
    args.initpoint = initpoint
    PC.newCurve(args)
    quiet_call(PC[name].forward)
    quiet_call(PC[name].backward)
    return PC[name]


def main():
    # --- H_bio window: alpha_I=0, freepar alpha_P ---
    h = landmark_H_bio(REPO_ROOT)
    ics = dict(h["state"])
    ode_h = build_alpha_P_vode_system(
        pars={**dict(h["pars"]), "g_I": 0.0},
        ics=ics,
        name="PK_ttx0_Hbio_window",
        alpha_P=h["alpha_P"],
    )
    init_h = {**ics, "alpha_P": h["alpha_P"]}
    print(f"EP-C around H_bio at alpha_P={h['alpha_P']:.6f} ...")
    curve_h = run_epc(
        ode_h,
        "EQ_Hbio_local",
        "alpha_P",
        init_h,
        max_points=120,
        step=5e-4,
        max_step=2e-3,
    )
    rows_h = extract_branch_rows(curve_h.sol, "alpha_P", alpha_I_fixed=0.0)
    save_csv(DATADIR / "epc_Hbio_local.csv", rows_h)
    print(f"  saved {len(rows_h)} points -> epc_Hbio_local.csv")
    print(
        f"  alpha_P range [{min(r['param'] for r in rows_h):.6f}, "
        f"{max(r['param'] for r in rows_h):.6f}]"
    )

    # --- LP_bio window: alpha_P=1, freepar alpha_I ---
    lp = landmark_LP_bio(REPO_ROOT)
    ics_lp = dict(lp["state"])
    ode_lp = build_alpha_I_vode_system(
        pars=dict(lp["pars"]),
        ics=ics_lp,
        name="PK_LP_bio_window",
        alpha_I=lp["alpha_I"],
    )
    init_lp = {**ics_lp, "alpha_I": lp["alpha_I"]}
    print(f"EP-C around LP_bio at alpha_I={lp['alpha_I']:.6f} ...")
    curve_lp = run_epc(
        ode_lp,
        "EQ_LPbio_local",
        "alpha_I",
        init_lp,
        max_points=200,
        step=2e-3,
        max_step=1e-2,
        min_step=1e-8,
    )
    rows_lp = extract_branch_rows(curve_lp.sol, "alpha_I", alpha_P_fixed=1.0)
    save_csv(DATADIR / "epc_LPbio_local.csv", rows_lp)
    print(f"  saved {len(rows_lp)} points -> epc_LPbio_local.csv")
    print(
        f"  alpha_I range [{min(r['param'] for r in rows_lp):.6f}, "
        f"{max(r['param'] for r in rows_lp):.6f}]"
    )
    print("done")


if __name__ == "__main__":
    main()

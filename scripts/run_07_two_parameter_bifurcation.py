#!/usr/bin/env python3
"""Run Section 3.7 two-parameter bifurcation analysis (Notebook 07 engine).

Primary plane: (alpha_I, alpha_P). Continuation types: H-C2 and LP-C.
Seeds from validated Notebook 05/06 CSV special points.
Intended to run inside Docker (curuy/math3888) with CWD on sys.path.
"""
from __future__ import annotations

import csv
import logging
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import PyDSTool as dst

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path.cwd().resolve()) not in sys.path:
    sys.path.insert(0, str(Path.cwd().resolve()))

from src.plant_kim_analysis import (
    G_I0,
    I_EP0,
    analyse_equilibrium,
    build_alpha_IP_vode_system,
    pars_from_alpha,
    scan_equilibrium_voltages,
)
from src.plant_kim_model import DEFAULT_ALGPARAMS, STATE_VARS

FIGDIR = REPO_ROOT / "outputs" / "figures" / "two_parameter_bifurcation"
DATADIR = REPO_ROOT / "outputs" / "data" / "two_parameter_bifurcation"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATADIR.mkdir(parents=True, exist_ok=True)

TTX_CSV = REPO_ROOT / "outputs" / "data" / "ttx_bifurcation" / "alphaI_epc_special_points.csv"
TTX0_CSV = (
    REPO_ROOT / "outputs" / "data" / "ouabain_bifurcation" / "ttx0_alphaP_epc_special_points.csv"
)

# Soft domain for storage / plotting focus
A_I_LO, A_I_HI = -0.25, 2.0
A_P_LO, A_P_HI = -0.25, 2.0

# Validation tolerances (plan gate)
VAL_ATOL_ALPHA = 5e-3
VAL_ATOL_V = 0.1

# Curve overlap / special dedup
CURVE_ATOL_A = 1e-3
CURVE_ATOL_V = 0.05


def quiet_call(fn, *args, **kwargs):
    buf_out, buf_err = StringIO(), StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            return fn(*args, **kwargs)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_by_label(rows: Sequence[Dict[str, str]], label: str) -> Dict[str, str]:
    for r in rows:
        if r.get("label") == label or r.get("report_label") == label:
            return r
    raise KeyError(f"label {label!r} not found in CSV")


def state_from_csv_row(row: Dict[str, str]) -> Dict[str, float]:
    return {k: float(row[k]) for k in STATE_VARS}


def save_csv(path: Path, rows: Sequence[Dict], fieldnames: Optional[Sequence[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
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
                if isinstance(v, (list, tuple)):
                    out[k] = ";".join(str(x) for x in v)
                elif isinstance(v, float):
                    out[k] = repr(v) if np.isfinite(v) else ""
                elif v is None:
                    out[k] = ""
                else:
                    out[k] = v
            w.writerow(out)


def collect_special_points(curve, kinds=("H", "LP", "BT", "ZH", "GH", "CP", "MX", "B")):
    rows = []
    for kind in kinds:
        for i in range(1, 30):
            lab = f"{kind}{i}"
            pt = curve.getSpecialPoint(lab)
            if pt is None:
                break
            row = {"label": lab, "type": kind, "software_label": lab}
            for key in pt.coordnames:
                try:
                    row[key] = float(pt[key])
                except Exception:
                    pass
            rows.append(row)
    return rows


def deduplicate_specials(
    rows,
    keys=("alpha_I", "alpha_P", "V"),
    atol=(CURVE_ATOL_A, CURVE_ATOL_A, CURVE_ATOL_V),
):
    unique = []
    for row in rows:
        matched = None
        for u in unique:
            if u.get("type") != row.get("type"):
                continue
            ok = True
            for k, tol in zip(keys, atol):
                if k not in u or k not in row:
                    ok = False
                    break
                if abs(float(u[k]) - float(row[k])) > tol:
                    ok = False
                    break
            if ok:
                matched = u
                break
        if matched is None:
            u = dict(row)
            u["raw_labels"] = [row.get("software_label", row.get("label"))]
            unique.append(u)
        else:
            matched.setdefault("raw_labels", []).append(
                row.get("software_label", row.get("label"))
            )
    return unique


def extract_curve_points(curve, curve_name: str, direction: str = "") -> List[Dict]:
    sol = curve.sol
    n = len(sol["V"])
    rows = []
    for i in range(n):
        rows.append(
            {
                "curve": curve_name,
                "direction": direction,
                "index": i,
                "alpha_I": float(sol["alpha_I"][i]),
                "alpha_P": float(sol["alpha_P"][i]),
                "V": float(sol["V"][i]),
                "g_I": float(sol["alpha_I"][i]) * G_I0,
                "I_ep": float(sol["alpha_P"][i]) * I_EP0,
            }
        )
    return rows


def curves_overlap(pts_a: Sequence[Dict], pts_b: Sequence[Dict], frac: float = 0.15) -> bool:
    """True if curves share a non-trivial fraction of nearby samples.

    Also returns True if the minimum pairwise distance is very small (same
    physical branch meeting in an exterior region), even when interior samples
    do not heavily overlap.
    """
    if not pts_a or not pts_b:
        return False
    hits = 0
    sample = pts_a[:: max(1, len(pts_a) // 40)]
    min_d = float("inf")
    for p in sample:
        for q in pts_b:
            d = (
                (p["alpha_I"] - q["alpha_I"]) ** 2
                + (p["alpha_P"] - q["alpha_P"]) ** 2
                + 0.01 * (p["V"] - q["V"]) ** 2
            ) ** 0.5
            if d < min_d:
                min_d = d
            if (
                abs(p["alpha_I"] - q["alpha_I"]) <= CURVE_ATOL_A
                and abs(p["alpha_P"] - q["alpha_P"]) <= CURVE_ATOL_A
                and abs(p["V"] - q["V"]) <= CURVE_ATOL_V
            ):
                hits += 1
                break
    if min_d < 0.02:
        return True
    return (hits / max(1, len(sample))) >= frac


def merge_curve_points(list_of_pt_lists: Sequence[Sequence[Dict]], name: str) -> List[Dict]:
    """Concatenate and lightly thin near-duplicates along a physical branch."""
    merged: List[Dict] = []
    for pts in list_of_pt_lists:
        for p in pts:
            dup = False
            for q in merged[-8:]:
                if (
                    abs(p["alpha_I"] - q["alpha_I"]) <= 5e-4
                    and abs(p["alpha_P"] - q["alpha_P"]) <= 5e-4
                    and abs(p["V"] - q["V"]) <= 0.02
                ):
                    dup = True
                    break
            if not dup:
                r = dict(p)
                r["curve"] = name
                merged.append(r)
    # Sort approximately along arc length in alpha plane from first point
    if not merged:
        return merged
    a0, b0 = merged[0]["alpha_I"], merged[0]["alpha_P"]
    merged.sort(
        key=lambda r: (
            (r["alpha_I"] - a0) ** 2 + (r["alpha_P"] - b0) ** 2,
            r["alpha_I"],
            r["alpha_P"],
        )
    )
    return merged


def nearest_on_curve(
    pts: Sequence[Dict],
    *,
    alpha_I: Optional[float] = None,
    alpha_P: Optional[float] = None,
) -> Optional[Dict]:
    best = None
    best_d = float("inf")
    for p in pts:
        d = 0.0
        if alpha_I is not None:
            d += (p["alpha_I"] - alpha_I) ** 2
        if alpha_P is not None:
            d += (p["alpha_P"] - alpha_P) ** 2
        if d < best_d:
            best_d = d
            best = p
    return best


def interpolate_slice_intersection(
    pts: Sequence[Dict],
    fixed: str,
    value: float,
) -> Optional[Dict]:
    """Linear interpolate where curve crosses fixed == value (alpha_I or alpha_P)."""
    if len(pts) < 2:
        return None
    best = None
    best_gap = float("inf")
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        fa, fb = a[fixed], b[fixed]
        if (fa - value) * (fb - value) > 0 and abs(fa - value) > 1e-12:
            continue
        if abs(fb - fa) < 1e-15:
            t = 0.0
        else:
            t = (value - fa) / (fb - fa)
        if t < -0.05 or t > 1.05:
            continue
        t = min(1.0, max(0.0, t))
        other = "alpha_P" if fixed == "alpha_I" else "alpha_I"
        inter = {
            "alpha_I": a["alpha_I"] + t * (b["alpha_I"] - a["alpha_I"]),
            "alpha_P": a["alpha_P"] + t * (b["alpha_P"] - a["alpha_P"]),
            "V": a["V"] + t * (b["V"] - a["V"]),
        }
        # Prefer crossings near the slice value
        gap = abs(inter[fixed] - value)
        if gap < best_gap:
            best_gap = gap
            best = inter
    if best is None:
        # Fallback: nearest point to the slice
        p = nearest_on_curve(pts, **{fixed: value})
        return p
    return best


def run_epc(
    ode,
    name: str,
    freepar: str,
    initpoint: Dict,
    max_points: int = 400,
    step: float = 0.01,
    max_step: float = 0.05,
):
    PC = dst.ContClass(ode)
    args = dst.args(name=name, type="EP-C")
    args.freepars = [freepar]
    args.initpoint = initpoint
    args.StepSize = step
    args.MinStepSize = 1e-6
    args.MaxStepSize = max_step
    args.MaxNumPoints = max_points
    args.LocBifPoints = "all"
    args.SaveEigen = True
    args.verbosity = 0
    PC.newCurve(args)
    print(f"Computing {name} forward...")
    quiet_call(PC[name].forward)
    print(f"Computing {name} backward...")
    quiet_call(PC[name].backward)
    return PC


def run_twopar(
    PC,
    name: str,
    curvetype: str,
    initpoint: str,
    loc_bif: Sequence[str],
    max_points: int = 200,
    step: float = 1e-3,
    max_step: float = 0.03,
):
    args = dst.args(name=name, type=curvetype, force=True)
    args.freepars = ["alpha_I", "alpha_P"]
    args.initpoint = initpoint
    args.StepSize = step
    args.MinStepSize = 1e-6
    args.MaxStepSize = max_step
    args.MaxNumPoints = max_points
    args.LocBifPoints = list(loc_bif)
    args.SaveEigen = True
    args.verbosity = 0
    PC.newCurve(args)
    print(f"Computing {name} ({curvetype}) forward from {initpoint}...")
    quiet_call(PC[name].forward)
    n_fwd = len(PC[name].sol["V"])
    print(f"  forward points: {n_fwd}")
    pts_fwd = extract_curve_points(PC[name], name, "forward")
    # Separate backward curve name to avoid clobbering; then merge
    bname = name + "_bwd"
    args2 = dst.args(name=bname, type=curvetype, force=True)
    args2.freepars = ["alpha_I", "alpha_P"]
    args2.initpoint = initpoint
    args2.StepSize = step
    args2.MinStepSize = 1e-6
    args2.MaxStepSize = max_step
    args2.MaxNumPoints = max_points
    args2.LocBifPoints = list(loc_bif)
    args2.SaveEigen = True
    args2.verbosity = 0
    PC.newCurve(args2)
    print(f"Computing {bname} ({curvetype}) backward from {initpoint}...")
    quiet_call(PC[bname].backward)
    n_bwd = len(PC[bname].sol["V"])
    print(f"  backward points: {n_bwd}")
    pts_bwd = extract_curve_points(PC[bname], name, "backward")
    specs = collect_special_points(PC[name], kinds=("BT", "ZH", "GH", "CP", "MX"))
    specs += collect_special_points(PC[bname], kinds=("BT", "ZH", "GH", "CP", "MX"))
    for s in specs:
        s["parent_curve"] = name
        s["seed_initpoint"] = initpoint
    return pts_fwd, pts_bwd, specs


def find_software_label_for_target(PC, eq_name: str, kind: str, alpha_key: str, target: float, v_target: float):
    best = None
    best_d = float("inf")
    for i in range(1, 20):
        lab = f"{kind}{i}"
        pt = PC[eq_name].getSpecialPoint(lab)
        if pt is None:
            break
        a = float(pt[alpha_key])
        v = float(pt["V"])
        d = abs(a - target) + 0.01 * abs(v - v_target)
        if d < best_d:
            best_d = d
            best = lab
    return best, best_d


def draw_bio_square(ax, transformed=False):
    if not transformed:
        xs = [0, 1, 1, 0, 0]
        ys = [0, 0, 1, 1, 0]
        ax.plot(xs, ys, "k-", lw=1.5, label="biological square")
    else:
        # g_I in [0,4], I_ep in [-0.22, 0]
        xs = [0, G_I0, G_I0, 0, 0]
        ys = [0, 0, I_EP0, I_EP0, 0]
        ax.plot(xs, ys, "k-", lw=1.5, label="biological image")


def main():
    print("executable:", sys.executable)
    print("PyDSTool:", getattr(dst, "__version__", "?"))
    print("DEFAULT_ALGPARAMS:", DEFAULT_ALGPARAMS)
    print("G_I0 =", G_I0, "I_EP0 =", I_EP0)
    print("FIGDIR:", FIGDIR)
    print("DATADIR:", DATADIR)

    ttx_rows = load_csv_rows(TTX_CSV)
    ttx0_rows = load_csv_rows(TTX0_CSV)
    lp_bio = row_by_label(ttx_rows, "LP_bio")
    h_bio = row_by_label(ttx0_rows, "H_bio")
    h_ext = row_by_label(ttx0_rows, "H_ext")

    print("\n=== Loaded 1D anchors ===")
    print(
        f"LP_bio: alpha_I={lp_bio['alpha_I']} V={lp_bio['V']} (alpha_P=1 fixed in NB05)"
    )
    print(f"H_bio: alpha_P={h_bio['alpha_P']} V={h_bio['V']} (alpha_I=0)")
    print(f"H_ext: alpha_P={h_ext['alpha_P']} V={h_ext['V']} (alpha_I=0)")

    # ------------------------------------------------------------------
    # Seed EP-C: complete-TTX slice (alpha_I=0, freepar alpha_P)
    # ------------------------------------------------------------------
    print("\n=== Seed EP-C: alpha_I=0, freepar alpha_P ===")
    pars_ttx = pars_from_alpha(0.0, 1.0)
    roots = scan_equilibrium_voltages(pars_ttx)
    ttx_cands = []
    for r in roots:
        info = analyse_equilibrium(r["V_star"], pars_ttx, r["root_type"])
        if info.get("accepted", True):
            ttx_cands.append(info)
    ttx_seed = min(ttx_cands, key=lambda info: abs(float(info["V_star"]) + 45.0))
    eq_ics_ttx = {k: float(ttx_seed[k]) for k in STATE_VARS}
    print("TTX0 seed V* =", eq_ics_ttx["V"])

    ode_ttx0 = build_alpha_IP_vode_system(
        ics=eq_ics_ttx, alpha_I=0.0, alpha_P=1.0, name="PlantKim_alphaIP_ttx0"
    )
    init_ttx0 = dict(eq_ics_ttx)
    init_ttx0["alpha_I"] = 0.0
    init_ttx0["alpha_P"] = 1.0
    PC_ttx0 = run_epc(
        ode_ttx0,
        "EQ_ttx0",
        "alpha_P",
        init_ttx0,
        max_points=450,
        step=0.01,
        max_step=0.05,
    )
    specs_ttx0 = collect_special_points(PC_ttx0["EQ_ttx0"], kinds=("H", "LP", "MX", "B"))
    print("TTX0 specials:", [(s["label"], s.get("alpha_P"), s.get("V")) for s in specs_ttx0])

    hbio_lab, d_bio = find_software_label_for_target(
        PC_ttx0, "EQ_ttx0", "H", "alpha_P", float(h_bio["alpha_P"]), float(h_bio["V"])
    )
    hext_lab, d_ext = find_software_label_for_target(
        PC_ttx0, "EQ_ttx0", "H", "alpha_P", float(h_ext["alpha_P"]), float(h_ext["V"])
    )
    print(f"Matched H_bio -> EQ_ttx0:{hbio_lab} (score {d_bio:.4g})")
    print(f"Matched H_ext -> EQ_ttx0:{hext_lab} (score {d_ext:.4g})")
    if hbio_lab is None or d_bio > 0.05:
        raise RuntimeError("Failed to recover H_bio on joint-system EP-C")
    if hext_lab is None or d_ext > 0.05:
        raise RuntimeError("Failed to recover H_ext on joint-system EP-C")

    # ------------------------------------------------------------------
    # Seed EP-C: TTX-only slice (alpha_P=1, freepar alpha_I)
    # ------------------------------------------------------------------
    print("\n=== Seed EP-C: alpha_P=1, freepar alpha_I ===")
    pars_ctrl = pars_from_alpha(1.0, 1.0)
    roots_c = scan_equilibrium_voltages(pars_ctrl)
    seed_c = None
    for r in roots_c:
        info = analyse_equilibrium(r["V_star"], pars_ctrl, r["root_type"])
        if abs(float(info["V_star"]) + 18.3) < 2.0:
            seed_c = info
            break
    if seed_c is None:
        raise RuntimeError("Could not locate untreated equilibrium seed")
    eq_ics = {k: float(seed_c[k]) for k in STATE_VARS}

    ode_ttx = build_alpha_IP_vode_system(
        ics=eq_ics, alpha_I=1.0, alpha_P=1.0, name="PlantKim_alphaIP_ttx"
    )
    init_ttx = dict(eq_ics)
    init_ttx["alpha_I"] = 1.0
    init_ttx["alpha_P"] = 1.0
    PC_ttx = run_epc(
        ode_ttx,
        "EQ_ttx",
        "alpha_I",
        init_ttx,
        max_points=450,
        step=0.01,
        max_step=0.05,
    )
    specs_ttx = collect_special_points(PC_ttx["EQ_ttx"], kinds=("H", "LP", "MX", "B"))
    print("TTX specials:", [(s["label"], s.get("alpha_I"), s.get("V")) for s in specs_ttx])
    lp_lab, d_lp = find_software_label_for_target(
        PC_ttx, "EQ_ttx", "LP", "alpha_I", float(lp_bio["alpha_I"]), float(lp_bio["V"])
    )
    print(f"Matched LP_bio -> EQ_ttx:{lp_lab} (score {d_lp:.4g})")
    if lp_lab is None or d_lp > 0.05:
        raise RuntimeError("Failed to recover LP_bio on joint-system EP-C")

    save_csv(DATADIR / "seed_ttx0_epc_special_points.csv", specs_ttx0)
    save_csv(DATADIR / "seed_ttx_epc_special_points.csv", specs_ttx)

    # ------------------------------------------------------------------
    # H-C2 from H_bio and H_ext
    # ------------------------------------------------------------------
    print("\n=== H-C2 from H_bio ===")
    # Use PC_ttx0 ContClass (has EQ_ttx0 with Hopfs)
    pts_hbio_f, pts_hbio_b, specs_hbio = run_twopar(
        PC_ttx0,
        "H_from_Hbio",
        "H-C2",
        f"EQ_ttx0:{hbio_lab}",
        ["BT", "ZH", "GH"],
        max_points=220,
        step=1e-3,
        max_step=0.03,
    )
    print("\n=== H-C2 from H_ext ===")
    pts_hext_f, pts_hext_b, specs_hext = run_twopar(
        PC_ttx0,
        "H_from_Hext",
        "H-C2",
        f"EQ_ttx0:{hext_lab}",
        ["BT", "ZH", "GH"],
        max_points=220,
        step=1e-3,
        max_step=0.03,
    )

    save_csv(DATADIR / "hopf_Hbio_raw.csv", pts_hbio_f + pts_hbio_b)
    save_csv(DATADIR / "hopf_Hext_raw.csv", pts_hext_f + pts_hext_b)

    hbio_pts = merge_curve_points([pts_hbio_f, pts_hbio_b], "H_physical")
    hext_pts = merge_curve_points([pts_hext_f, pts_hext_b], "H_from_Hext")
    same_h = curves_overlap(hbio_pts, hext_pts, frac=0.12)
    print(f"H_bio and H_ext curves overlap? {same_h}")
    if same_h:
        hopf_dedup = merge_curve_points([hbio_pts, hext_pts], "H_curve")
        hopf_note = "H_bio and H_ext seeds lie on the same physical Hopf curve (deduplicated)."
    else:
        hopf_dedup = hbio_pts  # primary biological curve
        # Keep H_ext as secondary if distinct
        save_csv(DATADIR / "hopf_curve_from_Hext_dedup.csv", hext_pts)
        hopf_note = (
            "H_bio and H_ext seeds produced distinct (or weakly overlapping) Hopf curves; "
            "primary report curve is from H_bio; H_ext curve saved separately."
        )
        # For the main map, also include both if distinct
        hopf_dedup = merge_curve_points([hbio_pts], "H_curve")
    print(hopf_note)
    with open(DATADIR / "hopf_dedup_note.txt", "w") as f:
        f.write(hopf_note + "\n")
        f.write(f"same_physical_curve={same_h}\n")

    save_csv(DATADIR / "hopf_curve_dedup.csv", hopf_dedup)
    if not same_h:
        # Also store combined for optional plotting of both
        both = merge_curve_points([hbio_pts, hext_pts], "H_all_seeds")
        save_csv(DATADIR / "hopf_curves_both_seeds.csv", both)

    hopf_specs_raw = specs_hbio + specs_hext
    for s in hopf_specs_raw:
        s["curve_family"] = "H"
    hopf_specs = deduplicate_specials(hopf_specs_raw)
    save_csv(DATADIR / "hopf_codim2_raw.csv", hopf_specs_raw)
    save_csv(DATADIR / "hopf_codim2_unique.csv", hopf_specs)

    # ------------------------------------------------------------------
    # LP-C from LP_bio
    # ------------------------------------------------------------------
    print("\n=== LP-C from LP_bio ===")
    pts_lp_f, pts_lp_b, specs_lp = run_twopar(
        PC_ttx,
        "LP_from_LPbio",
        "LP-C",
        f"EQ_ttx:{lp_lab}",
        ["BT", "ZH", "CP"],
        max_points=220,
        step=1e-3,
        max_step=0.03,
    )
    save_csv(DATADIR / "lp_LPbio_raw.csv", pts_lp_f + pts_lp_b)
    lp_dedup = merge_curve_points([pts_lp_f, pts_lp_b], "LP_curve")
    save_csv(DATADIR / "lp_curve_dedup.csv", lp_dedup)

    for s in specs_lp:
        s["curve_family"] = "LP"
    lp_specs = deduplicate_specials(specs_lp)
    save_csv(DATADIR / "lp_codim2_raw.csv", specs_lp)
    save_csv(DATADIR / "lp_codim2_unique.csv", lp_specs)

    all_codim2 = deduplicate_specials(hopf_specs_raw + specs_lp)
    save_csv(DATADIR / "codim2_special_points.csv", all_codim2)
    print("Unique codim-2 points:", len(all_codim2))
    for s in all_codim2:
        print(
            f"  {s.get('type')} @ alpha_I={s.get('alpha_I')} alpha_P={s.get('alpha_P')} "
            f"V={s.get('V')} parent={s.get('parent_curve')}"
        )

    # ------------------------------------------------------------------
    # Slice validation
    # ------------------------------------------------------------------
    print("\n=== Slice validation ===")
    # Primary Hopf curve for validation: prefer combined if same, else H_bio curve;
    # also check H_ext curve for H_ext anchor.
    h_for_val = hopf_dedup if same_h else hbio_pts
    h_ext_curve = hopf_dedup if same_h else hext_pts

    validations = []

    # LP ∩ alpha_P = 1 vs LP_bio
    lp_int = interpolate_slice_intersection(lp_dedup, "alpha_P", 1.0)
    if lp_int is None:
        validations.append(
            {
                "curve": "LP",
                "slice": "alpha_P=1",
                "target": "LP_bio",
                "status": "FAIL",
                "note": "no intersection",
            }
        )
    else:
        d_a = abs(lp_int["alpha_I"] - float(lp_bio["alpha_I"]))
        d_v = abs(lp_int["V"] - float(lp_bio["V"]))
        status = "PASS" if (d_a <= VAL_ATOL_ALPHA and d_v <= VAL_ATOL_V) else "FAIL"
        validations.append(
            {
                "curve": "LP",
                "slice": "alpha_P=1",
                "target": "LP_bio",
                "twoD_alpha_I": lp_int["alpha_I"],
                "twoD_alpha_P": lp_int["alpha_P"],
                "twoD_V": lp_int["V"],
                "oneD_alpha": float(lp_bio["alpha_I"]),
                "oneD_V": float(lp_bio["V"]),
                "diff_alpha": d_a,
                "diff_V": d_v,
                "status": status,
            }
        )

    # H ∩ alpha_I = 0 vs H_bio
    h_int = interpolate_slice_intersection(h_for_val, "alpha_I", 0.0)
    if h_int is None:
        validations.append(
            {
                "curve": "H",
                "slice": "alpha_I=0",
                "target": "H_bio",
                "status": "FAIL",
                "note": "no intersection",
            }
        )
    else:
        # Among intersections / nearest, pick closest to H_bio alpha_P
        # Also search all points with alpha_I near 0
        candidates = [p for p in h_for_val if abs(p["alpha_I"]) <= 0.02]
        if candidates:
            p = min(candidates, key=lambda r: abs(r["alpha_P"] - float(h_bio["alpha_P"])))
        else:
            p = h_int
        d_a = abs(p["alpha_P"] - float(h_bio["alpha_P"]))
        d_v = abs(p["V"] - float(h_bio["V"]))
        status = "PASS" if (d_a <= VAL_ATOL_ALPHA and d_v <= VAL_ATOL_V) else "FAIL"
        validations.append(
            {
                "curve": "H",
                "slice": "alpha_I=0",
                "target": "H_bio",
                "twoD_alpha_I": p["alpha_I"],
                "twoD_alpha_P": p["alpha_P"],
                "twoD_V": p["V"],
                "oneD_alpha": float(h_bio["alpha_P"]),
                "oneD_V": float(h_bio["V"]),
                "diff_alpha": d_a,
                "diff_V": d_v,
                "status": status,
            }
        )

    # H ∩ alpha_I = 0 vs H_ext
    candidates = [p for p in h_ext_curve if abs(p["alpha_I"]) <= 0.02]
    if candidates:
        p = min(candidates, key=lambda r: abs(r["alpha_P"] - float(h_ext["alpha_P"])))
        d_a = abs(p["alpha_P"] - float(h_ext["alpha_P"]))
        d_v = abs(p["V"] - float(h_ext["V"]))
        status = "PASS" if (d_a <= VAL_ATOL_ALPHA and d_v <= VAL_ATOL_V) else "FAIL"
        validations.append(
            {
                "curve": "H" if same_h else "H_from_Hext",
                "slice": "alpha_I=0",
                "target": "H_ext",
                "twoD_alpha_I": p["alpha_I"],
                "twoD_alpha_P": p["alpha_P"],
                "twoD_V": p["V"],
                "oneD_alpha": float(h_ext["alpha_P"]),
                "oneD_V": float(h_ext["V"]),
                "diff_alpha": d_a,
                "diff_V": d_v,
                "status": status,
            }
        )
    else:
        validations.append(
            {
                "curve": "H",
                "slice": "alpha_I=0",
                "target": "H_ext",
                "status": "SKIP",
                "note": "curve does not reach alpha_I~0 near H_ext",
            }
        )

    # Check alpha_I=1: any H in biological alpha_P?
    h_on_aI1 = [p for p in h_for_val if abs(p["alpha_I"] - 1.0) <= 0.02]
    bio_h_on_aI1 = [p for p in h_on_aI1 if 0.0 - 1e-3 <= p["alpha_P"] <= 1.0 + 1e-3]
    validations.append(
        {
            "curve": "H",
            "slice": "alpha_I=1",
            "target": "no_bio_H_expected",
            "n_near_slice": len(h_on_aI1),
            "n_bio_alpha_P": len(bio_h_on_aI1),
            "status": "PASS" if len(bio_h_on_aI1) == 0 else "WARN",
            "note": "NB06 ouabain-alone: no biological Hopf on untreated-connected branch",
        }
    )

    # Check alpha_P=1: any H in biological alpha_I?
    h_on_aP1 = [p for p in h_for_val if abs(p["alpha_P"] - 1.0) <= 0.02]
    bio_h_on_aP1 = [p for p in h_on_aP1 if 0.0 - 1e-3 <= p["alpha_I"] <= 1.0 + 1e-3]
    validations.append(
        {
            "curve": "H",
            "slice": "alpha_P=1",
            "target": "no_bio_H_expected",
            "n_near_slice": len(h_on_aP1),
            "n_bio_alpha_I": len(bio_h_on_aP1),
            "status": "PASS" if len(bio_h_on_aP1) == 0 else "WARN",
            "note": "NB05 TTX-only: no biological Hopf in alpha_I in [0,1]",
        }
    )

    save_csv(DATADIR / "slice_validation.csv", validations)
    for v in validations:
        print(v)

    gate_fail = any(
        v.get("status") == "FAIL" and v.get("target") in ("LP_bio", "H_bio") for v in validations
    )
    if gate_fail:
        print("\n*** VALIDATION GATE FAILED for LP_bio/H_bio — stopping before figures ***")
        save_csv(DATADIR / "validation_gate.txt", [{"status": "FAIL"}])
        raise SystemExit(2)
    print("\nValidation gate: PASS")

    # ------------------------------------------------------------------
    # Stability sample points
    # ------------------------------------------------------------------
    print("\n=== Stability samples ===")
    # Choose points based on curve extents inside the square
    sample_pts = [
        (1.0, 1.0, "control"),
        (0.0, 1.0, "complete_TTX"),
        (1.0, 0.0, "complete_ouabain"),
        (0.0, 0.0, "TTX_plus_ouabain"),
        (0.0, 0.15, "TTX0_below_Hbio"),
        (0.0, 0.5, "TTX0_between_Hopfs"),
        (0.3, 1.0, "TTX_partial_alphaP1"),
        (0.8, 1.0, "TTX_mild_alphaP1"),
        (0.5, 0.5, "mid_square"),
        (0.2, 0.2, "low_low"),
        (0.8, 0.2, "highI_lowP"),
        (0.2, 0.8, "lowI_highP"),
    ]
    stab_rows = []
    for aI, aP, name in sample_pts:
        pars = pars_from_alpha(aI, aP)
        roots = scan_equilibrium_voltages(pars)
        # Prefer root nearest to known regimes
        infos = []
        for r in roots:
            info = analyse_equilibrium(r["V_star"], pars, r["root_type"])
            if info.get("accepted", False):
                infos.append(info)
        if not infos:
            stab_rows.append(
                {
                    "name": name,
                    "alpha_I": aI,
                    "alpha_P": aP,
                    "n_accepted_eq": 0,
                    "stability": "none_found",
                }
            )
            continue
        # Report all accepted briefly; pick primary as most depolarized or mid
        for j, info in enumerate(infos):
            stab_rows.append(
                {
                    "name": name if j == 0 else f"{name}_eq{j+1}",
                    "alpha_I": aI,
                    "alpha_P": aP,
                    "n_accepted_eq": len(infos),
                    "V_star": info["V_star"],
                    "stability": info["stability"],
                    "max_real_part": info["max_real_part"],
                    "leading_real": info["leading_real"],
                    "leading_imag": info["leading_imag"],
                    "n_unstable": info["n_unstable"],
                }
            )
    save_csv(DATADIR / "stability_sample_points.csv", stab_rows)
    print(f"Wrote {len(stab_rows)} stability rows")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n=== Figures ===")
    anchors = {
        "LP_bio": (float(lp_bio["alpha_I"]), 1.0),
        "H_bio": (0.0, float(h_bio["alpha_P"])),
        "H_ext": (0.0, float(h_ext["alpha_P"])),
    }

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    draw_bio_square(ax)
    ax.axvline(0, color="0.6", ls="--", lw=0.8)
    ax.axhline(0, color="0.6", ls="--", lw=0.8)
    ax.axvline(1, color="0.75", ls=":", lw=0.8, label=r"$\alpha_I=1$ slice")
    ax.axhline(1, color="0.75", ls=":", lw=0.8, label=r"$\alpha_P=1$ slice")

    if hopf_dedup:
        ax.plot(
            [p["alpha_I"] for p in hopf_dedup],
            [p["alpha_P"] for p in hopf_dedup],
            color="C3",
            lw=2.0,
            label="Hopf curve",
        )
    if not same_h and hext_pts:
        ax.plot(
            [p["alpha_I"] for p in hext_pts],
            [p["alpha_P"] for p in hext_pts],
            color="C3",
            lw=1.5,
            ls="--",
            label="Hopf (H_ext seed)",
        )
    if lp_dedup:
        ax.plot(
            [p["alpha_I"] for p in lp_dedup],
            [p["alpha_P"] for p in lp_dedup],
            color="C0",
            lw=2.0,
            label="LP / fold curve",
        )

    ax.plot(1, 1, "ko", ms=7, label="control (1,1)")
    for lab, (x, y) in anchors.items():
        ax.plot(x, y, "k^", ms=7)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Codim-2 markers
    markers = {"GH": "s", "BT": "D", "CP": "P", "ZH": "X"}
    colors = {"GH": "darkorange", "BT": "purple", "CP": "green", "ZH": "brown"}
    seen_types = set()
    for s in all_codim2:
        t = s.get("type")
        if t not in markers or "alpha_I" not in s or "alpha_P" not in s:
            continue
        lab = t if t not in seen_types else None
        seen_types.add(t)
        ax.plot(
            s["alpha_I"],
            s["alpha_P"],
            markers[t],
            color=colors[t],
            ms=8,
            label=lab,
            zorder=5,
        )

    ax.set_xlabel(r"$\alpha_I$ (TTX scaling)")
    ax.set_ylabel(r"$\alpha_P$ (ouabain scaling)")
    ax.set_xlim(A_I_LO, A_I_HI)
    ax.set_ylim(A_P_LO, A_P_HI)
    ax.set_title(r"Two-parameter equilibrium bifurcations $(\alpha_I,\alpha_P)$")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_alphaIP_bifurcation_map.png", dpi=160)
    plt.close(fig)

    # Zoom on biological square
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    draw_bio_square(ax)
    ax.axvline(0, color="0.6", ls="--", lw=0.8)
    ax.axhline(0, color="0.6", ls="--", lw=0.8)
    if hopf_dedup:
        ax.plot(
            [p["alpha_I"] for p in hopf_dedup],
            [p["alpha_P"] for p in hopf_dedup],
            color="C3",
            lw=2.0,
            label="Hopf curve",
        )
    if not same_h and hext_pts:
        ax.plot(
            [p["alpha_I"] for p in hext_pts],
            [p["alpha_P"] for p in hext_pts],
            color="C3",
            lw=1.5,
            ls="--",
            label="Hopf (H_ext seed)",
        )
    if lp_dedup:
        ax.plot(
            [p["alpha_I"] for p in lp_dedup],
            [p["alpha_P"] for p in lp_dedup],
            color="C0",
            lw=2.0,
            label="LP / fold curve",
        )
    ax.plot(1, 1, "ko", ms=7, label="control (1,1)")
    for lab, (x, y) in [("LP_bio", anchors["LP_bio"]), ("H_bio", anchors["H_bio"])]:
        ax.plot(x, y, "k^", ms=7)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    for s in all_codim2:
        t = s.get("type")
        if t not in markers or "alpha_I" not in s:
            continue
        if not ( -0.05 <= s["alpha_I"] <= 1.05 and -0.05 <= s["alpha_P"] <= 1.05):
            continue
        ax.plot(s["alpha_I"], s["alpha_P"], markers[t], color=colors[t], ms=8, zorder=5)
        ax.annotate(t, (s["alpha_I"], s["alpha_P"]), textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.set_xlabel(r"$\alpha_I$")
    ax.set_ylabel(r"$\alpha_P$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Biological square focus")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_alphaIP_biological_square.png", dpi=160)
    plt.close(fig)

    # Transformed g_I–I_ep figure
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    draw_bio_square(ax, transformed=True)
    if hopf_dedup:
        ax.plot(
            [p["g_I"] for p in hopf_dedup],
            [p["I_ep"] for p in hopf_dedup],
            color="C3",
            lw=2.0,
            label="Hopf curve",
        )
    if not same_h and hext_pts:
        ax.plot(
            [p["alpha_I"] * G_I0 for p in hext_pts],
            [p["alpha_P"] * I_EP0 for p in hext_pts],
            color="C3",
            lw=1.5,
            ls="--",
            label="Hopf (H_ext seed)",
        )
    if lp_dedup:
        ax.plot(
            [p["g_I"] for p in lp_dedup],
            [p["I_ep"] for p in lp_dedup],
            color="C0",
            lw=2.0,
            label="LP / fold curve",
        )
    ax.plot(G_I0, I_EP0, "ko", ms=7, label="control")
    ax.plot(0.0, float(h_bio["alpha_P"]) * I_EP0, "k^", ms=7)
    ax.annotate("H_bio", (0.0, float(h_bio["alpha_P"]) * I_EP0), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.plot(float(lp_bio["alpha_I"]) * G_I0, I_EP0, "k^", ms=7)
    ax.annotate("LP_bio", (float(lp_bio["alpha_I"]) * G_I0, I_EP0), textcoords="offset points", xytext=(5, 5), fontsize=8)
    for s in all_codim2:
        t = s.get("type")
        if t not in markers or "alpha_I" not in s:
            continue
        ax.plot(
            float(s["alpha_I"]) * G_I0,
            float(s["alpha_P"]) * I_EP0,
            markers[t],
            color=colors[t],
            ms=8,
            label=t if t not in seen_types else None,
            zorder=5,
        )
    ax.set_xlabel(r"$g_I = G_{I0}\,\alpha_I$")
    ax.set_ylabel(r"$I_{\mathrm{ep}} = I_{EP0}\,\alpha_P$  (note $I_{EP0}<0$)")
    ax.set_title(r"Same geometry in physical parameters $(g_I,I_{\mathrm{ep}})$")
    ax.legend(loc="best", fontsize=8)
    # Show modest exterior margin in physical coords
    ax.set_xlim(A_I_LO * G_I0, A_I_HI * G_I0)
    # alpha_P up -> I_ep more negative: flip or keep natural
    ax.set_ylim(A_P_HI * I_EP0, A_P_LO * I_EP0)  # so up on plot is less negative / toward 0? 
    # Actually natural: I_ep axis from more negative (bottom) to less negative (top)
    # A_P_HI * I_EP0 is more negative. Better: ylim (more neg, less neg) = (A_P_HI*I_EP0, A_P_LO*I_EP0)
    # Wait I_EP0=-0.22, A_P_HI=2 -> -0.44; A_P_LO=-0.25 -> +0.055
    # So ylim(-0.44, 0.055) with increasing upward is correct without flip.
    ax.set_ylim(A_P_HI * I_EP0, A_P_LO * I_EP0)
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_gI_Iep_transformed_map.png", dpi=160)
    plt.close(fig)

    # Fix I_ep axis: matplotlib ylim(low, high) with low > high flips. Use ascending.
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    draw_bio_square(ax, transformed=True)
    if hopf_dedup:
        ax.plot(
            [p["g_I"] for p in hopf_dedup],
            [p["I_ep"] for p in hopf_dedup],
            color="C3",
            lw=2.0,
            label="Hopf curve",
        )
    if not same_h and hext_pts:
        ax.plot(
            [p["alpha_I"] * G_I0 for p in hext_pts],
            [p["alpha_P"] * I_EP0 for p in hext_pts],
            color="C3",
            lw=1.5,
            ls="--",
            label="Hopf (H_ext seed)",
        )
    if lp_dedup:
        ax.plot(
            [p["g_I"] for p in lp_dedup],
            [p["I_ep"] for p in lp_dedup],
            color="C0",
            lw=2.0,
            label="LP / fold curve",
        )
    ax.plot(G_I0, I_EP0, "ko", ms=7, label="control")
    ax.plot(0.0, float(h_bio["alpha_P"]) * I_EP0, "k^", ms=7)
    ax.annotate(
        "H_bio",
        (0.0, float(h_bio["alpha_P"]) * I_EP0),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
    )
    ax.plot(float(lp_bio["alpha_I"]) * G_I0, I_EP0, "k^", ms=7)
    ax.annotate(
        "LP_bio",
        (float(lp_bio["alpha_I"]) * G_I0, I_EP0),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
    )
    labeled = set()
    for s in all_codim2:
        t = s.get("type")
        if t not in markers or "alpha_I" not in s:
            continue
        lab = t if t not in labeled else None
        labeled.add(t)
        ax.plot(
            float(s["alpha_I"]) * G_I0,
            float(s["alpha_P"]) * I_EP0,
            markers[t],
            color=colors[t],
            ms=8,
            label=lab,
            zorder=5,
        )
    ax.set_xlabel(r"$g_I = G_{I0}\,\alpha_I$")
    ax.set_ylabel(r"$I_{\mathrm{ep}} = I_{EP0}\,\alpha_P$  ($I_{EP0}<0$; $\alpha_P\uparrow$ ⇒ more negative)")
    ax.set_title(r"Transformed map $(g_I,I_{\mathrm{ep}})$ — no new continuation")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(min(A_I_LO * G_I0, -0.5), max(A_I_HI * G_I0, 4.5))
    y0, y1 = A_P_HI * I_EP0, A_P_LO * I_EP0
    ax.set_ylim(min(y0, y1), max(y0, y1))
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_gI_Iep_transformed_map.png", dpi=160)
    plt.close(fig)

    print("Saved figures to", FIGDIR)
    print("DONE")


if __name__ == "__main__":
    main()

"""Analysis helpers for Plant--Kim drug-parameter sweeps and equilibria.

Scaled drug parameters are analysis-level only:

    g_I = alpha_I * g_I_baseline
    I_ep = alpha_P * I_ep_baseline

The canonical 7D equations in plant_kim_model.py are not modified.
ODE integration remains PyDSTool VODE via build_vode_system / simulate.
Equilibria use the exact gating substitution and a scalar current balance.
The Jacobian is a central finite difference of a NumPy vector field that
imports the validated gating functions.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from copy import deepcopy

import PyDSTool as dst

from src.plant_kim_model import (
    BASELINE_PARS,
    DEFAULT_ALGPARAMS,
    DEFAULT_V0,
    STATE_VARS,
    S_A,
    S_I,
    S_K,
    S_P,
    Z_A,
    Z_I,
    make_fnspecs,
    make_varspecs,
    steady_state_ics,
    tau_X_I,
    tau_X_K,
    tau_Y_I,
    build_vode_system,
    simulate,
)

SPIKE_THR = -20.0  # mV
BURST_GAP = 1500.0  # ms
EQ_V_RANGE = 0.5  # mV
EQ_MEAN_DVDT = 1e-3  # mV/ms
STABILITY_TOL = 1e-5
ROOT_DEDUP_MV = 0.05
TANGENT_F_TOL = 1e-6
RESIDUAL_TOL = 1e-8
H_V = 1e-3
H_G = 1e-6

G_I0 = float(BASELINE_PARS["g_I"])
I_EP0 = float(BASELINE_PARS["I_ep"])


def pars_from_alpha(alpha_I: float, alpha_P: float) -> Dict[str, float]:
    """Copy untreated parameters and scale only g_I and I_ep."""
    pars = dict(BASELINE_PARS)
    pars["g_I"] = float(alpha_I) * G_I0
    pars["I_ep"] = float(alpha_P) * I_EP0
    return pars


def make_alpha_I_varspecs() -> Dict[str, str]:
    """Continuation-only varspecs: inward conductance is alpha_I * g_I.

    Canonical plant_kim_model.make_varspecs is unchanged. Here g_I is held at
    the baseline value while alpha_I is the free ContClass parameter, so

        g_I_eff = alpha_I * g_I = alpha_I * g_I^(0)
    """
    specs = make_varspecs()
    specs["V"] = (
        "(1/C)*((alpha_I*g_I*X_I^3*Y_I + g_T)*(V_I - V)"
        " + (g_K*X_K^4 + g_A*X_A*Y_A + g_P*X_P)*(V_K - V)"
        " + g_L*(V_L - V) + I_ep + I_ext)"
    )
    return specs


def build_alpha_I_vode_system(
    pars: Optional[Mapping[str, float]] = None,
    ics: Optional[Mapping[str, float]] = None,
    name: str = "PlantKim7D_alphaI",
    algparams: Optional[Mapping[str, object]] = None,
    alpha_I: float = 1.0,
):
    """VODE system for ContClass continuation in alpha_I (not a model change).

    Parameters include alpha_I and freeze g_I at baseline G_I0. Effective
    transient inward conductance is alpha_I * g_I.
    """
    use_pars = deepcopy(BASELINE_PARS) if pars is None else dict(pars)
    use_pars["g_I"] = float(G_I0)
    use_pars["alpha_I"] = float(alpha_I)
    use_ics = steady_state_ics(DEFAULT_V0) if ics is None else dict(ics)

    DSargs = dst.args(name=name)
    DSargs.pars = use_pars
    DSargs.fnspecs = make_fnspecs()
    DSargs.varspecs = make_alpha_I_varspecs()
    DSargs.ics = use_ics
    ode_sys = dst.Generator.Vode_ODEsystem(DSargs)
    use_alg = dict(DEFAULT_ALGPARAMS) if algparams is None else dict(algparams)
    if use_alg:
        ode_sys.set(algparams=use_alg)
    return ode_sys


def make_alpha_P_varspecs() -> Dict[str, str]:
    """Continuation-only varspecs: electrogenic pump is alpha_P * I_ep.

    Canonical plant_kim_model.make_varspecs is unchanged. Here I_ep is held at
    the baseline value while alpha_P is the free ContClass parameter, so

        I_ep_eff = alpha_P * I_ep = alpha_P * I_ep^(0)
    """
    specs = make_varspecs()
    specs["V"] = (
        "(1/C)*((g_I*X_I^3*Y_I + g_T)*(V_I - V)"
        " + (g_K*X_K^4 + g_A*X_A*Y_A + g_P*X_P)*(V_K - V)"
        " + g_L*(V_L - V) + alpha_P*I_ep + I_ext)"
    )
    return specs


def build_alpha_P_vode_system(
    pars: Optional[Mapping[str, float]] = None,
    ics: Optional[Mapping[str, float]] = None,
    name: str = "PlantKim7D_alphaP",
    algparams: Optional[Mapping[str, object]] = None,
    alpha_P: float = 1.0,
):
    """VODE system for ContClass continuation in alpha_P (not a model change).

    Parameters include alpha_P and freeze I_ep at baseline I_EP0. Effective
    electrogenic pump current is alpha_P * I_ep. Callers may override g_I
    (e.g. g_I=0 for complete-TTX + ouabain) via pars=.
    """
    use_pars = deepcopy(BASELINE_PARS) if pars is None else dict(pars)
    use_pars["I_ep"] = float(I_EP0)
    use_pars["alpha_P"] = float(alpha_P)
    use_ics = steady_state_ics(DEFAULT_V0) if ics is None else dict(ics)

    DSargs = dst.args(name=name)
    DSargs.pars = use_pars
    DSargs.fnspecs = make_fnspecs()
    DSargs.varspecs = make_alpha_P_varspecs()
    DSargs.ics = use_ics
    ode_sys = dst.Generator.Vode_ODEsystem(DSargs)
    use_alg = dict(DEFAULT_ALGPARAMS) if algparams is None else dict(algparams)
    if use_alg:
        ode_sys.set(algparams=use_alg)
    return ode_sys


def make_alpha_IP_varspecs() -> Dict[str, str]:
    """Continuation-only varspecs with both alpha_I and alpha_P scalings.

    Canonical plant_kim_model.make_varspecs is unchanged. Here g_I and I_ep
    are held at baseline while ContClass free parameters are alpha_I and
    alpha_P, so

        g_I_eff  = alpha_I * g_I  = alpha_I * g_I^(0)
        I_ep_eff = alpha_P * I_ep = alpha_P * I_ep^(0)
    """
    specs = make_varspecs()
    specs["V"] = (
        "(1/C)*((alpha_I*g_I*X_I^3*Y_I + g_T)*(V_I - V)"
        " + (g_K*X_K^4 + g_A*X_A*Y_A + g_P*X_P)*(V_K - V)"
        " + g_L*(V_L - V) + alpha_P*I_ep + I_ext)"
    )
    return specs


def build_alpha_IP_vode_system(
    pars: Optional[Mapping[str, float]] = None,
    ics: Optional[Mapping[str, float]] = None,
    name: str = "PlantKim7D_alphaIP",
    algparams: Optional[Mapping[str, object]] = None,
    alpha_I: float = 1.0,
    alpha_P: float = 1.0,
):
    """VODE system for two-parameter ContClass work in (alpha_I, alpha_P).

    Freezes baseline g_I = G_I0 and I_ep = I_EP0. Effective drug parameters
    are alpha_I * g_I and alpha_P * I_ep. Not a scientific-model change.
    """
    use_pars = deepcopy(BASELINE_PARS) if pars is None else dict(pars)
    use_pars["g_I"] = float(G_I0)
    use_pars["I_ep"] = float(I_EP0)
    use_pars["alpha_I"] = float(alpha_I)
    use_pars["alpha_P"] = float(alpha_P)
    use_ics = steady_state_ics(DEFAULT_V0) if ics is None else dict(ics)

    DSargs = dst.args(name=name)
    DSargs.pars = use_pars
    DSargs.fnspecs = make_fnspecs()
    DSargs.varspecs = make_alpha_IP_varspecs()
    DSargs.ics = use_ics
    ode_sys = dst.Generator.Vode_ODEsystem(DSargs)
    use_alg = dict(DEFAULT_ALGPARAMS) if algparams is None else dict(algparams)
    if use_alg:
        ode_sys.set(algparams=use_alg)
    return ode_sys


def corner_table() -> List[Dict[str, float]]:
    """Four endpoint corners with both alpha and physical parameters."""
    corners = [
        ("untreated", 1.0, 1.0),
        ("TTX", 0.0, 1.0),
        ("ouabain", 1.0, 0.0),
        ("TTX+ouabain", 0.0, 0.0),
    ]
    rows = []
    for name, a_I, a_P in corners:
        pars = pars_from_alpha(a_I, a_P)
        rows.append(
            {
                "corner": name,
                "alpha_I": a_I,
                "alpha_P": a_P,
                "g_I": pars["g_I"],
                "I_ep": pars["I_ep"],
            }
        )
    return rows


def state_vector(state: Mapping[str, float]) -> np.ndarray:
    return np.array([float(state[k]) for k in STATE_VARS], dtype=float)


def vector_to_state(x: Sequence[float]) -> Dict[str, float]:
    return {k: float(x[i]) for i, k in enumerate(STATE_VARS)}


def current_balance(V, pars: Mapping[str, float]):
    """Bracketed membrane current F(V); zero iff Vdot = 0 when C != 0.

    Gates are replaced by their voltage-dependent steady states.
    """
    V = np.asarray(V, dtype=float)
    g_I = float(pars["g_I"])
    inward = (g_I * S_I(V) ** 3 * Z_I(V) + float(pars["g_T"])) * (
        float(pars["V_I"]) - V
    )
    outward = (
        float(pars["g_K"]) * S_K(V) ** 4
        + float(pars["g_A"]) * S_A(V) * Z_A(V)
        + float(pars["g_P"]) * S_P(V)
    ) * (float(pars["V_K"]) - V)
    leak = float(pars["g_L"]) * (float(pars["V_L"]) - V)
    return inward + outward + leak + float(pars["I_ep"]) + float(pars["I_ext"])


def rhs(state, pars: Mapping[str, float]) -> np.ndarray:
    """Canonical 7D vector field. state is a dict or a length-7 sequence."""
    if isinstance(state, Mapping):
        V = float(state["V"])
        X_K = float(state["X_K"])
        X_A = float(state["X_A"])
        Y_A = float(state["Y_A"])
        X_P = float(state["X_P"])
        X_I = float(state["X_I"])
        Y_I = float(state["Y_I"])
    else:
        V, X_K, X_A, Y_A, X_P, X_I, Y_I = (float(v) for v in state)

    C = float(pars["C"])
    vdot = (
        (float(pars["g_I"]) * X_I ** 3 * Y_I + float(pars["g_T"]))
        * (float(pars["V_I"]) - V)
        + (
            float(pars["g_K"]) * X_K ** 4
            + float(pars["g_A"]) * X_A * Y_A
            + float(pars["g_P"]) * X_P
        )
        * (float(pars["V_K"]) - V)
        + float(pars["g_L"]) * (float(pars["V_L"]) - V)
        + float(pars["I_ep"])
        + float(pars["I_ext"])
    ) / C
    return np.array(
        [
            vdot,
            (float(S_K(V)) - X_K) / float(tau_X_K(V)),
            (float(S_A(V)) - X_A) / float(pars["tau_X_A"]),
            (float(Z_A(V)) - Y_A) / float(pars["tau_Y_A"]),
            (float(S_P(V)) - X_P) / float(pars["tau_X_P"]),
            (float(S_I(V)) - X_I) / float(tau_X_I(V)),
            (float(Z_I(V)) - Y_I) / float(tau_Y_I(V)),
        ],
        dtype=float,
    )


def equilibrium_state(V: float) -> Dict[str, float]:
    return steady_state_ics(float(V))


def residual_norm(V: float, pars: Mapping[str, float]) -> float:
    return float(np.max(np.abs(rhs(equilibrium_state(V), pars))))


def _dedup_roots(roots: Sequence[float], tol: float = ROOT_DEDUP_MV) -> List[float]:
    if not roots:
        return []
    ordered = sorted(float(r) for r in roots)
    kept = [ordered[0]]
    for r in ordered[1:]:
        if abs(r - kept[-1]) > tol:
            kept.append(r)
    return kept


def _sign_change_roots(V: np.ndarray, F: np.ndarray, pars) -> List[float]:
    roots: List[float] = []
    for i in range(len(V) - 1):
        f0, f1 = float(F[i]), float(F[i + 1])
        if f0 == 0.0:
            roots.append(float(V[i]))
            continue
        if f0 * f1 < 0.0:
            a, b = float(V[i]), float(V[i + 1])
            roots.append(
                float(brentq(lambda v: float(current_balance(v, pars)), a, b))
            )
    if float(F[-1]) == 0.0:
        roots.append(float(V[-1]))
    return roots


def _tangent_candidates(V: np.ndarray, F: np.ndarray, pars, known: Sequence[float]) -> List[float]:
    """Local minima of |F| that do not change sign, refined and thresholded."""
    mag = np.abs(F)
    found: List[float] = []
    known = list(known)
    for i in range(1, len(V) - 1):
        if not (mag[i] <= mag[i - 1] and mag[i] <= mag[i + 1]):
            continue
        if mag[i] > 1e-3:
            continue
        if any(abs(float(V[i]) - r) <= ROOT_DEDUP_MV for r in known):
            continue
        a = float(V[i - 1])
        b = float(V[i + 1])
        opt = minimize_scalar(
            lambda v: abs(float(current_balance(v, pars))),
            bounds=(a, b),
            method="bounded",
            options={"xatol": 1e-8},
        )
        v_star = float(opt.x)
        f_star = abs(float(current_balance(v_star, pars)))
        if f_star >= TANGENT_F_TOL:
            continue
        if any(abs(v_star - r) <= ROOT_DEDUP_MV for r in known + found):
            continue
        # Confirm a local minimum of |F| on a slightly wider window.
        vv = np.linspace(v_star - 0.05, v_star + 0.05, 11)
        if abs(float(current_balance(v_star, pars))) > np.min(np.abs(current_balance(vv, pars))) + 1e-12:
            continue
        found.append(v_star)
    return found


def scan_equilibrium_voltages(
    pars: Mapping[str, float],
    v_lo: float = -80.0,
    v_hi: float = 40.0,
    dv: float = 0.1,
) -> List[Dict[str, float]]:
    """Find all simple roots of F and any tangent candidates on a voltage grid."""

    def evaluate(lo, hi):
        V = np.arange(lo, hi + 0.5 * dv, dv)
        F = np.asarray(current_balance(V, pars), dtype=float)
        return V, F

    V, F = evaluate(v_lo, v_hi)
    mag = np.abs(F)
    near_edge = False
    for i in range(len(V) - 1):
        edge = min(V[i], V[i + 1]) - v_lo <= 2.0 or v_hi - max(V[i], V[i + 1]) <= 2.0
        if edge and (
            F[i] * F[i + 1] < 0.0
            or (0 < i < len(V) - 1 and mag[i] <= mag[i - 1] and mag[i] <= mag[i + 1] and mag[i] < 1e-3)
        ):
            near_edge = True
            break
    used_lo, used_hi = v_lo, v_hi
    if near_edge:
        used_lo, used_hi = -120.0, 60.0
        V, F = evaluate(used_lo, used_hi)

    simple = _dedup_roots(_sign_change_roots(V, F, pars))
    tangent = _dedup_roots(_tangent_candidates(V, F, pars, simple))
    rows = []
    for v, kind in [(r, "simple") for r in simple] + [(r, "tangent_candidate") for r in tangent]:
        rows.append(
            {
                "V_star": float(v),
                "root_type": kind,
                "F": float(current_balance(v, pars)),
                "v_lo": used_lo,
                "v_hi": used_hi,
            }
        )
    rows.sort(key=lambda r: r["V_star"])
    return rows


def jacobian(
    state,
    pars: Mapping[str, float],
    h_V: float = H_V,
    h_g: float = H_G,
) -> np.ndarray:
    """Central finite-difference Jacobian. Voltage and gates use separate steps."""
    x = state_vector(state) if isinstance(state, Mapping) else np.asarray(state, dtype=float)
    h = np.array([h_V] + [h_g] * 6, dtype=float)
    J = np.zeros((7, 7), dtype=float)
    for j in range(7):
        xp = x.copy()
        xm = x.copy()
        xp[j] += h[j]
        xm[j] -= h[j]
        J[:, j] = (rhs(xp, pars) - rhs(xm, pars)) / (2.0 * h[j])
    return J


def eigenvalue_summary(eigs: np.ndarray, tol: float = STABILITY_TOL) -> Dict[str, object]:
    eigs = np.asarray(eigs)
    re = np.real(eigs)
    im = np.imag(eigs)
    k = int(np.argmax(re))
    leading = eigs[k]
    is_complex = abs(float(im[k])) > 1e-8
    max_re = float(re[k])
    if np.all(re < -tol):
        stability = "stable"
    elif np.any(re > tol):
        stability = "unstable"
    else:
        stability = "near_critical"
    return {
        "stability": stability,
        "max_real_part": max_re,
        "leading_real": float(np.real(leading)),
        "leading_imag": float(np.imag(leading)),
        "leading_is_complex": bool(is_complex),
        "n_unstable": int(np.sum(re > tol)),
        "eigenvalues": eigs,
    }


def analyse_equilibrium(V: float, pars: Mapping[str, float], root_type: str = "simple") -> Dict[str, object]:
    state = equilibrium_state(V)
    f = rhs(state, pars)
    res = float(np.max(np.abs(f)))
    accepted = res < RESIDUAL_TOL
    row: Dict[str, object] = {
        "V_star": float(V),
        "root_type": root_type,
        "residual_max": res,
        "accepted": accepted,
        **{k: state[k] for k in STATE_VARS},
    }
    if not accepted:
        row.update(
            {
                "stability": "rejected",
                "max_real_part": float("nan"),
                "leading_real": float("nan"),
                "leading_imag": float("nan"),
                "leading_is_complex": False,
                "n_unstable": 0,
                "eigenvalues": np.full(7, np.nan),
            }
        )
        return row
    eigs = np.linalg.eigvals(jacobian(state, pars))
    row.update(eigenvalue_summary(eigs))
    return row


def jacobian_step_check(V: float, pars: Mapping[str, float]) -> Dict[str, float]:
    """Leading real-part agreement across finite-difference steps."""
    state = equilibrium_state(V)
    out = {}
    for scale, tag in ((0.1, "fine"), (1.0, "base"), (10.0, "coarse")):
        J = jacobian(state, pars, h_V=H_V * scale, h_g=H_G * scale)
        summary = eigenvalue_summary(np.linalg.eigvals(J))
        out[tag] = float(summary["max_real_part"])
    out["spread"] = float(max(out.values()) - min(v for k, v in out.items() if k != "spread"))
    return out


def scan_slice(
    alphas: Sequence[float],
    *,
    vary: str,
    alpha_I_fixed: float = 1.0,
    alpha_P_fixed: float = 1.0,
) -> List[Dict[str, object]]:
    """Scan all accepted equilibria along one drug-parameter slice."""
    rows: List[Dict[str, object]] = []
    for a in alphas:
        if vary == "alpha_I":
            a_I, a_P = float(a), float(alpha_P_fixed)
        elif vary == "alpha_P":
            a_I, a_P = float(alpha_I_fixed), float(a)
        else:
            raise ValueError(vary)
        pars = pars_from_alpha(a_I, a_P)
        for root in scan_equilibrium_voltages(pars):
            info = analyse_equilibrium(root["V_star"], pars, root["root_type"])
            info["alpha_I"] = a_I
            info["alpha_P"] = a_P
            info["g_I"] = pars["g_I"]
            info["I_ep"] = pars["I_ep"]
            rows.append(info)
    return rows


def track_sign_changes(rows: Sequence[Mapping[str, object]], vary: str) -> List[Dict[str, object]]:
    """Nearest-neighbour eigenvalue tracking where V* matching is unique.

    Only report candidate crossings. Do not label them as bifurcations.
    The persistently small slow eigenvalue is not selected by |Re| ranking;
    a candidate must actually change sign along a tracked eigenvalue.
    """
    by_alpha: Dict[float, List[Mapping[str, object]]] = {}
    for row in rows:
        if not row.get("accepted", False):
            continue
        by_alpha.setdefault(float(row[vary]), []).append(row)
    alphas = sorted(by_alpha)
    events: List[Dict[str, object]] = []
    if len(alphas) < 2:
        return events

    def unique_pairs(left, right, v_tol=2.0):
        pairs = []
        used_r = set()
        for i, a in enumerate(left):
            dists = []
            for j, b in enumerate(right):
                if j in used_r:
                    continue
                dists.append((abs(float(a["V_star"]) - float(b["V_star"])), j))
            if not dists:
                return None
            dists.sort()
            if dists[0][0] > v_tol:
                return None
            if len(dists) > 1 and dists[1][0] - dists[0][0] < 0.2:
                return None
            used_r.add(dists[0][1])
            pairs.append((i, dists[0][1]))
        if len(pairs) != min(len(left), len(right)):
            return None
        if len(left) != len(right):
            return None
        return pairs

    for a0, a1 in zip(alphas[:-1], alphas[1:]):
        left = sorted(by_alpha[a0], key=lambda r: float(r["V_star"]))
        right = sorted(by_alpha[a1], key=lambda r: float(r["V_star"]))
        pairs = unique_pairs(left, right)
        if pairs is None:
            continue
        for i, j in pairs:
            e0 = np.asarray(left[i]["eigenvalues"])
            e1 = np.asarray(right[j]["eigenvalues"])
            unused = set(range(len(e1)))
            match = []
            for p, lam0 in enumerate(e0):
                best = None
                best_d = None
                for q in unused:
                    d = abs(lam0 - e1[q])
                    if best_d is None or d < best_d:
                        best, best_d = q, d
                unused.remove(best)
                match.append((p, best))
            for p, q in match:
                r0, r1 = float(np.real(e0[p])), float(np.real(e1[q]))
                i0, i1 = abs(float(np.imag(e0[p]))), abs(float(np.imag(e1[q])))
                crossed = (r0 > STABILITY_TOL and r1 < -STABILITY_TOL) or (
                    r0 < -STABILITY_TOL and r1 > STABILITY_TOL
                )
                if not crossed:
                    continue
                if i0 < 1e-8 and i1 < 1e-8:
                    kind = "candidate_real_zero_crossing"
                elif i0 > 1e-8 and i1 > 1e-8:
                    kind = "candidate_complex_real_part_crossing"
                else:
                    kind = "candidate_mixed_crossing"
                events.append(
                    {
                        "kind": kind,
                        "vary": vary,
                        "alpha_left": a0,
                        "alpha_right": a1,
                        "V_left": float(left[i]["V_star"]),
                        "V_right": float(right[j]["V_star"]),
                        "re_left": r0,
                        "re_right": r1,
                        "im_left": float(np.imag(e0[p])),
                        "im_right": float(np.imag(e1[q])),
                    }
                )
    return events


# ---------------------------------------------------------------------------
# Direct-simulation sweep metrics
# ---------------------------------------------------------------------------


def group_bursts(tspike, gap: float = BURST_GAP):
    if len(tspike) == 0:
        return []
    groups = [[float(tspike[0])]]
    for i in range(1, len(tspike)):
        if float(tspike[i]) - float(tspike[i - 1]) > gap:
            groups.append([float(tspike[i])])
        else:
            groups[-1].append(float(tspike[i]))
    return groups


def estimate_slow_period_ms(t, V, dt: float = 0.5) -> float:
    if len(V) < 20:
        return float("nan")
    w = max(3, int(round(50.0 / dt)))
    if w % 2 == 0:
        w += 1
    Vs = np.convolve(V, np.ones(w) / w, mode="same")
    half = 0.25 * (Vs.max() - Vs.min())
    peaks = []
    for i in range(1, len(Vs) - 1):
        if Vs[i] >= Vs[i - 1] and Vs[i] > Vs[i + 1] and Vs[i] > (Vs.mean() + 0.1 * half):
            if not peaks or (t[i] - peaks[-1]) > 500.0:
                peaks.append(t[i])
    if len(peaks) < 2:
        return float("nan")
    return float(np.median(np.diff(peaks)))


def _window(data, t0, t1):
    t = data["t"]
    m = (t >= t0) & (t <= t1)
    return t[m], data["V"][m]


def spike_times(t, V, thr: float = SPIKE_THR) -> np.ndarray:
    if len(V) < 2:
        return np.array([])
    cross = np.where((V[:-1] <= thr) & (V[1:] > thr))[0]
    return t[cross]


def mean_spike_peak(t, V, tspike, look_ms: float = 30.0) -> float:
    if len(tspike) == 0:
        return float("nan")
    peaks = []
    for ts in tspike:
        m = (t >= ts) & (t <= ts + look_ms)
        if np.any(m):
            peaks.append(float(np.max(V[m])))
    return float(np.mean(peaks)) if peaks else float("nan")


def classify_window(t, V) -> Dict[str, object]:
    if len(V) < 3 or not np.all(np.isfinite(V)):
        return {
            "regime": "FAIL",
            "V_min": float("nan"),
            "V_max": float("nan"),
            "amplitude": float("nan"),
            "n_spikes": 0,
            "firing_rate_hz": float("nan"),
            "mean_spike_peak": float("nan"),
            "period_s": float("nan"),
            "spikes_per_burst": float("nan"),
            "mean_isi_s": float("nan"),
            "isi_cv": float("nan"),
            "detail": "non-finite or empty window",
        }
    V_min = float(np.min(V))
    V_max = float(np.max(V))
    duration_s = float(t[-1] - t[0]) / 1000.0
    dVdt = np.diff(V) / np.diff(t)
    mean_abs_dVdt = float(np.mean(np.abs(dVdt)))
    tspike = spike_times(t, V)
    groups = group_bursts(tspike)
    sizes = [len(g) for g in groups]
    isis = np.diff(tspike)
    mean_isi_s = float(np.mean(isis) / 1000.0) if len(isis) else float("nan")
    isi_cv = (
        float(np.std(isis) / np.mean(isis))
        if len(isis) >= 2 and np.mean(isis) > 0
        else float("nan")
    )
    peak = mean_spike_peak(t, V, tspike)
    rate = (len(tspike) / duration_s) if duration_s > 0 else float("nan")

    base = {
        "V_min": V_min,
        "V_max": V_max,
        "amplitude": V_max - V_min,
        "n_spikes": int(len(tspike)),
        "firing_rate_hz": rate,
        "mean_spike_peak": peak,
        "mean_isi_s": mean_isi_s,
        "isi_cv": isi_cv,
    }
    if (V_max - V_min) < EQ_V_RANGE and mean_abs_dVdt < EQ_MEAN_DVDT:
        return {
            **base,
            "regime": "equilibrium",
            "period_s": float("nan"),
            "spikes_per_burst": float("nan"),
            "detail": f"V range={V_max - V_min:.3g}, mean|dV/dt|={mean_abs_dVdt:.3g}",
        }
    if len(tspike) >= 2:
        med_size = float(np.median(sizes))
        if len(groups) >= 2 and med_size >= 2:
            onsets = [g[0] for g in groups]
            periods = np.diff(onsets)
            period_s = float(np.median(periods) / 1000.0) if len(periods) else float("nan")
            return {
                **base,
                "regime": "bursting",
                "period_s": period_s,
                "spikes_per_burst": med_size,
                "detail": f"n_spikes={len(tspike)}, n_groups={len(groups)}, median_size={med_size}",
            }
        return {
            **base,
            "regime": "regular_beating",
            "period_s": mean_isi_s,
            "spikes_per_burst": 1.0,
            "detail": f"n_spikes={len(tspike)}, n_groups={len(groups)}",
        }
    if (V_max - V_min) >= EQ_V_RANGE:
        period_ms = estimate_slow_period_ms(t, V)
        return {
            **base,
            "regime": "slow_wave",
            "period_s": period_ms / 1000.0 if np.isfinite(period_ms) else float("nan"),
            "spikes_per_burst": float("nan"),
            "detail": f"V range={V_max - V_min:.3g}, n_spikes={len(tspike)}",
        }
    return {
        **base,
        "regime": "other",
        "period_s": float("nan"),
        "spikes_per_burst": float("nan"),
        "detail": f"unclassified V range={V_max - V_min:.3g}",
    }


def gates_sensible(data) -> Tuple[bool, str]:
    for k, arr in data.items():
        if k == "t":
            continue
        if not np.all(np.isfinite(arr)):
            return False, f"non-finite {k}"
        if k == "V":
            continue
        if float(np.min(arr)) < -0.05 or float(np.max(arr)) > 1.05:
            return False, f"gate {k} out of range"
    return True, "ok"


def metrics_on_window(data, t0: float, t1: float) -> Dict[str, object]:
    ok, msg = gates_sensible(data)
    if not ok:
        out = classify_window(np.array([]), np.array([]))
        out["detail"] = msg
        return out
    t, V = _window(data, t0, t1)
    return classify_window(t, V)


def needs_extension(
    data,
    t_end: float,
    t_cut: float,
    metrics_ms: float = 20000.0,
) -> bool:
    """Compare the last post-transient window with the preceding one.

    Both windows start at or after t_cut. The spike-rate test is used only
    when both windows are at least 15 s, so a short silent phase of a burst
    does not force every bursting run to be extended.
    """
    late0 = max(float(t_cut), float(t_end) - metrics_ms)
    prev_end = late0
    prev_start = max(float(t_cut), prev_end - metrics_ms)
    if prev_end - prev_start < 5000.0 or t_end - late0 < 5000.0:
        return False
    late = metrics_on_window(data, late0, t_end)
    prev = metrics_on_window(data, prev_start, prev_end)
    if late["regime"] == "FAIL" or prev["regime"] == "FAIL":
        return True
    if abs(late["V_min"] - prev["V_min"]) > 2.0 or abs(late["V_max"] - prev["V_max"]) > 2.0:
        return True
    if (t_end - late0) >= 15000.0 and (prev_end - prev_start) >= 15000.0:
        r0, r1 = prev["firing_rate_hz"], late["firing_rate_hz"]
        if np.isfinite(r0) and np.isfinite(r1) and max(r0, r1) > 0:
            if abs(r1 - r0) / max(r0, r1) > 0.25 and max(prev["n_spikes"], late["n_spikes"]) >= 2:
                return True
    return False


def localise_fold(alpha_lo: float, alpha_hi: float, vary: str, v_lo: float, v_hi: float) -> Dict[str, float]:
    """Bisect in alpha so the extremum of F in a voltage window is zero.

    Used only when adjacent grid samples differ by a pair of roots. This is a
    root-scan refinement, not continuation, and is not labelled a saddle-node.
    """

    def extremum(alpha: float) -> float:
        if vary == "alpha_I":
            pars = pars_from_alpha(alpha, 1.0)
        else:
            pars = pars_from_alpha(1.0, alpha)
        V = np.linspace(v_lo, v_hi, 4001)
        F = np.asarray(current_balance(V, pars), dtype=float)
        d = np.diff(F)
        crit = np.where((d[:-1] <= 0) & (d[1:] >= 0) | (d[:-1] >= 0) & (d[1:] <= 0))[0] + 1
        if len(crit) == 0:
            return float(F[int(np.argmin(np.abs(F)))])
        k = int(crit[int(np.argmin(np.abs(F[crit])))])
        return float(F[k])

    f_lo, f_hi = extremum(alpha_lo), extremum(alpha_hi)
    if f_lo * f_hi > 0.0:
        return {
            "alpha": float("nan"),
            "g_I": float("nan"),
            "I_ep": float("nan"),
            "V_star": float("nan"),
            "F": float("nan"),
            "note": "extremum did not change sign; no tangent bracket",
        }
    alpha = float(brentq(extremum, alpha_lo, alpha_hi, xtol=1e-5))
    if vary == "alpha_I":
        pars = pars_from_alpha(alpha, 1.0)
    else:
        pars = pars_from_alpha(1.0, alpha)
    V = np.linspace(v_lo, v_hi, 4001)
    F = np.asarray(current_balance(V, pars), dtype=float)
    d = np.diff(F)
    crit = np.where(((d[:-1] <= 0) & (d[1:] >= 0)) | ((d[:-1] >= 0) & (d[1:] <= 0)))[0] + 1
    k = int(crit[int(np.argmin(np.abs(F[crit])))]) if len(crit) else int(np.argmin(np.abs(F)))
    return {
        "alpha": alpha,
        "alpha_I": alpha if vary == "alpha_I" else 1.0,
        "alpha_P": 1.0 if vary == "alpha_I" else alpha,
        "g_I": float(pars["g_I"]),
        "I_ep": float(pars["I_ep"]),
        "V_star": float(V[k]),
        "F": float(F[k]),
        "note": "tangent_candidate between adjacent root-count change",
    }


def sweep_point(alpha_I, alpha_P, t_end, t_cut, name, extend_to=120000.0):
    print(
        f"Computing 7D {name}: alpha_I={alpha_I:.4f}, alpha_P={alpha_P:.4f}, T_END={t_end}",
        flush=True,
    )
    ode, data, pars = simulate_7d(alpha_I, alpha_P, t_end, name)
    used = float(t_end)
    extended = False
    if needs_extension(data, used, t_cut):
        print(f"  unsettled; extending to {extend_to} ms", flush=True)
        ode, data, pars = simulate_7d(alpha_I, alpha_P, extend_to, name + "_ext")
        used = float(extend_to)
        extended = True
    t1 = used
    t0 = max(float(t_cut), used - 20000.0)
    metrics = metrics_on_window(data, t0, t1)
    metrics.update(
        {
            "alpha_I": float(alpha_I),
            "alpha_P": float(alpha_P),
            "g_I": float(pars["g_I"]),
            "I_ep": float(pars["I_ep"]),
            "t_end_used": used,
            "t_metrics_start": t0,
            "extended": extended,
        }
    )
    print(
        f"  regime={metrics['regime']} V=[{metrics['V_min']:.3g}, {metrics['V_max']:.3g}] "
        f"n_spikes={metrics['n_spikes']}",
        flush=True,
    )
    return metrics, data


def simulate_7d(alpha_I: float, alpha_P: float, t_end: float, name: str):
    pars = pars_from_alpha(alpha_I, alpha_P)
    ode = build_vode_system(
        pars=pars,
        ics=steady_state_ics(DEFAULT_V0),
        name=name,
    )
    pts = simulate(ode, tdata=[0.0, float(t_end)], dt=0.5, trajname=name)
    data = {k: np.asarray(pts[k], dtype=float) for k in ["t", *STATE_VARS]}
    return ode, data, pars


def run_slice_sweep(alphas, vary, t_end, t_cut, extend_to=120000.0):
    """VODE sweep on one slice, plus at most two midpoint refinements at regime changes.

    Traces are kept only for endpoints, alpha=0.5, and values straddling a
    regime change. Full trajectories are not returned for every grid point.
    """
    alphas = [float(a) for a in alphas]
    records = []
    traces = {}

    def one(alpha):
        if vary == "alpha_I":
            a_I, a_P = float(alpha), 1.0
        elif vary == "alpha_P":
            a_I, a_P = 1.0, float(alpha)
        else:
            raise ValueError(vary)
        one.i += 1
        name = f"{vary}_{one.i:03d}"
        return sweep_point(a_I, a_P, t_end, t_cut, name, extend_to=extend_to)

    one.i = 0

    def keep_trace(alpha, data, force=False):
        if force or min(abs(alpha - x) for x in (0.0, 0.5, 1.0)) < 1e-8:
            traces[round(float(alpha), 6)] = {
                "t": data["t"],
                "V": data["V"],
            }

    for alpha in alphas:
        metrics, data = one(alpha)
        records.append(metrics)
        keep_trace(alpha, data)

    for _round in range(2):
        records.sort(key=lambda r: float(r[vary]))
        inserts = []
        for left, right in zip(records, records[1:]):
            if left["regime"] == right["regime"]:
                continue
            gap = abs(float(right[vary]) - float(left[vary]))
            if gap > 0.0125 + 1e-9:
                inserts.append(0.5 * (float(left[vary]) + float(right[vary])))
        if not inserts:
            break
        print(f"Refinement midpoints on {vary}: {inserts}", flush=True)
        for alpha in inserts:
            metrics, data = one(alpha)
            records.append(metrics)
            keep_trace(alpha, data, force=True)

    records.sort(key=lambda r: float(r[vary]))
    for left, right in zip(records, records[1:]):
        if left["regime"] != right["regime"]:
            for rec in (left, right):
                # Re-simulate only if the trace was not kept.
                a = float(rec[vary])
                if round(a, 6) not in traces:
                    _, data = one(a)
                    keep_trace(a, data, force=True)
    return records, traces

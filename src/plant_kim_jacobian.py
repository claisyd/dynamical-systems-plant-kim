"""Symbolic Jacobian and local bifurcation diagnostics for Plant--Kim 7D.

The symbolic RHS is transcribed from src/plant_kim_model.py (Canavier 1990
rate corrections) and validated against plant_kim_analysis.rhs / jacobian.
This module does not alter the canonical numerical model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import linear_sum_assignment

from src.plant_kim_analysis import (
    G_I0,
    H_G,
    H_V,
    I_EP0,
    jacobian as jacobian_fd,
    pars_from_alpha,
    rhs,
    state_vector,
    vector_to_state,
)
from src.plant_kim_model import BASELINE_PARS, STATE_VARS

# ---------------------------------------------------------------------------
# Symbolic construction
# ---------------------------------------------------------------------------

_PAR_NAMES = (
    "C",
    "g_I",
    "g_K",
    "g_L",
    "g_T",
    "g_A",
    "g_P",
    "V_I",
    "V_K",
    "V_L",
    "tau_X_A",
    "tau_Y_A",
    "tau_X_P",
    "I_ep",
    "I_ext",
)


def _sym_limit_safe_rate(num_coeff: float, linear_const: float, linear_V: float, V):
    """Build  num_coeff * x / (exp(x/10)-1) with removable singularity at x=0.

    Matches plant_kim_model.alpha_m / alpha_n limit handling via sympy Piecewise.
    """
    x = linear_const + linear_V * V
    regular = num_coeff * x / (sp.exp(x / 10) - 1)
    limit_val = num_coeff * 10  # lim x/(exp(x/10)-1) = 10
    return sp.Piecewise((limit_val, sp.Abs(x) < 1e-12), (regular, True))


def build_symbolic_system():
    """Return symbols, symbolic state vector, symbolic RHS, and Jacobian.

    State order matches STATE_VARS:
        (V, X_K, X_A, Y_A, X_P, X_I, Y_I)
    """
    V, X_K, X_A, Y_A, X_P, X_I, Y_I = sp.symbols(
        "V X_K X_A Y_A X_P X_I Y_I", real=True
    )
    x_sym = sp.Matrix([V, X_K, X_A, Y_A, X_P, X_I, Y_I])

    pars = {name: sp.symbols(name, real=True) for name in _PAR_NAMES}

    # Rates (Canavier-corrected), matching plant_kim_model
    alpha_m = _sym_limit_safe_rate(sp.Rational(1, 10), -26, -sp.Rational(12, 10), V)
    beta_m = 4 * sp.exp((-51 - sp.Rational(12, 10) * V) / 18)
    alpha_h = sp.Rational(7, 100) * sp.exp((-51 - sp.Rational(121, 100) * V) / 20)
    beta_h = 1 / (sp.exp((-21 - sp.Rational(13, 10) * V) / 10) + 1)
    alpha_n = _sym_limit_safe_rate(sp.Rational(1, 100), -21, -sp.Rational(121, 100), V)
    beta_n = sp.Rational(125, 1000) * sp.exp((-31 - V) / 80)

    S_I = alpha_m / (alpha_m + beta_m)
    Z_I = alpha_h / (alpha_h + beta_h)
    S_K = alpha_n / (alpha_n + beta_n)
    tau_X_I = sp.Rational(25, 2) / (alpha_m + beta_m)
    tau_Y_I = sp.Rational(25, 2) / (alpha_h + beta_h)
    tau_X_K = sp.Rational(25, 2) / (alpha_n + beta_n)
    S_A = 1 / (1 + sp.exp(-sp.Rational(8, 100) * (V + 45)))
    Z_A = 1 / (1 + sp.exp(sp.Rational(27, 100) * (V + 50)))
    S_P = 1 / (1 + sp.exp(-sp.Rational(7, 10) * (V + 47)))

    C = pars["C"]
    vdot = (
        (pars["g_I"] * X_I ** 3 * Y_I + pars["g_T"]) * (pars["V_I"] - V)
        + (pars["g_K"] * X_K ** 4 + pars["g_A"] * X_A * Y_A + pars["g_P"] * X_P)
        * (pars["V_K"] - V)
        + pars["g_L"] * (pars["V_L"] - V)
        + pars["I_ep"]
        + pars["I_ext"]
    ) / C

    F_sym = sp.Matrix(
        [
            vdot,
            (S_K - X_K) / tau_X_K,
            (S_A - X_A) / pars["tau_X_A"],
            (Z_A - Y_A) / pars["tau_Y_A"],
            (S_P - X_P) / pars["tau_X_P"],
            (S_I - X_I) / tau_X_I,
            (Z_I - Y_I) / tau_Y_I,
        ]
    )
    J_sym = F_sym.jacobian(x_sym)
    return {
        "x_sym": x_sym,
        "F_sym": F_sym,
        "J_sym": J_sym,
        "pars_sym": pars,
        "state_symbols": (V, X_K, X_A, Y_A, X_P, X_I, Y_I),
    }


_SYM = None
_F_NUM = None
_J_NUM = None
_F_MU_NUM = None  # dF/dI_ep (pump appears linearly) and dF/dg_I


def _ensure_lambdified():
    global _SYM, _F_NUM, _J_NUM, _F_MU_NUM
    if _F_NUM is not None:
        return
    _SYM = build_symbolic_system()
    args = list(_SYM["state_symbols"]) + [_SYM["pars_sym"][n] for n in _PAR_NAMES]
    _F_NUM = sp.lambdify(args, _SYM["F_sym"], modules="numpy")
    _J_NUM = sp.lambdify(args, _SYM["J_sym"], modules="numpy")

    # Parametric derivatives for SN / Hopf semi-analytics (g_I and I_ep only).
    F = _SYM["F_sym"]
    dF_dgI = F.diff(_SYM["pars_sym"]["g_I"])
    dF_dIep = F.diff(_SYM["pars_sym"]["I_ep"])
    _F_MU_NUM = {
        "g_I": sp.lambdify(args, dF_dgI, modules="numpy"),
        "I_ep": sp.lambdify(args, dF_dIep, modules="numpy"),
    }


def _arg_tuple(state, pars: Mapping[str, float]):
    x = state_vector(state) if isinstance(state, Mapping) else np.asarray(state, dtype=float)
    return tuple(float(v) for v in x) + tuple(float(pars[n]) for n in _PAR_NAMES)


def rhs_symbolic(state, pars: Mapping[str, float]) -> np.ndarray:
    """Evaluate the lambdified symbolic RHS."""
    _ensure_lambdified()
    out = np.asarray(_F_NUM(*_arg_tuple(state, pars)), dtype=float).reshape(7)
    return out


def jacobian_symbolic(state, pars: Mapping[str, float]) -> np.ndarray:
    """Evaluate the lambdified symbolic Jacobian."""
    _ensure_lambdified()
    out = np.asarray(_J_NUM(*_arg_tuple(state, pars)), dtype=float).reshape(7, 7)
    return out


def F_mu_symbolic(state, pars: Mapping[str, float], mu: str) -> np.ndarray:
    """Partial of F w.r.t. physical parameter mu in {'g_I','I_ep'}."""
    _ensure_lambdified()
    if mu not in _F_MU_NUM:
        raise ValueError(mu)
    out = np.asarray(_F_MU_NUM[mu](*_arg_tuple(state, pars)), dtype=float).reshape(7)
    return out


def hessian_action_symbolic(
    state, pars: Mapping[str, float], q: np.ndarray
) -> np.ndarray:
    """Compute D_x^2 F(x)[q,q] via symbolic second derivatives (lambdified once)."""
    _ensure_lambdified()
    # Build once and cache
    if not hasattr(hessian_action_symbolic, "_H"):
        x_sym = _SYM["x_sym"]
        F = _SYM["F_sym"]
        # For each component i: q^T H_i q where H_i = Hessian of F_i
        q_sym = sp.Matrix(sp.symbols(f"q0:{7}"))
        actions = []
        for i in range(7):
            Hi = sp.hessian(F[i], x_sym)
            actions.append((q_sym.T * Hi * q_sym)[0])
        args = list(_SYM["state_symbols"]) + [_SYM["pars_sym"][n] for n in _PAR_NAMES] + list(q_sym)
        hessian_action_symbolic._H = sp.lambdify(args, sp.Matrix(actions), modules="numpy")
    x = state_vector(state) if isinstance(state, Mapping) else np.asarray(state, dtype=float)
    args = tuple(float(v) for v in x) + tuple(float(pars[n]) for n in _PAR_NAMES) + tuple(
        float(v) for v in np.asarray(q, dtype=float).reshape(7)
    )
    return np.asarray(hessian_action_symbolic._H(*args), dtype=float).reshape(7)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class JacobianValidation:
    max_abs_F_diff: float
    max_abs_J_diff: float
    rel_frobenius_J: float
    max_eig_mismatch: float
    residual_max: float


def match_spectra(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Hungarian-match eigenvalues a -> b. Returns (perm of a, matched b, max |diff|)."""
    a = np.asarray(a, dtype=complex).ravel()
    b = np.asarray(b, dtype=complex).ravel()
    n = len(a)
    cost = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            cost[i, j] = abs(a[i] - b[j])
    ri, ci = linear_sum_assignment(cost)
    a_m = a[ri]
    b_m = b[ci]
    return a_m, b_m, float(np.max(np.abs(a_m - b_m)))


def validate_jacobian_at(
    state,
    pars: Mapping[str, float],
    pydstool_evals: Optional[np.ndarray] = None,
    h_V: float = H_V,
    h_g: float = H_G,
) -> JacobianValidation:
    """Cross-check symbolic F/J against NumPy rhs / FD jacobian (and optional PyDSTool)."""
    f_s = rhs_symbolic(state, pars)
    f_n = rhs(state, pars)
    J_s = jacobian_symbolic(state, pars)
    J_f = jacobian_fd(state, pars, h_V=h_V, h_g=h_g)
    dJ = J_s - J_f
    fro_f = np.linalg.norm(J_f, ord="fro")
    rel = float(np.linalg.norm(dJ, ord="fro") / max(fro_f, 1e-30))
    eigs_s = np.linalg.eigvals(J_s)
    eigs_f = np.linalg.eigvals(J_f)
    _, _, eig_mis = match_spectra(eigs_s, eigs_f)
    if pydstool_evals is not None:
        _, _, eig_mis_p = match_spectra(eigs_s, np.asarray(pydstool_evals, dtype=complex))
        eig_mis = max(eig_mis, eig_mis_p)
    return JacobianValidation(
        max_abs_F_diff=float(np.max(np.abs(f_s - f_n))),
        max_abs_J_diff=float(np.max(np.abs(dJ))),
        rel_frobenius_J=rel,
        max_eig_mismatch=eig_mis,
        residual_max=float(np.max(np.abs(f_n))),
    )


# ---------------------------------------------------------------------------
# Eigenvalue tracking
# ---------------------------------------------------------------------------


def track_eigenvalues(eig_sequence: Sequence[np.ndarray]) -> np.ndarray:
    """Match consecutive spectra; return array shape (n_points, 7) complex."""
    if not eig_sequence:
        return np.zeros((0, 7), dtype=complex)
    tracked = [np.asarray(eig_sequence[0], dtype=complex).ravel().copy()]
    for eigs in eig_sequence[1:]:
        prev = tracked[-1]
        cur = np.asarray(eigs, dtype=complex).ravel()
        n = len(prev)
        cost = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                cost[i, j] = abs(prev[i] - cur[j])
        ri, ci = linear_sum_assignment(cost)
        ordered = np.empty(n, dtype=complex)
        ordered[ri] = cur[ci]
        tracked.append(ordered)
    return np.vstack(tracked)


def identify_critical_real(eigs: np.ndarray) -> int:
    """Index of eigenvalue closest to 0 among (near-)real modes; fallback closest overall."""
    eigs = np.asarray(eigs, dtype=complex).ravel()
    realish = np.where(np.abs(np.imag(eigs)) < 1e-8)[0]
    if len(realish):
        return int(realish[np.argmin(np.abs(eigs[realish]))])
    return int(np.argmin(np.abs(eigs)))


def identify_critical_hopf_pair(eigs: np.ndarray) -> Tuple[int, int]:
    """Indices of the complex-conjugate pair with Re closest to 0 and Im > 0 / < 0."""
    eigs = np.asarray(eigs, dtype=complex).ravel()
    # Prefer pairs with |Im| > tol, minimise |Re|
    cands = []
    for i, lam in enumerate(eigs):
        if abs(lam.imag) > 1e-10:
            cands.append((abs(lam.real), -abs(lam.imag), i))
    if not cands:
        # fallback: two closest to imag axis
        order = np.argsort(np.abs(np.real(eigs)))
        return int(order[0]), int(order[1])
    cands.sort()
    i0 = cands[0][2]
    # partner: closest conjugate
    target = np.conj(eigs[i0])
    dists = [abs(eigs[j] - target) if j != i0 else np.inf for j in range(len(eigs))]
    i1 = int(np.argmin(dists))
    if eigs[i0].imag >= 0:
        return i0, i1
    return i1, i0


# ---------------------------------------------------------------------------
# Bifurcation diagnostics
# ---------------------------------------------------------------------------


def hopf_period_from_eigs(eigs: np.ndarray) -> Dict[str, float]:
    i_pos, _ = identify_critical_hopf_pair(eigs)
    lam = np.asarray(eigs, dtype=complex)[i_pos]
    omega = abs(float(lam.imag))
    T = float(2.0 * np.pi / omega) if omega > 0 else float("nan")
    return {
        "lambda_re": float(lam.real),
        "lambda_im": float(lam.imag),
        "omega_H": omega,
        "T_lin_ms": T,
    }


def spectral_gap_excluding(eigs: np.ndarray, exclude: Sequence[int], mode: str = "zero") -> float:
    """Min distance of non-excluded eigenvalues from 0 (mode='zero') or imag axis ('imag')."""
    eigs = np.asarray(eigs, dtype=complex).ravel()
    mask = np.ones(len(eigs), dtype=bool)
    for i in exclude:
        if 0 <= i < len(eigs):
            mask[i] = False
    rem = eigs[mask]
    if rem.size == 0:
        return float("nan")
    if mode == "zero":
        return float(np.min(np.abs(rem)))
    if mode == "imag":
        return float(np.min(np.abs(np.real(rem))))
    raise ValueError(mode)


def left_right_eigenvectors(J: np.ndarray, lam: complex) -> Tuple[np.ndarray, np.ndarray]:
    """Return (v, w) with J v = lam v, w^H J = lam w^H, and w^H v = 1.

    For complex ``denom = w^H v``, scale by ``w <- w / conj(denom)`` so that
    the new ``w^H v`` equals 1 (plain division by ``denom`` fails for complex).
    """
    J = np.asarray(J, dtype=complex)
    evals, rvecs = np.linalg.eig(J)
    k = int(np.argmin(np.abs(evals - lam)))
    v = rvecs[:, k].copy()
    evals_L, lvecs = np.linalg.eig(J.conj().T)
    kL = int(np.argmin(np.abs(evals_L - np.conj(lam))))
    w = lvecs[:, kL].copy()
    denom = np.vdot(w, v)  # w^H v
    if abs(denom) < 1e-30:
        raise RuntimeError("left/right eigenvector normalization failed")
    w = w / np.conj(denom)
    return v, w


def nullvectors_zero(J: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Right/left nullvectors for simple zero eigenvalue: Jq=0, p^T J=0, p^T q=1."""
    J = np.asarray(J, dtype=float)
    # SVD
    u, s, vh = np.linalg.svd(J)
    q = vh[-1, :].real.copy()
    p = u[:, -1].real.copy()
    # signs
    if abs(p @ q) < 1e-30:
        raise RuntimeError("nullvector normalization failed")
    p = p / (p @ q)
    return q, p


def sn_coefficients(
    state,
    pars: Mapping[str, float],
    mu_physical: str,
    dmu_dalpha: float,
) -> Dict[str, float]:
    """Compute a = p^T F_mu_alpha and b = 1/2 p^T D^2F[q,q].

    mu_physical is 'g_I' or 'I_ep'. dmu_dalpha converts to continuation parameter
    (e.g. d g_I / d alpha_I = G_I0).
    """
    J = jacobian_symbolic(state, pars)
    q, p = nullvectors_zero(J)
    F_mu = F_mu_symbolic(state, pars, mu_physical) * float(dmu_dalpha)
    a = float(p @ F_mu)
    Hqq = hessian_action_symbolic(state, pars, q)
    b = 0.5 * float(p @ Hqq)
    return {
        "a": a,
        "b": b,
        "sigma_min": float(np.min(np.linalg.svd(J, compute_uv=False))),
        "q_norm": float(np.linalg.norm(q)),
        "p_dot_q": float(p @ q),
    }


def dJ_dmu_total(
    state,
    pars: Mapping[str, float],
    mu_physical: str,
    dmu_dalpha: float,
    h_state: float = 1e-6,
    h_mu: float = 1e-7,
) -> np.ndarray:
    """Total dJ/d alpha including equilibrium dependence via dx*/dmu = -J^{-1} F_mu."""
    J = jacobian_symbolic(state, pars)
    F_mu = F_mu_symbolic(state, pars, mu_physical) * float(dmu_dalpha)
    # At a Hopf, J is invertible
    dx_dmu = -np.linalg.solve(J, F_mu)
    # Directional derivative of J along dx: finite difference of J(x + h dx)
    x = state_vector(state) if isinstance(state, Mapping) else np.asarray(state, dtype=float)
    # Also partial J / partial mu at fixed x via parameter FD
    pars_p = dict(pars)
    pars_m = dict(pars)
    delta = h_mu * float(dmu_dalpha)
    # physical parameter shift of size h_mu * |scale|, but we want d/d alpha
    # Use physical step then scale: dJ/d alpha = (dJ/d mu_phys) * dmu_dalpha
    step = h_mu
    pars_p[mu_physical] = float(pars[mu_physical]) + step
    pars_m[mu_physical] = float(pars[mu_physical]) - step
    J_mu = (jacobian_symbolic(x, pars_p) - jacobian_symbolic(x, pars_m)) / (2.0 * step)
    J_mu *= float(dmu_dalpha)

    # State directional derivative
    dx = dx_dmu
    nrm = np.linalg.norm(dx)
    if nrm < 1e-18:
        J_x = np.zeros_like(J)
    else:
        h = h_state
        xp = x + h * dx
        xm = x - h * dx
        J_x = (jacobian_symbolic(xp, pars) - jacobian_symbolic(xm, pars)) / (2.0 * h)
    return J_mu + J_x


def hopf_eigenvalue_derivative(
    state,
    pars: Mapping[str, float],
    mu_physical: str,
    dmu_dalpha: float,
) -> complex:
    """Semi-analytical d lambda / d alpha at a Hopf: w^H (dJ/dalpha) v."""
    J = jacobian_symbolic(state, pars)
    eigs = np.linalg.eigvals(J)
    i_pos, _ = identify_critical_hopf_pair(eigs)
    lam = eigs[i_pos]
    v, w = left_right_eigenvectors(J, lam)
    dJ = dJ_dmu_total(state, pars, mu_physical, dmu_dalpha)
    return complex(np.vdot(w, dJ @ v))  # w already normalized so w^H v = 1


def _align_spectrum_to_ref(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=complex).ravel()
    ref = np.asarray(ref, dtype=complex).ravel()
    n = len(ref)
    cost = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            cost[i, j] = abs(src[i] - ref[j])
    ri, ci = linear_sum_assignment(cost)
    out = np.empty(n, dtype=complex)
    out[ci] = src[ri]
    return out


def hopf_transversality_fd(
    state,
    alpha_I: float,
    alpha_P: float,
    h_values: Sequence[float],
    freepar: str = "alpha_P",
) -> List[Dict[str, float]]:
    """Centred FD of d Re(lambda_crit)/d freepar at several step sizes."""
    pars0 = pars_from_alpha(alpha_I, alpha_P)
    e0 = np.linalg.eigvals(jacobian_symbolic(state, pars0))
    i0, _ = identify_critical_hopf_pair(e0)
    rows = []
    for h in h_values:
        h = float(h)
        if freepar == "alpha_P":
            pars_p = pars_from_alpha(alpha_I, alpha_P + h)
            pars_m = pars_from_alpha(alpha_I, alpha_P - h)
        elif freepar == "alpha_I":
            pars_p = pars_from_alpha(alpha_I + h, alpha_P)
            pars_m = pars_from_alpha(alpha_I - h, alpha_P)
        else:
            raise ValueError(freepar)
        xp = _refine_equilibrium(state, pars_p)
        xm = _refine_equilibrium(state, pars_m)
        ep_a = _align_spectrum_to_ref(np.linalg.eigvals(jacobian_symbolic(xp, pars_p)), e0)
        em_a = _align_spectrum_to_ref(np.linalg.eigvals(jacobian_symbolic(xm, pars_m)), e0)
        d_re = (ep_a[i0].real - em_a[i0].real) / (2.0 * h)
        rows.append(
            {
                "h": h,
                "dRe_dmu": float(d_re),
                "Re_plus": float(ep_a[i0].real),
                "Re_minus": float(em_a[i0].real),
                "Im_plus": float(ep_a[i0].imag),
                "Im_minus": float(em_a[i0].imag),
            }
        )
    return rows


def _refine_equilibrium(state, pars: Mapping[str, float], niter: int = 12) -> np.ndarray:
    """Newton refine full 7D equilibrium starting from state."""
    x = state_vector(state) if isinstance(state, Mapping) else np.asarray(state, dtype=float).copy()
    for _ in range(niter):
        f = rhs_symbolic(x, pars)
        if np.max(np.abs(f)) < 1e-12:
            break
        J = jacobian_symbolic(x, pars)
        try:
            dx = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            dx = -np.linalg.pinv(J) @ f
        x = x + dx
    return x


# ---------------------------------------------------------------------------
# CSV / landmark helpers
# ---------------------------------------------------------------------------


def load_special_csv(path) -> List[Dict[str, str]]:
    import csv
    from pathlib import Path

    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def row_by_label(rows: Sequence[Mapping], label: str) -> Mapping:
    for r in rows:
        if r.get("label") == label or r.get("report_label") == label:
            return r
    raise KeyError(label)


def state_from_row(row: Mapping) -> Dict[str, float]:
    return {k: float(row[k]) for k in STATE_VARS}


def landmark_H_bio(repo_root=None):
    from pathlib import Path

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    rows = load_special_csv(
        root / "outputs/data/ouabain_bifurcation/ttx0_alphaP_epc_special_points.csv"
    )
    row = row_by_label(rows, "H_bio")
    state = state_from_row(row)
    alpha_P = float(row["alpha_P"])
    return {
        "label": "H_bio",
        "alpha_I": 0.0,
        "alpha_P": alpha_P,
        "state": state,
        "row": row,
        "pars": pars_from_alpha(0.0, alpha_P),
    }


def landmark_LP_bio(repo_root=None):
    from pathlib import Path

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    rows = load_special_csv(
        root / "outputs/data/ttx_bifurcation/alphaI_epc_special_points.csv"
    )
    row = row_by_label(rows, "LP_bio")
    state = state_from_row(row)
    alpha_I = float(row["alpha_I"])
    return {
        "label": "LP_bio",
        "alpha_I": alpha_I,
        "alpha_P": 1.0,
        "state": state,
        "row": row,
        "pars": pars_from_alpha(alpha_I, 1.0),
    }


def format_spectrum_table(eigs: np.ndarray, critical_idx: Sequence[int]) -> List[Dict]:
    rows = []
    eigs = np.asarray(eigs, dtype=complex).ravel()
    for i, lam in enumerate(eigs):
        rows.append(
            {
                "index": i,
                "real": float(lam.real),
                "imag": float(lam.imag),
                "abs": float(abs(lam)),
                "critical": i in set(critical_idx),
            }
        )
    rows.sort(key=lambda r: (not r["critical"], abs(r["real"]), -abs(r["imag"])))
    return rows

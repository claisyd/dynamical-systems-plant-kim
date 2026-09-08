"""Untreated 7D Plant--Kim (1976) R15 bursting model for PyDSTool VODE.

Canonical local source: Notes/model.md (cross-checked against Plant--Kim 1976
Tables I and II, with Canavier et al. 1990 typographical corrections to
alpha_m, beta_m, and beta_h).

Implementation choices (not tabulated in Plant--Kim):
- I_ext = 0 for untreated spontaneous activity
- default ICs: V0 = -50 mV with gates at steady state at V0
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import PyDSTool as dst

STATE_VARS: Tuple[str, ...] = ("V", "X_K", "X_A", "Y_A", "X_P", "X_I", "Y_I")

# Baseline untreated parameters from Plant--Kim 1976 Tables I and II.
# Conductances in mho x 10^{-3}; potentials in mV; C in uF; currents in uA;
# time constants in ms.
BASELINE_PARS: Dict[str, float] = {
    "C": 1.0,
    "g_I": 4.0,
    "g_K": 0.30,
    "g_L": 0.003,
    "g_T": 0.008,
    "g_A": 0.06,
    "g_P": 0.015,
    "V_I": 30.0,
    "V_K": -75.0,
    "V_L": -40.0,
    "tau_X_A": 10.0,
    "tau_Y_A": 235.0,
    "tau_X_P": 8000.0,
    "I_ep": -0.22,
    "I_ext": 0.0,  # implementation choice: untreated spontaneous bursting
}

DEFAULT_V0: float = -50.0  # implementation choice for initial voltage


# ---------------------------------------------------------------------------
# NumPy gating / rate helpers (Stage A diagnostics; must mirror fnspecs)
# ---------------------------------------------------------------------------

def alpha_m(V):
    """Plant--Kim Table I alpha_m(V), with verified denominator correction.

    The printed Table I denominator is easily misread as exp((26-V)/10)-1, which
    makes S_I leave [0,1]. The HH-consistent form matching the numerator (and the
    alpha_n pattern in the same table) is

        0.1*(-26-1.2*V) / (exp((-26-1.2*V)/10) - 1)

    as also used by Canavier et al. (1990) after correcting Plant--Kim typesetting.
    Removable singularity at V = -26/1.2: limit is 1.0.
    """
    V = np.asarray(V, dtype=float)
    x = -26.0 - 1.2 * V
    # lim_{x->0} x/(exp(x/10)-1) = 10, so 0.1*10 = 1.0
    near = np.isclose(x, 0.0, atol=1e-10)
    out = np.empty(np.shape(V), dtype=float)
    out[near] = 1.0
    xn = x[~near]
    out[~near] = 0.1 * xn / np.expm1(xn / 10.0)
    return out


def beta_m(V):
    """Plant--Kim Table I beta_m(V), with Canavier (1990) sign correction.

    Printed Table I: 4*exp((51-1.2*V)/18). That form keeps S_I near 0 for all
    physiologically relevant V. Canavier et al. (1990) document a missing minus
    on the constant in the exponential for bm, giving:

        4*exp((-51-1.2*V)/18)

    which restores HH-consistent activation around spike threshold.
    """
    V = np.asarray(V, dtype=float)
    return 4.0 * np.exp((-51.0 - 1.2 * V) / 18.0)


def alpha_h(V):
    V = np.asarray(V, dtype=float)
    return 0.07 * np.exp((-51.0 - 1.21 * V) / 20.0)


def beta_h(V):
    """Plant--Kim Table I beta_h(V), with Canavier (1990) sign correction.

    Printed: 1/(exp((21-1.3*V)/10)+1). Canavier inserts a minus before the
    constant in the exponential argument: exp((-21-1.3*V)/10).
    """
    V = np.asarray(V, dtype=float)
    return 1.0 / (np.exp((-21.0 - 1.3 * V) / 10.0) + 1.0)


def alpha_n(V):
    """Plant--Kim Table I alpha_n(V); removable singularity when denom -> 0."""
    V = np.asarray(V, dtype=float)
    x = -21.0 - 1.21 * V
    # lim_{x->0} 0.01*x/(exp(x/10)-1) = 0.01*10 = 0.1
    near = np.isclose(x, 0.0, atol=1e-10)
    out = np.empty(np.shape(V), dtype=float)
    out[near] = 0.1
    xn = x[~near]
    out[~near] = 0.01 * xn / np.expm1(xn / 10.0)
    return out


def beta_n(V):
    V = np.asarray(V, dtype=float)
    return 0.125 * np.exp((-31.0 - V) / 80.0)


def S_I(V):
    a, b = alpha_m(V), beta_m(V)
    return a / (a + b)


def Z_I(V):
    a, b = alpha_h(V), beta_h(V)
    return a / (a + b)


def S_K(V):
    a, b = alpha_n(V), beta_n(V)
    return a / (a + b)


def tau_X_I(V):
    return 12.5 / (alpha_m(V) + beta_m(V))


def tau_Y_I(V):
    return 12.5 / (alpha_h(V) + beta_h(V))


def tau_X_K(V):
    return 12.5 / (alpha_n(V) + beta_n(V))


def S_A(V):
    V = np.asarray(V, dtype=float)
    return 1.0 / (1.0 + np.exp(-0.08 * (V + 45.0)))


def Z_A(V):
    V = np.asarray(V, dtype=float)
    return 1.0 / (1.0 + np.exp(0.27 * (V + 50.0)))


def S_P(V):
    V = np.asarray(V, dtype=float)
    return 1.0 / (1.0 + np.exp(-0.7 * (V + 47.0)))


def steady_state_ics(V0: float = DEFAULT_V0) -> Dict[str, float]:
    """Gates at steady state at V0 (implementation choice if V0 is default)."""
    V0 = float(V0)
    return {
        "V": V0,
        "X_K": float(S_K(V0)),
        "X_A": float(S_A(V0)),
        "Y_A": float(Z_A(V0)),
        "X_P": float(S_P(V0)),
        "X_I": float(S_I(V0)),
        "Y_I": float(Z_I(V0)),
    }


# ---------------------------------------------------------------------------
# PyDSTool specs (Labs-style string fnspecs / varspecs)
# ---------------------------------------------------------------------------

def make_fnspecs() -> Dict[str, tuple]:
    """Auxiliary functions for PyDSTool, matching Notes/model.md / Table I--II."""
    return {
        # Rate corrections: Canavier et al. (1990) typographical fixes to Table I
        "alpha_m": (["V"], "0.1*(-26 - 1.2*V)/(exp((-26 - 1.2*V)/10) - 1)"),
        "beta_m": (["V"], "4*exp((-51 - 1.2*V)/18)"),
        "alpha_h": (["V"], "0.07*exp((-51 - 1.21*V)/20)"),
        "beta_h": (["V"], "1/(exp((-21 - 1.3*V)/10) + 1)"),
        "alpha_n": (["V"], "0.01*(-21 - 1.21*V)/(exp((-21 - 1.21*V)/10) - 1)"),
        "beta_n": (["V"], "0.125*exp((-31 - V)/80)"),
        "S_I": (["V"], "alpha_m(V)/(alpha_m(V) + beta_m(V))"),
        "Z_I": (["V"], "alpha_h(V)/(alpha_h(V) + beta_h(V))"),
        "S_K": (["V"], "alpha_n(V)/(alpha_n(V) + beta_n(V))"),
        "tau_X_I": (["V"], "12.5/(alpha_m(V) + beta_m(V))"),
        "tau_Y_I": (["V"], "12.5/(alpha_h(V) + beta_h(V))"),
        "tau_X_K": (["V"], "12.5/(alpha_n(V) + beta_n(V))"),
        "S_A": (["V"], "1/(1 + exp(-0.08*(V + 45)))"),
        "Z_A": (["V"], "1/(1 + exp(0.27*(V + 50)))"),
        "S_P": (["V"], "1/(1 + exp(-0.7*(V + 47)))"),
    }


def make_varspecs() -> Dict[str, str]:
    """7D Plant--Kim ODEs as PyDSTool varspecs strings."""
    return {
        "V": (
            "(1/C)*((g_I*X_I^3*Y_I + g_T)*(V_I - V)"
            " + (g_K*X_K^4 + g_A*X_A*Y_A + g_P*X_P)*(V_K - V)"
            " + g_L*(V_L - V) + I_ep + I_ext)"
        ),
        "X_K": "(S_K(V) - X_K)/tau_X_K(V)",
        "X_A": "(S_A(V) - X_A)/tau_X_A",
        "Y_A": "(Z_A(V) - Y_A)/tau_Y_A",
        "X_P": "(S_P(V) - X_P)/tau_X_P",
        "X_I": "(S_I(V) - X_I)/tau_X_I(V)",
        "Y_I": "(Z_I(V) - Y_I)/tau_Y_I(V)",
    }


# Practical VODE settings for long (multi-second) runs. Defaults in PyDSTool
# (rtol=1e-9, atol=1e-12) are unnecessarily tight for exploratory baseline traces.
DEFAULT_ALGPARAMS: Dict[str, object] = {
    "rtol": 1e-6,
    "atol": 1e-8,
    "stiff": True,
    "init_step": 0.1,
    "max_step": 1.0,  # Plant--Kim used ~1 ms steps; resolve spike timescale
}


def build_vode_system(
    pars: Optional[Mapping[str, float]] = None,
    ics: Optional[Mapping[str, float]] = None,
    name: str = "PlantKim7D",
    algparams: Optional[Mapping[str, object]] = None,
):
    """Construct a PyDSTool Vode_ODEsystem for the untreated 7D model."""
    use_pars = deepcopy(BASELINE_PARS) if pars is None else dict(pars)
    use_ics = steady_state_ics(DEFAULT_V0) if ics is None else dict(ics)

    DSargs = dst.args(name=name)
    DSargs.pars = use_pars
    DSargs.fnspecs = make_fnspecs()
    DSargs.varspecs = make_varspecs()
    DSargs.ics = use_ics
    ode_sys = dst.Generator.Vode_ODEsystem(DSargs)
    use_alg = dict(DEFAULT_ALGPARAMS) if algparams is None else dict(algparams)
    if use_alg:
        ode_sys.set(algparams=use_alg)
    return ode_sys


def simulate(
    ode_sys,
    tdata: Sequence[float],
    dt: Optional[float] = None,
    trajname: str = "baseline",
):
    """Integrate with VODE using the known-working Labs pattern.

    ode_sys.set(tdata) -> compute -> sample
    """
    ode_sys.set(tdata=list(tdata))
    trajectory = ode_sys.compute(trajname)
    if dt is None:
        return trajectory.sample()
    return trajectory.sample(dt=dt)

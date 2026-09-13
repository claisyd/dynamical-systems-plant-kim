"""Exact 5D Plant--Kim reduction under complete TTX (g_I = 0).

At g_I = 0 the transient inward current g_I * X_I^3 * Y_I * (V_I - V)
vanishes, so X_I and Y_I decouple from the remaining subsystem

    (V, X_K, X_A, Y_A, X_P).

This module is a sanity-check reduction only. Canonical scientific endpoint
simulations remain the validated 7D model in plant_kim_model.py with g_I = 0.

Reuse: baseline parameters, VODE defaults, gating helpers for ICs, and the
shared simulate() helper from the 7D module. Rate/gate formulas are not
retyped; fnspecs are a subset of the 7D strings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Mapping, Optional, Tuple

import PyDSTool as dst

from src.plant_kim_model import (
    BASELINE_PARS,
    DEFAULT_ALGPARAMS,
    DEFAULT_V0,
    make_fnspecs,
    simulate,  # noqa: F401 — re-exported for callers
    steady_state_ics,
)

STATE_VARS_5D: Tuple[str, ...] = ("V", "X_K", "X_A", "Y_A", "X_P")

# Parameters appearing in the reduced ODEs (g_I never enters the 5D equations).
_PARS_5D_KEYS = (
    "C",
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

# Fnspecs still required after dropping X_I / Y_I auxiliaries.
_FNSPECS_5D_KEYS = (
    "alpha_n",
    "beta_n",
    "S_K",
    "tau_X_K",
    "S_A",
    "Z_A",
    "S_P",
)


def default_pars_5d() -> Dict[str, float]:
    """Baseline parameters restricted to those used by the 5D ODEs."""
    full = deepcopy(BASELINE_PARS)
    return {k: float(full[k]) for k in _PARS_5D_KEYS}


def steady_state_ics_5d(V0: float = DEFAULT_V0) -> Dict[str, float]:
    """Steady-state ICs for the five common state variables only."""
    full = steady_state_ics(V0)
    return {k: float(full[k]) for k in STATE_VARS_5D}


def make_fnspecs_5d() -> Dict[str, tuple]:
    """Subset of 7D fnspecs needed when X_I / Y_I are absent."""
    full = make_fnspecs()
    return {k: full[k] for k in _FNSPECS_5D_KEYS}


def make_varspecs_5d() -> Dict[str, str]:
    """5D ODEs: g_I = 0 restriction of the canonical voltage equation."""
    return {
        "V": (
            "(1/C)*((g_T)*(V_I - V)"
            " + (g_K*X_K^4 + g_A*X_A*Y_A + g_P*X_P)*(V_K - V)"
            " + g_L*(V_L - V) + I_ep + I_ext)"
        ),
        "X_K": "(S_K(V) - X_K)/tau_X_K(V)",
        "X_A": "(S_A(V) - X_A)/tau_X_A",
        "Y_A": "(Z_A(V) - Y_A)/tau_Y_A",
        "X_P": "(S_P(V) - X_P)/tau_X_P",
    }


def build_vode_system_5d(
    pars: Optional[Mapping[str, float]] = None,
    ics: Optional[Mapping[str, float]] = None,
    name: str = "PlantKimTTX5D",
    algparams: Optional[Mapping[str, object]] = None,
):
    """Construct a PyDSTool Vode_ODEsystem for the exact TTX 5D reduction.

    Note: pars= and algparams= fully replace defaults (same contract as 7D).
    """
    use_pars = default_pars_5d() if pars is None else dict(pars)
    use_ics = steady_state_ics_5d(DEFAULT_V0) if ics is None else dict(ics)

    DSargs = dst.args(name=name)
    DSargs.pars = use_pars
    DSargs.fnspecs = make_fnspecs_5d()
    DSargs.varspecs = make_varspecs_5d()
    DSargs.ics = use_ics
    ode_sys = dst.Generator.Vode_ODEsystem(DSargs)
    use_alg = dict(DEFAULT_ALGPARAMS) if algparams is None else dict(algparams)
    if use_alg:
        ode_sys.set(algparams=use_alg)
    return ode_sys

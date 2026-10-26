#!/usr/bin/env python3
"""Compact 2x2 report figures for TTX and ouabain membrane-voltage traces.

Reuses project-canonical VODE simulation via ``simulate_7d`` (same ICs,
parameters, and integrator settings as Notebook 04). Does not modify the
Plant--Kim equations or any bifurcation/continuation outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.plant_kim_analysis import (  # noqa: E402
    G_I0,
    I_EP0,
    simulate_7d,
)
from src.plant_kim_model import DEFAULT_V0  # noqa: E402

DATA_DIR = REPO / "outputs" / "data" / "compact_time_traces"
FIG_DIR = REPO / "ReportFigs" / "compact_time_traces"

TTX_ALPHAS = (1.0, 0.25, 0.20, 0.0)
TTX_TITLES = (
    r"$\alpha_I = 1$ (control)",
    r"$\alpha_I = 0.25$",
    r"$\alpha_I = 0.20$",
    r"$\alpha_I = 0$ (complete TTX)",
)
TTX_T_END_MS = 40000.0
TTX_XLIM = (0.0, 40.0)

OUA_ALPHAS = (1.0, 0.5, 0.2, 0.0)
OUA_TITLES = (
    r"$\alpha_P = 1$ (control)",
    r"$\alpha_P = 0.5$",
    r"$\alpha_P = 0.2$",
    r"$\alpha_P = 0$ (complete ouabain)",
)
OUA_T_END_MS = 60000.0
OUA_XLIM = (0.0, 60.0)

FIGSIZE = (10.5, 6.2)
LINE_LW = 0.8
DPI = 300
Y_PAD = 2.0


def _simulate_panel(alpha_I: float, alpha_P: float, t_end: float, name: str):
    ode, data, pars = simulate_7d(alpha_I, alpha_P, t_end, name)
    t_s = np.asarray(data["t"], dtype=float) / 1000.0
    V = np.asarray(data["V"], dtype=float)
    return t_s, V, pars


def _shared_ylim(voltages, pad: float = Y_PAD):
    vmin = min(float(np.min(V)) for V in voltages)
    vmax = max(float(np.max(V)) for V in voltages)
    return (vmin - pad, vmax + pad)


def _plot_2x2(
    traces,
    titles,
    xlim,
    ylim,
    suptitle: str,
    out_path: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, (t_s, V), title in zip(axes_flat, traces, titles):
        ax.plot(t_s, V, color="C0", lw=LINE_LW)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=9)

    for ax in axes[1, :]:
        ax.set_xlabel(r"$t$ (s)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$V$ (mV)", fontsize=10)

    fig.suptitle(suptitle, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def _save_npz(path: Path, alphas, traces, alpha_key: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {alpha_key: np.asarray(alphas, dtype=float)}
    for a, (t_s, V) in zip(alphas, traces):
        tag = f"{a:g}".replace(".", "p")
        payload[f"t_{tag}"] = np.asarray(t_s, dtype=float)
        payload[f"V_{tag}"] = np.asarray(V, dtype=float)
    np.savez_compressed(path, **payload)
    print(f"Wrote {path}", flush=True)


def make_ttx_figure():
    print(
        f"TTX traces: alpha_P=1, alpha_I={list(TTX_ALPHAS)}, "
        f"t_end={TTX_T_END_MS} ms, V0={DEFAULT_V0}, "
        f"g_I0={G_I0}, I_ep0={I_EP0}",
        flush=True,
    )
    traces = []
    for a in TTX_ALPHAS:
        name = f"compact_ttx_aI{a:g}".replace(".", "p")
        t_s, V, pars = _simulate_panel(a, 1.0, TTX_T_END_MS, name)
        assert abs(pars["I_ep"] - I_EP0) < 1e-12
        assert abs(pars["I_ext"]) < 1e-12
        print(
            f"  alpha_I={a:g}: g_I={pars['g_I']:.4g}, "
            f"V=[{V.min():.3g}, {V.max():.3g}]",
            flush=True,
        )
        traces.append((t_s, V))

    _save_npz(DATA_DIR / "ttx_alphaI_traces.npz", TTX_ALPHAS, traces, "alpha_I")
    ylim = _shared_ylim([V for _, V in traces])
    _plot_2x2(
        traces,
        TTX_TITLES,
        TTX_XLIM,
        ylim,
        "Simulated membrane voltage under TTX inhibition",
        FIG_DIR / "ttx_alphaI_time_traces_compact.png",
    )
    return traces, ylim


def make_ouabain_figure():
    print(
        f"Ouabain traces: alpha_I=1, alpha_P={list(OUA_ALPHAS)}, "
        f"t_end={OUA_T_END_MS} ms, V0={DEFAULT_V0}, "
        f"g_I0={G_I0}, I_ep0={I_EP0}",
        flush=True,
    )
    traces = []
    for a in OUA_ALPHAS:
        name = f"compact_oua_aP{a:g}".replace(".", "p")
        t_s, V, pars = _simulate_panel(1.0, a, OUA_T_END_MS, name)
        assert abs(pars["g_I"] - G_I0) < 1e-12
        assert abs(pars["I_ext"]) < 1e-12
        print(
            f"  alpha_P={a:g}: I_ep={pars['I_ep']:.4g}, "
            f"V=[{V.min():.3g}, {V.max():.3g}]",
            flush=True,
        )
        traces.append((t_s, V))

    _save_npz(DATA_DIR / "ouabain_alphaP_traces.npz", OUA_ALPHAS, traces, "alpha_P")
    ylim = _shared_ylim([V for _, V in traces])
    _plot_2x2(
        traces,
        OUA_TITLES,
        OUA_XLIM,
        ylim,
        "Simulated membrane voltage under ouabain inhibition",
        FIG_DIR / "ouabain_alphaP_time_traces_compact.png",
    )
    return traces, ylim


def _quick_checks(ttx_traces, oua_traces):
    """Print qualitative checks against expected regimes."""
    # Late-window spike proxy: peaks above -10 mV after t = 10 s.
    def late_has_spikes(t_s, V, thr=-10.0, t0=10.0):
        mask = t_s >= t0
        return bool(np.any(V[mask] > thr))

    a25_t, a25_V = ttx_traces[1]
    a20_t, a20_V = ttx_traces[2]
    a0_t, a0_V = ttx_traces[3]
    print("Verification (TTX):", flush=True)
    print(
        f"  alpha_I=0.25 late spikes above -10 mV: "
        f"{late_has_spikes(a25_t, a25_V)} (expect True)",
        flush=True,
    )
    print(
        f"  alpha_I=0.20 late spikes above -10 mV: "
        f"{late_has_spikes(a20_t, a20_V)} (expect False)",
        flush=True,
    )
    print(
        f"  alpha_I=0 late V range: [{a0_V[a0_t >= 10].min():.3g}, "
        f"{a0_V[a0_t >= 10].max():.3g}] (expect slow-wave, no fast spikes)",
        flush=True,
    )

    p0_t, p0_V = oua_traces[3]
    print("Verification (ouabain):", flush=True)
    print(
        f"  alpha_P=0 late spikes above -10 mV: "
        f"{late_has_spikes(p0_t, p0_V)} (expect True, regular beating)",
        flush=True,
    )
    for label, (t_s, V) in zip(("0.5", "0.2", "0"), oua_traces[1:]):
        print(
            f"  alpha_P={label}: V=[{V.min():.3g}, {V.max():.3g}], "
            f"late max={V[t_s >= 10].max():.3g}",
            flush=True,
        )


def main():
    ttx_traces, ttx_ylim = make_ttx_figure()
    oua_traces, oua_ylim = make_ouabain_figure()
    print(f"TTX shared ylim: {ttx_ylim}", flush=True)
    print(f"Ouabain shared ylim: {oua_ylim}", flush=True)
    _quick_checks(ttx_traces, oua_traces)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

# MATH3888 — Plant–Kim R15 Neuron Modelling Project

## Project Overview

This repository contains the computational work for our MATH3888 project on the dynamics of the *Aplysia* R15 bursting neuron using the Plant–Kim model.

The main mathematical model is the seven-dimensional Plant–Kim normal bursting model. The project will initially focus on:

1. implementing and validating the baseline 7D model;
2. reproducing untreated bursting;
3. reproducing the effects of TTX and ouabain;
4. performing coarse parameter sweeps;
5. computing equilibria and assessing their stability;
6. performing one-parameter bifurcation analysis;
7. potentially performing two-parameter continuation or fast–slow analysis as an extension.

The current priority is **environment verification, model implementation, and simulation groundwork**. Formal AUTO bifurcation analysis comes later.

---

# Important Computational Rules

These rules should be followed unless explicitly changed by the project members.

## Canonical project environment

The project contains a local virtual environment:

`.venv/`

This is the preferred Python environment for the project. PyDSTool has already been successfully installed and imported in this environment.

### Canonical Python interpreter

- Canonical Python interpreter:

      ./.venv/bin/python

- Absolute path on this machine:

      /Users/calvinlai/Desktop/MATH3888/.venv/bin/python

- Prefer this explicit `.venv` interpreter. Do **not** use bare `python` or `python3` for project work.

- You may still activate the environment with `source .venv/bin/activate` if convenient, but prefer the explicit path above for scripts and agent commands.

### Canonical Jupyter kernel

- Internal kernel name: `pydstool`
- Display name: `Python 3.9 (PyDSTool)`
- This kernel launches the project `.venv/bin/python` (absolute path in its `kernel.json`).
- Use this kernel for project notebooks.
- Do **not** use the generic `python3` kernel for project notebooks unless it is explicitly verified to be the canonical project kernel. The venv-local `python3` kernel currently uses bare `"python"` in `argv[0]` and is PATH-dependent.
- Prefer explicit `.venv` commands rather than bare `python`, `python3`, `jupyter`, or `jupyter lab`.
- Preferred JupyterLab command:

      ./.venv/bin/jupyter-lab

  equivalently:

      ./.venv/bin/python -m jupyter lab

The workspace Cursor/VS Code interpreter is configured via `.vscode/settings.json` to `${workspaceFolder}/.venv/bin/python`.

---

## PyDSTool execution rule

- PyDSTool may segfault with exit code **139** under Cursor's restricted sandbox.
- It has been confirmed to work under unrestricted/local execution.
- If PyDSTool fails only inside the sandbox, do **not** conclude that the environment is broken.
- Prefer unrestricted/local execution using the explicit `.venv/bin/python` interpreter for actual PyDSTool computation.

---

## PyDSTool version metadata note

- `PyDSTool.__version__` may report `1.23.5`.
- Installed distribution metadata reports PyDSTool `0.91.0`.
- This discrepancy is known.
- Do **not** attempt to "fix" it by reinstalling or upgrading PyDSTool.

---

## PyDSTool Solver Rule

For direct ODE simulation using the current local environment:

**USE VODE ONLY.**

Do not use:

- `Radau`;
- `Dopri`;
- `Dopri853`.

These solvers have previously caused compatibility/compilation problems in the current macOS setup.

Unless explicitly instructed otherwise, use:

    PyDSTool.Generator.Vode_ODEsystem

for direct ODE integration.

---

## PyDSTool Documentation / Syntax Rule

PyDSTool is an older library and some documentation is incomplete or difficult to locate.

**Do not invent or guess PyDSTool APIs.**

When uncertain about PyDSTool syntax, use the following sources in approximately this order:

1. existing working examples in this repository;
2. guided workshop material in `Labs/`;
3. `Code/practice.ipynb`;
4. installed PyDSTool source or documentation if necessary;
5. external documentation only when the above are insufficient.

Prefer adapting syntax that is already known to work in this environment.

If the correct PyDSTool API cannot be determined confidently, stop and explain the uncertainty rather than silently inventing syntax.

---

# Repository Structure

## Hidden / Configuration Files

### `.venv/`

Local Python virtual environment containing the working PyDSTool installation.

This directory should generally not be edited manually.

### `.gitignore`

Git ignore configuration.

### `.cursorignore`

Cursor indexing exclusions.

The virtual environment may be ignored by Cursor indexing, but its Python executable may still be used to run the project.

---

# `Code/`

Contains previous computational examples.

## `textbook.ipynb`

Implementation of figures/examples from the dynamical systems textbook.

This notebook **does not use PyDSTool**.

It may be useful for general mathematical or plotting ideas, but should normally be ignored as a reference for PyDSTool syntax.

## `practice.ipynb`

Contains basic bifurcation examples implemented using PyDSTool, including:

- saddle-node bifurcation;
- transcritical bifurcation;
- pitchfork bifurcation.

This is a useful reference for known-working PyDSTool syntax.

---

# `Labs/`

Contains guided computational labs from MATH3888 workshops.

These include:

- material provided by the tutor;
- implementations completed during or after workshops.

This directory is one of the **main references for PyDSTool usage and syntax**.

When implementing continuation or other PyDSTool functionality later, inspect the relevant lab examples before attempting to construct new API calls.

---

# `Notes/`

Contains project planning and mathematical specifications.

## `checklist.md`

Condensed computational checklist describing the main workflow.

Use this for a quick overview of the planned computational stages.

## `detailed_checklist.md`

Expanded version of the computational checklist.

This contains detailed explanations of:

- baseline simulation;
- robustness checks;
- drug endpoint validation;
- parameter sweeps;
- equilibria and Jacobian stability;
- one-parameter continuation;
- two-parameter continuation;
- optional fast–slow and hysteresis analysis.

**Read this file before beginning a major computational stage.**

## `model.md`

Contains the mathematical specification of the Plant–Kim model, including:

- the seven-dimensional dynamical system;
- state-variable definitions;
- ionic current terms;
- model parameters and gating-function information.

Treat this as the main local model specification when implementing the system.

Do not alter the mathematical model or parameter values without justification.

---

# `References/`

Contains literature relevant to the project.

The two primary references for the mathematical model are:

- Plant & Kim (1975);
- Plant & Kim (1976).

These should be treated as the main scientific references for:

- the R15 model;
- parameter values;
- gating functions;
- untreated bursting;
- TTX effects;
- ouabain effects.

Other papers are stored in subdirectories such as:

- `Other References/`;
- `R15 Papers/`.

These may be useful for supporting context but are less central to the initial implementation.

When there is disagreement between an informal note and the original Plant–Kim paper, flag the discrepancy rather than guessing which is correct.

---

# `notebooks/`

Currently reserved for the main computational notebooks produced for this project.

Current notebooks:

    notebooks/
    ├── 01_baseline_model.ipynb
    ├── 02_robustness_phase_planes.ipynb
    └── 03_drug_endpoints.ipynb

Possible later notebooks (not yet created):

    notebooks/
    ├── 04_parameter_sweeps.ipynb
    └── 05_bifurcation_analysis.ipynb

---

# `src/`

Reserved for reusable Python source code.

The aim is eventually to avoid duplicating the full Plant–Kim model across multiple notebooks.

Existing reusable model modules:

    src/
    ├── plant_kim_model.py
    └── plant_kim_ttx_5d.py

Additional helpers may be added later if needed. Do not create unnecessary abstractions before they are required.

---

# `outputs/`

Reserved for generated outputs.

Possible structure:

    outputs/
    ├── figures/
    └── data/

Generated simulation figures, sweep summaries, or exported numerical results may eventually be stored here.

---

# Other Files / Folders

Miscellaneous files that are unrelated to the current computational analysis may be ignored unless specifically requested.

---

# `docker-compose.yml`

A Docker configuration is provided for an alternative PyDSTool environment with AUTO support.

Use this environment for:

- AUTO continuation;
- periodic-orbit continuation (`LC-C`);
- Hopf bifurcations requiring AUTO;
- more advanced bifurcation calculations.

Keep the host `.venv` for ordinary VODE/PyDSTool work (baseline simulation, sweeps, EP-C where available). Do **not** modify or replace the host `.venv` to obtain AUTO.

### Verified Docker / AUTO smoke test (2026-09-18)

Image: `curuy/math3888:latest` (compose service `jupyter-service`, container name `jupyter-lab`).

Start / stop from the repository root:

```bash
docker compose up -d
docker compose down
```

JupyterLab is then at `http://localhost:8888` with token `password`. The project tree is bind-mounted at `/notebooks`.

Inside the container:

- Python: `3.9.25` at `/usr/local/bin/python3`
- PyDSTool: import OK (reports `1.19.5`)
- AUTO: bundled under `PyDSTool/PyCont/auto` and compiled into the process CWD on first `LC-C`

Minimal proven LC-C example: Selkov oscillator EP-C → Hopf → `LC-C` pattern from `Labs/Week 7 - Lab7 example.ipynb`.

Important AUTO detail: PyDSTool writes `auto_*.py` / `_auto_*.so` into the **current working directory**, which must be importable (Jupyter notebooks already put CWD on `sys.path`; a script launched from another directory needs e.g. `PYTHONPATH="$(pwd)"` or an equivalent `sys.path` insert). Add generated AUTO artefacts to `.gitignore` rather than committing them.

---

# Model Implementation Principles

When implementation begins:

1. Implement the original untreated **7D Plant–Kim model first**.
2. Follow the equations and parameters in `Notes/model.md` and the original Plant–Kim references.
3. Do not initially:
   - nondimensionalise the model;
   - reduce its dimension;
   - make quasi-steady-state approximations;
   - alter parameters to force bursting.
4. Put gating and time-constant functions into clearly defined auxiliary functions where practical.
5. Validate the gating functions before trusting the complete ODE simulation.
6. Reproduce baseline bursting before attempting drug sweeps or bifurcation analysis.
7. If the baseline model does not reproduce the expected behaviour, investigate:
   - transcription errors;
   - parameter values;
   - units;
   - initial conditions;
   - solver configuration;
   - gating-function definitions;

   before changing the mathematical model.

---

# Exact TTX Reduction

Complete TTX blockade corresponds to

$$
g_I=0.
$$

At this endpoint, the variables $X_I$ and $Y_I$ no longer feed back into the remaining system, so the effective dynamics can be reduced exactly from 7D to 5D.

This reduction should **only** be used at complete TTX blockade.

For

$$
g_I>0,
$$

the full 7D model must be retained.

---

# Drug Parameters

Later analysis may introduce scaled drug-strength parameters

$$
g_I=\alpha_I g_I^{(0)},
$$

and

$$
I_{\mathrm{ep}}=\alpha_P I_{\mathrm{ep}}^{(0)},
$$

where

$$
\alpha_I,\alpha_P\in[0,1].
$$

Interpretation:

- $\alpha_I=1$: untreated inward conductance;
- $\alpha_I=0$: complete TTX blockade;
- $\alpha_P=1$: normal electrogenic pump strength;
- $\alpha_P=0$: complete ouabain blockade.

These parameters are primarily intended for later parameter sweeps and bifurcation analysis.

---

# Current Stage

The project is currently at the **environment verification and baseline implementation preparation stage**.

The immediate goals are:

1. verify that Cursor can use the local `.venv`;
2. verify that PyDSTool imports correctly;
3. verify that Jupyter/JupyterLab are available;
4. identify the correct Jupyter kernel for the `.venv`;
5. inspect the existing PyDSTool examples and repository structure;
6. confirm that the environment is ready for implementation.

**Do not begin full Plant–Kim model implementation until explicitly requested.**

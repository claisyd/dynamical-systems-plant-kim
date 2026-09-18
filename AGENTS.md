# MATH3888 Agent Instructions

## Project Context

This repository contains the computational work for the MATH3888 capstone project on the Plant--Kim model of *Aplysia* R15 neuronal bursting.

The main computational tasks are:

1. reproduce the baseline 7D Plant--Kim normal bursting model;
2. validate untreated, TTX, ouabain, and combined-drug behaviour;
3. perform coarse parameter sweeps;
4. compute equilibria and assess local stability;
5. perform one-parameter bifurcation continuation;
6. optionally perform two-parameter continuation and/or fast--slow analysis.

PyDSTool is the primary numerical package because it is used in the course workshops and recommended by the teaching staff.

---

# 1. Canonical Python / Jupyter Environment

## Python

The canonical Python interpreter for this repository is:

`./.venv/bin/python`

Verified Python version:

`Python 3.9.20`

Always prefer explicit project-environment commands, for example:

`./.venv/bin/python script.py`

or:

`./.venv/bin/python -m <module>`

Do **not** assume that bare `python` or `python3` resolves to the correct environment.

Do **not** reinstall, upgrade, or downgrade Python, NumPy, SciPy, PyDSTool, Jupyter, ipykernel, or related numerical dependencies unless explicitly requested.

If an import or execution problem occurs, first verify the interpreter with:

`./.venv/bin/python -c "import sys; print(sys.executable)"`

The expected executable is the absolute path to this repository's `.venv/bin/python`.

## Jupyter

The canonical project Jupyter kernel has been verified as:

- internal kernel name: `pydstool`
- display name: `Python 3.9 (PyDSTool)`

It points to the same project `.venv/bin/python`.

Use this kernel for project notebooks.

Do **not** use the generic `python3` kernel for project notebooks. A venv-local/generic `python3` kernelspec may contain a bare `"python"` in `argv[0]`, making it PATH-dependent; it is intentionally not the canonical project kernel.

Preferred JupyterLab launch command:

`./.venv/bin/jupyter-lab`

Do not rely on a bare `jupyter lab` command when environment identity matters.

---

# 2. Verified PyDSTool State

The environment has already passed a smoke test:

- PyDSTool import: PASS
- VODE ODE construction: PASS
- integration / compute: PASS
- trajectory sampling: PASS
- known-working example: saddle-node pattern in `Code/practice.ipynb`

The successful pattern was:

`dst.args -> Vode_ODEsystem -> set(tdata) -> compute -> sample`

The same style is also represented in the workshop material under `Labs/`.

## Solver Rule

For direct ODE simulation in the local environment:

**USE VODE ONLY unless explicitly instructed otherwise.**

Do not use:

- Radau
- Dopri
- Dopri853

These have previously caused compatibility/compilation problems in this macOS/PyDSTool setup.

## Sandbox Note

PyDSTool may segfault with exit code 139 under a restricted Cursor execution sandbox while working correctly under ordinary local/unrestricted execution.

If PyDSTool fails only in a restricted sandbox:

1. do not immediately conclude that the `.venv` is broken;
2. rerun using the explicit local `.venv/bin/python` interpreter;
3. report the discrepancy.

## PyDSTool Version Metadata

A known metadata discrepancy may exist:

- `PyDSTool.__version__` may report `1.23.5`;
- installed distribution metadata reports PyDSTool `0.91.0`.

Do not attempt to "fix" this by reinstalling or upgrading PyDSTool.

---

# 3. Source-of-Truth Order

Before implementing or changing the scientific model, inspect the relevant local material.

Use sources in approximately this order:

1. `Notes/model.md` -- local mathematical/model specification;
2. Plant--Kim 1975 and Plant--Kim 1976 in `References/` -- primary scientific sources;
3. `Notes/detailed_checklist.md` -- detailed computational plan;
4. `Notes/checklist.md` -- concise computational plan;
5. `Code/practice.ipynb` -- known-working PyDSTool syntax;
6. `Labs/` -- workshop/tutor examples and additional known-working patterns.

`Code/textbook.ipynb` does **not** use PyDSTool and should not normally be used as a PyDSTool API reference.

PyDSTool is an older library. **Do not invent or guess PyDSTool APIs.**

When syntax is uncertain:

1. inspect existing working repository examples;
2. inspect the relevant workshop lab;
3. inspect installed PyDSTool source/documentation if necessary;
4. use external documentation only when needed.

If the correct API remains uncertain, report the uncertainty instead of silently inventing syntax.

---

# 4. Scientific Modelling Rules

The canonical baseline model is the original **7D Plant--Kim normal bursting model** with state vector

`(V, X_K, X_A, Y_A, X_P, X_I, Y_I)`.

Do not alter the mathematical model, equations, parameters, units, or initial conditions merely to make numerical code run.

Always distinguish between:

1. implementation changes;
2. numerical-method changes;
3. mathematical-model changes.

Parameter values must be traceable to the Plant--Kim literature or project documentation.

Before changing any parameter or model equation:

- state what is being changed;
- explain why;
- identify whether it is an implementation correction, numerical choice, or scientific/model change.

Preserve a baseline implementation before conducting experiments.

## TTX Reduction

Complete TTX blockade corresponds to:

`g_I = 0`

Only at this endpoint do `X_I` and `Y_I` cease to feed back into the remaining dynamics, allowing the effective model to reduce exactly from 7D to 5D.

Do **not** use the 5D reduction when `g_I > 0`.

## Scaled Drug Parameters

Later parameter sweeps/continuation may use:

`g_I = alpha_I * g_I^(0)`

and

`I_ep = alpha_P * I_ep^(0)`

with:

`alpha_I, alpha_P in [0, 1]`

Interpretation:

- `alpha_I = 1`: untreated inward conductance;
- `alpha_I = 0`: complete TTX blockade;
- `alpha_P = 1`: normal electrogenic pump strength;
- `alpha_P = 0`: complete ouabain blockade.

Do not introduce these parameters prematurely if the current task is only baseline implementation.

---

# 5. Current Computational Workflow

Use `Notes/detailed_checklist.md` as the detailed workflow.

The intended high-level order is:

1. environment verification;
2. implement the original 7D model;
3. inspect gating functions and time constants;
4. reproduce untreated normal bursting;
5. check transients, numerical robustness, and initial-condition robustness;
6. inspect useful phase-plane projections;
7. validate TTX, ouabain, and combined-drug endpoints;
8. perform coarse parameter sweeps and extract summaries such as `V_min`, `V_max`, period, and spike count where appropriate;
9. compute equilibria and assess Jacobian eigenvalues;
10. perform one-parameter continuation;
11. optionally perform two-parameter continuation and/or fast--slow analysis.

Do not jump ahead to later stages unless explicitly requested.

---

# 6. Docker / AUTO

`docker-compose.yml` provides an alternative environment that may be useful later for AUTO continuation and advanced bifurcation calculations.

Do not automatically:

- start Docker;
- rebuild Docker;
- modify `docker-compose.yml`;
- switch environments;
- start AUTO continuation.

Prefer the local `.venv` for baseline implementation, direct VODE simulation, parameter sweeps, and initial exploratory analysis.

Use the Docker/AUTO environment only when explicitly requested or when later continuation work requires it (especially `LC-C`).

When AUTO/`LC-C` is requested, start with:

`docker compose up -d`

from the repository root (image `curuy/math3888:latest`). A 2026-09-18 smoke test confirmed Selkov periodic-orbit continuation works in that container. See `README.md` (docker-compose section) for Jupyter URL/token and the CWD/`sys.path` requirement. Do **not** modify the host `.venv` to obtain AUTO.

---

# 7. Repository / File Discipline

Before making a substantial change:

1. read this file and the relevant `README.md`;
2. read the relevant checklist/model notes;
3. inspect existing code before creating a parallel implementation;
4. state the intended mathematical/computational change;
5. implement the smallest reasonable change;
6. run the relevant code using the canonical `.venv`;
7. check output for numerical and qualitative plausibility.

Prefer reusable model code in `src/` and analysis notebooks in `notebooks/` once implementation begins, while avoiding unnecessary abstraction before the baseline model works.

Do not delete existing experimental results unless explicitly requested.

Avoid unnecessary refactoring during scientific experiments.

Keep computation separate from scientific interpretation where practical.

### Notebook figures, captions, and HTML reports

Analysis notebooks under `notebooks/` are the primary scientific narrative. Treat figures as part of that narrative, not as side artefacts that exist only on disk.

For every figure that is written to `outputs/figures/`:

1. include the image in the notebook in the **appropriate section** (immediately after the cell that produces it, or in a nearby dedicated figure cell);
2. give a short markdown caption / description stating what the plot shows and any essential caveats (units, biological vs mathematical parameter ranges, duplicate-detection notes, etc.);
3. prefer durable embedding via notebook markdown attachments (`attachment:filename.png`) so images survive HTML export, rather than relying only on fragile relative paths;
4. when regenerating a report, render HTML **after** figures have been attached so `outputs/reports/*.html` contains the images inline.

Blank, near-empty, or superseded diagnostic plots must not be left in the notebook or HTML report once a clearer figure exists.

When building or repairing notebooks from generator scripts, keep figure captions in sync with the generator / presentation helper (for example `scripts/fix_05_ttx_presentation.py` for Notebook 05) so re-runs remain consistent.

---

# 8. Dependency Snapshot

`requirements-lock.txt` in the repository root is intended to be a snapshot of the exact packages installed in the working `.venv`.

Do not hand-edit it to guess dependency versions.

If explicitly asked to refresh it, generate it from the canonical environment with:

`./.venv/bin/python -m pip freeze > requirements-lock.txt`

A change to `requirements-lock.txt` should normally accompany an intentional dependency/environment change.

The `.venv/` directory itself should remain untracked by Git.

---

# 9. Git / Version-Control Rules for Agents

Agents may inspect:

- `git status`
- `git diff`
- `git log`

when useful.

Do not automatically:

- commit;
- push;
- force-push;
- reset;
- rebase;
- delete branches;
- discard uncommitted work

unless explicitly requested.

Before a substantial edit, preserve the user's existing work and avoid overwriting unrelated changes.

After completing work, report:

- files changed;
- commands executed;
- whether execution succeeded;
- numerical checks performed;
- assumptions;
- unresolved issues.

---

# 10. Default Behaviour When Something Fails

If a numerical or environment step fails:

1. verify the correct interpreter/kernel;
2. compare against a known-working PyDSTool example;
3. inspect transcription, units, initial conditions, and solver settings;
4. distinguish environment failure from model failure;
5. do not change the scientific model simply to suppress an error;
6. report the smallest safe next step.

When in doubt, preserve the working environment and ask before making destructive or dependency-level changes.

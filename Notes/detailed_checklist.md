# Computational and Bifurcation Analysis Checklist

The overall computational workflow should approximately be

$$
\text{Model implementation}
\rightarrow
\text{Baseline validation}
\rightarrow
\text{Drug endpoint validation}
\rightarrow
\text{Coarse parameter sweeps}
\rightarrow
\text{Equilibria and stability}
\rightarrow
\text{One-parameter continuation}
\rightarrow
\text{Optional two-parameter continuation}.
$$

The original untreated Plant--Kim model is seven-dimensional, with state vector

$$
\mathbf{x}
=
(V,X_K,X_A,Y_A,X_P,X_I,Y_I).
$$

Nondimensionalisation may reduce the number of independent parameter groups and remove physical units, but it does **not generally reduce the number of state variables or the dimension of the dynamical system itself**. It may also be unnecessary here since retaining the dimensional formulation preserves the physiological interpretation of parameters such as $g_I$ and $I_{\mathrm{ep}}$.

An exact reduction from 7D to 5D occurs under complete TTX blockade, $g_I=0$, because $X_I$ and $Y_I$ then no longer feed back into the remaining system. This reduction is therefore specific to the TTX endpoint and cannot in general be used for $g_I>0$.

**Implementation note.** The baseline code in `src/plant_kim_model.py` / `notebooks/01_baseline_model.ipynb` uses Plant--Kim (1976) Tables I--II together with the Canavier et al. (1990) corrections to apparent typographical errors in the Table I rate functions $\alpha_m$, $\beta_m$, and $\beta_h$. Details are recorded in `Notes/model.md`.

**Progress (baseline + drug endpoints).** Sections 1.1--1.5 and 2.1--2.5 are complete: untreated 7D VODE bursting is reproduced, robustness-checked, and phase-plane projections documented in `notebooks/02_robustness_phase_planes.ipynb`; drug endpoints (TTX, ouabain, combined) and the exact 5D TTX reduction sanity check are documented in `notebooks/03_drug_endpoints.ipynb` / `src/plant_kim_ttx_5d.py`. Later sweep / continuation items remain open.

**Robustness / phase-plane note.** Across sensible ICs ($V_0\in\{-60,-50,-40\}$ with steady-state gates; modest $X_P$ offset) and VODE settings (baseline / tighter / looser), the post-transient attractor is the same qualitative bursting regime: about 4 spikes per mature burst, period $\sim 9.0$ s, $V$ approximately $[-51.6, 27.5]$ mV.

**Drug-endpoint note.** At the hard endpoints (`notebooks/03_drug_endpoints.ipynb`): TTX ($g_I=0$) removes fast APs and leaves a slow wave (period $\sim 10.0$ s, $V$ approximately $[-52.5,-38.9]$ mV); ouabain ($I_{\mathrm{ep}}=0$) yields regular beating (ISI $\sim 0.45$ s); combined TTX+ouabain converges to equilibrium (late $V$ range $\sim 10^{-7}$ mV); 7D vs exact 5D TTX common-state errors are within VODE tolerance (max $|\Delta V|\sim 2\times 10^{-3}$ mV).

---

# 1. Baseline Simulations and Implementation

## 1.1 Implement the original 7D Plant--Kim model ✅

- Implement the full untreated seven-dimensional Plant--Kim system using the state variables

  $$
  V,\quad X_K,\quad X_A,\quad Y_A,\quad X_P,\quad X_I,\quad Y_I.
  $$

- Enter all model parameters using the values reported in the Plant--Kim model, including:
  - membrane capacitance $C$;
  - reversal potentials $V_I,V_K,V_L$;
  - conductances $g_I,g_T,g_K,g_A,g_P,g_L$;
  - electrogenic pump current $I_{\mathrm{ep}}$;
  - applied current $I_{\mathrm{ext}}$;
  - any additional constants used in the gating functions.

- Implement the voltage equation exactly as specified by the model before attempting any simplification or reduction.

- Implement the gating functions and time constants as auxiliary functions rather than placing every expression directly inside the ODE definitions. These include functions of the form

  $$
  S_i(V),\qquad Z_i(V),\qquad \tau_i(V).
  $$

- The gating equations should retain the form

  $$
  \dot X_i=\frac{S_i(V)-X_i}{\tau_{X_i}(V)},
  $$

  and similarly for the inactivation variables,

  $$
  \dot Y_i=\frac{Z_i(V)-Y_i}{\tau_{Y_i}(V)}.
  $$

- Use the full 7D system as the canonical baseline implementation. Do **not** initially:
  - nondimensionalise the equations;
  - assume any gating variable is at steady state;
  - remove fast variables;
  - use the 5D TTX reduction;
  - otherwise approximate the original model.

- In PyDSTool, this will broadly involve defining:
  - `varspecs`;
  - `pars`;
  - `ics`;
  - auxiliary functions;
  - the integration time domain;
  - the ODE generator, initially using VODE if this remains the most reliable solver in our setup.

---

## 1.2 Inspect the gating functions and time constants ✅

Before interpreting any full model simulation, check that the individual gating functions have been entered correctly.

- Plot each activation and inactivation function against voltage:

  $$
  S_i(V),\qquad Z_i(V).
  $$

- Use a voltage interval wide enough to contain the voltages encountered during R15 activity.

- Check that the activation and inactivation functions:
  - generally lie in $[0,1]$;
  - have the expected monotonic or sigmoidal shapes;
  - agree qualitatively with their definitions in Plant--Kim;
  - do not contain obvious discontinuities or transcription errors.

- Plot each voltage-dependent time constant

  $$
  \tau_i(V).
  $$

- Check that

  $$
  \tau_i(V)>0
  $$

  throughout the relevant voltage range.

- Compare the relative magnitudes of the time constants. This may already indicate which gating variables act on:
  - fast spike timescales;
  - intermediate timescales;
  - slow bursting timescales.

- In particular, examine the behaviour of the very slow potassium activation variable $X_P$, since this may later be useful for interpreting the slow--fast structure of the bursting system.

- This step should also help identify errors such as:
  - missing minus signs;
  - incorrect exponential arguments;
  - misplaced brackets;
  - incorrect units;
  - incorrect parameter values.

---

## 1.3 Run the baseline simulation of normal R15 bursting ✅

Once the individual functions appear correct:

- Use the untreated baseline parameter values

  $$
  g_I=g_I^{(0)},\qquad
  I_{\mathrm{ep}}=I_{\mathrm{ep}}^{(0)}.
  $$

- Run the full 7D system over a sufficiently long time interval.

- Initially plot the voltage trace

  $$
  V(t).
  $$

- The first major validation target is to reproduce the qualitative **normal bursting behaviour of the untreated R15 neuron**.

- Run the simulation for sufficiently long that any initial transient can be discarded.

- Do not immediately interpret the beginning of the trajectory. Instead:
  - inspect the whole time trace;
  - identify approximately when the transient ends;
  - analyse a later section in which the dynamics appear to have approached the long-term attractor.

- Once $V(t)$ looks reasonable, plot the remaining six state variables:

  $$
  X_K(t),\quad
  X_A(t),\quad
  Y_A(t),\quad
  X_P(t),\quad
  X_I(t),\quad
  Y_I(t).
  $$

- Inspect how the gating variables behave during:
  - the silent phase;
  - the start of a burst;
  - individual action potentials;
  - termination of a burst;
  - recovery before the next burst.

- Note which variables vary rapidly with individual spikes and which change primarily over the entire burst cycle.

---

## 1.4 Check numerical and initial-condition robustness ✅

Completed in `notebooks/02_robustness_phase_planes.ipynb` (IC + VODE-tolerance checks; post-transient metrics).

Once normal bursting has been reproduced, check that it is not a numerical artefact or an artefact of a single initial condition.

### Numerical robustness

- Repeat the baseline simulation using tighter numerical tolerances and/or more restrictive integration step settings.

- Compare the resulting trajectories qualitatively.

- Exact pointwise agreement is not necessary because oscillatory systems may accumulate phase differences over long integrations.

- Instead, verify that quantities such as the following remain approximately stable:
  - qualitative dynamical regime;
  - burst period;
  - number of spikes per burst;
  - voltage range;
  - general waveform.

### Initial-condition robustness

- Repeat the simulation using several reasonable initial conditions.

- Check whether the trajectories converge toward the same long-term bursting attractor.

- If different initial conditions converge toward genuinely different long-term behaviours, do **not** automatically treat this as an error. It may indicate:
  - multiple attractors;
  - bistability;
  - dependence on initial conditions.

- Such behaviour would be worth noting for later bifurcation analysis.

---

## 1.5 Make phase-plane projections ✅

Completed in `notebooks/02_robustness_phase_planes.ipynb`: post-transient six-panel $V$ vs $\{X_K,X_A,Y_A,X_P,X_I,Y_I\}$ plus enlarged $V$–$X_P$ slow-variable projection.

Although the system is seven-dimensional, useful two-dimensional projections can still be plotted.

- Plot voltage against each gating variable, for example:

  $$
  (V,X_K),\quad
  (V,X_A),\quad
  (V,Y_A),\quad
  (V,X_P),\quad
  (V,X_I),\quad
  (V,Y_I).
  $$

- In particular, examine

  $$
  (V,X_P),
  $$

  since $X_P$ is associated with the very slow potassium current and may provide a useful view of the bursting cycle.

- Use these plots to identify:
  - closed periodic trajectories;
  - slow sections of the orbit;
  - fast spike excursions;
  - differences between bursting and later drug-induced regimes.

- We do not necessarily need to include all phase-plane plots in the final report; they are initially exploratory diagnostics.

---

# 2. Drug Effect Validations

The purpose of this stage is to determine whether the implemented model reproduces the expected qualitative behaviours at the experimentally relevant drug endpoints.

At this stage we are **not yet performing formal bifurcation analysis**.

---

## 2.1 Confirm untreated normal bursting ✅

- Use the baseline values

  $$
  g_I=g_I^{(0)},\qquad
  I_{\mathrm{ep}}=I_{\mathrm{ep}}^{(0)}.
  $$

- Confirm that the long-term voltage behaviour corresponds to normal R15 bursting.

- This untreated solution will act as the reference trajectory against which the drug cases are compared.

---

## 2.2 TTX endpoint ✅

Complete TTX blockade is represented by

$$
g_I=0,
$$

with all other parameters initially held at their baseline values.

- Simulate the full 7D system with

  $$
  g_I=0,\qquad
  I_{\mathrm{ep}}=I_{\mathrm{ep}}^{(0)}.
  $$

- Run the system sufficiently long to remove transients.

- Plot $V(t)$ and relevant gating variables.

- Check whether the fast action potentials disappear and whether the remaining dynamics exhibit the expected slow-wave oscillation.

- This provides an initial validation of the first hypothesis:

  $$
  \text{normal bursting}
  \longrightarrow
  \text{slow-wave oscillation}
  $$

  as the transient inward conductance is suppressed.

**Result.** Validated in `notebooks/03_drug_endpoints.ipynb` on the canonical 7D model: post-transient slow wave without fast spikes (period $\sim 10.0$ s, $V$ approximately $[-52.5,-38.9]$ mV).

---

## 2.3 Compare the 7D TTX system against the exact 5D TTX reduction ✅

At

$$
g_I=0,
$$

the term involving $X_I$ and $Y_I$ disappears from the voltage equation.

Therefore, $X_I$ and $Y_I$ no longer feed back into the dynamics of

$$
V,\quad X_K,\quad X_A,\quad Y_A,\quad X_P.
$$

- Construct the corresponding reduced 5D model.

- Simulate both:
  - the full 7D system with $g_I=0$;
  - the reduced 5D TTX system.

- Compare the common variables:

  $$
  V,\quad X_K,\quad X_A,\quad Y_A,\quad X_P.
  $$

- Their trajectories should agree up to numerical error.

- This acts as a useful implementation sanity check.

- Note that this exact reduction is only valid when

  $$
  g_I=0.
  $$

  For any $g_I>0$, $X_I$ and $Y_I$ again influence the voltage equation and the full 7D system must be used.

**Result.** Exact 5D reduction in `src/plant_kim_ttx_5d.py`; 7D vs 5D common-state agreement within VODE tolerance (max $|\Delta V|\sim 2\times 10^{-3}$ mV) in `notebooks/03_drug_endpoints.ipynb`.

---

## 2.4 Ouabain endpoint ✅

Complete inhibition of the electrogenic pump by ouabain is represented by

$$
I_{\mathrm{ep}}=0.
$$

- Simulate the full 7D system with

  $$
  g_I=g_I^{(0)},\qquad
  I_{\mathrm{ep}}=0.
  $$

- Again run the model sufficiently long to discard transients.

- Plot $V(t)$ and relevant gating variables.

- Check whether the dynamics move from normal bursting toward the expected regular beating/tonic activity.

- This provides an endpoint validation of the second hypothesis:

  $$
  \text{normal bursting}
  \longrightarrow
  \text{regular beating}.
  $$

**Result.** Validated in `notebooks/03_drug_endpoints.ipynb`: regular beating with ISI $\sim 0.45$ s and $V$ approximately $[-49.3, 26.6]$ mV.

---

## 2.5 Both interventions ✅

Set

$$
g_I=0,\qquad
I_{\mathrm{ep}}=0.
$$

- Simulate the model under simultaneous complete TTX and pump blockade.

- Determine the resulting long-term behaviour rather than assuming the answer in advance.

- Our current hypothesis predicts that inhibition of the pump under TTX will suppress the remaining slow oscillation and may drive the system toward a **stable equilibrium**.

- Check whether the trajectory:
  - continues oscillating;
  - changes to another oscillatory regime;
  - approaches a constant voltage;
  - exhibits any transient oscillations before settling.

- This endpoint provides an initial qualitative check of Hypothesis 3 before the two-parameter analysis is attempted.

**Result.** Validated in `notebooks/03_drug_endpoints.ipynb`: post-transient trajectory converges to a stable equilibrium (late-window $V$ range $\sim 10^{-7}$ mV).

---

# 3. Bifurcation Studies

This section begins the transition from direct simulation toward formal bifurcation analysis.

It is important to distinguish between:

$$
\boxed{\text{parameter sweep}=\text{repeated direct ODE simulations}}
$$

and

$$
\boxed{\text{continuation}=\text{following solution branches numerically using AUTO}.}
$$

The parameter sweeps should therefore be performed before formal continuation.

---

## 3.1 Introduce scaled drug parameters

Define a scaled TTX parameter

$$
\alpha_I\in[0,1]
$$

such that

$$
g_I=\alpha_I g_I^{(0)}.
$$

Therefore:

$$
\alpha_I=1
\quad\Longleftrightarrow\quad
\text{untreated transient inward conductance},
$$

and

$$
\alpha_I=0
\quad\Longleftrightarrow\quad
\text{complete TTX blockade}.
$$

Similarly, define a scaled pump-strength parameter

$$
\alpha_P\in[0,1]
$$

such that

$$
I_{\mathrm{ep}}
=
\alpha_P I_{\mathrm{ep}}^{(0)}.
$$

Therefore:

$$
\alpha_P=1
\quad\Longleftrightarrow\quad
\text{normal pump activity},
$$

and

$$
\alpha_P=0
\quad\Longleftrightarrow\quad
\text{complete ouabain blockade}.
$$

This avoids complications from the sign convention of $I_{\mathrm{ep}}$ and gives both drug parameters the same interpretation:

$$
(\alpha_I,\alpha_P)\in[0,1]^2.
$$

The four corners are then

$$
(1,1)=\text{untreated},
$$

$$
(0,1)=\text{TTX},
$$

$$
(1,0)=\text{ouabain},
$$

$$
(0,0)=\text{TTX + ouabain}.
$$

---

## 3.2 Perform coarse one-parameter simulation sweeps

Before using AUTO, perform ordinary ODE simulations at a sequence of parameter values.

### TTX sweep

Hold

$$
\alpha_P=1
$$

and decrease

$$
\alpha_I:1\rightarrow0.
$$

For example, initially use a coarse grid such as

$$
\alpha_I
=
1,\ 0.9,\ 0.8,\ldots,0.
$$

### Ouabain sweep

Hold

$$
\alpha_I=1
$$

and decrease

$$
\alpha_P:1\rightarrow0.
$$

For example,

$$
\alpha_P
=
1,\ 0.9,\ 0.8,\ldots,0.
$$

### For every parameter value

- Integrate for sufficiently long that the transient is removed.

- Analyse only the long-term trajectory.

- Record quantitative summaries such as:

  $$
  V_{\min},
  \qquad
  V_{\max},
  \qquad
  T,
  $$

  where $T$ is the oscillation or burst period when this is well defined.

- Potentially also record:
  - number of spikes per burst;
  - burst duration;
  - silent-phase duration;
  - interspike interval;
  - mean voltage.

- Plot quantities such as

  $$
  V_{\min}(\alpha_I),\qquad
  V_{\max}(\alpha_I)
  $$

  and similarly for $\alpha_P$.

- These plots are **not yet formal bifurcation diagrams**, but they may reveal approximate locations where the qualitative dynamics change.

- If a transition appears to occur between two coarse parameter values, repeat the sweep with a finer grid in that region.

- The main purpose of this stage is to identify approximately where AUTO continuation is likely to encounter interesting behaviour.

---

## 3.3 Compute equilibria and assess their stability

Use the coarse parameter sweeps to identify parameter values that are worth investigating more carefully.

For a fixed pair $(\alpha_I,\alpha_P)$, an equilibrium satisfies

$$
\dot{\mathbf{x}}=0.
$$

For the gating variables,

$$
X_i^*=S_i(V^*),
$$

and

$$
Y_i^*=Z_i(V^*).
$$

Therefore, the equilibrium gating values can be substituted into the voltage equation, reducing the equilibrium calculation to the corresponding current-balance condition for $V^*$.

### Compute equilibria at relevant parameter settings

Initially consider:
- the untreated baseline;
- the TTX endpoint;
- the ouabain endpoint;
- the combined-drug endpoint;
- selected values near transitions suggested by the coarse parameter sweeps.

### Stability analysis

For each equilibrium $\mathbf{x}^*$, calculate the Jacobian

$$
J(\mathbf{x}^*)
=
D_{\mathbf{x}}F(\mathbf{x}^*).
$$

Compute its seven eigenvalues

$$
\lambda_1,\ldots,\lambda_7.
$$

The equilibrium is locally asymptotically stable if

$$
\Re(\lambda_i)<0
\qquad
\text{for all }i.
$$

It is unstable if at least one eigenvalue satisfies

$$
\Re(\lambda_i)>0.
$$

Look particularly for:
- a real eigenvalue approaching zero;
- a complex-conjugate pair whose real parts approach zero.

For example, a complex pair satisfying approximately

$$
\lambda_{1,2}\approx \pm i\omega
$$

may indicate that a Hopf bifurcation lies nearby.

This step provides preliminary stability information before formal continuation and gives useful checks against the bifurcations later detected by AUTO.

---

## 3.4 TTX one-parameter bifurcation study

For the formal TTX continuation, hold

$$
\alpha_P=1
$$

and use

$$
\alpha_I
$$

as the continuation parameter.

Follow

$$
\alpha_I:1\rightarrow0.
$$

### Equilibrium continuation

- Start from a known equilibrium.
- Continue the equilibrium branch as $\alpha_I$ changes.
- Track the stability of the equilibrium along the branch.
- Detect bifurcation points automatically where possible.

Possible bifurcations may include:
- Hopf bifurcations;
- saddle-node/fold bifurcations;
- other equilibrium bifurcations.

Do **not** assume beforehand which bifurcation must occur.

### Periodic-orbit continuation

Since both normal bursting and the TTX slow wave are oscillatory behaviours, periodic-orbit continuation is likely to be important.

Where appropriate:
- start or identify a periodic branch;
- continue the periodic solution as $\alpha_I$ decreases;
- track quantities such as

  $$
  V_{\min},\qquad
  V_{\max},\qquad
  T.
  $$

- Look for bifurcations of periodic orbits, which may include:
  - folds of periodic orbits;
  - period-doubling bifurcations;
  - torus/Neimark--Sacker bifurcations;
  - possible global bifurcation behaviour.

The main question is:

$$
\text{Which bifurcation structure mediates }
\text{bursting}\rightarrow\text{slow-wave oscillation}?
$$

We should not assume that there is necessarily one single bifurcation at which the system suddenly changes from bursting to slow-wave behaviour.

---

## 3.5 Ouabain one-parameter bifurcation study

For the formal ouabain continuation, hold

$$
\alpha_I=1
$$

and continue in

$$
\alpha_P:1\rightarrow0.
$$

### Equilibrium continuation

- Continue the equilibrium branch as pump strength decreases.
- Track eigenvalues and stability.
- Detect equilibrium bifurcations where present.

### Periodic-orbit continuation

- Continue relevant periodic solutions as $\alpha_P$ decreases.
- Track quantities such as

  $$
  V_{\min},\qquad
  V_{\max},\qquad
  T.
  $$

- Examine how the bursting orbit changes as the hyperpolarising electrogenic pump contribution is removed.

The main question is:

$$
\text{Which bifurcation structure mediates }
\text{bursting}\rightarrow\text{regular beating}?
$$

Again, do not assume in advance that a single bifurcation necessarily separates the two regimes.

---

## 3.6 Ouabain continuation under complete TTX

Before attempting the full two-parameter study, investigate the one-dimensional slice corresponding to complete TTX.

Set

$$
\alpha_I=0
$$

and continue

$$
\alpha_P:1\rightarrow0.
$$

This corresponds to progressively inhibiting the electrogenic pump while the transient inward current is already completely blocked.

The expected qualitative transition from Hypothesis 3 is approximately

$$
\text{TTX slow-wave oscillation}
\longrightarrow
\text{stable equilibrium}.
$$

- Continue the relevant equilibrium branch.
- Track its stability.
- Continue the slow-wave periodic orbit where possible.
- Determine how the oscillation disappears as $\alpha_P$ decreases.
- Identify any bifurcation separating the slow-wave regime from the stable equilibrium.

Because

$$
\alpha_I=0
\quad\Longleftrightarrow\quad
g_I=0,
$$

this analysis can also be performed using the exact reduced 5D TTX model.

---

## 3.7 Optional two-parameter continuation

Only attempt this after the one-parameter continuation studies have identified useful codimension-one bifurcation points.

The parameter plane is

$$
(\alpha_I,\alpha_P)\in[0,1]^2.
$$

The aim is **not** simply to simulate every point in the square.

Instead, select bifurcation points found during the one-parameter analyses and continue those bifurcations as curves in the two-parameter plane.

For example, if a Hopf bifurcation is detected at

$$
(\alpha_I^*,1),
$$

allow $\alpha_P$ to vary and continue that Hopf point to obtain a curve schematically of the form

$$
H(\alpha_I,\alpha_P)=0.
$$

Similarly, other suitable codimension-one bifurcations may potentially be continued as curves.

The resulting parameter plane may contain regions corresponding to qualitatively different dynamical behaviours, for example:

$$
\text{bursting},
\qquad
\text{regular beating},
\qquad
\text{slow-wave oscillation},
\qquad
\text{stable equilibrium}.
$$

The two-parameter analysis would then help answer how the effects of TTX and ouabain interact rather than considering each drug separately.

This should be regarded as an **advanced extension**, not something that must be completed before the one-parameter studies are understood.

---

# 4. Miscellaneous / Optional Analyses

These do not need to be performed in strict chronological order and can be considered depending on what is observed in the main analysis.

## 4.1 Bistability and hysteresis

Near an apparent transition:

- sweep the parameter downward,

  $$
  1\rightarrow0,
  $$

  using the final state from one simulation as the initial condition for the next;

- then sweep it back upward,

  $$
  0\rightarrow1.
  $$

If the forward and backward sweeps:
- transition at different parameter values; or
- converge to different attractors over the same parameter range,

this may indicate bistability or hysteresis.

---

## 4.2 Track equilibrium eigenvalues

During equilibrium calculations and continuation:

- monitor the eigenvalues of the Jacobian;
- in particular, plot or inspect the eigenvalues with the largest real parts.

This helps explain *why* an equilibrium gains or loses stability rather than merely recording that AUTO detected a bifurcation.

---

## 4.3 Fast--slow analysis

A possible advanced alternative to the full two-parameter continuation is to exploit the separation of timescales in the model.

If $X_P$ is sufficiently slow compared with the remaining variables, the system may be viewed schematically as

$$
\dot{\mathbf{x}}_f
=
F(\mathbf{x}_f,X_P),
$$

$$
\dot X_P
=
\varepsilon G(\mathbf{x}_f,X_P),
\qquad
0<\varepsilon\ll1.
$$

Then $X_P$ can temporarily be treated as a parameter of the fast subsystem.

This may provide a mechanistic explanation of how the slow variable drives transitions between silent and spiking phases during bursting.

This should only be attempted after the baseline simulation and main one-parameter bifurcation studies are working reliably.

Later we can decide whether the more useful advanced extension is:

$$
\boxed{\text{two-parameter drug continuation}}
$$

or

$$
\boxed{\text{fast--slow bursting analysis}},
$$

rather than automatically attempting both.
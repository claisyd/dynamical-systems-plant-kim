We should note that non-dimensionalising the dynamical system may reduce the number of independent parameter groups and removes physical units, but it does *not generally reduce the number of state variables or the dimension of the dynamical system itself*. It may not be worth doing non-dimensionalisation either since we will lose the physiological meaning behind all the variables.

One possible advanced reduction is to split the dynamical system to a fast-slow time scale, which may be done if we have time after the main parameter continuations. Speaking of which, I list out below the bifurcation studies that we should do for our hypotheses.

1. TTX: define $\alpha_I \in [0, 1]$ so that $g_I=\alpha_I g_I^{(0)}$ where $g_I^{(0)}$ is the initial value of the parameter $g_I$. Then, we set $\alpha_I \to 0$ and see what bifurcations occur there; we examine which bifurcations mediate the transition; possible bifurcations include Hopf, saddle-node, or bifurcations of periodic orbits.
2. Ouabain: similarly define $\alpha_P\in [0, 1]$ so that $I_{\text{ep}}=\alpha_PI_{\text{ep}}^{(0)}$ and set $\alpha_P\to 0$ rather than deal with the negative sign of $I_{\text{ep}}$, and see what bifurcations happen. 
3. Both drugs: this would be a two-parameter bifurcation on $(\alpha_I, \alpha_P)$. Before that, we can also set immediately $\alpha_I=0$ and see what bifurcations happen on $\alpha_P$ alone; the two-parameter bifurcation can be an extension to this as well.

# Simulation Pipeline

**Implementation note.** The baseline 7D implementation uses Plant--Kim (1976) Tables I--II with the Canavier et al. (1990) corrections to apparent typographical errors in Table I rate functions ($\alpha_m$, $\beta_m$, $\beta_h$). See `Notes/model.md` and `src/plant_kim_model.py`.

**Robustness / phase-plane note** (`notebooks/02_robustness_phase_planes.ipynb`): sensible IC and VODE-tolerance perturbations converge to the same post-transient bursting regime — about 4 spikes per mature burst, period $\sim 9.0$ s, $V$ approximately $[-51.6, 27.5]$ mV.

**Drug-endpoint note** (`notebooks/03_drug_endpoints.ipynb`): complete TTX ($g_I=0$) yields slow-wave oscillation without fast spikes (period $\sim 10.0$ s); ouabain ($I_{\mathrm{ep}}=0$) yields regular beating (ISI $\sim 0.45$ s); TTX+ouabain converges to a stable equilibrium; exact 5D TTX reduction matches 7D common states to within VODE tolerance (max $|\Delta V|\sim 2\times 10^{-3}$ mV). See also `src/plant_kim_ttx_5d.py`.

## Baseline Simulations and Implementations

1. [x] Implement the original 7D model, putting the gating functions as auxiliary functions
2. [x] Inspect gating functions $S_i, Z_i$ against $V$ (should lie in $[0, 1]$ and have sigmoidal shapes) and plot time constants $\tau_i$ (should be positive)
3. [x] Run baseline simulation of normal R15 bursting: mainly plot $V(t)$ and the other variables (7 figures)
4. [x] Check different initial conditions, check robustness by comparing tolerance (we should see stable qualitative regime, burst period, spike count, and voltage range) and make sure we are not just looking at the transient (check latter part of time trace)
5. [x] Phase plane projections: plot $V$ against the other variables.

## Drug Effect Validations

1. [x] Confirm untreated R15 normal bursting (see above point 3)
2. [x] TTX endpoint: Set $g_I=0$ and everything else as baseline; hope to recover slow-wave oscillation
3. [x] Ouabain endpoint: Set $I_\text{ep}=0$ and everything else as baseline; hope to recover regular beating
4. [x] Both interventions: Set $g_I=0$ and $I_{\text{ep}}=0$ and see what happens; we hypothesise that the trajectory may approach a stable equilibrium
5. [x] Check TTX endpoint against the 5D model as well for a sanity check (for $g_I=0$)

## Bifurcation Studies

1. [ ] Rewrite model to introduce $\alpha_I$ and $\alpha_P$
2. [ ] Parameter sweep of $\alpha_I \to 0$ and $\alpha_P\to 0$; here we extract quantitative values like $V_{\text{min}},\ V_{\text{max}}$, and period throughout and e.g., plot $V_{\text{min}}$ and $V_{\text{max}}$ against $g_I$.
3. [ ] Compute equilibria at the relevant parameter settings and assess their stability using the Jacobian eigenvalues
4. [ ] TTX Bifurcation Study: continue $\alpha_I \to 0$ and $\alpha_P=1$
5. [ ] Ouabain Bifurcation Study: continue $\alpha_P\to 0$ and $\alpha_I =1$
6. [ ] Ouabain under TTX: continue $\alpha_P\to 0$ and $\alpha_I=0$
7. [ ] Two-parameter continuation: continue selected bifurcation curves in the $(\alpha_I, \alpha_P)\in [0, 1]^2$ parameter plane

## Miscellaneous

1. [ ] Bistability/hysteresis: at a potential transition, sweep the parameter downward ($1 \to 0$) and then upward ($0 \to 1$); If the forward and backward sweeps give different transition parameter values or converge to different attractors over the same parameter range, this suggests bistability/hysteresis
2. [ ] Track equilibrium eigenvalues: plot/examine how the leading real parts change to understand why an equilibrium gains or loses stability.
3. [ ] Fast-slow analysis: potential replacement for two-parameter continuation; decide later.

---
output:
  html_document: default
  pdf_document: default
---
# MATH3888 Individual Component Plan
## Mathematical Analysis of Hopf Eigenvalues and Local Bifurcation Conditions

### Working individual-project question

**Can the local mathematical mechanisms underlying the numerically detected Hopf and saddle-node bifurcations in the Plant–Kim neuron model be characterised directly from the Jacobian spectrum, and can those local calculations independently validate the continuation results?**

This individual component combines three extensions:

1. **Jacobian/eigenvalue mechanism behind the bifurcations**
2. **Hopf frequency prediction from eigenvalues**
3. **Numerical verification of local Hopf and saddle-node bifurcation conditions**

The aim is not to repeat the group bifurcation diagrams. Instead, the project should explain *why* the continuation software labels a point as a Hopf or limit point by analysing the local linearisation and its derivatives.

---

# 1. Recommended primary case studies

## A. Primary Hopf case: complete TTX + ouabain

Use the biologically relevant Hopf already found on the complete-TTX slice:

- Fix complete TTX: `alpha_I = 0`.
- Vary pump activity through `alpha_P`.
- Analyse the biological Hopf `H_bio` found in Notebook 06.

Why this is a strong case:

- it lies on a biologically meaningful drug path;
- the equilibrium changes stability there;
- an AUTO limit-cycle branch is already available for comparison;
- the LC period gives an independent numerical benchmark for the eigenvalue-based period prediction.

## B. Primary saddle-node case: TTX alone

Use the biological TTX fold already found in Notebook 05:

- Fix untreated pump: `alpha_P = 1`.
- Vary `alpha_I`.
- Analyse `LP_bio`, near `alpha_I ≈ 0.614`.

Why this is a strong case:

- it gives a complementary local mechanism to Hopf;
- at a Hopf, a *complex-conjugate pair* crosses the imaginary axis;
- at a saddle-node, a *real eigenvalue* reaches zero.

This contrast should form one of the central mathematical themes of the report.

## C. Secondary cases, only if useful

Possible additional checks:

- the exterior Hopf on the complete-TTX `alpha_P` branch;
- one exterior TTX Hopf;
- the exterior ouabain-alone Hopf;
- codimension-2 points discovered later in the two-parameter analysis.

These should remain secondary unless they reveal qualitatively new behaviour.

---

# 2. Core mathematical setup

Write the model abstractly as
$$\dot{x}=F(x;\mu), \qquad x\in\mathbb{R}^7$$

where `mu` is the selected continuation parameter, such as `alpha_P` or `alpha_I`.

An equilibrium satisfies
$$F(x^*(\mu);\mu)=0.$$

The Jacobian is


$$J(x;\mu)=D_xF(x;\mu)=
\left[
\frac{\partial F_i}{\partial x_j}
\right].$$

The local dynamics near an equilibrium are

$$
\dot{\xi}
=
J(x^*;\mu)\xi
+
O(\|\xi\|^2).$$


The eigenvalues of `J` determine local equilibrium stability.

---

# 3. Part 1 — Jacobian and eigenvalue mechanism

## 3A. Analytical tasks

- [ ] Write the seven-dimensional state vector clearly.
- [ ] Write the ODE system in the form `dot{x}=F(x;mu)`.
- [ ] Define the Jacobian `J = D_x F`.
- [ ] Derive representative entries by hand to explain the structure of the matrix.
- [ ] For a standard gating equation
  $$
  \dot X=\frac{X_\infty(V)-X}{\tau_X(V)},
  $$
  explain analytically why
  $$
  \frac{\partial \dot X}{\partial X}
  =
  -\frac{1}{\tau_X(V)}.$$
- [ ] Explain how voltage couples into the gating-variable rows through derivatives of `X_inf(V)` and `tau_X(V)`.
- [ ] Explain how gating variables enter the voltage row through ionic-current derivatives.
- [ ] Keep the full 7x7 symbolic Jacobian in an appendix rather than the main text.

## 3B. Computational tasks for Cursor

### Symbolic Jacobian

Recommended implementation:

- use `sympy`;
- define the canonical RHS symbolically;
- compute
  ```python
  J_sym = F_sym.jacobian(x_sym)
  ```
- convert the symbolic expression into a fast numerical function using `sympy.lambdify`.

Important implementation rule:

**Do not create a symbolic model that silently differs from `src/plant_kim_model.py`.**

The symbolic equations should be cross-checked against the canonical source implementation.

### Independent Jacobian validation

At representative equilibrium points, compare:

1. symbolic/lambdified Jacobian;
2. centred finite-difference Jacobian;
3. eigenvalues stored by PyDSTool during EP-C.

Use, for example,

$$
[J_{\rm FD}]_{ij}
\approx
\frac{F_i(x+h e_j)-F_i(x-h e_j)}{2h}.
$$

Useful diagnostics:

- maximum absolute matrix error;
- relative Frobenius-norm error;
- maximum eigenvalue mismatch after matching spectra.

Expected validation chain:

$$
\boxed{
J_{\rm symbolic}
\leftrightarrow
J_{\rm FD}
\leftrightarrow
\text{PyDSTool spectrum}
}
$$

### Eigenvalue tracking

For each equilibrium along a local parameter window around a bifurcation:

- compute all seven eigenvalues;
- match eigenvalues between neighbouring parameter points rather than independently sorting them;
- track the critical eigenvalue/eigenpair continuously.

For small spectra, nearest-neighbour matching in the complex plane may be sufficient. A Hungarian-assignment method can be used if labels switch.

### Expected outputs

- [ ] Jacobian validation table.
- [ ] Full spectrum at the bifurcation.
- [ ] Critical eigenvalue/eigenpair plotted against the continuation parameter.
- [ ] Maximum real part of the spectrum plotted against the continuation parameter.
- [ ] Comparison against the existing PyDSTool stability classification.

---

# 4. Hopf case — local spectral mechanism

For the complete-TTX + ouabain Hopf, write the critical pair as

$$
\lambda_\pm(\alpha_P)
=
a(\alpha_P)
\pm i\omega(\alpha_P).
$$

At the Hopf,

$$
a(\alpha_P^H)=0,
\qquad
\omega(\alpha_P^H)\neq 0.
$$

## 4A. Analytical tasks

- [ ] Explain why a complex pair crossing the imaginary axis changes equilibrium stability.
- [ ] Distinguish the Hopf mechanism from a zero-real-eigenvalue saddle-node mechanism.
- [ ] Explain why the imaginary part is the local angular frequency.
- [ ] State the generic Hopf spectral conditions.

## 4B. Computational tasks

Around `H_bio`:

- [ ] sample equilibria on both sides of the Hopf;
- [ ] evaluate the Jacobian;
- [ ] track the critical complex pair;
- [ ] plot `Re(lambda_crit)` versus `alpha_P`;
- [ ] plot `Im(lambda_crit)` versus `alpha_P`;
- [ ] verify that the real part changes sign;
- [ ] verify that the imaginary part remains nonzero;
- [ ] verify that all other eigenvalues remain away from the imaginary axis.

A central figure should show

$$
\Re\lambda_\pm \to 0
$$

while

$$
|\Im\lambda_\pm|>0.
$$

---

# 5. Saddle-node case — local spectral mechanism

For the biological TTX fold `LP_bio`, the expected critical mechanism is

$$
\lambda_1(\alpha_I)\to0,
\qquad
\lambda_1\in\mathbb{R}.
$$

## 5A. Analytical tasks

- [ ] Explain why a simple zero eigenvalue is the relevant local linear signature of a generic saddle-node.
- [ ] Contrast it explicitly with the Hopf critical complex pair.
- [ ] Introduce right and left nullvectors:
  $$
  Jq=0,
  \qquad
  p^\top J=0.
  $$

## 5B. Computational tasks

Around `LP_bio`:

- [ ] evaluate equilibria on both sides where they exist;
- [ ] compute the spectrum;
- [ ] track the critical real eigenvalue;
- [ ] show that it approaches zero at the fold;
- [ ] show that the other eigenvalues remain bounded away from zero.

Recommended comparison figure:

**Hopf panel**
$$
\Re(\lambda_\pm)\to0,\qquad \Im(\lambda_\pm)\neq0
$$

versus

**saddle-node panel**
$$
\lambda_1\to0,\qquad \lambda_1\in\mathbb R.
$$

---

# 6. Part 2 — Hopf frequency prediction

At the Hopf,

$$
\lambda_\pm=\pm i\omega_H.
$$

The linearised oscillation therefore has predicted period

$$
\boxed{
T_{\rm lin}
=
\frac{2\pi}{\omega_H}.
}
$$

The Plant–Kim model time unit is milliseconds, so the predicted period should be in milliseconds.

## 6A. Analytical tasks

- [ ] Derive `T = 2*pi/omega` from the local solution `exp(± i omega t)`.
- [ ] Explain why the prediction is local and should agree best with limit cycles of very small amplitude near the Hopf.
- [ ] Explain why the prediction need not match large-amplitude cycles far from the Hopf.

## 6B. Computational tasks

- [ ] calculate `omega_H = abs(Im(lambda_H))`;
- [ ] calculate `T_lin = 2*pi/omega_H`;
- [ ] extract AUTO periods on the LC branch close to the Hopf;
- [ ] compute LC amplitude
  $$
  A=V_{\max}-V_{\min};
  $$
- [ ] compare AUTO period against `T_lin` as `A -> 0`;
- [ ] report relative error.

Suggested table:

| distance from Hopf | LC amplitude | AUTO period | eigenvalue prediction | relative error |
|---:|---:|---:|---:|---:|

Suggested result:

$$
T_{\rm AUTO}
\longrightarrow
T_{\rm lin}
\quad
\text{as}
\quad
A\to0.
$$

This gives a strong analytical-to-numerical consistency check.

---

# 7. Part 3 — numerical verification of Hopf theorem conditions

The goal is not merely to say "AUTO labelled this point H".

Instead, verify the local conditions directly.

## 7A. Equilibrium residual

Check

$$
F(x_H;\mu_H)=0.
$$

Record

$$
\|F(x_H;\mu_H)\|.
$$

## 7B. Simple imaginary pair

Verify one simple pair

$$
\lambda_{1,2}
=
\pm i\omega_H,
\qquad
\omega_H>0,
$$

while all remaining eigenvalues satisfy

$$
\Re\lambda_j\neq0.
$$

## 7C. Hopf transversality

Check

$$
\boxed{
\frac{d}{d\mu}
\Re\lambda(\mu)
\bigg|_{\mu=\mu_H}
\neq0.
}
$$

### Core numerical method

Use centred finite differences:

$$
\frac{
\Re\lambda(\mu_H+h)
-
\Re\lambda(\mu_H-h)
}{
2h
}.
$$

Repeat for several values of `h` and check numerical stability.

### Stronger semi-analytical method

Let

$$
Jv=\lambda v,
\qquad
w^*J=\lambda w^*,
$$

with

$$
w^*v=1.
$$

Then

$$
\frac{d\lambda}{d\mu}
=
w^*
\frac{dJ}{d\mu}
v.
$$

Because the equilibrium itself depends on `mu`,

$$
F(x^*(\mu),\mu)=0
$$

implies, at a Hopf where `J` is invertible,

$$
\frac{dx^*}{d\mu}
=
-J^{-1}F_\mu.
$$

Hence

$$
\frac{dJ}{d\mu}
=
J_\mu
+
D_xJ
\left[
\frac{dx^*}{d\mu}
\right].
$$

This provides an independent transversality calculation.

---

# 8. Numerical verification of saddle-node conditions

At the TTX fold, verify the generic saddle-node conditions.

## 8A. Equilibrium condition

$$
F(x_{LP};\mu_{LP})=0.
$$

## 8B. Simple zero eigenvalue

Verify

$$
\lambda_1=0
$$

is simple and all other eigenvalues are nonzero.

Let

$$
Jq=0,
\qquad
p^\top J=0,
\qquad
p^\top q=1.
$$

## 8C. Parameter transversality

Compute

$$
\boxed{
a
=
p^\top F_\mu
}
$$

and verify

$$
a\neq0.
$$

## 8D. Quadratic nondegeneracy

Compute

$$
\boxed{
b
=
\frac12
p^\top
D_x^2F(x_{LP})[q,q]
}
$$

and verify

$$
b\neq0.
$$

`SymPy` can generate the Hessian/second-derivative tensors required for this calculation.

---

# 9. Strong optional result — local fold normal form

If the saddle-node coefficients are successfully computed, compare the numerical branch to the local normal form

$$
\dot u
=
a(\mu-\mu_{LP})
+
bu^2
+\cdots.
$$

Equilibria satisfy approximately

$$
a(\mu-\mu_{LP})+bu^2=0,
$$

so

$$
u
\sim
\pm
\sqrt{
-\frac{a}{b}
(\mu-\mu_{LP})
}.
$$

Possible extension:

- fit the numerical EP-C branch close to `LP_bio`;
- compare it with the predicted square-root scaling.

This would provide an analytical explanation for the local folded shape.

---

# 10. Optional Hopf eigenvector analysis

For the critical Hopf eigenvector

$$
Jq=i\omega q,
$$

inspect:

- magnitudes `|q_j|`;
- phases relative to voltage,
  $$
  \arg(q_j)-\arg(q_V).
  $$

This can reveal which state variables dominate the local oscillatory mode.

This is optional and should only be included if the pattern is interpretable.

---

# 11. Stretch objective — first Lyapunov coefficient

A generic Hopf additionally requires nondegeneracy, often expressed through

$$
\ell_1\neq0.
$$

Computing `l1` requires second and third derivatives of the vector field.

This is **not required for the core project**.

It becomes especially relevant if the two-parameter continuation later discovers a generalized Hopf / Bautin point, where

$$
\ell_1=0.
$$

Possible use:

- compute `l1` at `H_bio`;
- determine local Hopf criticality;
- if a GH point is found, track `l1` along the Hopf curve and verify its sign change.

---

# 12. Relationship to the two-parameter continuation

This individual component does **not** depend on the two-parameter analysis finding a codimension-2 point.

The current one-parameter results already provide:

- a biologically relevant Hopf;
- a biologically relevant saddle-node;
- equilibrium data;
- periodic-orbit data.

However, the two-parameter analysis may provide optional extensions.

## If only ordinary Hopf and LP curves are found

Possible individual extension:

- compute the critical frequency `omega_H` at several points along the Hopf curve;
- examine how the local oscillation period varies with both drug parameters.

## If a generalized Hopf / Bautin point is found

Possible extension:

- compute the first Lyapunov coefficient;
- show it approaches/crosses zero along the Hopf curve.

## If a Bogdanov–Takens point is found

Possible extension:

- use the Jacobian machinery to show a double-zero spectral configuration.

## If a cusp is found

Possible extension:

- analyse degeneration of the saddle-node quadratic coefficient.

These are optional upgrades, not prerequisites.

---

# 13. Computational implementation checklist for Cursor

## Core implementation

- [ ] Read canonical equations from `src/plant_kim_model.py`.
- [ ] Build a matching symbolic `SymPy` representation.
- [ ] Construct symbolic Jacobian.
- [ ] Lambdify the Jacobian.
- [ ] Build an independent centred finite-difference Jacobian.
- [ ] Validate symbolic vs finite-difference Jacobians.
- [ ] Validate resulting spectra against PyDSTool EP-C eigenvalues.
- [ ] Build robust eigenvalue matching/tracking.
- [ ] Analyse complete-TTX `H_bio`.
- [ ] Analyse TTX `LP_bio`.
- [ ] Produce report-quality eigenvalue plots.
- [ ] Compute `T_lin = 2*pi/omega_H`.
- [ ] Compare with AUTO LC periods near Hopf.
- [ ] Numerically estimate Hopf transversality using several finite-difference step sizes.
- [ ] Check simple-zero condition at LP.
- [ ] Save all numerical summaries as CSV/JSON where useful.
- [ ] Render a clean HTML analysis notebook.

## Strong extension

- [ ] Compute left/right eigenvectors.
- [ ] Compute semi-analytical Hopf eigenvalue derivative.
- [ ] Compute saddle-node `a = p^T F_mu`.
- [ ] Compute saddle-node `b = 1/2 p^T D^2F[q,q]`.
- [ ] Compare local fold normal form against EP-C geometry.
- [ ] Analyse Hopf eigenvector amplitudes/phases.

## Stretch

- [ ] Compute first Lyapunov coefficient `l1`.
- [ ] Connect `l1` to a GH/Bautin point if discovered.
- [ ] Track critical frequency / transversality along a two-parameter Hopf curve.

---

# 14. Analytical writing checklist

The final report should explain, not merely display:

- [ ] what the Jacobian represents;
- [ ] why its eigenvalues determine local stability;
- [ ] why Hopf requires a complex pair crossing the imaginary axis;
- [ ] why a saddle-node has a simple zero eigenvalue;
- [ ] why the imaginary part of the Hopf eigenvalue predicts frequency;
- [ ] why transversality matters;
- [ ] what the saddle-node nondegeneracy coefficients mean;
- [ ] why local linear analysis is strongest near bifurcation points;
- [ ] why local bifurcation analysis alone cannot explain the entire bursting orbit far from equilibrium.

---

# 15. Recommended staged scope

## Core — should be sufficient for a coherent individual report

1. Derive and validate the Jacobian.
2. Track the critical Hopf pair.
3. Track the critical saddle-node eigenvalue.
4. Predict Hopf period from the imaginary eigenvalue.
5. Compare against AUTO periods near Hopf.
6. Verify simple-pair/simple-zero conditions.
7. Verify Hopf transversality numerically.

## Strong extension — recommended if implementation is smooth

8. Compute saddle-node transversality/nondegeneracy coefficients.
9. Compare the local fold normal form with EP-C.
10. Compute Hopf transversality via left/right eigenvectors.
11. Analyse the critical Hopf eigenvector.

## Stretch

12. Compute the first Lyapunov coefficient.
13. Connect to GH/BT/CP results from two-parameter continuation if relevant.
14. Track local spectral quantities along the two-parameter Hopf curve.

---

# 16. Tentative individual report structure

## 1. Introduction and biological motivation

- Plant–Kim neuron model.
- TTX and ouabain interventions.
- Group continuation results motivating the individual analysis.
- Individual research question.

## 2. Mathematical model and local linearisation

- ODE formulation.
- Equilibria.
- Jacobian derivation.
- Representative symbolic entries.

## 3. Numerical implementation and validation

- SymPy Jacobian.
- Lambdification.
- Finite-difference validation.
- PyDSTool spectrum cross-check.
- Eigenvalue tracking.

## 4. Saddle-node mechanism under TTX

- `LP_bio`.
- Critical real eigenvalue.
- Simple-zero condition.
- Optional `a`, `b` coefficients.
- Optional local fold normal form.

## 5. Hopf mechanism under complete TTX + ouabain

- `H_bio`.
- Critical complex pair.
- Stability crossing.
- Spectral isolation.

## 6. Predicting oscillation frequency

- derivation of `T = 2*pi/omega`;
- AUTO comparison;
- near-Hopf convergence.

## 7. Verification of local bifurcation conditions

- Hopf spectral conditions;
- transversality;
- saddle-node simple-zero condition;
- saddle-node nondegeneracy;
- optional first Lyapunov coefficient.

## 8. Optional connection to codimension-2 continuation

Only include if the two-parameter results add a natural mathematical extension.

## 9. Physiological interpretation and limitations

- what local stability explains;
- what it does not explain;
- relation to the group time-domain and continuation analysis.

## 10. Conclusion

Summarise the local mathematical mechanisms underlying the selected bifurcations.

---

# 17. Proposed final scope statement

> **Use the biological complete-TTX/ouabain Hopf and biological TTX saddle-node as complementary case studies. Derive and validate the Plant–Kim Jacobian, track their critical eigenvalues, predict the Hopf oscillation period from the imaginary eigenvalue, compare this prediction with AUTO limit-cycle periods, and numerically verify the generic local Hopf and saddle-node conditions. Add two-parameter spectral analysis only where the codimension-2 results provide a clear mathematical motivation.**

This scope is intended as a working plan rather than a fixed final checklist. It can be adjusted after the two-parameter continuation and once the individual-report rubric is released.

# Bifurcation Analysis Reference

This file is a working reference for the MATH3888 Plant--Kim continuation
analysis.

It summarises terminology used in lectures and continuation software.
It should NOT be treated as a substitute for the course notes, PyDSTool/AUTO
documentation, or mathematical verification.

When analysing a continuation result:

1. inspect the actual solution branch;
2. inspect Jacobian/Floquet information where appropriate;
3. check the point type reported by the continuation software;
4. do not infer a bifurcation solely from the visual appearance of a plot.

---

## 1. Continuation curve types

### EP-C: Equilibrium Point Continuation

EP-C follows equilibria as a model parameter is varied.

Typical quantities plotted include:

- equilibrium membrane voltage \(V^*\);
- continuation parameter;
- equilibrium stability;
- detected codimension-1 equilibrium bifurcations.

Important possible points include:

- `LP`: limit point / fold / saddle-node of equilibria;
- `H`: Hopf bifurcation.

A Hopf point detected on an EP-C branch indicates that a conjugate pair of
eigenvalues crosses the imaginary axis.

EP-C alone does not generally determine the full periodic-orbit branch or
whether the resulting Hopf bifurcation is supercritical or subcritical.

---

### LC-C: Limit Cycle Continuation

LC-C follows periodic solutions, usually starting from a Hopf bifurcation.

Useful quantities include:

- minimum and maximum membrane voltage;
- oscillation amplitude;
- period;
- periodic-orbit stability.

LC-C may detect bifurcations of periodic orbits such as limit points of cycles.

For this project, periodic-orbit continuation is expected to use AUTO through
the provided Docker environment.

Docker / AUTO environment status (smoke-tested 2026-09-18): `docker compose up -d`
with image `curuy/math3888:latest` successfully ran a Selkov `LC-C` continuation
inside the container. See the `docker-compose.yml` section of `README.md` for
startup commands and the CWD / `sys.path` requirement when invoking AUTO.

---

# 2. Codimension-1 bifurcations

A codimension-1 bifurcation generically requires variation of one parameter.

## LP -- Limit Point / Saddle-Node / Fold of Equilibria

Two equilibrium solutions collide and disappear or are created.

At a generic saddle-node, one eigenvalue passes through zero.

In a one-parameter equilibrium continuation diagram this usually appears as a
turning point of the equilibrium branch.

PyDSTool commonly labels these points `LP`.

---

## H -- Hopf Bifurcation

A complex-conjugate pair of eigenvalues crosses the imaginary axis.

A periodic orbit may be created or destroyed.

The equilibrium continuation identifies the Hopf location, but periodic-orbit
continuation is required to investigate the resulting limit-cycle branch.

The Hopf may be supercritical or subcritical; do not infer this only from the
EP-C curve.

---

## LPC -- Limit Point of Cycles

Also called a fold or saddle-node of limit cycles.

Two periodic orbits meet and disappear or are created.

This is a bifurcation of periodic solutions and is therefore associated with
LC-C rather than ordinary equilibrium continuation.

---

## HOM -- Homoclinic Bifurcation

A periodic orbit approaches an equilibrium, usually a saddle, and becomes a
homoclinic orbit.

The period often grows very large when approaching a homoclinic bifurcation.

Numerically, continuation may become difficult close to this point.

A very large period alone is suggestive but is not sufficient proof of a
homoclinic bifurcation.

---

## SNIC -- Saddle-Node on an Invariant Circle

A saddle-node bifurcation occurs on an invariant circle and creates or destroys
oscillations.

The oscillation period typically becomes very large near the SNIC.

This is especially relevant in excitable systems because it may describe an
onset or loss of repetitive firing.

---

## Transcritical and Pitchfork Bifurcations

These involve intersecting equilibrium branches and exchange or change of
stability.

Continuation software may identify these through branch-point information,
depending on the curve type and software implementation.

Do not assume a particular PyDSTool/AUTO label without checking the actual
continuation output and documentation.

---

# 3. Codimension-2 bifurcations

A codimension-2 bifurcation generically requires variation of two independent
parameters.

These become relevant when two-parameter continuation is performed later in
the project.

---

## BT -- Bogdanov--Takens Point

A Bogdanov--Takens point is associated with a double zero eigenvalue.

It can organise several nearby bifurcation curves, commonly including
saddle-node, Hopf, and homoclinic phenomena.

AUTO/PyDSTool may report this using the label `BT`.

---

## GH -- Generalised Hopf / Bautin Point

A generalised Hopf (Bautin) point occurs when the character of a Hopf
bifurcation changes.

For example, the Hopf may change between supercritical and subcritical.

A limit-point-of-cycles curve can emerge from a generalised Hopf point.

The usual continuation label is `GH`.

---

## CP -- Cusp Point

A cusp is a degenerate saddle-node bifurcation.

In a two-parameter diagram, two saddle-node/fold branches can meet or turn at
a cusp.

The usual continuation label is `CP`.

---

## Saddle-Node Homoclinic Interaction

Codimension-2 interactions involving saddle-node and homoclinic phenomena can
also occur.

These may not be implemented as directly detectable special points in the
PyDSTool workflow used for this project.

If suspected, they should be investigated from the surrounding bifurcation
curves rather than assigned solely from visual appearance.

---

# 4. Numerical termination: MX

`MX` should NOT automatically be interpreted as a physical bifurcation.

It generally indicates that a numerical/continuation limit has been reached or
that the continuation algorithm has terminated because of a computational
bound.

For periodic-orbit continuation, an `MX` point occurring while the period is
becoming extremely large may occur near a global bifurcation such as a
homoclinic orbit or SNIC.

However:

> MX by itself is numerical evidence of termination, not proof of a
> bifurcation type.

Always inspect period, orbit shape, equilibrium structure, and nearby
continuation results.

---

# 5. Drug parameterisations used in this project

## TTX

The transient inward current may be represented either by direct continuation
of its maximal conductance \(g_I\), or by a dimensionless scaling parameter
\(\alpha_I\).

If the model uses

\[
g_I^{\mathrm{eff}} = \alpha_I g_I^{(0)},
\]

then:

- \(\alpha_I = 1\): baseline/control conductance;
- \(\alpha_I = 0\): complete TTX block;
- \(0 < \alpha_I < 1\): partial blockade.

Values outside \([0,1]\) may be useful for mathematical continuation but do not
represent ordinary TTX blockade.

Direct continuation in \(g_I\) can be used to study the broader mathematical
bifurcation structure and to cross-check continuation results.

When the alpha formulation is a simple multiplicative rescaling, bifurcation
locations in the two parameterisations should transform according to

\[
\alpha_I = \frac{g_I^{\mathrm{eff}}}{g_I^{(0)}}.
\]

A disagreement inside the corresponding parameter ranges should trigger an
implementation check.

---

## Ouabain

Pump inhibition may similarly be represented through the pump-current
parameter \(I_{ep}\) or a dimensionless multiplier \(\alpha_P\).

If

\[
I_{ep}^{\mathrm{eff}} = \alpha_P I_{ep}^{(0)},
\]

then:

- \(\alpha_P = 1\): baseline pump activity;
- \(\alpha_P = 0\): complete pump inhibition;
- \(0 < \alpha_P < 1\): partial pump inhibition.

Again, values outside the biological interval may be explored mathematically
but should not automatically be interpreted as experimentally achievable
ouabain exposure.

---

# 6. Recommended analysis workflow

For each intervention:

1. Verify the baseline model and equilibrium.
2. Construct the EP-C curve.
3. Continue in BOTH forward and backward parameter directions.
4. Detect and record all special points.
5. Record:
   - continuation parameter value;
   - equilibrium membrane voltage;
   - stability;
   - relevant eigenvalues.
6. Cross-check bifurcation locations independently where possible.
7. Distinguish the mathematically explored parameter range from the
   biologically meaningful intervention range.
8. For each relevant Hopf point, perform LC-C using AUTO.
9. Record limit-cycle:
   - minimum voltage;
   - maximum voltage;
   - amplitude;
   - period;
   - stability.
10. Investigate detected periodic-orbit bifurcations.
11. Only after the one-parameter analysis is reliable, proceed to
    two-parameter continuation.

---

# 7. Current TTX cross-check target

An independently generated continuation curve for direct continuation in
\(g_I\) currently provides a useful validation target.

The reference curve contains:

- two Hopf points (`H1`, `H2`);
- two equilibrium limit points (`LP1`, `LP2`);
- continuation in both forward and backward directions.

Future implementations should aim to reproduce the same qualitative branch
structure before interpreting new continuation results.

Do not rely on old notebook output if the notebook predates the corrected
forward/backward continuation implementation.

The reference image supplied by the group should be treated as the current
qualitative target.
Plant and Kim (1975) proposed a slow-wave mechanism for the Aplysia R15 neuron under TTX. Their 1976 paper developed this into a quantitative modified Hodgkin–Huxley model capable of reproducing both normal bursting and the TTX-induced slow wave. We propose to use the untreated seven-dimensional Plant–Kim (1976) system as our baseline model. The effect of TTX may be represented by reducing the transient inward conductance $g_I$, with complete blockade corresponding to $g_I=0$. The variables $X_I$ and $Y_I$ then decouple from the remaining system, yielding a five-dimensional model under TTX. The effect of ouabain may similarly be modelled by reducing the magnitude of the electrogenic pump current $I_\text{ep}$, with complete pump blockade corresponding to $I_\text{ep}=0$.

Hence, our 7D dynamical system for the baseline R15 bursting model will be as follows: $$\begin{align*}
    \dot{V}&=\dfrac{1}{C}\big[(g_IX_I^3Y_I+g_T)(V_I-V)+(g_KX_K^4+g_AX_AY_A+g_PX_P)(V_K-V)+g_L(V_L-V)+I_{\text{ep}}+I_{\text{ext}}\big]\\ 
    \dot{X}_K&=\dfrac{1}{\tau_{X_K}(V)}\big[S_K(V)-X_K\big], \qquad \dot{X}_P=\dfrac{1}{\tau_{X_P}}\big[S_P(V)-X_P\big], \qquad \dot{X}_A=\dfrac{1}{\tau_{X_A}(V)}\big[S_A(V)-X_A\big],\\ 
    \dot{Y}_A&=\dfrac{1}{\tau_{Y_A}(V)}\big[Z_A(V)-Y_A\big], \qquad \dot{X}_I=\dfrac{1}{\tau_{X_I}(V)}\big[S_I(V)-X_I\big], \qquad \dot{Y}_I=\dfrac{1}{\tau_{Y_I}(V)}\big[Z_I(V)-Y_I\big].
\end{align*}$$
## State Variables

| Variable | Description                                                                                  |
| -------- | -------------------------------------------------------------------------------------------- |
| $V(t)$   | Membrane potential of the R15 neuron.                                                        |
| $X_K(t)$ | Activation of the fast potassium $K^+$-current.                                              |
| $X_A(t)$ | Activation of the inactivating $A$-type potassium current.                                   |
| $Y_A(t)$ | Inactivation of the $A$-type potassium current.                                              |
| $X_P(t)$ | Activation of the very slow potassium current responsible for slow feedback.                 |
| $X_I(t)$ | Activation of the fast transient inward current responsible for action-potential generation. |
| $Y_I(t)$ | Inactivation of the fast transient inward current.                                           |
<div style="page-break-after: always;"></div>


## Ionic and Applied Current Terms

| Current term       | Description                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| $g_IX_I^3Y_I(V_I-V)$ | Fast transient inward current responsible for action-potential spikes; blocked by TTX.           |
| $g_T(V_I−V)$       | Persistent inward current that remains in the presence of TTX and promotes depolarisation.       |
| $g_KX_K^4(V_K−V)$  | Fast outward $K^+$ current involved in membrane repolarisation.                                  |
| $g_AX_AY_A(V_K−V)$ | Inactivating $A$-type outward $K^+$ current, controlled by both activation and inactivation.     |
| $g_PX_P(V_K−V)$    | Very slow outward $K^+$ current that provides slow feedback underlying the membrane oscillation. |
| $g_L(V_L−V)$       | Passive leak current driving the voltage toward the leak reversal potential $V_L$.               |
| $I_\text{ep}$      | Hyperpolarising electrogenic pump current.                                                       |
| $I_\text{ext}$     | Externally applied current controlled experimentally.                                            |
## Model Parameters and Functions

| Symbol                | Description                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| $C$                   | Membrane capacitance.                                                       |
| $V_I,V_K,V_L$         | Reversal potentials for inward, potassium, and leak currents.               |
| $g_T,g_K,g_A,g_P,g_L$ | Conductance parameters associated with the corresponding membrane currents. |
| $S_i(V)$              | Voltage-dependent steady-state activation function.                         |
| $Z_i(V)$              | Voltage-dependent steady-state inactivation function.                       |
| $\tau_{X_i}(V)$       | Activation time constant for $X_i$.                                         |
| $\tau_{Y_i}(V)$       | Inactivation time constant for $Y_i$.                                       |

# Hypotheses

Based on the experimentally observed effects of TTX and ouabain on R15 activity, we formulate the following hypotheses concerning how changes in the corresponding model parameters may alter the system’s dynamical regime.

Hypothesis 1: Effect of TTX
As $g_I\downarrow0$, we expect the spike-generating transient inward current to weaken, leading to a reduction and eventual loss of action potentials and the emergence of the underlying slow membrane oscillation. We therefore aim to determine the bifurcation(s) mediating the transition from bursting to slow-wave oscillation as $g_I$ is reduced.

Hypothesis 2: Effect of Ouabain
As $|I_{\text{ep}}|\downarrow0$, we expect the loss of the hyperpolarising electrogenic pump contribution to destabilise the normal bursting regime and promote a transition toward regular beating. We therefore aim to determine the bifurcation(s) separating these regimes as the pump current is reduced.

Hypothesis 3: Effect of both drugs
Under TTX, we expect further inhibition of the electrogenic pump by ouabain to suppress the remaining slow oscillation and drive the system toward a stable equilibrium. This motivates a two-parameter bifurcation study in $(g_I, I_{\text{ep}})$, with the aim of identifying parameter regions associated with qualitatively different behaviours such as bursting, beating, slow-wave oscillation and stable equilibrium.

****

References used:
- Plant-Kim 1975: On the Mechanism Underlying Bursting in the Aplysia Abdominal Ganglion R15 Cell
- Plant-Kim 1976: Mathematical Description of a Bursting Pacemaker Neuron by a Modification of the Hodgkin-Huxley Equations
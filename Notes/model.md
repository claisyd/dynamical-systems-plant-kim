Plant and Kim (1975) proposed a slow-wave mechanism for the Aplysia R15 neuron under TTX. Their 1976 paper developed this into a quantitative modified Hodgkin–Huxley model capable of reproducing both normal bursting and the TTX-induced slow wave. We propose to use the untreated seven-dimensional Plant–Kim (1976) system as our baseline model. The effect of TTX may be represented by reducing the transient inward conductance $g_I$, with complete blockade corresponding to $g_I=0$. The variables $X_I$ and $Y_I$ then decouple from the remaining system, yielding a five-dimensional model under TTX. The effect of ouabain may similarly be modelled by reducing the magnitude of the electrogenic pump current $I_\text{ep}$, with complete pump blockade corresponding to $I_\text{ep}=0$.

Hence, our 7D dynamical system for the baseline R15 bursting model will be as follows: $$\begin{align*}
    \dot{V}&=\dfrac{1}{C}\big[(g_IX_I^3Y_I+g_T)(V_I-V)+(g_KX_K^4+g_AX_AY_A+g_PX_P)(V_K-V)+g_L(V_L-V)+I_{\text{ep}}+I_{\text{ext}}\big]\\ 
    \dot{X}_K&=\dfrac{1}{\tau_{X_K}(V)}\big[S_K(V)-X_K\big], \qquad \dot{X}_P=\dfrac{1}{\tau_{X_P}}\big[S_P(V)-X_P\big], \qquad \dot{X}_A=\dfrac{1}{\tau_{X_A}}\big[S_A(V)-X_A\big],\\ 
    \dot{Y}_A&=\dfrac{1}{\tau_{Y_A}}\big[Z_A(V)-Y_A\big], \qquad \dot{X}_I=\dfrac{1}{\tau_{X_I}(V)}\big[S_I(V)-X_I\big], \qquad \dot{Y}_I=\dfrac{1}{\tau_{Y_I}(V)}\big[Z_I(V)-Y_I\big].
\end{align*}$$

## State Variables

| Variable | Description                                                                                  |
| -------- | -------------------------------------------------------------------------------------------- |
| $V(t)$   | Membrane potential of the R15 neuron.                                                        |
| $X_K(t)$ | Activation of the fast potassium $K^+$-current.                                              |
| $X_A(t)$ | Activation of the inactivating $A$-type potassium current.                                   |
| $Y_A(t)$ | Inactivation of the $A$-type potassium current.                                              |
| $X_P(t)$ | Activation of the very slow potassium current responsible for slow feedback.                 |
| $X_I(t)$ | Activation of the fast transient inward current responsible for action-potential generation. |
| $Y_I(t)$ | Inactivation of the fast transient inward current.                                           |
<div style="page-break-after: always;"></div>


## Ionic and Applied Current Terms

| Current term       | Description                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| $g_IX_I^3Y_I(V_I-V)$ | Fast transient inward current responsible for action-potential spikes; blocked by TTX.           |
| $g_T(V_I−V)$       | Persistent inward current that remains in the presence of TTX and promotes depolarisation.       |
| $g_KX_K^4(V_K−V)$  | Fast outward $K^+$ current involved in membrane repolarisation.                                  |
| $g_AX_AY_A(V_K−V)$ | Inactivating $A$-type outward $K^+$ current, controlled by both activation and inactivation.     |
| $g_PX_P(V_K−V)$    | Very slow outward $K^+$ current that provides slow feedback underlying the membrane oscillation. |
| $g_L(V_L−V)$       | Passive leak current driving the voltage toward the leak reversal potential $V_L$.               |
| $I_\text{ep}$      | Hyperpolarising electrogenic pump current.                                                       |
| $I_\text{ext}$     | Externally applied current controlled experimentally.                                            |
## Model Parameters and Functions

| Symbol | Description |
| --- | --- |
| $C$ | Membrane capacitance. |
| $V_I,V_K,V_L$ | Reversal potentials for inward, potassium, and leak currents. |
| $g_T,g_K,g_A,g_P,g_L,g_I$ | Conductance parameters associated with the corresponding membrane currents. |
| $S_i(V)$ | Voltage-dependent steady-state activation function. |
| $Z_i(V)$ | Voltage-dependent steady-state inactivation function. |
| $\tau_{X_I}(V),\tau_{Y_I}(V),\tau_{X_K}(V)$ | Voltage-dependent time constants inherited from the modified Hodgkin--Huxley component. |
| $\tau_{X_A},\tau_{Y_A},\tau_{X_P}$ | Constant time scales for the added $A$- and $P$-current gates. |

### Plant--Kim (1976) Table I: Modified Hodgkin--Huxley parameters and functions

Conductances in Table I are quoted in units of $\mathrm{mho}\times 10^{-3}$, reversal potentials in mV, and $C$ in $\mu\mathrm{F}$.

| Quantity | Value / definition |
| --- | --- |
| $g_I$ | $4.0$ |
| $g_K$ | $0.30$ |
| $g_L$ | $0.003$ |
| $V_I$ | $30\ \mathrm{mV}$ |
| $V_K$ | $-75\ \mathrm{mV}$ |
| $V_L$ | $-40\ \mathrm{mV}$ |
| $C$ | $1\ \mu\mathrm{F}$ |
| $S_I(V)$ | $\displaystyle \frac{\alpha_m(V)}{\alpha_m(V)+\beta_m(V)}$ |
| $\tau_{X_I}(V)$ | $\displaystyle \frac{12.5}{\alpha_m(V)+\beta_m(V)}$ |
| $Z_I(V)$ | $\displaystyle \frac{\alpha_h(V)}{\alpha_h(V)+\beta_h(V)}$ |
| $\tau_{Y_I}(V)$ | $\displaystyle \frac{12.5}{\alpha_h(V)+\beta_h(V)}$ |
| $S_K(V)$ | $\displaystyle \frac{\alpha_n(V)}{\alpha_n(V)+\beta_n(V)}$ |
| $\tau_{X_K}(V)$ | $\displaystyle \frac{12.5}{\alpha_n(V)+\beta_n(V)}$ |
| $\alpha_m(V)$ | $\displaystyle \frac{0.1(-26-1.2V)}{\exp\!\left(\frac{26-V}{10}\right)-1}$ |
| $\beta_m(V)$ | $\displaystyle 4\exp\!\left(\frac{51-1.2V}{18}\right)$ |
| $\alpha_h(V)$ | $\displaystyle 0.07\exp\!\left(\frac{-51-1.21V}{20}\right)$ |
| $\beta_h(V)$ | $\displaystyle \frac{1}{\exp\!\left(\frac{21-1.3V}{10}\right)+1}$ |
| $\alpha_n(V)$ | $\displaystyle \frac{0.01(-21-1.21V)}{\exp\!\left(\frac{-21-1.21V}{10}\right)-1}$ |
| $\beta_n(V)$ | $\displaystyle 0.125\exp\!\left(\frac{-31-V}{80}\right)$ |

### Plant--Kim (1976) Table II: Additional parameters for the R15 model

Conductances in Table II are quoted in units of $\mathrm{mho}\times 10^{-3}$.

| Quantity | Value / definition |
| --- | --- |
| $g_T$ | $0.008$ |
| $g_A$ | $0.06$ |
| $g_P$ | $0.015$ |
| $S_A(V)$ | $\displaystyle \left[1+\exp\!\left(-0.08(V+45)\right)\right]^{-1}$ |
| $Z_A(V)$ | $\displaystyle \left[1+\exp\!\left(0.27(V+50)\right)\right]^{-1}$ |
| $S_P(V)$ | $\displaystyle \left[1+\exp\!\left(-0.7(V+47)\right)\right]^{-1}$ |
| $\tau_{X_A}$ | $10\ \mathrm{ms}$ |
| $\tau_{Y_A}$ | $235\ \mathrm{ms}$ |
| $\tau_{X_P}$ | $8000\ \mathrm{ms}$ |
| $I_{\mathrm{ep}}$ | $-0.22\ \mu\mathrm{A}$ |

# Hypotheses

Based on the experimentally observed effects of TTX and ouabain on R15 activity, we formulate the following hypotheses concerning how changes in the corresponding model parameters may alter the system’s dynamical regime.

Hypothesis 1: Effect of TTX
As $g_I\downarrow0$, we expect the spike-generating transient inward current to weaken, leading to a reduction and eventual loss of action potentials and the emergence of the underlying slow membrane oscillation. We therefore aim to determine the bifurcation(s) mediating the transition from bursting to slow-wave oscillation as $g_I$ is reduced.

Hypothesis 2: Effect of Ouabain
As $|I_{\text{ep}}|\downarrow0$, we expect the loss of the hyperpolarising electrogenic pump contribution to destabilise the normal bursting regime and promote a transition toward regular beating. We therefore aim to determine the bifurcation(s) separating these regimes as the pump current is reduced.

Hypothesis 3: Effect of both drugs
Under TTX, we expect further inhibition of the electrogenic pump by ouabain to suppress the remaining slow oscillation and drive the system toward a stable equilibrium. This motivates a two-parameter bifurcation study in $(g_I, I_{\text{ep}})$, with the aim of identifying parameter regions associated with qualitatively different behaviours such as bursting, beating, slow-wave oscillation and stable equilibrium.

****

References used:
- Plant-Kim 1975: On the Mechanism Underlying Bursting in the Aplysia Abdominal Ganglion R15 Cell
- Plant-Kim 1976: Mathematical Description of a Bursting Pacemaker Neuron by a Modification of the Hodgkin-Huxley Equations
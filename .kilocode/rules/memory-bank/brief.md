# Model Objective

**Project introduction:** Mental health promotion programmes are categorized into universal, selective, and indicated based on their intended audiences. A universal programme is intended to engage all the general audiences. A selective programme is indicated for groups at a higher risk of having mental health disorders. An indicated programme is reserved for patients with specific health conditions. Physical activity, psychosocial intervention, and lifestyle modification had small to moderate effect in promoting mental health.

**Research question:** How cost-effective is the workplace-based universal mental health promotion programmes on the community level?

**Project impetus:** Mental health public intervention programmes have been widely investigated, mainly as school-based and workplace-based mental health promotion programmes. Due to quality and methodological differences of the programmes, health-economics evaluation of universal, selective, and indicated mental health promotion programmes remains a controversial affair.

**Project objective:** This project aim to implement an agent-based model to evaluate the comparison of cost-effectiveness in the universal, selective, and indicated mental health promotion programmes. To resolve programme differences and provide better generalisability, our model will consider several common scenarios when evaluating the cost-effectiveness. Scenarios will be based on intervention approaches, baseline mental disorder prevalence, baseline stressor indicators, and baseline psychopharmaca uses.

**Project results:** The output of this project is a protocol document and a scientific manuscript. The protocol document will highlight model conceptualization, parameterization, and validation. The scientific manuscript will provide a comparison of three mental health promotion approaches by detailing cost and benefit associated with reaching a certain level of resilience.

**Outside scope of the project:** This model will only consider resilience as a residual between stressor and mental health outcome. We limit the scope of stressor to Perceived Stress Scale-10, depressive disorder to Patient Health Questionnaire-9, anxiety to Generalized Anxiety Disorder Scale-7, and burnout to Maslach Burnout Inventory.

**Effects:** Compare the cost-effectiveness of workplace-based universal mental health promotion programmes in improving resilience and reducing the prevalence of major depressive disorder and anxiety disorder.

# Model Explanation

**Perceive Stress:**

1. **Life event is initialized** with three continuous attributes (normalized to $[0,1]$):

   * controllability $c\in[0,1]$ (how much the agent can influence the event)
   * predictability $p\in[0,1]$ (how foreseeable the event is)
   * overload $o\in[0,1]$ (perceived burden / intensity relative to capacity)

2. **Apply Weight function** $W(c,p,o)$: this maps the triple $(c,p,o)$ to a single scalar $z\in\mathbb{R}$ that then produces **challenge** and **hindrance** stress values on a shared scale. Challenge and hindrance are *opposite polar* of the same scale:

   $$
   \begin{aligned}
   z_i &= g(c_i,p_i,o_i) \\
   \text{challenge}_i &= f_{\text{ch}}(z_i)\in[0,1] \\
   \text{hindrance}_i &= 1 - \text{challenge}_i
   \end{aligned}
   $$

   The mapping is built so that:

   * $c\uparrow, p\uparrow, o\downarrow \Rightarrow \text{challenge}\uparrow$ (100% challenge when $c=1,p=1,o=0$)
   * $c\downarrow, p\downarrow, o\uparrow \Rightarrow \text{hindrance}\uparrow$ (100% hindrance when $c=0,p=0,o=1$)

   A convenient parametric choice (used below) is:

   $$
   z = \omega_c c + \omega_p p - \omega_o o + b,
   $$

   then

   $$
   \text{challenge} = \sigma\big( \gamma \cdot z \big),
   $$

   with $\sigma(x)=\frac{1}{1+e^{-x}}$ and $\gamma$ chosen so $\sigma$ pushes extreme triples to near 0 or 1. Finally set $\text{hindrance}=1-\text{challenge}$.

3. **Apply Agent Threshold (n23):** the rule deciding whether the agent is “stressed out” takes *two* inputs:

   * the appraised **overall stress** $L$ (from event magnitude + challenge/hindrance mapping), and
   * the **hindrance** component (because hindrance specifically reduces coping capacity / threshold).
     A compact function:

   $$
   \text{stressed}_i = \mathbf{1}\Big( L_i > T^{\text{stress}}_i - \lambda_H \cdot \text{hindrance}_i \Big)
   $$

   where $\lambda_H\ge0$ scales how much hindrance lowers the effective threshold. Conversely, challenge increases threshold via:

   $$
   T^{\text{stress,eff}}_i = T^{\text{stress}}_i + \lambda_C \cdot \text{challenge}_i - \lambda_H\cdot\text{hindrance}_i.
   $$

   This makes the influence explicit: challenge raises the agent’s momentary stress buffer (more likely to handle event), hindrance reduces it.

# Concrete Functional Forms

Use normalized variables in $[0,1]$.

1. **Apply-weight function**

   $$
   z_i = \omega_c c_i + \omega_p p_i - \omega_o o_i + b
   $$

   $$
   \text{challenge}_i = \sigma(\gamma z_i) \qquad\text{(sigmoid)} 
   $$

   $$
   \text{hindrance}_i = 1 - \text{challenge}_i
   $$

   Typical parameter choices: $\omega_c=\omega_p=\omega_o=1$, $b=0$, and $\gamma=6$ (steep sigmoid to push extremes). You can treat $\omega$s and $\gamma$ as tunable.

2. **Overall appraised load $L$**
   Combine event magnitude $s\in[0,1]$ and the challenge/hindrance polarity:

   $$
   L_i = s \cdot \big( \underbrace{\alpha_{\text{ch}} \cdot (1-\text{challenge}_i) + \alpha_{\text{hd}} \cdot \text{hindrance}_i}_{\text{weighting of how challenge/hindrance map to net stress}} \big)
   $$

   Here $\alpha_{\text{ch}} \ll \alpha_{\text{hd}}$ if challenge is less stressful per unit magnitude than hindrance. A compact alternative is:

   $$
   L_i = s \cdot (1 + \delta\cdot(\text{hindrance}_i-\text{challenge}_i)).
   $$

3. **Effective stress threshold**

   $$
   T^{\text{eff}}_i = T^{\text{stress}}_i + \lambda_C\cdot \text{challenge}_i - \lambda_H\cdot \text{hindrance}_i
   $$

   Agent becomes stressed if $L_i > T^{\text{eff}}_i$.

# Mapping to PSS-10

Create three composite scores per agent from the PSS-10-like item responses:
* **Controllability $c$**: average of items/concepts about perceived control over events.
* **Predictability $p$**: average of items/concepts about predictability/unexpectedness.
* **Overload $o$**: average of items/concepts about feeling overwhelmed/overloaded.

# Parameterization Plan

## Decide variable scales & normalization

Use normalized scales $[0,1]$ for all agent-internal variables (resources $R$, distress $D$, c/p/o). This simplifies interpretation and parameter ranges.

## Parameters to specify / estimate

**Appraisal & perception**

* $\omega_c, \omega_p, \omega_o, b, \gamma$ (apply-weight function)
* $\lambda_C, \lambda_H$ (how challenge/hindrance shift threshold)

**Event process**

* Shock arrival rate $\lambda_{shock}$ (Poisson)
* Shock magnitude distribution $s\sim \text{Beta}(\alpha_s,\beta_s)$ or truncated Normal on \[0,1]

**Resource dynamics**

* initial resources $R_i(0)$ distribution (e.g., Beta)
* regeneration rate $\gamma_R\in[0,1]$ per day
* cost scalar $\kappa$, cost function exponent $\gamma_c$

**Protective factor efficacy**

* $\alpha_{soc},\alpha_{fam},\alpha_{int},\alpha_{cap}$ (per-unit efficacy in reducing distress)
* $\rho_{soc},\rho_{fam},\rho_{int},\rho_{cap}$ (replenishment per unit allocated)

**Behavioral decision**: Softmax temperature $\beta$ (decision stochasticity)

**Network**

* network type (ER / small-world / scale-free), mean degree $\langle k\rangle$
* helping probability $p_{\text{help}}$, support size distribution

**Adaptation**: Adaptation threshold $T^{\text{adapt}}$, learning rate $\eta_{\text{adapt}}$, rewiring prob $p_{\text{rewire}}$

## How to sample the parameter space

1. **Exploratory sweeps:** Latin Hypercube Sampling (LHS) across the most influential parameters: $\gamma_R, \kappa, \alpha_{soc}, \rho_{soc}, \lambda_{shock}, \omega_c,\omega_p,\omega_o,\gamma,\lambda_C,\lambda_H$. Use $n=200-1000$ sampled parameter sets and run $m=20$ stochastic replicates each.
2. **Global sensitivity analysis:** compute Sobol indices or use variance-based decomposition on scalar resilience metrics (return time $\tau$, basin stability, FTLE, area under recovery curve). This identifies which parameters matter most.
3. **Local sensitivity / PRCC:** compute partial rank correlation coefficients across parameter sweeps to get parameter–output monotonic relationships.

## Calibration to qualitative scoping-review insights

When lacking longitudinal data, calibrate to *pattern targets* extracted from available articles. Examples of pattern targets:

* Typical recovery time range (e.g., “most recover within X–Y days”) — translate to target $\tau$ percentile;
* Proportion resilient after standardized shock sequences (e.g., “\~40% show minimal functional decline”) — target basin stability;
* Observed intervention effect sizes (e.g., training coping increases resilience by Z%) — use as constraints.

Use **Approximate Bayesian Computation (ABC)** or **Simulated Method of Moments (SMM)**:

* Define distance metric between simulated pattern statistics and scoping-review target statistics;
* Sample parameter space (via LHS or prior draws) and accept parameter vectors where distance < threshold;
* Resulting posterior over parameters gives plausible parameter regions consistent with qualitative knowledge.

## Robustness & validation checks

* Re-run top plausible parameter sets under alternative network topologies and shock regimes.
* Check for boundary / degenerate behavior (e.g., resources always deplete to zero, agents never adapt).
* Report results across ensemble runs and quantify stochastic variability (confidence bands).

# Suggested numeric parameter table

All variables normalized to $[0,1]$ unless noted. These are modifiable starting recommendations.

| Parameter                              |                                        Description |         Default | Reasonable range     |
| -------------------------------------- | -------------------------------------------------: | --------------: | -------------------- |
| $R_i(0)$                               |                       Initial resource (per agent) | 0.6 (Beta-like) | 0.2 - 1.0            |
| $\gamma_R$                             |              Resource passive regeneration per day |            0.05 | 0.0 - 0.2            |
| $\kappa$                               |                Cost scalar of allocating resources |            0.15 | 0.01 - 0.5           |
| $\gamma_c$                             |                          Cost exponent (convexity) |             1.5 | 1.0 - 3.0            |
| $\alpha_{soc}$                         |    Efficacy of social support in reducing distress |             0.4 | 0.1 - 0.8            |
| $\rho_{soc}$                           | Replenishment contribution per social support unit |            0.08 | 0.0 - 0.3            |
| $\alpha_{fam}$                         |                         Efficacy of family support |             0.5 | 0.1 - 0.9            |
| $\rho_{fam}$                           |                          Replenishment from family |            0.12 | 0.0 - 0.4            |
| $\alpha_{int}$                         |                    Efficacy of formal intervention |             0.6 | 0.2 - 0.95           |
| $\rho_{int}$                           |              Replenishes resources (e.g., therapy) |            0.15 | 0.0 - 0.5            |
| $\alpha_{cap}$                         |           Efficacy of psychological capital (self) |            0.35 | 0.05 - 0.8           |
| $\rho_{cap}$                           |                        Self-recovery replenishment |            0.04 | 0.0 - 0.2            |
| $T^{\text{stress}}_i$                  |                          Baseline stress threshold |             0.5 | 0.2 - 0.9            |
| $\lambda_C$                            |            Scaling of challenge→threshold increase |            0.15 | 0.0 - 0.5            |
| $\lambda_H$                            |            Scaling of hindrance→threshold decrease |            0.25 | 0.0 - 0.6            |
| $\omega_c,\omega_p,\omega_o$           |                          apply-weight coefficients |        1.0 each | 0.2 - 2.0            |
| $\gamma$                               |                  Sigmoid steepness in apply-weight |             6.0 | 1.0 - 12.0           |
| $\lambda_{shock}$                      |            Poisson rate of shock per agent per day |            0.02 | 0.005 - 0.2          |
| Shock magnitude $s$                    |                           mean magnitude of shocks |             0.4 | 0.05 - 0.9           |
| $\beta$                                | Softmax inverse temperature (decision determinism) |             8.0 | 1 - 50               |
| Network mean degree $\langle k\rangle$ |                                  average neighbors |               6 | 2 - 30               |
| $p_{\text{help}}$                      |     per-neighbor probability of helping when asked |            0.25 | 0 - 0.8              |
| $p_{\text{rewire}}$                    |               rewiring prob per epoch (adaptation) |            0.01 | 0 - 0.2              |
| $T^{\text{adapt}}$                     |   adaptation trigger threshold (repeated breaches) | 3 events in 14d | 1 - 14 events in 30d |

Notes:

* Defaults are deliberately moderate (not extreme).
* Use multiple orders of magnitude in the sampling range for parameters with low certainty (e.g., $\kappa,\gamma_R$). For these low-certainty parameters:
  * Instead of guessing a narrow range (say 0.05-0.15), it’s better to sample across orders of magnitude (e.g., 0.001-0.5).
  * This avoids “baking in” an assumption prematurely and allows you to discover which regimes of parameter space produce realistic dynamics.

| Parameter                                                                           | Evidence & Supporting Articles                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **$R\_i(0)$ (Initial resource)**                                                  | Hobfoll, S.E. (1989). *Conservation of Resources: A New Attempt at Conceptualizing Stress* (Am. Psychologist, 44:513–524). Defines resources as finite, with initial levels varying widely across individuals, consistent with 0.2–1.0 range. DOI: [10.1037/0003-066X.44.3.513](https://doi.org/10.1037/0003-066X.44.3.513)                                |
| **$\gamma\_R$ (Resource regeneration)**                                           | Kalisch, R. et al. (2015). *A conceptual framework for the neurobiological study of resilience* (Behav. Brain Sci., 38\:e92). Discusses ongoing recovery processes at daily/weekly scales, supporting passive regeneration in range 0.0–0.2. DOI: [10.1017/S0140525X1400082X](https://doi.org/10.1017/S0140525X1400082X)                                   |
| **$\kappa$, $\gamma\_c$ (Cost scalar & exponent)**                              | Gross, J.J. (2015). *Emotion regulation: Current status and future prospects* (Psychological Inquiry, 26:1–26). Notes cognitive/effort costs of regulation are nonlinear, motivating convex cost forms. DOI: [10.1080/1047840X.2014.940781](https://doi.org/10.1080/1047840X.2014.940781)                                                                  |
| **$\alpha\_{soc}, \rho\_{soc}$ (Social support)**                                 | Harandi, T.F. et al. (2017). *The effect of social support on mental health: A meta-analysis* (Epidemiol. Health, 39\:e2017037). Finds moderate pooled effect sizes (r ≈ .2–.3), consistent with defaults 0.4 efficacy, 0–0.3 replenishment. DOI: [10.4178/epih.e2017037](https://doi.org/10.4178/epih.e2017037)                                           |
| **$\alpha\_{fam}, \rho\_{fam}$ (Family support)**                                 | Van Harmelen, A.L. et al. (2017). *Friendship and family support during adolescence...* (PLoS One, 12\:e0181462). Shows family support strongly buffers stress effects, with effect sizes often higher than peers. DOI: [10.1371/journal.pone.0181462](https://doi.org/10.1371/journal.pone.0181462)                                                       |
| **$\alpha\_{int}, \rho\_{int}$ (Formal interventions)**                           | Robertson, I.T. et al. (2015). *Resilience training in the workplace* (J. Occup. Health Psychol., 20:397–411). Reports medium-to-large effect sizes (g \~0.5–0.9), supporting range 0.2–0.95. DOI: [10.1037/a0038763](https://doi.org/10.1037/a0038763)                                                                                                    |
| **$\alpha\_{cap}, \rho\_{cap}$ (Psychological capital)**                          | Luthans, F. et al. (2007). *Psychological capital: Measurement and relationship with performance* (Personnel Psychology, 60:541–572). PsyCap predictive effects typically r = .2–.4; PsyCap shown trainable. DOI: [10.1111/j.1744-6570.2007.00083.x](https://doi.org/10.1111/j.1744-6570.2007.00083.x)                                                     |
| **$T^{stress}\_i$, $\lambda\_C$, $\lambda\_H$ (Stress thresholds)**           | Lazarus, R.S. & Folkman, S. (1984). *Stress, Appraisal, and Coping.* Defines challenge/hindrance appraisals shifting effective thresholds. No direct numbers, but supports ranges (0.2–0.9 baseline, ±0.5 modulation). ISBN: 9780826141927                                                                                                                 |
| **$\omega\_c,\omega\_p,\omega\_o$, $\gamma$ (Appraisal coefficients, sigmoid)** | Cohen, S. et al. (1983). *A global measure of perceived stress* (J. Health Soc. Behav., 24:385–396). Source of PSS-10 instrument; controllability, predictability, overload mapped to appraisal weights. DOI: [10.2307/2136404](https://doi.org/10.2307/2136404)                                                                                           |
| **$\lambda\_{shock}$, $s$ (Shock rate and magnitude)**                          | Bonanno, G.A. et al. (2015). *Resilience to loss and chronic grief: Trajectories of adjustment...* (J. Pers. Soc. Psychol., 83:1150–1164). Empirical distribution of life stressors shows events cluster around monthly/yearly; justifies Poisson with mean 0.005–0.2/day. DOI: [10.1037/0022-3514.83.5.1150](https://doi.org/10.1037/0022-3514.83.5.1150) |
| **$\beta$ (Decision determinism)**                                                | Camerer, C.F. (1999). *Behavioral economics: Loss aversion...* (J. Econ. Lit., 37:41–58). Softmax/logit choice models with inverse temperature in range 1–50 are standard in decision modeling. DOI: [10.1257/jel.37.4.1107](https://doi.org/10.1257/jel.37.4.1107)                                                                                        |
| **Network mean degree $\langle k \rangle$**                                       | Christakis, N.A. & Fowler, J.H. (2007). *The spread of obesity in a large social network...* (NEJM, 357:370–379). Average network degree in large cohorts \~6; range 2–30 realistic for social-ecological ABMs. DOI: [10.1056/NEJMsa066082](https://doi.org/10.1056/NEJMsa066082)                                                                          |
| **$p\_{\text{help}}$ (Helping probability)**                                      | Kaniasty, K. & Norris, F.H. (1993). *A test of the social support deterioration model...* (J. Pers. Soc. Psychol., 64:395–408). Probability of receiving aid varies widely (0–0.8 observed across contexts). DOI: [10.1037/0022-3514.64.3.395](https://doi.org/10.1037/0022-3514.64.3.395)                                                                 |
| **$p\_{\text{rewire}}$ (Rewiring probability)**                                   | Gross, T. & Blasius, B. (2008). *Adaptive coevolutionary networks: a review* (J. R. Soc. Interface, 5:259–271). Empirical and model-based rewiring often occurs at 0–0.2 per epoch in adaptive networks. DOI: [10.1098/rsif.2007.1229](https://doi.org/10.1098/rsif.2007.1229)                                                                             |
| **$T^{adapt}$ (Adaptation trigger)**                                              | Kalisch, R. et al. (2017). *A conceptual framework for resilience research* (Behav. Brain Sci., 38\:e92). Suggests adaptation occurs after repeated threshold breaches (1–14 exposures typical), supporting the 3-in-14d default. DOI: [10.1017/S0140525X1500084X](https://doi.org/10.1017/S0140525X1500084X)                                              |

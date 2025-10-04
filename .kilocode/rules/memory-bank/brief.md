# Model Objective

**Project introduction:** Mental health promotion programmes are categorized into universal, selective, and indicated based on their intended audiences. A universal programme is intended to engage all the general audiences. A selective programme is indicated for groups at a higher risk of having mental health disorders. An indicated programme is reserved for patients with specific health conditions. Physical activity, psychosocial intervention, and lifestyle modification had small to moderate effect in promoting mental health.

**Research question:** How cost-effective is the workplace-based universal mental health promotion programmes on the community level?

**Project impetus:** Mental health public intervention programmes have been widely investigated, mainly as school-based and workplace-based mental health promotion programmes. Due to quality and methodological differences of the programmes, health-economics evaluation of universal, selective, and indicated mental health promotion programmes remains a controversial affair.

**Project objective:** This project aims to implement an agent-based model to evaluate the comparison of cost-effectiveness in the universal, selective, and indicated mental health promotion programmes. To resolve programme differences and provide better generalisability, our model considers several common scenarios when evaluating the cost-effectiveness. Scenarios are based on intervention approaches, baseline mental disorder prevalence, baseline stressor indicators, and baseline psychopharmaca uses.

**Project results:** The output of this project is a protocol document and a scientific manuscript. The protocol document will highlight model conceptualization, parameterization, and validation. The scientific manuscript will provide a comparison of three mental health promotion approaches by detailing cost and benefit associated with reaching a certain level of resilience.

**Current Implementation Status:** The model is fully implemented as a production-ready Python ABM using the Mesa framework with comprehensive PSS-10 integration, advanced configuration management, extensive testing infrastructure, and research-grade documentation.

**Outside scope of the project:** This model considers resilience as a residual between stressor and mental health outcome. We limit the scope of stressor to Perceived Stress Scale-10, depressive disorder to Patient Health Questionnaire-9, anxiety to Generalized Anxiety Disorder Scale-7, and burnout to Maslach Burnout Inventory.

**Effects:** Compare the cost-effectiveness of workplace-based universal mental health promotion programmes in improving resilience and reducing the prevalence of major depressive disorder and anxiety disorder.

# Model Explanation

**Perceive Stress:**

1. **Life event is initialized** with two continuous attributes (normalized to $[0,1]$):

   * controllability $c\in[0,1]$ (how much the agent can influence the event)
   * overload $o\in[0,1]$ (perceived burden / intensity relative to capacity)

2. **Apply Weight function** $W(c,o)$: this maps the triple $(c,o)$ to a single scalar $z\in\mathbb{R}$ that then produces **challenge** and **hindrance** stress values on a shared scale. Challenge and hindrance are *opposite polar* of the same scale:

   $$
   \begin{aligned}
   z_i &= g(c_i,o_i) \\
   \text{challenge}_i &= f_{\text{ch}}(z_i)\in[0,1] \\
   \text{hindrance}_i &= 1 - \text{challenge}_i
   \end{aligned}
   $$

   The mapping is built so that:

   * $c\uparrow, o\downarrow \Rightarrow \text{challenge}\uparrow$ (100% challenge when $c=1,p=1,o=0$)
   * $c\downarrow, o\uparrow \Rightarrow \text{hindrance}\uparrow$ (100% hindrance when $c=0,p=0,o=1$)

   A convenient parametric choice (used below) is:

   $$
   z = \omega_c c - \omega_o o + b,
   $$

   then

   $$
   \text{challenge} = \sigma\big( \gamma \cdot z \big),
   $$

   with $\sigma(x)=\frac{1}{1+e^{-x}}$ and $\gamma$ chosen so $\sigma$ pushes extreme triples to near 0 or 1. Finally set $\text{hindrance}=1-\text{challenge}$.

3. **Apply Agent Threshold (n23):** the rule deciding whether the agent is “stressed out” takes *two* inputs:
   * The total PSS-10 score
   * The total PSS-10 threshold
   * Stressed out is when the PSS-10 score >= PSS-10 threhold

# Concrete Functional Forms

Use normalized variables in $[0,1]$.

1. **Apply-weight function**

   $$
   z_i = \omega_c c_i - \omega_o o_i + b
   $$

   $$
   \text{challenge}_i = \sigma(\gamma z_i) \qquad\text{(sigmoid)} 
   $$

   $$
   \text{hindrance}_i = 1 - \text{challenge}_i
   $$

   Typical parameter choices: $\omega_c=\omega_o=1$, $b=0$, and $\gamma=6$ (steep sigmoid to push extremes). You can treat $\omega$s and $\gamma$ as tunable.

2. **Overall appraised load $L$**
   Combine event magnitude $s\in[0,1]$ and the challenge/hindrance polarity:

   $$
   L_i = s \cdot \big( \underbrace{\alpha_{\text{ch}} \cdot (1-\text{challenge}_i) + \alpha_{\text{hd}} \cdot \text{hindrance}_i}_{\text{weighting of how challenge/hindrance map to net stress}} \big)
   $$

   Here $\alpha_{\text{ch}} \ll \alpha_{\text{hd}}$ if challenge is less stressful per unit magnitude than hindrance. A compact alternative is:

   $$
   L_i = s \cdot (1 + \delta\cdot(\text{hindrance}_i-\text{challenge}_i)).
   $$

# Mapping to PSS-10

The PSS-10 (Perceived Stress Scale-10) is fully integrated into the model with:
* **Complete bifactor model implementation** - orthogonal general stress factor and specific factors
* **Dimension score generation** - controllability, overload, and predictability components
* **Empirical validation testing** - comprehensive test suite with 4 specialized PSS-10 test files
* **Configuration integration** - PSS-10 parameters fully configurable via environment variables

PSS-10 creates three composite scores per agent from item responses:
* **Controllability $c$**: average of items/concepts about perceived control over events
* **Overload $o$**: average of items/concepts about feeling overwhelmed/overloaded

# Current Implementation Status

## Core ABM Implementation
- **Framework:** Fully functional Python ABM using Mesa framework with all theoretical components implemented
- **Agent Model:** Complete Person class with all specified state variables (resources, distress, stress threshold, network position, protective factors) and behaviors
- **Network Structure:** Watts-Strogatz small-world network with configurable parameters via environment variables
- **Stress Processing:** Complete stress event generation, challenge/hindrance appraisal, and threshold evaluation mechanisms
- **Resource Dynamics:** Full protective factor allocation system with social support, family support, formal interventions, and psychological capital
- **Social Interactions:** Network-based support system with help requests and mutual aid mechanisms
- **Adaptation Mechanisms:** Learning and network rewiring based on stress experiences

## PSS-10 Integration
- **Bifactor Model:** Complete implementation with orthogonal general stress factor and specific factors
- **Dimension Generation:** Automated creation of controllability, overload, and predictability scores
- **Empirical Validation:** Comprehensive testing framework with 4 specialized PSS-10 test files
- **Configuration Support:** Full parameter integration with environment variable system

## Configuration Management System
- **Environment Variables:** 50+ configurable parameters with type conversion and validation
- **Parameter Documentation:** Comprehensive CONFIGURATION.md with usage scenarios and best practices
- **Shell Utilities:** Automated configuration extraction and environment file synchronization scripts
- **Research Integration:** Support for parameter sweeps, sensitivity analysis, and calibration workflows

## Testing Infrastructure
- **Test Suite:** 30 specialized test files covering unit tests, integration tests, configuration validation, mechanism testing, and PSS-10 validation
- **CI/CD Pipeline:** GitHub Actions workflow with automated testing and coverage reporting
- **Coverage Requirements:** 80% minimum test coverage with HTML report generation and Codecov integration
- **Debug Tools:** Specialized debugging utilities for threshold evaluation and stress processing troubleshooting
- **Test Categories:** Unit tests (`test_math_utils.py`, `test_stress_utils.py`, `test_affect_utils.py`), Integration tests (7 files), Configuration tests (5 files), Mechanism tests (4 files), Validation tests (5 files), PSS-10 tests (4 files)

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
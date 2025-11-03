# Resilience Dynamics and Triggering Events

_See [`.kilocode/rules/math/notation.md`](../../.kilocode/rules/math/notation.md) for symbol definitions and conventions._

## Overview

Resilience represents an individual's capacity to adapt and recover from stress events. The model examines multiple factors that influence resilience levels, including stress event outcomes, social support, protective factors, and natural tendencies to return to baseline equilibrium.

## Core Resilience State

### State Variables

- **Resilience Level**: Current capacity to adapt and recover
- **Baseline Resilience**: An individual's natural equilibrium point
- **Consecutive Hindrances**: Counter for cumulative hindrance events
- **Stress Breach Count**: Counter for threshold breaches triggering adaptation

## Triggering Events

### Stress Event Outcomes

Resilience changes are triggered by stress event processing outcomes:

#### Successful Coping
**Trigger Condition**: Successfully managing stress events

**Resilience Impact**:
Successfully coping with challenging events builds resilience significantly, while even hindrance events provide minor learning opportunities when handled successfully.

#### Failed Coping
**Trigger Condition**: Unsuccessful stress management

**Resilience Impact**:
Failed coping with hindrance events significantly depletes resilience, while challenge events have minimal negative impact when coping fails.

### Social Support Reception

**Trigger Condition**: Receiving social support from social connections

**Resilience Impact**:
Social support provides a boost to resilience, with the magnitude depending on the quality and effectiveness of the support received.

### Cumulative Overload Effects

**Trigger Condition**: Experiencing multiple consecutive hindrance events

**Resilience Impact**:
When hindrance events accumulate beyond a threshold, they create an overload effect that reduces resilience, representing how chronic stress can overwhelm coping capacity.

## Resilience Change Mechanisms

### Challenge-Hindrance Based Changes

The core resilience update mechanism integrates challenge and hindrance effects, with different outcomes depending on coping success:

**Challenge-Hindrance Resilience Effect:**

$$\Delta \mathfrak{R}_{\chi\zeta} = \begin{cases}
0.3 \cdot \chi + 0.1 \cdot \zeta & \text{if coping successful} \\
-0.1 \cdot \chi - 0.4 \cdot \zeta & \text{if coping failed}
\end{cases}$$

Where:
- $\Delta \mathfrak{R}_{\chi\zeta}$ is resilience change from challenge/hindrance
- $\chi \in [0,1]$ is challenge component
- $\zeta \in [0,1]$ is hindrance component

**Implementation**: [`compute_challenge_hindrance_resilience_effect()`](../../src/python/affect_utils.py#L471-L510) in `affect_utils.py`

- **Successful Coping**: Challenge events provide significant resilience building, while hindrance events offer minor benefits
- **Failed Coping**: Hindrance events significantly deplete resilience, while challenge events have minimal negative impact

### Protective Factor Boost

**Mechanism**: Resources allocated to protective factors provide resilience benefits, with greater benefits when resilience is low and most needed.

**Protective Factor Resilience Boost:**

$$\Delta \mathfrak{R}_p = \sum_{f \in F} e_f \cdot (\mathfrak{R}_{\text{0}} - \mathfrak{R}_c) \cdot \theta_{\text{boost}}$$

Where:
- $\Delta \mathfrak{R}_p$ is resilience boost from protective factors
- $F = \{\mathrm{soc}, \mathrm{fam}, \mathrm{int}, \mathrm{cap}\}$ is set of protective factors
- $e_f \in [0,1]$ is efficacy of factor $f$
- $\mathfrak{R}_{\text{0}} \in [0,1]$ is baseline resilience
- $\mathfrak{R}_c \in [0,1]$ is current resilience
- $\theta_{\text{boost}} > 0$ is boost rate parameter

**Protective Factors**:
- **Social Support**: Efficacy in providing emotional support
- **Family Support**: Efficacy of family relationships
- **Formal Intervention**: Efficacy of professional help
- **Psychological Capital**: Self-efficacy and coping skills

### Homeostatic Adjustment

**Mechanism**: Resilience tends to return to baseline levels over time, representing natural psychological adaptation and recovery processes.

**Homeostatic Resilience Adjustment:**

$$\mathfrak{R}_{t+1} = \mathfrak{R}_t + \lambda_{\text{resilience}} \cdot (\mathfrak{R}_{\text{0}} - \mathfrak{R}_t)$$

Where:
- $\mathfrak{R}_t \in [0,1]$ is current resilience
- $\lambda_{\text{resilience}} \in [0,1]$ is homeostatic rate
- $\mathfrak{R}_{\text{0}} \in [0,1]$ is baseline resilience

## Coping Success Determination

### Base Coping Probability

**Foundation**: Each individual has a baseline coping success probability that represents their inherent ability to manage stress.

**Coping Probability Equation:**

$$p_{\mathrm{coping}} = p_b + \theta_{\text{cope,}\chi} \cdot \chi - \theta_{\text{cope,}\zeta} \cdot \zeta + \delta_{\text{cope,soc}} \cdot \frac{1}{k} \sum_{j=1}^k A_j$$

**Coping Success Determination:**

$$\mathrm{coped\ successfully} = \begin{cases}
1 & \text{if } U \sim \mathcal{U}(0,1) < p_{\mathrm{coping}} \\
0 & \text{otherwise}
\end{cases}$$

Where:
- $p_b \in [0,1]$ is base coping probability
- $\theta_{\text{cope,}\chi} > 0$ is challenge bonus parameter
- $\theta_{\text{cope,}\zeta} > 0$ is hindrance penalty parameter
- $\delta_{\text{cope,soc}} \in [0,1]$ is social influence factor
- $\chi \in [0,1]$ is challenge component
- $\zeta \in [0,1]$ is hindrance component
- $A_j \in [-1,1]$ is neighbor $j$'s affect
- $U$ is uniform random variable

**Implementation**: [`compute_coping_probability()`](../../src/python/affect_utils.py#L424-L468) in `affect_utils.py`

### Challenge-Hindrance Effects

**Challenge Influence**:
- **Increases coping probability**: Challenging events motivate better coping
- **Effect**: Challenge events enhance coping likelihood

**Hindrance Influence**:
- **Decreases coping probability**: Hindrance events overwhelm coping capacity
- **Effect**: Hindrance events reduce coping likelihood

### Social Influence on Coping

**Mechanism**: The emotional state of social connections influences coping success, with positive social environments improving coping and negative environments hindering it.

## Resource Dynamics Integration

### Resource Consumption for Coping

**Successful Coping**:
Successful coping requires resource investment, representing the energy and effort needed to manage stress effectively.

**Failed Coping**:
- No resource consumption for failed coping attempts

### Resource Regeneration

**Passive Regeneration**:
Resources naturally regenerate toward maximum capacity over time, representing rest and natural recovery processes.

**Affect Influence on Regeneration**:
Positive emotional states enhance resource regeneration, while negative states may slow the process.

## Network Adaptation Triggers

### Stress Breach Adaptation

**Trigger Condition**:
When individuals experience repeated stress beyond a threshold, they begin adapting their social connections to better suit their needs.

**Adaptation Mechanisms**:
1. **Rewiring**: Connect to similar individuals (homophily)
2. **Support Effectiveness**: Strengthen ties with effective supporters
3. **Stress-Based Homophily**: Prefer connections with similar stress levels

**Network Adaptation Condition:**

$$\mathrm{adapt\ network} = \begin{cases}
1 & \text{if } c_{\text{breach}} \geq \eta_{\mathrm{adapt}} \\
0 & \text{otherwise}
\end{cases}$$

### Consecutive Hindrances Tracking

**Mechanism**:
The system tracks consecutive hindrance events, with cumulative effects when hindrances persist over time.

**Overload Trigger**:
When hindrance events accumulate beyond a threshold, they create an overload effect that reduces resilience.

**Cumulative Overload Effect:**

$$\Delta \mathfrak{R}_o = \begin{cases}
-0.2 \cdot \min\left(\frac{h_c}{\eta_{\text{res,overload}}}, 2.0\right) & \text{if } h_c \geq \eta_{\text{res,overload}} \\
0 & \text{otherwise}
\end{cases}$$

Where:
- $\Delta \mathfrak{R}_o$ is overload resilience change
- $h_c \in \mathbb{N}$ is consecutive hindrances count
- $\eta_{\text{res,overload}} \in \mathbb{N}$ is overload threshold

## Daily Reset Mechanisms

### End-of-Day Processing

**Daily Resilience Reset**:
At the end of each day, resilience is adjusted toward baseline levels, representing natural psychological adaptation and recovery processes.

**Stress Event Summary**:
The system tracks daily stress events, including average stress levels, maximum stress, number of events, and coping success rates.

## Integration with Other Systems

### Affect System Integration

**Bidirectional Influence**:
- **Affect → Resilience**: Positive affect improves resource regeneration
- **Resilience → Affect**: High resilience provides coping confidence

### Stress System Integration

**Stress Event Processing**:
1. Event occurs → Appraisal → Coping attempt → Resilience update
2. Resource consumption based on coping outcome
3. Network adaptation based on stress patterns

### Social Network Integration

**Social Influence on Resilience**:
- Neighbor affect influences coping probability
- Social support provides direct resilience boost
- Network adaptation based on support effectiveness

**Integrated Resilience Update:**

$$\mathfrak{R}_{t+1} = \mathfrak{R}_t + \Delta \mathfrak{R}_{\chi\zeta} + \Delta \mathfrak{R}_p + \Delta \mathfrak{R}_o + \Delta \mathfrak{R}_s + \lambda_{\text{resilience}} \cdot (\mathfrak{R}_{\text{0}} - \mathfrak{R}_t)$$

**Final Resilience Clamping:**

$$\mathfrak{R}_{t+1} = \mathrm{clamp}(\mathfrak{R}_{t+1}, 0, 1)$$

Where:
- $\mathfrak{R}_t \in [0,1]$ is current resilience
- $\Delta \mathfrak{R}_{\chi\zeta}$ is challenge-hindrance effect
- $\Delta \mathfrak{R}_p$ is protective factor boost
- $\Delta \mathfrak{R}_o$ is overload effect
- $\Delta \mathfrak{R}_s$ is social support effect
- $\lambda_{\text{resilience}} \in [0,1]$ is homeostatic rate
- $\mathfrak{R}_{\text{0}} \in [0,1]$ is baseline resilience

**Implementation**: [`update_resilience_dynamics()`](../../src/python/affect_utils.py#L1189-L1230) in `affect_utils.py`

## Validation Metrics

### Key Resilience Metrics

1. **Recovery Time**: Time to return to baseline after stress events
2. **Basin Stability**: Proportion of individuals maintaining resilience above threshold
3. **Coping Success Rate**: Overall success rate across all stress events
4. **Overload Frequency**: Rate of cumulative overload events

### Correlation Validation Framework

The resilience dynamics system includes validated correlations with other agent variables:

**Validated Correlation Ranges:**
- Resilience ↔ Affect: $r \in [-0.5, 0.5]$ (variable correlation)
- Resilience ↔ Resources: $r \in [-1.0, 1.0]$ (any correlation)
- Resilience ↔ PSS-10: $r \in [-1.0, 0.5]$ (negative to weak correlation)

**Implementation**: [`test_correlation_validation.py`](../../src/python/tests/test_correlation_validation.py) validates these correlation ranges and statistical significance.

### Calibration Targets

- **Literature-based**: Recovery times from psychological studies
- **Intervention Effects**: Expected improvement from mental health programs
- **Population Patterns**: Realistic distribution of resilience levels
- **Stress-Response Relationships**: Appropriate sensitivity to stress events
- **Correlation Stability**: Theoretical relationships maintained across parameter configurations
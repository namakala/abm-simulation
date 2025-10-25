# Step-by-Step Calculations for Resilience, Stress, and Affect

_See [`.kilocode/rules/math/notation.md`](../../.kilocode/rules/math/notation.md) for symbol definitions and conventions._

## Overview

This document provides detailed step-by-step calculations for how resilience, stress, and affect are computed in each simulation step. The calculations integrate multiple mechanisms including stress events, social interactions, resource dynamics, and homeostatic processes.

## Daily Simulation Step Structure

Each simulation day follows a structured sequence of operations:

1. **Initialization**: Set initial values and get neighbor states
2. **Subevent Processing**: Execute random sequence of interactions and stress events
3. **Integration**: Combine daily challenge/hindrance effects
4. **Dynamics Update**: Apply affect and resilience dynamics
5. **Resource Management**: Update resource levels and protective factors
6. **Homeostasis**: Apply homeostatic adjustments
7. **Daily Reset**: Reset daily tracking variables

## Detailed Calculation Steps

### Step 1: Daily Initialization

**Initial State Capture**:
At the beginning of each day, the system captures initial values for affect and resilience, gathers information about neighbor emotional states, and initializes tracking variables for daily events.

**Baseline State Integration**:
Each agent maintains baseline values that represent their natural equilibrium points, established during initialization and used throughout the simulation for homeostatic regulation and behavioral reference.

**Baseline Value Definitions**:
- **Baseline Resilience** $R_{\text{0}} \in [0,1]$: Agent's natural resilience capacity, established at initialization using sigmoid transformation of normal distribution
- **Baseline Affect** $A_{\text{0}} \in [-1,1]$: Agent's natural emotional equilibrium, established at initialization using tanh transformation of normal distribution
- **Baseline Resources**: Maximum resource capacity (typically 1.0) toward which regeneration occurs
- **Baseline PSS-10**: Individual stress assessment baseline derived from initial stress dimensions

**Mathematical Baseline Generation**:

**Resilience Baseline**:
$$R_{\text{0}} = \sigma\left(\frac{X - \mu_{R,\text{init}}}{\sigma_{R,\text{init}}}\right)$$
Where:
- $X \sim \mathcal{N}(\mu_{R,\text{init}}, \sigma_{R,\text{init}}^2)$ is normally distributed
- $\sigma(x) = \frac{1}{1+e^{-x}}$ is sigmoid function ensuring [0,1] range
- $\mu_{R,\text{init}} = 0.5$ (default mean resilience)
- $\sigma_{R,\text{init}} = 0.2$ (default standard deviation)

**Affect Baseline**:
$$A_{\text{0}} = \tanh\left(\frac{X - \mu_{A,\text{init}}}{\sigma_{A,\text{init}}}\right)$$
Where:
- $X \sim \mathcal{N}(\mu_{A,\text{init}}, \sigma_{A,\text{init}}^2)$ is normally distributed
- $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ ensures [-1,1] range
- $\mu_{A,\text{init}} = 0.0$ (default neutral affect)
- $\sigma_{A,\text{init}} = 0.5$ (default standard deviation)

**PSS-10 Baseline Generation**:
$$c_\Psi, o_\Psi \sim \mathcal{N}\left(\begin{bmatrix} \mu_c \\ \mu_o \end{bmatrix}, \begin{bmatrix} \sigma_c^2 & \rho_\Psi \sigma_c \sigma_o \\ \rho_\Psi \sigma_c \sigma_o & \sigma_o^2 \end{bmatrix}\right)$$
Where:
- $\rho_\Psi \in [-1,1]$ is bifactor correlation (default: 0.3)
- $\sigma_c, \sigma_o > 0$ are dimension standard deviations
- $\mu_c, \mu_o \in [0,1]$ are dimension means

**Implementation Details**:
- **Configuration Integration**: All baseline parameters loaded from environment variables via unified configuration system
- **Reproducible Initialization**: Seeded random number generation ensures consistent baseline generation across simulation runs
- **Population Variation**: Individual differences created through configurable normal distribution parameters
- **Validation**: Baseline values validated for proper ranges and distributions in comprehensive test suite

**Integration with Daily Dynamics**:
Baseline values serve as reference points for:
- **Homeostatic Regulation**: Natural tendency to return to baseline equilibrium
- **Behavioral Adaptation**: Learning and adjustment relative to baseline capacity
- **Intervention Effects**: Measuring improvement relative to individual baselines
- **Population Analysis**: Understanding individual differences in baseline characteristics

### Step 2: Subevent Generation and Execution

**Subevent Count Determination**:
Each day includes a variable number of subevents (typically 7-8) that can be either social interactions or stress events, with the sequence randomized for each day.

**Daily Subevent Generation:**

$$n_s \sim \max(\mathcal{P}(\lambda_s), 1)$$

Where:
- $n_s \in \mathbb{N}$ is number of subevents per day
- $\lambda_s > 0$ is subevent rate parameter
- $\mathcal{P}(\lambda)$ is Poisson distribution

**Implementation**: [`sample_poisson()`](src/python/math_utils.py:126) in `math_utils.py`

**Action Processing**:
The system processes each subevent, tracking social interactions and accumulating challenge/hindrance effects from stress events.

**Daily Challenge/Hindrance Integration:**

$$\bar{\chi}_d = \frac{1}{n_e} \sum_{i=1}^{n_e} \chi_i$$

$$\bar{\zeta}_d = \frac{1}{n_e} \sum_{i=1}^{n_e} \zeta_i$$

Where:
- $\bar{\chi}_d, \bar{\zeta}_d \in [0,1]$ are daily average challenge/hindrance
- $n_e$ is number of stress events in day
- $\chi_i, \zeta_i \in [0,1]$ are challenge/hindrance for event $i$

### Step 3: Daily Challenge/Hindrance Integration

**Normalization by Event Count**:
Daily challenge and hindrance values are averaged across all stress events in the day to create representative daily values.

**Interpretation**: Daily challenge/hindrance represent average event appraisal across all stress events in the day.

### Step 4: Affect Dynamics Calculation

**Integrated Affect Update**:
Affect changes result from multiple components working together:

- **Peer Influence**: Social influence from neighbors
- **Event Appraisal**: Emotional impact of stress events
- **Homeostasis**: Natural tendency to return to baseline

These components combine to determine daily affect changes.

**Integrated Affect Update:**

$$A_{t+1} = A_t + \Delta A_p + \Delta A_e + \Delta A_h$$

**Peer Influence:**

$$\Delta A_p = \frac{1}{k} \sum_{j=1}^{k} \alpha_p \cdot (A_j - A_t) \cdot \mathbb{1}_{j \leq k_{\text{influence}}}$$

**Event Appraisal Effect:**

$$\Delta A_e = \alpha_e \cdot \bar{\chi}_d \cdot (1 - A_t) - \alpha_e \cdot \bar{\zeta}_d \cdot \max(0.1, A_t + 1)$$

**Homeostasis Effect:**

$$\Delta A_h = \lambda_{\text{affect}} \cdot (A_{\text{0}} - A_t)$$

Where:
- $A_t \in [-1,1]$ is current affect
- $\Delta A_p$ is peer influence effect
- $\Delta A_e$ is event appraisal effect
- $\Delta A_h$ is homeostasis effect
- $k$ is number of neighbors
- $k_{\text{influence}}$ is number of influencing neighbors
- $\alpha_p, \alpha_e \in [0,1]$ are influence rates
- $\lambda_{\text{affect}} \in [0,1]$ is homeostatic rate
- $A_{\text{0}} \in [-1,1]$ is baseline affect

**Implementation**: [`update_affect_dynamics()`](src/python/affect_utils.py:830) in `affect_utils.py`

**Detailed Component Calculations**:

#### Peer Influence
Social influence from neighbors affects emotional state, with the extent depending on the number of influential neighbors and the strength of their emotional states.

#### Event Appraisal Effect
Stress events influence emotional state based on their challenge/hindrance nature, with challenging events potentially motivating and hindrance events potentially demotivating.

#### Homeostasis Effect
Emotional states naturally tend to return to baseline levels over time, representing psychological adaptation and equilibrium-seeking tendencies.

### Step 5: Resilience Dynamics Calculation

**Integrated Resilience Update**:
Resilience changes result from multiple components working together:

- **Coping Success Effect**: Impact of successful stress management
- **Social Support Effect**: Benefits from supportive relationships
- **Overload Effect**: Consequences of cumulative hindrance events

These components combine to determine daily resilience changes.

**Integrated Resilience Update:**

$$R_{t+1} = R_t + \Delta R_{\chi\zeta} + \Delta R_p + \Delta R_o + \Delta R_s + \lambda_{\text{resilience}} \cdot (R_{\text{0}} - R_t)$$

**Challenge-Hindrance Effect:**

$$\Delta R_{\chi\zeta} = \begin{cases}
0.3 \cdot \bar{\chi}_d + 0.1 \cdot \bar{\zeta}_d & \text{if coping successful} \\
-0.1 \cdot \bar{\chi}_d - 0.4 \cdot \bar{\zeta}_d & \text{if coping failed}
\end{cases}$$

**Protective Factor Boost:**

$$\Delta R_p = \sum_{f \in F} e_f \cdot (R_{\text{0}} - R_t) \cdot \theta_{\text{boost}}$$

**Overload Effect:**

$$\Delta R_o = \begin{cases}
-0.2 \cdot \min\left(\frac{h_c}{\eta_{\text{res,overload}}}, 2.0\right) & \text{if } h_c \geq \eta_{\text{res,overload}} \\
0 & \text{otherwise}
\end{cases}$$

Where:
- $R_t \in [0,1]$ is current resilience
- $\Delta R_{\chi\zeta}$ is challenge-hindrance effect
- $\Delta R_p$ is protective factor boost
- $\Delta R_o$ is overload effect
- $\Delta R_s$ is social support effect
- $\lambda_{\text{resilience}} \in [0,1]$ is homeostatic rate
- $R_{\text{0}} \in [0,1]$ is baseline resilience
- $F = \{\mathrm{soc}, \mathrm{fam}, \mathrm{int}, \mathrm{cap}\}$ is set of protective factors
- $e_f \in [0,1]$ is efficacy of factor $f$
- $\theta_{\text{boost}} > 0$ is boost rate parameter
- $h_c \in \mathbb{N}$ is consecutive hindrances count
- $\eta_{\text{res,overload}} \in \mathbb{N}$ is overload threshold

**Detailed Component Calculations**:

#### Coping Success Effect
Coping outcomes influence resilience based on challenge/hindrance characteristics, with successful coping building resilience and failed coping depleting it.

#### Social Support Effect
Social support received during interactions provides a boost to resilience, representing how supportive relationships help build coping capacity.

#### Overload Effect
When hindrance events accumulate beyond a threshold, they create an overload effect that reduces resilience, representing how chronic stress can overwhelm coping capacity.

### Step 6: Resource Dynamics Integration

**Resource Regeneration**:
Resources naturally regenerate toward maximum capacity, with positive affect enhancing the regeneration process. Successful coping consumes resources, representing the energy invested in stress management.

**Resource Regeneration Function**:
Resources regenerate linearly toward maximum capacity, representing natural recovery and rest processes.

### Step 7: Protective Factor Management

**Resource Allocation to Protective Factors**:
Individuals allocate available resources across protective factors using a decision-making process that considers current efficacy levels and allocation temperature.

**Efficacy Updates**:
Resource investment improves protective factor efficacy, with higher returns when current efficacy is lower, representing realistic learning and development processes.

### Step 8: Homeostatic Adjustment

**Homeostatic Pull to Baseline**:
All systems have a natural tendency to return to baseline equilibrium levels, representing psychological adaptation and recovery processes.

**Application to Both Systems**:
Both affect and resilience are adjusted toward their baseline levels, with the rate of adjustment determining how quickly equilibrium is restored.

**Homeostatic Adjustment:**

$$x_{t+1} = x_t + \lambda_x \cdot (x_{\text{0}} - x_t)$$

**Clamping:**

$$x_{t+1} = \begin{cases}
\mathrm{clamp}(x_{t+1}, -1, 1) & \text{if } x = A \\
\mathrm{clamp}(x_{t+1}, 0, 1) & \text{if } x = R
\end{cases}$$

Where:
- $x_t$ is current value (affect $A_t$ or resilience $R_t$)
- $\lambda_x \in [0,1]$ is homeostatic rate ($\lambda_{\text{affect}}$ for affect, $\lambda_{\text{resilience}}$ for resilience)
- $x_{\text{0}}$ is baseline value ($A_{\text{0}}$ or $R_{\text{0}}$)

**Implementation**: [`compute_homeostatic_adjustment()`](src/python/affect_utils.py:744) in `affect_utils.py`

**Stress Decay:**

$$S_{t+1} = S_t \cdot (1 - \delta_{\text{stress}})$$

Where:
- $S_t \in [0,1]$ is current stress level
- $\delta_{\text{stress}} \in [0,1]$ is stress decay rate

### Step 9: Daily Reset and Tracking

**End-of-Day Processing**:
At the end of each day, the system applies daily resets to affect and stress levels, stores daily summaries for analysis, and resets tracking variables for the next day.

**Stress Decay**:
Stress levels naturally decay over time when no new stress events occur, representing psychological adaptation and forgetting of past stressors.

### Stress Decay Rate

**Parameter**: `STRESS_DECAY_RATE` (default: 0.05, range: 0.0-1.0)

**Description**: Controls the rate at which stress levels decay over time when no new stress events occur, representing natural recovery processes and psychological adaptation.

**Interpretation**:
- **High values**: Rapid stress decay, representing effective natural recovery
- **Low values**: Slow stress decay, representing persistent stress effects
- **Research context**: Calibrated against psychological research on stress recovery

**Theoretical Foundation**:
- **Natural Recovery**: Psychological tendency to return to baseline stress levels
- **Adaptation Process**: How individuals habituate to ongoing stressors
- **Memory Effects**: Fading of stressful memories and their emotional impact

**Integration with Other Systems**:
- **Stress Events**: New events add to current stress before decay
- **Consecutive Hindrances**: Decay affects hindrance tracking
- **Network Adaptation**: Persistent stress may trigger network changes

### Step 10: Stress Assessment Score Computation and Agent State Integration

**Assessment Integration Steps**:

The stress assessment integration occurs at the end of each simulation step and serves as both a measurement tool and a validation mechanism.

**Detailed Assessment Update Process**:

1. **Stress Dimension Input**: Current stress controllability and overload values
2. **Dimension Score Generation**: Create correlated dimension scores
3. **Item Response Generation**: Generate individual assessment item responses
4. **Total Score Calculation**: Sum all items to get total assessment score
5. **Reverse Mapping**: Update stress dimensions based on assessment responses
6. **History Tracking**: Store assessment trajectory for analysis and validation

**Assessment Dimension Score Generation**:
Dimension scores are generated using correlated distributions that reflect the empirical structure of stress responses.

**Example Dimension Score Generation**:
For an individual with moderate stress levels, the system generates slightly correlated dimension scores that reflect realistic stress response patterns.

**Assessment Item Response Generation**:
Individual item responses are generated based on dimension scores and factor loadings, implementing the empirical structure of stress assessment scales.

**Stress Level Update from Assessment**:
Assessment responses create a feedback loop that influences underlying stress dimensions, enabling empirical calibration and validation.

**Example Assessment Integration in Agent Step**:
The assessment integration occurs as part of the daily step process, ensuring continuous measurement and feedback.

**Parameters**:
- `correlation`: 0.3 (correlation between stress dimensions)
- `controllability_sd`: 1.0 (variability in controllability dimension)
- `overload_sd`: 1.0 (variability in overload dimension)

## Complete Integration Example

### Example: High-Challenge Event Day

**Scenario**: Individual experiences a high-challenge, low-hindrance event with positive social connections

**Step-by-Step Calculations**:

1. **Event Generation**:
   - High controllability, Low overload

2. **Appraisal**:
   - High challenge, Low hindrance

3. **Threshold Evaluation**:
   - Event exceeds stress threshold, triggering stress response

4. **Social Interaction**:
   - Positive neighbor emotional states provide social influence

5. **Affect Update**:
   - Combined effects result in positive affect change

6. **Resilience Update**:
   - Social support provides resilience boost

7. **Resource Update**:
   - Regeneration enhanced by positive affect

### Example: High-Hindrance Event Day

**Scenario**: Individual experiences a high-hindrance, low-challenge event with negative social connections

**Step-by-Step Calculations**:

1. **Event Generation**:
   - Low controllability, High overload

2. **Appraisal**:
   - Low challenge, High hindrance

3. **Threshold Evaluation**:
   - Event exceeds stress threshold, triggering stress response

4. **Coping Determination**:
   - Low coping success probability due to hindrance and negative social influence

5. **State Updates**:
   - Combined effects result in negative affect and resilience changes

6. **Final Values**:
   - Overall negative impact on mental health state

## Mathematical Integration Summary

### Core Integration Equations

**Affect Dynamics**:
Affect changes result from peer influence, event appraisal, and homeostatic processes.

**Resilience Dynamics**:
Resilience changes result from coping success, social support, overload effects, and challenge/hindrance outcomes.

**Stress Processing**:
Stress levels change based on event characteristics and natural decay processes.

### Parameter Integration

All calculations use the unified configuration system, ensuring consistent parameter usage across all calculation steps and maintaining research reproducibility requirements.
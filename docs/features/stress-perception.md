# Stress Perception Mechanisms

_See [`.kilocode/rules/math/notation.md`](../../.kilocode/rules/math/notation.md) for symbol definitions and conventions._

## Overview

Stress perception in the agent-based model follows a comprehensive pipeline that transforms life events into psychological stress responses. The system implements the theoretical challenge-hindrance framework, where events are appraised based on their controllability and overload characteristics.

## Event Generation

### Life Event Structure

Each stress event is characterized by two fundamental attributes:

- **Controllability**: Degree to which the individual can influence the event outcome
- **Overload**: Perceived burden relative to the individual's capacity

### Event Generation Process

Events are generated using a process that creates realistic combinations of controllability and overload, representing the varied nature of real-life stressors.

## Challenge-Hindrance Appraisal

### Weight Function Application

The core appraisal mechanism maps the two event attributes to challenge and hindrance components using a mathematical function that transforms controllability and overload into psychological impact measures.

**Challenge-Hindrance Appraisal Equation:**

$$z = \omega_c \cdot c - \omega_o \cdot o + b$$

$$\chi = \sigma(\gamma \cdot z)$$

$$\zeta = 1 - \chi$$

Where:
- $c \in [0,1]$ is controllability
- $o \in [0,1]$ is overload
- $\omega_c, \omega_o \in \mathbb{R}$ are weight parameters
- $b \in \mathbb{R}$ is bias term
- $\gamma > 0$ controls sigmoid steepness
- $\sigma(x) = \frac{1}{1+e^{-x}}$ is sigmoid function

**Implementation**: [`apply_weights()`](../../src/python/stress_utils.py#L126-L152) in `stress_utils.py`

### Parameter Configuration

The appraisal system uses several parameters that control how events are interpreted:

- **Controllability weight** $\omega_c$: How much controllability influences the appraisal
- **Overload weight** $\omega_o$: How much overload influences the appraisal
- **Bias term** $b$: Baseline shift in the appraisal function
- **Sigmoid steepness** $\gamma$: How decisively events are classified as challenge or hindrance

### Challenge-Hindrance Mapping Logic

The appraisal system implements specific mapping rules:

- **High Challenge Events**: High controllability combined with low overload
   - Example: "A challenging project at work that I can control"
   - Results in predominantly challenge appraisal

- **High Hindrance Events**: Low controllability combined with high overload
   - Example: "Unexpected financial crisis that overwhelms me"
   - Results in predominantly hindrance appraisal

## Stress Threshold Evaluation

### Effective Threshold Calculation

The decision to become stressed uses a dynamic threshold mechanism that adjusts based on the challenge and hindrance characteristics of events.

**Stress Threshold Evaluation:**

$$\eta_{\mathrm{eff}} = \eta_{\text{0}} + \eta_{\chi} \cdot \chi - \eta_{\zeta} \cdot \zeta$$

**Stress Classification:**

$$\mathrm{stressed} = \begin{cases}
1 & \text{if } L > \eta_{\mathrm{eff}} \\
0 & \text{otherwise}
\end{cases}$$

Where:
- $\eta_{\text{0}} \in [0,1]$ is base stress threshold
- $\eta_{\chi} > 0$ is challenge threshold modifier
- $\eta_{\zeta} > 0$ is hindrance threshold modifier
- $\chi \in [0,1]$ is challenge component
- $\zeta \in [0,1]$ is hindrance component

**Implementation**: [`evaluate_stress_threshold()`](../../src/python/stress_utils.py#L189-L224) in `stress_utils.py`

### Appraised Stress Load

The overall stress load is computed from challenge-hindrance polarity, representing how the balance between challenge and hindrance influences the overall stress impact.

**Appraised Stress Load:**

$$L = 1 + \delta \cdot (\zeta - \chi)$$

Where:
- $\delta > 0$ controls polarity effect strength
- $\zeta - \chi \in [-1,1]$ represents hindrance vs challenge balance

## Threshold Dynamics

### Challenge Effects
- **Increases effective threshold**: Challenge events make individuals more resilient to stress
- **Interpretation**: Challenging events build coping capacity

### Hindrance Effects
- **Decreases effective threshold**: Hindrance events reduce coping capacity
- **Interpretation**: Hindrance events deplete psychological resources

## PSS-10 Integration

### Bifactor Model Implementation

The model implements a comprehensive PSS-10 integration using an empirically grounded bifactor model that accurately represents the two-factor structure of perceived stress:

**PSS-10 Bifactor Model:**

**Dimension Score Generation:**

$$c_\Psi, o_\Psi \sim \mathcal{N}\left(\begin{bmatrix} c \\ o \end{bmatrix}, \begin{bmatrix} \sigma_c^2 & \rho_\Psi \sigma_c \sigma_o \\ \rho_\Psi \sigma_c \sigma_o & \sigma_o^2 \end{bmatrix}\right)$$

**Item Response Generation:**

$$\Psi_i = \mathrm{clamp}\left(\mathrm{round}\left(\mu_{\Psi,i} + (\lambda_{c,\Psi,i}(1-c_\Psi) + \lambda_{o,\Psi,i} o_\Psi - 0.5) \cdot 0.5 + \epsilon\right), 0, 4\right)$$

**Total PSS-10 Score:**

$$\Psi = \sum_{i=1}^{10} \begin{cases}
\Psi_i & \text{if } i \notin \{4,5,7,8\} \\
4 - \Psi_i & \text{if } i \in \{4,5,7,8\}
\end{cases}$$

Where:
- $c_\Psi, o_\Psi \in [0,1]$ are PSS-10 dimension scores
- $\rho_\Psi \in [-1,1]$ is dimension correlation
- $\sigma_c, \sigma_o > 0$ are dimension standard deviations
- $\mu_{\Psi,i} \in [0,4]$ is empirical item mean
- $\lambda_{c,\Psi,i}, \lambda_{o,\Psi,i} \in [0,1]$ are factor loadings
- $\epsilon \sim \mathcal{N}(0, 0.1)$ is measurement error

**Implementation**: [`generate_pss10_dimension_scores()`](../../src/python/stress_utils.py#L420-L483), [`generate_pss10_item_response()`](../../src/python/stress_utils.py#L486-L552), [`compute_pss10_score()`](../../src/python/stress_utils.py#L358-L393) in `stress_utils.py`

**Core Components:**
- **Controllability Dimension**: Items 4, 5, 7, 8 (reverse scored) - measures perceived control over life events
- **Overload Dimension**: Items 1, 2, 3, 6, 9, 10 - measures feeling overwhelmed by demands
- **Bifactor Correlation**: Configurable correlation between dimensions (default: 0.3)
- **Total Score**: Sum of all 10 items (0-40 scale)

### PSS-10 State Variables

Agents maintain comprehensive PSS-10 state for tracking perceived stress and enabling empirical validation.

### PSS-10 Initialization Process

When agents are created, PSS-10 state is initialized through a comprehensive process that integrates with the overall agent baseline initialization system:

1. **Baseline Stress Level Setting**: Initial stress levels are set based on agent characteristics using the same mathematical transformations as other baseline variables
2. **PSS-10 Response Generation**: Initial PSS-10 responses are generated from baseline stress levels using empirically grounded bifactor model
3. **Bidirectional Synchronization**: PSS-10 scores and stress levels are synchronized with feedback loops
4. **Integration with Other Baselines**: PSS-10 initialization coordinates with resilience and affect baseline generation

#### Complete Baseline Integration Process

**Step 1: Agent-Wide Baseline Generation**
All baseline values are generated simultaneously using the unified initialization framework.

The unified initialization framework ensures that all agent baseline values (resilience, affect, resources, PSS-10 dimensions) are generated consistently using the same seeded random number generator and mathematical transformations. This approach guarantees reproducible initialization while allowing for individual variation through configurable parameters.

**Implementation**: [`__init__()`](../../src/python/agent.py#L68-L170) in `agent.py`

**Step 2: PSS-10 Dimension Generation**
PSS-10 dimensions are generated using multivariate normal distribution correlated with other baseline characteristics:

$$c_\Psi, o_\Psi \sim \mathcal{N}\left(\begin{bmatrix} \mu_c \\ \mu_o \end{bmatrix}, \begin{bmatrix} \sigma_c^2 & \rho_\Psi \sigma_c \sigma_o \\ \rho_\Psi \sigma_c \sigma_o & \sigma_o^2 \end{bmatrix}\right)$$

Where:
- $\mu_c, \mu_o \in [0,1]$ are PSS-10 dimension means (default: 0.5)
- $\sigma_c, \sigma_o > 0$ are dimension standard deviations (default: 0.25)
- $\rho_\Psi \in [-1,1]$ is bifactor correlation (default: 0.3)

**Step 3: Item Response Generation**
Individual PSS-10 item responses are generated using empirically derived factor loadings:

$$\Psi_i = \mathrm{clamp}\left(\mathrm{round}\left(\mu_{\Psi,i} + (\lambda_{c,\Psi,i}(1-c_\Psi) + \lambda_{o,\Psi,i} o_\Psi - 0.5) \cdot 0.5 + \epsilon\right), 0, 4\right)$$

Where:
- $\mu_{\Psi,i} \in [0,4]$ is empirical item mean for item $i$
- $\lambda_{c,\Psi,i}, \lambda_{o,\Psi,i} \in [0,1]$ are factor loadings for controllability and overload
- $\epsilon \sim \mathcal{N}(0, 0.1)$ is measurement error
- Items 4, 5, 7, 8 are reverse scored: $\Psi_i' = 4 - \Psi_i$

**Step 4: Total Score Calculation**
Total PSS-10 score is computed with proper reverse scoring:

$$\Psi = \sum_{i=1}^{10} \begin{cases}
\Psi_i & \text{if } i \notin \{4,5,7,8\} \\
4 - \Psi_i & \text{if } i \in \{4,5,7,8\}
\end{cases}$$

**Step 5: Stress Level Derivation**
Initial stress level is computed from PSS-10 score and dimensions:

$$S = \frac{\Psi - \Psi_{\min}}{\Psi_{\max} - \Psi_{\min}} \cdot (1 - c_\Psi) + \frac{\Psi - \Psi_{\min}}{\Psi_{\max} - \Psi_{\min}} \cdot o_\Psi$$

Where $\Psi_{\min} = 0$, $\Psi_{\max} = 40$.

#### Integration with Resilience and Affect Baselines

**Mathematical Consistency**:
All baseline values use consistent transformation functions:
- **Resilience/Resources**: Sigmoid transformation for [0,1] bounds
- **Affect**: Tanh transformation for [-1,1] bounds
- **PSS-10**: Multivariate normal with empirical factor structure

**Parameter Coordination**:
Baseline generation coordinates through the unified configuration system:
- **Resilience**: $\mu_{R,\text{init}} = 0.5$, $\sigma_{R,\text{init}} = 0.2$
- **Affect**: $\mu_{A,\text{init}} = 0.0$, $\sigma_{A,\text{init}} = 0.5$
- **PSS-10**: $\mu_c = \mu_o = 0.5$, $\sigma_c = \sigma_o = 0.25$, $\rho_\Psi = 0.3$

**Implementation Details**:
- **Reproducible Generation**: All baselines use the same seeded random number generator
- **Population Variation**: Individual differences created through configurable distribution parameters
- **Validation**: Comprehensive testing ensures proper ranges and realistic distributions
- **Research Integration**: Parameters calibrated against empirical psychological research

### Dimension Score Generation

The dimension scores are generated using a multivariate normal distribution to model the correlation between controllability and overload dimensions, ensuring realistic stress response patterns.

**Implementation**: [`generate_pss10_dimension_scores()`](../../src/python/stress_utils.py#L420-L483) in `stress_utils.py`

### Item Response Generation

Individual PSS-10 item responses are generated by applying empirically derived factor loadings to the dimension scores, incorporating measurement error and proper reverse scoring for specified items.

**Implementation**: [`generate_pss10_item_response()`](../../src/python/stress_utils.py#L486-L552) in `stress_utils.py`

### Feedback Loop Integration

**Bidirectional Stress-PSS-10 Integration:**

1. **Stress → PSS-10**: Current stress levels generate PSS-10 responses

   The stress dimensions (controllability and overload) are used to generate PSS-10 responses through the bifactor model, creating a forward mapping from internal stress state to observable assessment scores.

   **Implementation**: [`generate_pss10_from_stress_dimensions()`](../../src/python/stress_utils.py#L718-L785) in `stress_utils.py`

2. **PSS-10 → Stress**: PSS-10 responses map back to stress dimensions

   PSS-10 responses are used to update the underlying stress dimensions through feedback mechanisms, ensuring that the assessment scores influence the agent's internal stress state.

   **Implementation**: [`update_stress_dimensions_from_pss10_feedback()`](../../src/python/stress_utils.py#L788-L862) in `stress_utils.py`

3. **Dynamic Updates**: PSS-10 scores update each simulation step based on current stress state

   The PSS-10 integration is updated daily during the agent step process, maintaining the bidirectional loop between stress perception and assessment.

   **Implementation**: [`step()`](../../src/python/agent.py#L204-L391) in `agent.py`

### Deterministic Generation

The deterministic generation subsection describes how PSS-10 generation can use deterministic seeds for reproducible results, ensuring consistent outputs for testing and validation.

**Deterministic Seed Equation:**

$$s = \mathrm{hash}(\mathrm{md5}(c \cdot 10^{10} \| o \cdot 10^{10} \| \rho \cdot 10^{10})) \mod 2^{32}$$

Where:
- $c \in [0,1]$ is controllability
- $o \in [0,1]$ is overload
- $\rho \in [-1,1]$ is correlation
- $s \in \mathbb{N}$ is the deterministic seed

**Implementation**: [`generate_pss10_responses()`](../../src/python/stress_utils.py#L555-L617) in `stress_utils.py`

## Configuration Parameters

### Stress Event Parameters

The stress event parameters subsection lists environment variables controlling stress event generation, including means, standard deviations, and distribution parameters.

**Implementation**: [`get_config()`](../../src/python/config.py#L647-L661) in `config.py`

### Appraisal Parameters

The appraisal parameters subsection details environment variables for the challenge-hindrance appraisal mechanism, including weights, bias, and sigmoid parameters.

**Implementation**: [`get_config()`](../../src/python/config.py#L647-L661) in `config.py`

### PSS-10 Parameters

The PSS-10 parameters subsection provides environment variables for the bifactor model, including item means, standard deviations, factor loadings, and correlation.

**Implementation**: [`get_config()`](../../src/python/config.py#L647-L661) in `config.py`

### Threshold Parameters

The threshold parameters subsection lists environment variables for stress threshold evaluation, including base threshold and scaling parameters.

**Implementation**: [`get_config()`](../../src/python/config.py#L647-L661) in `config.py`

## Agent Management Integration

The agent management integration subsection explains how PSS-10 is integrated into the agent lifecycle, from initialization to daily updates and behavioral influence.

**Implementation**: [`__init__()`](../../src/python/agent.py#L68-L170) in `agent.py` and [`step()`](../../src/python/agent.py#L204-L391) in `agent.py`

### PSS-10 and Agent Lifecycle

PSS-10 integration is deeply embedded in the agent lifecycle, providing empirical grounding for stress perception and enabling validation against real-world stress measurements.

#### Agent Creation and Initialization

The agent creation and initialization subsection describes how PSS-10 state is set up during agent initialization, including dimension generation and score calculation.

**Implementation**: [`__init__()`](../../src/python/agent.py#L68-L170) in `agent.py`

#### Daily Agent Step Integration

The daily agent step integration subsection explains how PSS-10 is updated daily based on current stress state, maintaining the bidirectional feedback loop.

**Implementation**: [`step()`](../../src/python/agent.py#L204-L391) in `agent.py`

#### PSS-10 Influence on Agent Behavior

The PSS-10 influence on agent behavior subsection details how PSS-10 scores affect agent decision-making and stress responses, providing empirical grounding for behavioral changes.

**Implementation**: [`step()`](../../src/python/agent.py#L204-L391) in `agent.py`

## Implementation Details

### Key Functions

1. **`generate_stress_event()`**: Creates random stress events with proper distributions
2. **`apply_weights()`**: Computes challenge/hindrance from event attributes
3. **`compute_appraised_stress()`**: Calculates overall stress load
4. **`evaluate_stress_threshold()`**: Determines if agent becomes stressed
5. **`process_stress_event()`**: Complete stress processing pipeline
6. **`initialize_pss10_state()`**: Sets up PSS-10 state for new agents
7. **`update_pss10_integration()`**: Updates PSS-10 based on current stress

### Integration Points

- **Agent State**: Updates `current_stress`, `daily_stress_events`, `consecutive_hindrances`
- **PSS-10 State**: Maintains `pss10_responses`, `pss10`, `stress_controllability`, `stress_overload`
- **Resource Dynamics**: Consumes resources for successful coping
- **Network Adaptation**: Triggers network changes based on stress patterns
- **Affect Dynamics**: Influences affect through challenge/hindrance outcomes
- **Agent Lifecycle**: PSS-10 initialization and daily updates

## Validation and Calibration

### Pattern Targets

The stress perception mechanism is calibrated against:

1. **Recovery Time Patterns**: Typical recovery times from stress events
2. **Challenge-Hindrance Ratios**: Distribution of challenge vs hindrance events
3. **PSS-10 Score Distributions**: Realistic stress scale responses
4. **Threshold Sensitivity**: Appropriate stress response rates

### Sensitivity Analysis

Key parameters for sensitivity analysis:
- Challenge/hindrance weight coefficients (`ω_c, ω_o`)
- Threshold scaling parameters (`λ_C, λ_H`)
- Sigmoid steepness (`γ`)
- Event generation parameters (`alpha, beta`)
- PSS-10 factor loadings and correlations

### Agent-Specific Volatility

Each agent is assigned a unique volatility parameter that introduces individual differences in stress response variability:

**Volatility Initialization:**

$$v_i \sim \mathcal{B}(1, 1)$$

Where:
- $v_i \in [0,1]$ is the volatility for agent $i$
- $\mathcal{B}(\alpha, \beta)$ is the Beta distribution (uniform on [0,1])

**Volatility Effects on Stress Dimensions:**

Volatility scales the magnitude of stress dimension updates, representing individual differences in sensitivity to stress events:

$$\Delta c' = \Delta c \cdot v_i$$
$$\Delta o' = \Delta o \cdot v_i$$

Where:
- $\Delta c, \Delta o$ are unscaled changes in controllability and overload
- $v_i$ is agent's volatility level

**Implementation Details:**
- **Initialization**: Assigned during agent creation using seeded random number generator
- **Usage**: Applied in stress dimension updates to modulate response intensity
- **Research Applications**: Enables modeling of individual differences in stress sensitivity and resilience variability

**Implementation**: [`update_stress_dimensions_from_event()`](../../src/python/stress_utils.py#L919-L1009) in `stress_utils.py`
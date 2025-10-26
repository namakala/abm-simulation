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

**Implementation**: [`apply_weights()`](src/python/stress_utils.py:110) in `stress_utils.py`

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

**Implementation**: [`evaluate_stress_threshold()`](src/python/stress_utils.py:173) in `stress_utils.py`

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

**Implementation**: [`generate_pss10_dimension_scores()`](src/python/stress_utils.py:413), [`generate_pss10_item_response()`](src/python/stress_utils.py:479), [`compute_pss10_score()`](src/python/stress_utils.py:351) in `stress_utils.py`

**Core Components:**
- **Controllability Dimension**: Items 4, 5, 7, 8 (reverse scored) - measures perceived control over life events
- **Overload Dimension**: Items 1, 2, 3, 6, 9, 10 - measures feeling overwhelmed by demands
- **Bifactor Correlation**: Configurable correlation between dimensions (default: 0.3)
- **Total Score**: Sum of all 10 items (0-40 scale)

### PSS-10 State Variables

Agents maintain comprehensive PSS-10 state for tracking perceived stress and enabling empirical validation:

```python
# PSS-10 state variables in Person agent
self.stress_controllability = 0.5  # Controllability stress level ∈ [0,1]
self.stress_overload = 0.5  # Overload stress level ∈ [0,1]
self.pss10 = 10  # Total PSS-10 score (0-40)
self.pss10_responses = {}  # Individual PSS-10 item responses (1-10)
self.pss10_dimension_scores = {}  # Controllability and overload dimension scores
```

### PSS-10 Initialization Process

When agents are created, PSS-10 state is initialized through a comprehensive process that integrates with the overall agent baseline initialization system:

1. **Baseline Stress Level Setting**: Initial stress levels are set based on agent characteristics using the same mathematical transformations as other baseline variables
2. **PSS-10 Response Generation**: Initial PSS-10 responses are generated from baseline stress levels using empirically grounded bifactor model
3. **Bidirectional Synchronization**: PSS-10 scores and stress levels are synchronized with feedback loops
4. **Integration with Other Baselines**: PSS-10 initialization coordinates with resilience and affect baseline generation

#### Complete Baseline Integration Process

**Step 1: Agent-Wide Baseline Generation**
All baseline values are generated simultaneously using the unified initialization framework:

```python
def __init__(self, model, config=None):
    """Initialize agent with complete baseline state integration."""
    super().__init__(model)

    # Generate all baseline values using transformation pipeline
    self.baseline_resilience = sigmoid_transform(
        mean=config['initial_resilience_mean'],
        std=config['initial_resilience_sd'],
        rng=self._rng
    )
    self.resilience = self.baseline_resilience

    self.baseline_affect = tanh_transform(
        mean=config['initial_affect_mean'],
        std=config['initial_affect_sd'],
        rng=self._rng
    )
    self.affect = self.baseline_affect

    # Initialize PSS-10 state with stress dimensions
    self._initialize_pss10_scores()

    # Initialize stress level from PSS-10
    self._initialize_stress_from_pss10()
```

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

**Initialization Code Example:**
```python
def initialize_pss10_state(self, controllability=0.5, overload=0.5):
    """Initialize PSS-10 state for new agent with full baseline integration."""
    # Set baseline stress dimensions (correlated with other baselines)
    self.stress_controllability = controllability
    self.stress_overload = overload

    # Generate initial PSS-10 responses using bifactor model
    self.pss10_responses = generate_pss10_responses(
        controllability=controllability,
        overload=overload,
        rng=self._rng
    )

    # Compute initial total score with reverse scoring
    self.pss10 = compute_pss10_score(self.pss10_responses)

    # Store dimension scores for tracking and validation
    self.pss10_dimension_scores = {
        'controllability': controllability,
        'overload': overload
    }

    # Initialize stress level from PSS-10 (bidirectional sync)
    self._initialize_stress_from_pss10()
```

### Dimension Score Generation

**Multivariate Normal Distribution:**
```python
def generate_pss10_dimension_scores(controllability, overload, correlation, rng, deterministic=False):
    # Create covariance matrix for bivariate normal distribution
    var_c = controllability_sd ** 2
    var_o = overload_sd ** 2
    cov = correlation * np.sqrt(var_c * var_o)

    # Mean vector and covariance matrix
    mean_vector = np.array([controllability, overload])
    cov_matrix = np.array([[var_c, cov], [cov, var_o]])

    # Sample from multivariate normal distribution
    correlated_scores = rng.multivariate_normal(mean_vector, cov_matrix)

    return max(0.0, min(1.0, correlated_scores[0])), max(0.0, min(1.0, correlated_scores[1]))
```

### Item Response Generation

**Empirically Grounded Response Generation:**
```python
def generate_pss10_item_response(item_mean, item_sd, controllability_loading, overload_loading,
                                controllability_score, overload_score, reverse_scored, rng, deterministic=False):
    # Linear combination of dimension scores weighted by factor loadings
    stress_component = (controllability_loading * (1.0 - controllability_score) +
                       overload_loading * overload_score)

    # Normalize by total loading
    total_loading = max(controllability_loading + overload_loading, 1e-10)
    normalized_stress = stress_component / total_loading

    # Sample from normal distribution around empirical mean
    adjusted_mean = item_mean + (normalized_stress - 0.5) * 0.5
    raw_response = rng.normal(adjusted_mean, item_sd)

    # Apply reverse scoring if needed
    if reverse_scored:
        raw_response = 4.0 - raw_response

    return int(round(clamp(raw_response, 0.0, 4.0)))
```

### Feedback Loop Integration

**Bidirectional Stress-PSS-10 Integration:**

1. **Stress → PSS-10**: Current stress levels generate PSS-10 responses
   ```python
   # Generate PSS-10 responses from current stress state
   new_responses = generate_pss10_responses(
       controllability=self.stress_controllability,
       overload=self.stress_overload,
       rng=self._rng
   )
   self.pss10_responses = new_responses
   self.pss10 = compute_pss10_score(new_responses)
   ```

2. **PSS-10 → Stress**: PSS-10 responses map back to stress dimensions
   ```python
   # Update stress dimensions from PSS-10 responses
   controllability_items = [4, 5, 7, 8]  # Reverse scored items
   overload_items = [1, 2, 3, 6, 9, 10]  # Regular scored items

   # Calculate controllability stress (reverse scored items)
   controllability_scores = []
   for item_num in controllability_items:
       response = self.pss10_responses[item_num]
       controllability_scores.append(1.0 - (response / 4.0))  # Reverse scoring
   self.stress_controllability = np.mean(controllability_scores)

   # Calculate overload stress (regular scored items)
   overload_scores = []
   for item_num in overload_items:
       response = self.pss10_responses[item_num]
       overload_scores.append(response / 4.0)  # Normal scoring
   self.stress_overload = np.mean(overload_scores)
   ```

3. **Dynamic Updates**: PSS-10 scores update each simulation step based on current stress state
   ```python
   def update_pss10_integration(self):
       """Update PSS-10 state based on current stress levels."""
       # Step 1: Generate new responses from current stress
       new_responses = generate_pss10_responses(
           controllability=self.stress_controllability,
           overload=self.stress_overload,
           rng=self._rng
       )

       # Step 2: Update PSS-10 state
       self.pss10_responses = new_responses
       self.pss10 = compute_pss10_score(new_responses)

       # Step 3: Update stress dimensions from new responses
       self._update_stress_from_pss10()
   ```

### Deterministic Generation

**Reproducible PSS-10 Generation:**
```python
# Deterministic seed generation for reproducible testing
if deterministic:
    input_str = f"{controllability:.10f}_{overload:.10f}_{correlation:.10f}"
    seed = int(hashlib.md5(input_str.encode()).hexdigest(), 16) % (2**32)
    local_rng = np.random.default_rng(seed)
```

## Configuration Parameters

### Stress Event Parameters
```python
stress_event_config = {
    'controllability_mean': 0.5,    # Mean controllability level
    'overload_mean': 0.5,          # Mean overload level
    'beta_alpha': 2.0,             # Beta distribution shape parameter
    'beta_beta': 2.0               # Beta distribution shape parameter
}
```

### Appraisal Parameters
```python
appraisal_config = {
    'omega_c': 1.0,    # Controllability weight
    'omega_o': 1.0,    # Overload weight
    'bias': 0.0,       # Bias term
    'gamma': 6.0       # Sigmoid steepness
}
```

### PSS-10 Parameters
```python
pss10_config = {
    'item_means': [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5],  # Empirical item means
    'item_sds': [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8],     # Empirical item SDs
    'load_controllability': [0.2, 0.8, 0.1, 0.7, 0.6, 0.1, 0.8, 0.6, 0.7, 0.1],  # Factor loadings
    'load_overload': [0.7, 0.3, 0.8, 0.2, 0.4, 0.9, 0.2, 0.3, 0.4, 0.9],         # Factor loadings
    'bifactor_correlation': 0.3,  # Correlation between dimensions
    'controllability_sd': 1.0,    # SD for controllability dimension
    'overload_sd': 1.0            # SD for overload dimension
}
```

### Threshold Parameters
```python
threshold_config = {
    'base_threshold': 0.5,      # Baseline stress threshold
    'challenge_scale': 0.15,    # Challenge threshold modifier
    'hindrance_scale': 0.25,    # Hindrance threshold modifier
    'stress_threshold': 0.5     # Overall stress threshold
}
```

## Agent Management Integration

### PSS-10 and Agent Lifecycle

PSS-10 integration is deeply embedded in the agent lifecycle, providing empirical grounding for stress perception and enabling validation against real-world stress measurements:

#### Agent Creation and Initialization
```python
def __init__(self, unique_id, model, rng):
    """Initialize agent with PSS-10 integration."""
    super().__init__(unique_id, model)

    # Initialize stress dimensions (can be customized per agent)
    initial_controllability = rng.uniform(0.3, 0.7)  # Individual variation
    initial_overload = rng.uniform(0.3, 0.7)

    # Initialize PSS-10 state
    self.initialize_pss10_state(
        controllability=initial_controllability,
        overload=initial_overload
    )

    # PSS-10 influences initial stress threshold
    self.stress_threshold = self._compute_initial_threshold()
```

#### Daily Agent Step Integration
```python
def step(self):
    """Complete daily step with PSS-10 integration."""
    # 1. Initialize daily state
    self._initialize_daily_state()

    # 2. Process stress events and social interactions
    self._process_daily_events()

    # 3. Update PSS-10 based on current stress state
    self.update_pss10_integration()

    # 4. Apply stress effects to other systems
    self._apply_stress_effects()

    # 5. Daily reset and tracking
    self._daily_reset()
```

#### PSS-10 Influence on Agent Behavior
```python
def _compute_initial_threshold(self):
    """Compute initial stress threshold influenced by PSS-10."""
    # Higher baseline stress → Lower stress threshold
    pss10_effect = (self.pss10 - 10) / 40.0  # Normalize to [-0.25, 0.25]
    threshold = 0.5 + pss10_effect * 0.2  # PSS-10 influences threshold by ±0.1

    return clamp(threshold, 0.2, 0.8)
```

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

**Implementation**: [`update_stress_dimensions_from_event()`](src/python/stress_utils.py:919) in `stress_utils.py`
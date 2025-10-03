# Stress Perception Mechanisms

## Overview

Stress perception in the agent-based model follows a comprehensive pipeline that transforms life events into psychological stress responses. The system implements the theoretical challenge-hindrance framework, where events are appraised based on their controllability and overload characteristics, with full PSS-10 integration for empirical grounding.

## Event Generation

### Life Event Structure

Each stress event is characterized by two fundamental attributes:

- **Controllability (c ∈ [0,1])**: Degree to which the agent can influence the event outcome
- **Overload (o ∈ [0,1])**: Perceived burden relative to the agent's capacity

### Event Generation Process

Events are generated using a Poisson process with the following characteristics:

```python
# Event generation uses beta distribution for bounded [0,1] values
controllability = rng.beta(alpha, beta)
overload = rng.beta(alpha, beta)
```

**Key Parameters:**
- `alpha, beta`: Shape parameters for beta distribution (default: 2.0, 2.0)

## Challenge-Hindrance Appraisal

### Weight Function Application

The core appraisal mechanism maps the two event attributes to challenge and hindrance components:

**Mathematical Foundation:**
```
z = ω_c × c - ω_o × o + b
challenge = σ(γ × z)
hindrance = 1 - challenge
```

Where:
- `σ(x) = 1/(1+e^(-x))` is the sigmoid function
- `ω_c, ω_o`: Weight coefficients for controllability and overload
- `b`: Bias term
- `γ`: Sigmoid steepness parameter

### Parameter Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `ω_c` | 1.0 | 0.2-2.0 | Controllability weight |
| `ω_o` | 1.0 | 0.2-2.0 | Overload weight |
| `b` | 0.0 | -1.0-1.0 | Bias term |
| `γ` | 6.0 | 1.0-12.0 | Sigmoid steepness |

### Challenge-Hindrance Mapping Logic

The appraisal system implements specific mapping rules:

- **High Challenge Events**: High controllability + Low overload
  - Example: "A challenging project at work that I can control"
  - Results in: challenge ≈ 1.0, hindrance ≈ 0.0

- **High Hindrance Events**: Low controllability + High overload
  - Example: "Unexpected financial crisis that overwhelms me"
  - Results in: challenge ≈ 0.0, hindrance ≈ 1.0

## Stress Threshold Evaluation

### Effective Threshold Calculation

The decision to become stressed uses a dynamic threshold mechanism:

**Mathematical Foundation:**
```
T_eff = T_base + λ_C × challenge - λ_H × hindrance
stressed = (L > T_eff)
```

Where:
- `T_base`: Baseline stress threshold (default: 0.5)
- `λ_C`: Challenge scaling parameter (default: 0.15)
- `λ_H`: Hindrance scaling parameter (default: 0.25)
- `L`: Appraised stress load

### Appraised Stress Load

The overall stress load is computed from challenge-hindrance polarity:

**Mathematical Foundation:**
```
L = 1 + δ × (hindrance - challenge)
```

Where:
- `δ`: Polarity effect parameter (default: 0.3)

## Threshold Dynamics

### Challenge Effects
- **Increases effective threshold**: Challenge events make agents more resilient to stress
- **Mathematical effect**: `+λ_C × challenge` to threshold
- **Interpretation**: Challenging events build coping capacity

### Hindrance Effects
- **Decreases effective threshold**: Hindrance events reduce coping capacity
- **Mathematical effect**: `-λ_H × hindrance` from threshold
- **Interpretation**: Hindrance events deplete psychological resources

## PSS-10 Integration

### Bifactor Model Implementation

The model implements a comprehensive PSS-10 integration using an empirically grounded bifactor model that accurately represents the two-factor structure of perceived stress:

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

When agents are created, PSS-10 state is initialized through a two-step process:

1. **Baseline Stress Level Setting**: Initial stress levels are set based on agent characteristics
2. **PSS-10 Response Generation**: Initial PSS-10 responses are generated from baseline stress levels
3. **Bidirectional Synchronization**: PSS-10 scores and stress levels are synchronized

**Initialization Code Example:**
```python
def initialize_pss10_state(self, controllability=0.5, overload=0.5):
    """Initialize PSS-10 state for new agent."""
    # Set baseline stress dimensions
    self.stress_controllability = controllability
    self.stress_overload = overload

    # Generate initial PSS-10 responses
    self.pss10_responses = generate_pss10_responses(
        controllability=controllability,
        overload=overload,
        rng=self._rng
    )

    # Compute initial total score
    self.pss10 = compute_pss10_score(self.pss10_responses)

    # Store dimension scores for tracking
    self.pss10_dimension_scores = {
        'controllability': controllability,
        'overload': overload
    }
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
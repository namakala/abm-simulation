# Stress Perception Mechanisms

## Overview

Stress perception in the agent-based model follows a comprehensive pipeline that transforms life events into psychological stress responses. The system implements the theoretical challenge-hindrance framework, where events are appraised based on their controllability, predictability, and overload characteristics.

## Event Generation

### Life Event Structure

Each stress event is characterized by three fundamental attributes:

- **Controllability (c ∈ [0,1])**: Degree to which the agent can influence the event outcome
- **Predictability (p ∈ [0,1])**: How foreseeable or expected the event is
- **Overload (o ∈ [0,1])**: Perceived burden relative to the agent's capacity

### Event Generation Process

Events are generated using a Poisson process with the following characteristics:

```python
# Event generation uses beta distribution for bounded [0,1] values
controllability = rng.beta(alpha, beta)
predictability = rng.beta(alpha, beta)
overload = rng.beta(alpha, beta)
magnitude = min(rng.exponential(scale), 1.0)  # Exponential distribution capped at 1.0
```

**Key Parameters:**
- `alpha, beta`: Shape parameters for beta distribution (default: 2.0, 2.0)
- `scale`: Scale parameter for exponential distribution (default: 0.4)

## Challenge-Hindrance Appraisal

### Weight Function Application

The core appraisal mechanism maps the three event attributes to challenge and hindrance components:

**Mathematical Foundation:**
```
z = ω_c × c + ω_p × p - ω_o × o + b
challenge = σ(γ × z)
hindrance = 1 - challenge
```

Where:
- `σ(x) = 1/(1+e^(-x))` is the sigmoid function
- `ω_c, ω_p, ω_o`: Weight coefficients for controllability, predictability, overload
- `b`: Bias term
- `γ`: Sigmoid steepness parameter

### Parameter Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `ω_c` | 1.0 | 0.2-2.0 | Controllability weight |
| `ω_p` | 1.0 | 0.2-2.0 | Predictability weight |
| `ω_o` | 1.0 | 0.2-2.0 | Overload weight |
| `b` | 0.0 | -1.0-1.0 | Bias term |
| `γ` | 6.0 | 1.0-12.0 | Sigmoid steepness |

### Challenge-Hindrance Mapping Logic

The appraisal system implements specific mapping rules:

- **High Challenge Events**: High controllability + High predictability + Low overload
  - Example: "A challenging project at work that I can control and saw coming"
  - Results in: challenge ≈ 1.0, hindrance ≈ 0.0

- **High Hindrance Events**: Low controllability + Low predictability + High overload
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

The overall stress load combines event magnitude with challenge-hindrance polarity:

**Mathematical Foundation:**
```
L = s × (1 + δ × (hindrance - challenge))
```

Where:
- `s`: Event magnitude ∈ [0,1]
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

### Mapping to Perceived Stress Scale

The model maps agent stress states to PSS-10 questionnaire responses:

**Component Mapping:**
- **Controllability**: Maps to items about control (items 2, 4, 5, 7, 8, 9)
- **Predictability**: Maps to items about unexpected events (item 1)
- **Overload**: Maps to items about feeling overwhelmed (items 3, 6, 10)

### Response Generation

```python
# PSS-10 response calculation
base_score = (weight_c × (1-c) + weight_p × (1-p) + weight_o × o + weight_d × d) / total_weight
response_value = int(round(clamp(base_score + variability, 0, 1) × 4))
```

## Configuration Parameters

### Stress Event Parameters
```python
stress_event_config = {
    'controllability_mean': 0.5,    # Mean controllability level
    'predictability_mean': 0.5,     # Mean predictability level
    'overload_mean': 0.5,          # Mean overload level
    'magnitude_scale': 0.4,        # Scale for event magnitude
    'beta_alpha': 2.0,             # Beta distribution shape parameter
    'beta_beta': 2.0               # Beta distribution shape parameter
}
```

### Appraisal Parameters
```python
appraisal_config = {
    'omega_c': 1.0,    # Controllability weight
    'omega_p': 1.0,    # Predictability weight
    'omega_o': 1.0,    # Overload weight
    'bias': 0.0,       # Bias term
    'gamma': 6.0       # Sigmoid steepness
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

## Implementation Details

### Key Functions

1. **`generate_stress_event()`**: Creates random stress events with proper distributions
2. **`apply_weights()`**: Computes challenge/hindrance from event attributes
3. **`compute_appraised_stress()`**: Calculates overall stress load
4. **`evaluate_stress_threshold()`**: Determines if agent becomes stressed
5. **`process_stress_event()`**: Complete stress processing pipeline

### Integration Points

- **Agent State**: Updates `current_stress`, `daily_stress_events`, `consecutive_hindrances`
- **Resource Dynamics**: Consumes resources for successful coping
- **Network Adaptation**: Triggers network changes based on stress patterns
- **Affect Dynamics**: Influences affect through challenge/hindrance outcomes

## Validation and Calibration

### Pattern Targets

The stress perception mechanism is calibrated against:

1. **Recovery Time Patterns**: Typical recovery times from stress events
2. **Challenge-Hindrance Ratios**: Distribution of challenge vs hindrance events
3. **PSS-10 Score Distributions**: Realistic stress scale responses
4. **Threshold Sensitivity**: Appropriate stress response rates

### Sensitivity Analysis

Key parameters for sensitivity analysis:
- Challenge/hindrance weight coefficients (`ω_c, ω_p, ω_o`)
- Threshold scaling parameters (`λ_C, λ_H`)
- Sigmoid steepness (`γ`)
- Event generation parameters (`alpha, beta, scale`)
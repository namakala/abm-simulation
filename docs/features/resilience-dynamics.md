# Resilience Dynamics and Triggering Events

## Overview

Resilience in the agent-based model represents an agent's capacity to adapt and recover from stress events. The system implements multiple mechanisms that influence resilience levels, including stress event outcomes, social support, protective factors, and homeostatic processes.

## Core Resilience State

### State Variables

- **Resilience Level (R ∈ [0,1])**: Current resilience capacity
- **Baseline Resilience (R_baseline ∈ [0,1])**: Agent's natural equilibrium point
- **Consecutive Hindrances**: Counter for cumulative hindrance events
- **Stress Breach Count**: Counter for threshold breaches triggering adaptation

## Triggering Events

### Stress Event Outcomes

Resilience changes are triggered by stress event processing outcomes:

#### Successful Coping
**Trigger Condition**: `coped_successfully = True`

**Resilience Impact**:
```
ΔR = challenge_effect + hindrance_effect
Where:
- challenge_effect = 0.3 × challenge  (Large positive effect)
- hindrance_effect = 0.1 × hindrance  (Small positive effect)
```

**Interpretation**: Successfully coping with challenging events builds resilience significantly, while even hindrance events provide minor learning opportunities when handled successfully.

#### Failed Coping
**Trigger Condition**: `coped_successfully = False`

**Resilience Impact**:
```
ΔR = challenge_effect + hindrance_effect
Where:
- challenge_effect = -0.1 × challenge  (Small negative effect)
- hindrance_effect = -0.4 × hindrance  (Large negative effect)
```

**Interpretation**: Failed coping with hindrance events significantly depletes resilience, while challenge events have minimal negative impact when coping fails.

### Social Support Reception

**Trigger Condition**: Agent receives social support from neighbors

**Resilience Impact**:
```
ΔR = social_support_rate × support_effectiveness
```

**Parameters**:
- `social_support_rate`: Base rate of resilience gain (default: 0.1)
- `support_effectiveness`: Quality of received support (0.0-1.0)

### Cumulative Overload Effects

**Trigger Condition**: `consecutive_hindrances ≥ overload_threshold`

**Resilience Impact**:
```
ΔR = -0.2 × overload_intensity
Where:
overload_intensity = min(consecutive_hindrances / influencing_hindrance, 2.0)
```

**Parameters**:
- `overload_threshold`: Events required to trigger overload (default: 3)
- `influencing_hindrance`: Scaling factor for overload intensity (default: 10)

## Resilience Change Mechanisms

### Challenge-Hindrance Based Changes

The core resilience update mechanism integrates challenge and hindrance effects:

```python
def compute_challenge_hindrance_resilience_effect(challenge, hindrance, coped_successfully):
    if coped_successfully:
        # Success: challenge greatly helps, hindrance slightly helps
        hindrance_effect = 0.1 * hindrance   # Small positive
        challenge_effect = 0.3 * challenge   # Large positive
    else:
        # Failure: hindrance greatly hurts, challenge slightly hurts
        hindrance_effect = -0.4 * hindrance  # Large negative
        challenge_effect = -0.1 * challenge  # Small negative

    return hindrance_effect + challenge_effect
```

### Protective Factor Boost

**Mechanism**: Resources allocated to protective factors provide resilience benefits:

```python
def get_resilience_boost_from_protective_factors():
    total_boost = 0.0

    for factor, efficacy in protective_factors.items():
        if efficacy > 0:
            # Boost higher when resilience is low (more needed)
            need_multiplier = max(0.1, 1.0 - current_resilience)
            boost_rate = 0.1
            total_boost += efficacy * need_multiplier * boost_rate

    return total_boost
```

**Protective Factors**:
- **Social Support**: Efficacy in providing emotional support
- **Family Support**: Efficacy of family relationships
- **Formal Intervention**: Efficacy of professional help
- **Psychological Capital**: Self-efficacy and coping skills

### Homeostatic Adjustment

**Mechanism**: Resilience tends to return to baseline levels over time:

```python
def compute_homeostatic_adjustment(initial_resilience, final_resilience, rate):
    distance = rate * abs(final_resilience - initial_resilience)

    if final_resilience > initial_resilience:
        return final_resilience - distance  # Pull down
    else:
        return final_resilience + distance  # Pull up
```

**Parameters**:
- `homeostatic_rate`: Rate of return to baseline (default: 0.05)

## Coping Success Determination

### Base Coping Probability

**Foundation**: Each agent has a baseline coping success probability:

```python
base_coping_probability = config.base_probability  # Default: 0.6
```

### Challenge-Hindrance Effects

**Challenge Influence**:
- **Increases coping probability**: Challenging events motivate better coping
- **Effect**: `+challenge_bonus × challenge`

**Hindrance Influence**:
- **Decreases coping probability**: Hindrance events overwhelm coping capacity
- **Effect**: `-hindrance_penalty × hindrance`

### Social Influence on Coping

**Mechanism**: Neighbor affect states influence coping success:

```python
def compute_coping_probability(challenge, hindrance, neighbor_affects):
    # Base probability
    base_prob = config.base_coping_probability

    # Challenge/hindrance effects
    challenge_effect = config.challenge_bonus * challenge
    hindrance_effect = -config.hindrance_penalty * hindrance

    # Social influence
    social_effect = 0.0
    if neighbor_affects:
        avg_neighbor_affect = mean(neighbor_affects)
        social_effect = config.social_influence_factor * avg_neighbor_affect

    # Combine effects
    coping_prob = base_prob + challenge_effect + hindrance_effect + social_effect

    return clamp(coping_prob, 0.0, 1.0)
```

**Parameters**:
- `challenge_bonus`: Challenge effect on coping (default: 0.2)
- `hindrance_penalty`: Hindrance effect on coping (default: 0.3)
- `social_influence_factor`: Social effect on coping (default: 0.15)

## Resource Dynamics Integration

### Resource Consumption for Coping

**Successful Coping**:
```python
if coped_successfully:
    resource_cost = config.resource_cost  # Default: 0.1
    agent.resources = max(0.0, agent.resources - resource_cost)
```

**Failed Coping**:
- No resource consumption for failed coping attempts

### Resource Regeneration

**Passive Regeneration**:
```python
def compute_resource_regeneration(current_resources):
    # Linear regeneration toward maximum
    regeneration = base_regeneration_rate * (1.0 - current_resources)
    return regeneration
```

**Affect Influence on Regeneration**:
```python
# Positive affect boosts regeneration
affect_multiplier = 1.0 + 0.2 * max(0.0, agent.affect)
regenerated_resources = base_regeneration * affect_multiplier
```

## Network Adaptation Triggers

### Stress Breach Adaptation

**Trigger Condition**:
```python
if stress_breach_count >= adaptation_threshold:
    # Trigger network adaptation
    adaptation_threshold = 3  # Default
```

**Adaptation Mechanisms**:
1. **Rewiring**: Connect to similar agents (homophily)
2. **Support Effectiveness**: Strengthen ties with effective supporters
3. **Stress-Based Homophily**: Prefer connections with similar stress levels

### Consecutive Hindrances Tracking

**Mechanism**:
```python
# Track consecutive hindrance events
if hindrance > challenge:
    agent.consecutive_hindrances += 1
else:
    agent.consecutive_hindrances = 0  # Reset counter
```

**Overload Trigger**:
```python
if consecutive_hindrances >= overload_threshold:
    # Apply overload resilience penalty
    overload_penalty = 0.2 * (consecutive_hindrances / 10.0)
    resilience = max(0.0, resilience - overload_penalty)
```

## Daily Reset Mechanisms

### End-of-Day Processing

**Daily Resilience Reset**:
```python
def daily_reset(current_resilience, baseline_resilience):
    # Apply homeostatic pull toward baseline
    distance = baseline_resilience - current_resilience
    reset_amount = daily_decay_rate * distance
    new_resilience = current_resilience + reset_amount

    return clamp(new_resilience, 0.0, 1.0)
```

**Stress Event Summary**:
```python
daily_summary = {
    'avg_stress': mean(stress_levels),
    'max_stress': max(stress_levels),
    'num_events': count(events),
    'coping_success_rate': mean(coping_outcomes)
}
```

## Configuration Parameters

### Core Resilience Parameters
```python
resilience_config = {
    'initial_resilience': 0.6,           # Starting resilience level
    'baseline_resilience': 0.6,          # Natural equilibrium point
    'coping_success_rate': 0.1,          # Base resilience gain from success
    'social_support_rate': 0.1,          # Resilience gain from social support
    'resilience_boost_rate': 0.1,        # Rate of resilience boost from protective factors
    'overload_threshold': 3,             # Events needed for overload
    'homeostatic_rate': 0.05             # Rate of return to baseline
}
```

### Coping Parameters
```python
coping_config = {
    'base_probability': 0.6,             # Base coping success rate
    'challenge_bonus': 0.2,              # Challenge effect on coping
    'hindrance_penalty': 0.3,            # Hindrance effect on coping
    'social_influence_factor': 0.15      # Social influence on coping
}
```

### Resource Parameters
```python
resource_config = {
    'base_regeneration': 0.05,           # Base regeneration rate
    'resource_cost': 0.1,                # Cost per successful coping
    'allocation_rate': 0.3               # Fraction allocated to protective factors
}
```

### Resilience Boost Rate (`RESILIENCE_BOOST_RATE`)

**Parameter**: `RESILIENCE_BOOST_RATE` (default: 0.1, range: 0.0-1.0)

**Description**: Controls the rate at which protective factors contribute to resilience building. This parameter determines how effectively an agent's investment in protective factors (social support, family support, formal interventions, psychological capital) translates into increased resilience capacity.

**Mathematical Effect**:
```
resilience_boost = efficacy × need_multiplier × RESILIENCE_BOOST_RATE
```

**Interpretation**:
- **High values (0.2-0.5)**: Protective factors quickly build resilience, representing highly effective interventions
- **Low values (0.0-0.1)**: Protective factors have minimal impact on resilience, representing less effective interventions
- **Research context**: This parameter can be calibrated against intervention effect sizes from mental health program evaluations

**Usage in Model**:
```python
# In resource allocation and resilience update
need_multiplier = max(0.1, 1.0 - current_resilience)
boost_rate = config.get('agent_parameters', 'resilience_boost_rate')
total_boost = efficacy * need_multiplier * boost_rate
```

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

## Validation Metrics

### Key Resilience Metrics

1. **Recovery Time**: Time to return to baseline after stress events
2. **Basin Stability**: Proportion of agents maintaining resilience above threshold
3. **Coping Success Rate**: Overall success rate across all stress events
4. **Overload Frequency**: Rate of cumulative overload events

### Calibration Targets

- **Literature-based**: Recovery times from psychological studies
- **Intervention Effects**: Expected improvement from mental health programs
- **Population Patterns**: Realistic distribution of resilience levels
- **Stress-Response Relationships**: Appropriate sensitivity to stress events
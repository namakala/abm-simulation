# Step-by-Step Calculations for Resilience, Stress, and Affect

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
```python
# Store initial values at beginning of day
initial_affect = self.affect
initial_resilience = self.resilience

# Get neighbor affects for social influence
neighbor_affects = self._get_neighbor_affects()

# Initialize daily tracking
daily_challenge = 0.0
daily_hindrance = 0.0
stress_events_count = 0
social_interactions_count = 0
```

**Neighbor Affect Collection**:
```python
def _get_neighbor_affects(self):
    neighbors = list(self.model.grid.get_neighbors(self.pos, include_center=False))
    return [neighbor.affect for neighbor in neighbors if hasattr(neighbor, 'affect')]
```

### Step 2: Subevent Generation and Execution

**Subevent Count Determination**:
```python
# Poisson sampling for number of daily subevents
n_subevents = sample_poisson(
    lam=config.get('agent', 'subevents_per_day'),  # Default: 7-8 events
    rng=self._rng,
    min_value=1
)

# Generate random sequence of actions
actions = [self._rng.choice(["interact", "stress"]) for _ in range(n_subevents)]
self._rng.shuffle(actions)
```

**Action Processing**:
```python
for action in actions:
    if action == "interact":
        self.interact()
        social_interactions_count += 1
    elif action == "stress":
        challenge, hindrance = self.stressful_event()
        daily_challenge += challenge
        daily_hindrance += hindrance
        stress_events_count += 1
```

### Step 3: Daily Challenge/Hindrance Integration

**Normalization by Event Count**:
```python
if stress_events_count > 0:
    daily_challenge /= stress_events_count
    daily_hindrance /= stress_events_count
else:
    daily_challenge = 0.0
    daily_hindrance = 0.0
```

**Interpretation**: Daily challenge/hindrance represent average event appraisal across all stress events in the day.

### Step 4: Affect Dynamics Calculation

**Integrated Affect Update**:
```python
def update_affect_dynamics(current_affect, baseline_affect, neighbor_affects,
                         challenge, hindrance, affect_config):
    # Component 1: Peer influence
    peer_effect = compute_peer_influence(current_affect, neighbor_affects, affect_config)

    # Component 2: Event appraisal
    appraisal_effect = compute_event_appraisal_effect(challenge, hindrance, current_affect, affect_config)

    # Component 3: Homeostasis
    homeostasis_effect = compute_homeostasis_effect(current_affect, baseline_affect, affect_config)

    # Combine effects
    total_effect = peer_effect + appraisal_effect + homeostasis_effect
    new_affect = current_affect + total_effect

    return clamp(new_affect, -1.0, 1.0)
```

**Detailed Component Calculations**:

#### Peer Influence
```python
def compute_peer_influence(self_affect, neighbor_affects, config):
    if not neighbor_affects:
        return 0.0

    n_neighbors = min(len(neighbor_affects), config.influencing_neighbors)
    selected_affects = neighbor_affects[:n_neighbors]

    influences = []
    for neighbor_affect in selected_affects:
        raw_influence = config.peer_influence_rate * (neighbor_affect - self_affect)
        influences.append(raw_influence)

    return mean(influences)
```

**Parameters**:
- `peer_influence_rate`: 0.1 (base social influence)
- `influencing_neighbors`: 5 (max neighbors considered)

#### Event Appraisal Effect
```python
def compute_event_appraisal_effect(challenge, hindrance, current_affect, config):
    # Challenge tends to improve affect (motivating)
    challenge_effect = config.event_appraisal_rate * challenge * (1.0 - current_affect)

    # Hindrance tends to worsen affect (demotivating)
    hindrance_effect = -config.event_appraisal_rate * hindrance * max(0.1, current_affect + 1.0)

    return challenge_effect + hindrance_effect
```

**Parameters**:
- `event_appraisal_rate`: 0.15 (challenge/hindrance effect strength)

#### Homeostasis Effect
```python
def compute_homeostasis_effect(current_affect, baseline_affect, config):
    distance_from_baseline = baseline_affect - current_affect
    homeostasis_strength = config.homeostatic_rate * abs(distance_from_baseline)

    if distance_from_baseline > 0:
        return homeostasis_strength  # Push up if below baseline
    else:
        return -homeostasis_strength  # Push down if above baseline
```

**Parameters**:
- `homeostatic_rate`: 0.05 (rate of return to baseline)

### Step 5: Resilience Dynamics Calculation

**Integrated Resilience Update**:
```python
def update_resilience_dynamics(current_resilience, coped_successfully,
                              received_social_support, consecutive_hindrances, config):
    # Component 1: Coping success effect
    coping_effect = 0.0
    if coped_successfully:
        coping_effect = config.coping_success_rate

    # Component 2: Social support effect
    social_support_effect = 0.0
    if received_social_support:
        social_support_effect = config.social_support_rate

    # Component 3: Overload effect
    overload_effect = compute_cumulative_overload(consecutive_hindrances, config)

    # Combine effects
    total_effect = coping_effect + social_support_effect + overload_effect
    new_resilience = current_resilience + total_effect

    return clamp(new_resilience, 0.0, 1.0)
```

**Detailed Component Calculations**:

#### Coping Success Effect
```python
# In the new stress processing mechanism, coping success is determined per event
# and integrated into resilience through challenge/hindrance effects

def compute_challenge_hindrance_resilience_effect(challenge, hindrance, coped_successfully):
    if coped_successfully:
        hindrance_effect = 0.1 * hindrance   # Small positive
        challenge_effect = 0.3 * challenge   # Large positive
    else:
        hindrance_effect = -0.4 * hindrance  # Large negative
        challenge_effect = -0.1 * challenge  # Small negative

    return hindrance_effect + challenge_effect
```

#### Social Support Effect
```python
# Social support received during interactions
received_social_support = social_interactions_count > 0 and self._rng.random() < 0.3

if received_social_support:
    support_boost = config.social_support_rate  # Default: 0.1
    resilience += support_boost
```

#### Overload Effect
```python
def compute_cumulative_overload(consecutive_hindrances, config):
    if consecutive_hindrances < config.overload_threshold:
        return 0.0

    overload_intensity = min(consecutive_hindrances / config.influencing_hindrance, 2.0)
    return -0.2 * overload_intensity
```

**Parameters**:
- `coping_success_rate`: 0.1 (resilience gain from successful coping)
- `social_support_rate`: 0.1 (resilience gain from social support)
- `overload_threshold`: 3 (events needed for overload)
- `influencing_hindrance`: 10 (scaling factor for overload)

### Step 6: Resource Dynamics Integration

**Resource Regeneration**:
```python
# Base regeneration
base_regeneration = compute_resource_regeneration(self.resources, regen_params)

# Affect influence on regeneration
affect_multiplier = 1.0 + 0.2 * max(0.0, self.affect)  # Positive affect boosts
self.resources += base_regeneration * affect_multiplier

# Resource consumption for coping
if coped_successfully:
    resource_cost = config.get('agent', 'resource_cost')  # Default: 0.1
    self.resources = max(0.0, self.resources - resource_cost)
```

**Resource Regeneration Function**:
```python
def compute_resource_regeneration(current_resources, config):
    # Linear regeneration toward maximum
    regeneration = config.base_regeneration * (1.0 - current_resources)
    return regeneration
```

**Parameters**:
- `base_regeneration`: 0.05 (base regeneration rate)
- `resource_cost`: 0.1 (cost per successful coping)

### Step 7: Protective Factor Management

**Resource Allocation to Protective Factors**:
```python
def allocate_protective_resources(available_resources, protective_factors, rng):
    # Softmax decision making based on current efficacy
    factors = ['social_support', 'family_support', 'formal_intervention', 'psychological_capital']
    efficacies = [protective_factors[f] for f in factors]

    # Temperature-scaled softmax
    temperature = config.get('utility', 'softmax_temperature')  # Default: 2.0
    logits = np.array(efficacies) / temperature
    softmax_weights = np.exp(logits) / np.sum(np.exp(logits))

    # Allocate proportionally
    allocations = {
        factor: available_resources * weight
        for factor, weight in zip(factors, softmax_weights)
    }

    return allocations
```

**Efficacy Updates**:
```python
for factor, allocation in allocations.items():
    if allocation > 0:
        current_efficacy = self.protective_factors[factor]
        improvement_rate = 0.5
        investment_effectiveness = 1.0 - current_efficacy  # Higher return when efficacy low

        efficacy_increase = allocation * improvement_rate * investment_effectiveness
        self.protective_factors[factor] = min(1.0, current_efficacy + efficacy_increase)

        self.resources -= allocation  # Consume resources for allocation
```

### Step 8: Homeostatic Adjustment

**Homeostatic Pull to Baseline**:
```python
def compute_homeostatic_adjustment(initial_value, final_value, homeostatic_rate, value_type):
    distance = homeostatic_rate * abs(final_value - initial_value)

    if final_value > initial_value:
        adjusted_value = final_value - distance  # Pull down
    elif final_value < initial_value:
        adjusted_value = final_value + distance  # Pull up
    else:
        adjusted_value = final_value

    # Apply bounds based on value type
    if value_type == 'affect':
        return clamp(adjusted_value, -1.0, 1.0)
    else:  # resilience
        return clamp(adjusted_value, 0.0, 1.0)
```

**Application to Both Systems**:
```python
# Apply to affect
self.affect = compute_homeostatic_adjustment(
    initial_value=self.baseline_affect,  # Fixed baseline
    final_value=self.affect,
    homeostatic_rate=affect_homeostatic_rate,
    value_type='affect'
)

# Apply to resilience
self.resilience = compute_homeostatic_adjustment(
    initial_value=self.baseline_resilience,  # Fixed baseline
    final_value=self.resilience,
    homeostatic_rate=resilience_homeostatic_rate,
    value_type='resilience'
)
```

**Parameters**:
- `affect_homeostatic_rate`: 0.05 (affect return to baseline)
- `resilience_homeostatic_rate`: 0.05 (resilience return to baseline)

### Step 9: Daily Reset and Tracking

**End-of-Day Processing**:
```python
def _daily_reset(self, current_day):
    # Apply daily affect reset to baseline
    self.affect = compute_daily_affect_reset(
        current_affect=self.affect,
        baseline_affect=self.baseline_affect,
        config=stress_config
    )

    # Apply stress decay
    self.current_stress = compute_stress_decay(
        current_stress=self.current_stress,
        config=stress_config
    )

    # Store daily summary
    if self.daily_stress_events:
        daily_summary = {
            'day': current_day,
            'avg_stress': mean([e['stress_level'] for e in self.daily_stress_events]),
            'max_stress': max([e['stress_level'] for e in self.daily_stress_events]),
            'num_events': len(self.daily_stress_events),
            'coping_success_rate': mean([e['coped_successfully'] for e in self.daily_stress_events])
        }
        self.stress_history.append(daily_summary)

    # Reset daily tracking
    self.daily_stress_events = []
    self.last_reset_day = current_day
```

**Stress Decay**:
```python
def compute_stress_decay(current_stress, config):
    # Exponential decay toward zero
    decayed_stress = current_stress * (1.0 - config.stress_decay_rate)
    return clamp(decayed_stress, 0.0, 1.0)
```

**Parameters**:
- `stress_decay_rate`: 0.05 (daily stress decay rate)

## Complete Integration Example

### Example: High-Challenge Event Day

**Scenario**: Agent experiences a high-challenge, low-hindrance event with positive neighbors

**Step-by-Step Calculations**:

1. **Event Generation**:
   - Controllability: 0.8, Predictability: 0.7, Overload: 0.2
   - Magnitude: 0.6

2. **Appraisal**:
   - z = 1.0×0.8 + 1.0×0.7 - 1.0×0.2 + 0.0 = 1.3
   - challenge = σ(6.0×1.3) = σ(7.8) ≈ 0.9996
   - hindrance = 1 - 0.9996 ≈ 0.0004

3. **Threshold Evaluation**:
   - L = 0.6 × (1 + 0.3×(0.0004 - 0.9996)) ≈ 0.6 × (1 - 0.3) ≈ 0.42
   - T_eff = 0.5 + 0.15×0.9996 - 0.25×0.0004 ≈ 0.5 + 0.15 ≈ 0.65
   - Stressed? 0.42 > 0.65? No

4. **Social Interaction**:
   - Neighbor affects: [0.2, 0.1, 0.3] (avg: 0.2)
   - Social influence: 0.1 × (0.2 - 0.0) = 0.02

5. **Affect Update**:
   - Peer effect: 0.02
   - Appraisal effect: 0.15 × 0.9996 × (1 - 0.0) ≈ 0.15
   - Homeostasis effect: 0.05 × (0.0 - 0.0) = 0.0
   - Total: 0.02 + 0.15 + 0.0 = 0.17
   - New affect: 0.0 + 0.17 = 0.17

6. **Resilience Update**:
   - Coping effect: 0.0 (no stress event)
   - Social support: 0.1 (received support)
   - Overload: 0.0 (no hindrances)
   - Total: 0.1
   - New resilience: 0.6 + 0.1 = 0.7

7. **Resource Update**:
   - Regeneration: 0.05 × (1 - 0.8) = 0.01
   - Affect multiplier: 1.0 + 0.2 × 0.17 ≈ 1.034
   - Final resources: 0.8 + 0.01 × 1.034 ≈ 0.81

### Example: High-Hindrance Event Day

**Scenario**: Agent experiences a high-hindrance, low-challenge event with negative neighbors

**Step-by-Step Calculations**:

1. **Event Generation**:
   - Controllability: 0.1, Predictability: 0.2, Overload: 0.9
   - Magnitude: 0.8

2. **Appraisal**:
   - z = 1.0×0.1 + 1.0×0.2 - 1.0×0.9 + 0.0 = -0.6
   - challenge = σ(6.0×-0.6) = σ(-3.6) ≈ 0.0266
   - hindrance = 1 - 0.0266 ≈ 0.9734

3. **Threshold Evaluation**:
   - L = 0.8 × (1 + 0.3×(0.9734 - 0.0266)) ≈ 0.8 × (1 + 0.283) ≈ 1.026 (capped at 1.0)
   - T_eff = 0.5 + 0.15×0.0266 - 0.25×0.9734 ≈ 0.5 + 0.004 - 0.243 ≈ 0.261
   - Stressed? 1.0 > 0.261? Yes

4. **Coping Determination**:
   - Base probability: 0.6
   - Challenge effect: 0.2 × 0.0266 ≈ 0.005
   - Hindrance effect: -0.3 × 0.9734 ≈ -0.292
   - Social effect: 0.15 × (-0.1) ≈ -0.015
   - Total probability: 0.6 + 0.005 - 0.292 - 0.015 ≈ 0.298
   - Coping successful? Random < 0.298? No

5. **State Updates**:
   - Affect change: -0.2 × 0.9734 ≈ -0.195
   - Resilience change: -0.4 × 0.9734 - 0.1 × 0.0266 ≈ -0.389 - 0.003 ≈ -0.392
   - Stress change: 0.3 × (1 + 0.9734) ≈ 0.3 × 1.973 ≈ 0.592

6. **Final Values**:
   - New affect: 0.0 - 0.195 = -0.195
   - New resilience: 0.6 - 0.392 = 0.208
   - New stress: 0.0 + 0.592 = 0.592

## Mathematical Integration Summary

### Core Integration Equations

**Affect Dynamics**:
```
A_{t+1} = A_t + π(A_t, N_t) + α(C_t, H_t, A_t) + η(A_t, A_baseline)
```

**Resilience Dynamics**:
```
R_{t+1} = R_t + κ(S_t) + σ(P_t) + ο(H_t) + ρ(C_t, H_t, S_t)
```

**Stress Processing**:
```
S_{t+1} = S_t + λ(C_t, H_t, P_t) - δ(S_t)
```

Where:
- `π`: Peer influence function
- `α`: Event appraisal function
- `η`: Homeostasis function
- `κ`: Coping success effect
- `σ`: Social support effect
- `ο`: Overload effect
- `ρ`: Challenge/hindrance resilience effect
- `λ`: Stress update function
- `δ`: Stress decay function

### Parameter Integration

All calculations use the unified configuration system:

```python
config = get_config()
affect_config = AffectDynamicsConfig()
resilience_config = ResilienceDynamicsConfig()
stress_config = StressProcessingConfig()
```

This ensures consistent parameter usage across all calculation steps and maintains the research reproducibility requirements.
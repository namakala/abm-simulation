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

### Stress Decay Rate (`STRESS_DECAY_RATE`)

**Parameter**: `STRESS_DECAY_RATE` (default: 0.05, range: 0.0-1.0)

**Description**: Controls the rate at which an agent's current stress level decays over time when no new stress events occur. This parameter represents natural stress recovery processes, forgetting of past stressors, and psychological adaptation to ongoing stress.

**Mathematical Effect**:
```
new_stress = current_stress × (1.0 - STRESS_DECAY_RATE)
```

**Interpretation**:
- **High values (0.1-1.0)**: Rapid stress decay, representing effective natural recovery mechanisms or high stress tolerance
- **Low values (0.0-0.05)**: Slow stress decay, representing persistent stress effects or difficulty recovering from stressors
- **Research context**: This parameter can be calibrated against psychological research on stress recovery times and resilience patterns

**Usage in Model**:
```python
# In daily reset process
def compute_stress_decay(current_stress, config):
    # Exponential decay toward zero
    decayed_stress = current_stress * (1.0 - config.stress_decay_rate)
    return clamp(decayed_stress, 0.0, 1.0)

# Apply during daily reset
self.current_stress = compute_stress_decay(self.current_stress, config)
```

**Theoretical Foundation**:
- **Natural Recovery**: Represents the psychological tendency to return to baseline stress levels over time
- **Adaptation Process**: Models how individuals habituate to ongoing stressors
- **Memory Effects**: Accounts for the fading of stressful memories and their emotional impact

**Integration with Other Systems**:
- **Stress Events**: New stress events add to current stress before decay is applied
- **Consecutive Hindrances**: Decay also affects the consecutive hindrances counter
- **Network Adaptation**: Persistent high stress (despite decay) may still trigger network changes

### Step 10: PSS-10 Score Computation and Agent State Integration

**PSS-10 Integration Steps**:

The PSS-10 integration occurs at the end of each simulation step and serves as both a measurement tool and a validation mechanism:

```python
def compute_pss10_score(self):
    """
    Recompute and update PSS-10 scores based on current stress levels.

    This function should be called at the end of each iteration step to:
    1. Generate new PSS-10 responses from current stress_controllability and stress_overload
    2. Update pss10_responses and pss10 score
    3. Update stress levels based on new PSS-10 responses
    4. Enable empirical validation and pattern matching
    """
    # Generate new PSS-10 responses from current stress levels
    new_responses = generate_pss10_responses(
        controllability=self.stress_controllability,
        overload=self.stress_overload,
        rng=self._rng
    )

    # Update PSS-10 state
    self.pss10_responses = new_responses
    self.pss10 = compute_pss10_score(new_responses)

    # Update stress levels based on new PSS-10 responses
    self._update_stress_levels_from_pss10()

    # Track PSS-10 history for analysis
    self.pss10_history.append({
        'step': self.model.schedule.steps,
        'pss10_total': self.pss10,
        'controllability': self.stress_controllability,
        'overload': self.stress_overload
    })
```

**Detailed PSS-10 Update Process**:

1. **Stress Dimension Input**: Current `stress_controllability` and `stress_overload` values
2. **Dimension Score Generation**: Create correlated dimension scores using bifactor model
3. **Item Response Generation**: Generate individual PSS-10 item responses (0-4 scale)
4. **Total Score Calculation**: Sum all 10 items to get total PSS-10 score (0-40)
5. **Reverse Mapping**: Update stress dimensions based on PSS-10 responses
6. **History Tracking**: Store PSS-10 trajectory for analysis and validation

**PSS-10 Dimension Score Generation**:
```python
def generate_pss10_dimension_scores(controllability, overload, correlation, rng, deterministic=False):
    """
    Generate correlated controllability and overload dimension scores.

    This implements the bifactor model where controllability and overload
    are correlated dimensions underlying PSS-10 responses.
    """
    # Create covariance matrix for bivariate normal distribution
    var_c = controllability_sd ** 2  # Variance of controllability dimension
    var_o = overload_sd ** 2         # Variance of overload dimension
    cov = correlation * np.sqrt(var_c * var_o)  # Covariance between dimensions

    # Mean vector and covariance matrix for multivariate normal
    mean_vector = np.array([controllability, overload])
    cov_matrix = np.array([[var_c, cov], [cov, var_o]])

    # Sample from multivariate normal distribution
    correlated_scores = rng.multivariate_normal(mean_vector, cov_matrix)

    # Clamp to [0,1] bounds
    controllability_score = max(0.0, min(1.0, correlated_scores[0]))
    overload_score = max(0.0, min(1.0, correlated_scores[1]))

    return controllability_score, overload_score
```

**Example Dimension Score Generation**:
```python
# Agent with moderate stress levels
controllability = 0.6  # Moderate perceived control
overload = 0.4         # Low-moderate overload
correlation = 0.3      # Moderate correlation between dimensions

# Generate correlated dimension scores
controllability_score, overload_score = generate_pss10_dimension_scores(
    controllability, overload, correlation, rng
)

# Example output: (0.58, 0.42) - slightly correlated as expected
```

**PSS-10 Item Response Generation**:
```python
def generate_pss10_item_response(item_mean, item_sd, controllability_loading, overload_loading,
                                controllability_score, overload_score, reverse_scored, rng, deterministic=False):
    """
    Generate a single PSS-10 item response based on dimension scores and factor loadings.

    This implements the bifactor model where each item loads on both controllability
    and overload dimensions with specified factor loadings.
    """
    # Linear combination of dimension scores weighted by factor loadings
    # Note: controllability is reverse-coded in the stress component
    stress_component = (controllability_loading * (1.0 - controllability_score) +
                       overload_loading * overload_score)

    # Normalize by total loading to get stress contribution
    total_loading = max(controllability_loading + overload_loading, 1e-10)
    normalized_stress = stress_component / total_loading

    # Adjust empirical mean based on stress level
    # Higher stress → Higher PSS-10 responses
    adjusted_mean = item_mean + (normalized_stress - 0.5) * 0.5
    raw_response = rng.normal(adjusted_mean, item_sd)

    # Apply reverse scoring if needed (for controllability items)
    if reverse_scored:
        raw_response = 4.0 - raw_response

    return int(round(clamp(raw_response, 0.0, 4.0)))

def generate_pss10_responses(controllability, overload, rng, deterministic=False):
    """
    Generate complete set of PSS-10 responses for an agent.

    Returns dictionary mapping item numbers (1-10) to response values (0-4).
    """
    responses = {}

    # PSS-10 item parameters (empirically derived)
    pss10_items = [
        {'item_num': 1, 'mean': 2.1, 'sd': 1.1, 'controllability_loading': 0.2, 'overload_loading': 0.7, 'reverse_scored': False},
        {'item_num': 2, 'mean': 1.8, 'sd': 0.9, 'controllability_loading': 0.8, 'overload_loading': 0.3, 'reverse_scored': False},
        {'item_num': 3, 'mean': 2.3, 'sd': 1.2, 'controllability_loading': 0.1, 'overload_loading': 0.8, 'reverse_scored': False},
        {'item_num': 4, 'mean': 1.9, 'sd': 1.0, 'controllability_loading': 0.7, 'overload_loading': 0.2, 'reverse_scored': True},
        {'item_num': 5, 'mean': 2.2, 'sd': 1.1, 'controllability_loading': 0.6, 'overload_loading': 0.4, 'reverse_scored': True},
        {'item_num': 6, 'mean': 1.7, 'sd': 0.8, 'controllability_loading': 0.1, 'overload_loading': 0.9, 'reverse_scored': False},
        {'item_num': 7, 'mean': 2.0, 'sd': 1.0, 'controllability_loading': 0.8, 'overload_loading': 0.2, 'reverse_scored': True},
        {'item_num': 8, 'mean': 1.6, 'sd': 0.9, 'controllability_loading': 0.6, 'overload_loading': 0.3, 'reverse_scored': True},
        {'item_num': 9, 'mean': 2.4, 'sd': 1.3, 'controllability_loading': 0.7, 'overload_loading': 0.4, 'reverse_scored': False},
        {'item_num': 10, 'mean': 1.5, 'sd': 0.8, 'controllability_loading': 0.1, 'overload_loading': 0.9, 'reverse_scored': False}
    ]

    # Generate response for each item
    for item in pss10_items:
        response = generate_pss10_item_response(
            item_mean=item['mean'],
            item_sd=item['sd'],
            controllability_loading=item['controllability_loading'],
            overload_loading=item['overload_loading'],
            controllability_score=controllability,
            overload_score=overload,
            reverse_scored=item['reverse_scored'],
            rng=rng,
            deterministic=deterministic
        )
        responses[item['item_num']] = response

    return responses
```

**Stress Level Update from PSS-10**:
```python
def _update_stress_levels_from_pss10(self):
    """Update stress levels based on current PSS-10 responses.

    This creates a feedback loop where PSS-10 measurements influence
    the underlying stress dimensions, enabling empirical calibration.
    """
    if not self.pss10_responses:
        return

    # Calculate controllability stress from relevant PSS-10 items
    # Items 4, 5, 7, 8 are reverse scored (higher response = lower controllability stress)
    controllability_items = [4, 5, 7, 8]
    controllability_scores = []
    for item_num in controllability_items:
        if item_num in self.pss10_responses:
            response = self.pss10_responses[item_num]
            # Reverse scoring: high PSS-10 response = low controllability stress
            controllability_stress = 1.0 - (response / 4.0)  # Normalize to [0,1]
            controllability_scores.append(controllability_stress)

    self.stress_controllability = np.mean(controllability_scores) if controllability_scores else 0.5

    # Calculate overload stress from relevant PSS-10 items
    # Items 1, 2, 3, 6, 9, 10 are regularly scored (higher response = higher overload stress)
    overload_items = [1, 2, 3, 6, 9, 10]
    overload_scores = []
    for item_num in overload_items:
        if item_num in self.pss10_responses:
            response = self.pss10_responses[item_num]
            # Normal scoring: high PSS-10 response = high overload stress
            overload_stress = response / 4.0  # Normalize to [0,1]
            overload_scores.append(overload_stress)

    self.stress_overload = np.mean(overload_scores) if overload_scores else 0.5

    # Update dimension scores for tracking
    self.pss10_dimension_scores = {
        'controllability': self.stress_controllability,
        'overload': self.stress_overload
    }
```

**Example PSS-10 Integration in Agent Step**:
```python
def step(self):
    """Complete agent step with integrated PSS-10 processing."""

    # 1. Initialize daily tracking
    initial_pss10 = self.pss10
    initial_stress_controllability = self.stress_controllability
    initial_stress_overload = self.stress_overload

    # 2. Process daily events (stress events, social interactions)
    for _ in range(self.daily_subevents):
        if self._rng.random() < 0.5:
            # Process stress event
            challenge, hindrance = self.stressful_event()
            # ... stress processing logic
        else:
            # Process social interaction
            self.interact()
            # ... interaction logic

    # 3. Update PSS-10 based on new stress state
    old_pss10 = self.pss10
    self.update_pss10_integration()

    # 4. Track PSS-10 changes for analysis
    pss10_change = self.pss10 - old_pss10
    if abs(pss10_change) > 0:
        self.pss10_changes.append({
            'step': self.model.schedule.steps,
            'pss10_before': old_pss10,
            'pss10_after': self.pss10,
            'change': pss10_change,
            'primary_stressor': self._identify_primary_stressor()
        })

    # 5. Apply PSS-10 influence on agent behavior
    self._apply_pss10_behavioral_effects()
```

**Parameters**:
- `bifactor_correlation`: 0.3 (correlation between controllability and overload dimensions)
- `controllability_sd`: 1.0 (standard deviation for controllability dimension)
- `overload_sd`: 1.0 (standard deviation for overload dimension)

## Complete Integration Example

### Example: High-Challenge Event Day

**Scenario**: Agent experiences a high-challenge, low-hindrance event with positive neighbors

**Step-by-Step Calculations**:

1. **Event Generation**:
   - Controllability: 0.8, Overload: 0.2

2. **Appraisal**:
   - z = 1.0×0.8 - 1.0×0.2 + 0.0 = 0.6
   - challenge = σ(6.0×0.6) = σ(3.6) ≈ 0.973
   - hindrance = 1 - 0.973 ≈ 0.027

3. **Threshold Evaluation**:
   - L = 1 + 0.3×(0.027 - 0.973) ≈ 1 - 0.283 ≈ 0.717
   - T_eff = 0.5 + 0.15×0.973 - 0.25×0.027 ≈ 0.5 + 0.146 - 0.007 ≈ 0.639
   - Stressed? 0.717 > 0.639? Yes

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
   - Controllability: 0.1, Overload: 0.9

2. **Appraisal**:
   - z = 1.0×0.1 - 1.0×0.9 + 0.0 = -0.8
   - challenge = σ(6.0×-0.8) = σ(-4.8) ≈ 0.008
   - hindrance = 1 - 0.008 ≈ 0.992

3. **Threshold Evaluation**:
   - L = 1 + 0.3×(0.992 - 0.008) ≈ 1 + 0.296 ≈ 1.296 (capped at 1.0)
   - T_eff = 0.5 + 0.15×0.008 - 0.25×0.992 ≈ 0.5 + 0.001 - 0.248 ≈ 0.253
   - Stressed? 1.0 > 0.253? Yes

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
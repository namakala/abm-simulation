# Affect Dynamics and Influencing Factors

## Overview

Affect in the agent-based model represents emotional state on a continuum from negative to positive valence. The system implements multiple influencing factors including social interactions, stress events, peer influence, and homeostatic processes.

## Core Affect State

### State Variables

- **Affect Level (A ∈ [-1,1])**: Current emotional state (-1 = very negative, +1 = very positive)
- **Baseline Affect (A_baseline ∈ [-1,1])**: Agent's natural emotional equilibrium
- **Daily Initial Affect**: Affect level at the start of each day for tracking

## Social Influence Mechanisms

### Peer Influence

**Core Mechanism**: Agents influence each other's affect through social interactions:

```python
def compute_peer_influence(self_affect, neighbor_affects, config):
    if not neighbor_affects:
        return 0.0

    # Limit to specified number of influencing neighbors
    n_neighbors = min(len(neighbor_affects), config.influencing_neighbors)
    selected_affects = neighbor_affects[:n_neighbors]

    # Compute influence from each neighbor
    influences = []
    for neighbor_affect in selected_affects:
        # Positive neighbor pulls self upward, negative pulls downward
        raw_influence = config.peer_influence_rate * (neighbor_affect - self_affect)
        influences.append(raw_influence)

    return mean(influences)
```

**Key Parameters**:
- `peer_influence_rate`: Base rate of social influence (default: 0.1)
- `influencing_neighbors`: Maximum neighbors considered (default: 5)

### Interaction Effects

**Mutual Influence**: Social interactions create bidirectional affect changes:

```python
def process_interaction(self_affect, partner_affect, config):
    # Each agent influences the other
    self_influence = compute_social_influence(partner_affect, self_affect, config)
    partner_influence = compute_social_influence(self_affect, partner_affect, config)

    # Apply asymmetric weighting: negative affects have stronger impact
    if self_influence < 0:
        self_influence *= 1.5  # Negative influence 50% stronger
    if partner_influence < 0:
        partner_influence *= 1.5  # Negative influence 50% stronger

    return self_affect + self_influence, partner_affect + partner_influence
```

**Asymmetric Impact**: Negative emotional states have 50% stronger influence than positive states, reflecting realistic social dynamics where negative interactions are more impactful.

## Stress Event Impact on Affect

### Challenge-Hindrance Effects

**Successful Coping**:
```python
if coped_successfully:
    affect_change = challenge_bonus * challenge
else:
    affect_change = -hindrance_penalty * hindrance
```

**Interpretation**:
- **Challenge Success**: Provides positive affect boost proportional to challenge level
- **Hindrance Failure**: Creates negative affect impact proportional to hindrance level

### Event Appraisal Effects

**Core Mechanism**: Challenge and hindrance appraisals directly influence affect:

```python
def compute_event_appraisal_effect(challenge, hindrance, current_affect, config):
    # Challenge tends to improve affect (motivating)
    challenge_effect = config.event_appraisal_rate * challenge * (1.0 - current_affect)

    # Hindrance tends to worsen affect (demotivating)
    hindrance_effect = -config.event_appraisal_rate * hindrance * max(0.1, current_affect + 1.0)

    return challenge_effect + hindrance_effect
```

**Mathematical Foundation**:
- Challenge provides motivational boost when affect is low
- Hindrance creates stronger negative impact when agent is already struggling

## Homeostatic Mechanisms

### Daily Homeostasis

**Core Mechanism**: Affect tends to return to baseline levels:

```python
def compute_homeostasis_effect(current_affect, baseline_affect, config):
    # Homeostasis strength increases with distance from baseline
    distance_from_baseline = baseline_affect - current_affect
    homeostasis_strength = config.homeostatic_rate * abs(distance_from_baseline)

    # Direction toward baseline
    if distance_from_baseline > 0:
        return homeostasis_strength  # Push up if below baseline
    else:
        return -homeostasis_strength  # Push down if above baseline
```

**Daily Reset**: At the end of each day, affect is reset toward baseline:

```python
def compute_daily_affect_reset(current_affect, baseline_affect, config):
    distance = baseline_affect - current_affect
    reset_amount = config.daily_decay_rate * distance
    new_affect = current_affect + reset_amount

    return clamp(new_affect, -1.0, 1.0)
```

### Baseline Affect Dynamics

**Fixed Baseline**: Each agent has a stable baseline affect that represents their natural emotional equilibrium:

- **Initialization**: Set at agent creation based on individual differences
- **Stability**: Remains constant throughout simulation (represents personality)
- **Homeostatic Target**: All dynamics pull affect toward this fixed point

## Integrated Affect Dynamics

### Complete Update Process

**Daily Affect Update**:
```python
def update_affect_dynamics(current_affect, baseline_affect, neighbor_affects,
                          challenge=0.0, hindrance=0.0, config=None):
    # Compute individual effect components
    peer_effect = compute_peer_influence(current_affect, neighbor_affects, config)
    appraisal_effect = compute_event_appraisal_effect(challenge, hindrance, current_affect, config)
    homeostasis_effect = compute_homeostasis_effect(current_affect, baseline_affect, config)

    # Combine all effects
    total_effect = peer_effect + appraisal_effect + homeostasis_effect

    # Apply the change
    new_affect = current_affect + total_effect

    return clamp(new_affect, -1.0, 1.0)
```

### Effect Component Weighting

| Component | Default Rate | Range | Description |
|-----------|--------------|-------|-------------|
| Peer Influence | 0.1 | 0.0-0.5 | Social contagion of affect |
| Event Appraisal | 0.15 | 0.0-0.3 | Challenge/hindrance emotional impact |
| Homeostasis | 0.05 | 0.0-0.2 | Return to baseline tendency |

## Resilience-Affect Interactions

### Bidirectional Influence

**Affect → Resilience**:
- **Resource Regeneration**: Positive affect boosts resource recovery
- **Mechanism**: `affect_multiplier = 1.0 + 0.2 × max(0.0, affect)`

**Resilience → Affect**:
- **Coping Confidence**: High resilience provides emotional buffer
- **Mechanism**: Integrated through challenge/hindrance appraisal

### Threshold Effects

**Affect Thresholds**: Some resilience effects only occur when affect exceeds thresholds:

```python
if abs(partner_affect) > affect_threshold:
    # Apply resilience influence from social interaction
    resilience_change = partner_affect * resilience_influence_rate
```

**Parameters**:
- `affect_threshold`: Minimum affect level for social influence (default: 0.3)
- `resilience_influence_rate`: Rate of resilience change from affect (default: 0.1)

## Stress Processing Integration

### New Stress Mechanism

**Enhanced Stress Processing**:
```python
def process_stress_event_with_new_mechanism(current_affect, current_resilience,
                                           challenge, hindrance, neighbor_affects):
    # Compute coping probability with social influence
    coping_prob = compute_coping_probability(challenge, hindrance, neighbor_affects)

    # Determine coping outcome
    coped_successfully = random() < coping_prob

    # Update affect based on outcome
    if coped_successfully:
        affect_change = 0.1 * challenge  # Challenge provides positive boost
    else:
        affect_change = -0.2 * hindrance  # Hindrance creates negative impact

    new_affect = current_affect + affect_change

    return new_affect, new_resilience, new_stress, coped_successfully
```

### Coping-Affect Feedback Loop

**Successful Coping**:
- **Immediate Effect**: Positive affect boost from challenge
- **Resource Cost**: Consumes resources, potentially reducing future coping capacity
- **Long-term**: Builds resilience for future stress events

**Failed Coping**:
- **Immediate Effect**: Negative affect impact from hindrance
- **No Resource Cost**: Conserves resources for future attempts
- **Long-term**: Depletes resilience, increases future failure probability

## Social Network Effects

### Neighbor Selection

**Network Structure**: Agents interact with neighbors in a grid network:

```python
def get_neighbor_affects(agent):
    neighbors = model.grid.get_neighbors(agent.pos, include_center=False)
    return [neighbor.affect for neighbor in neighbors if hasattr(neighbor, 'affect')]
```

**Network Influence**:
- **Local Clustering**: Similar affect states cluster in neighborhoods
- **Social Contagion**: Affect spreads through local social networks
- **Homophily**: Similar agents tend to connect and influence each other

### Interaction Frequency

**Subevent Structure**: Each day consists of multiple interaction opportunities:

```python
# Daily subevents with mixed stress/interaction events
n_subevents = poisson_sample(lam=subevents_per_day)  # Default: 5-10 events
actions = random_choice(["interact", "stress"], n_subevents)
```

**Interaction Probability**: Approximately 50% of daily events are social interactions.

## Configuration Parameters

### Core Affect Parameters
```python
affect_config = {
    'initial_affect': 0.0,              # Starting affect level
    'baseline_affect': 0.0,             # Natural equilibrium point
    'peer_influence_rate': 0.1,         # Social influence strength
    'event_appraisal_rate': 0.15,       # Challenge/hindrance effect rate
    'homeostatic_rate': 0.05,           # Return to baseline rate
    'influencing_neighbors': 5          # Max neighbors for influence
}
```

### Interaction Parameters
```python
interaction_config = {
    'influence_rate': 0.1,              # Base interaction influence
    'resilience_influence': 0.1,        # Affect-resilience interaction
    'max_neighbors': 8,                 # Maximum neighbor connections
    'negative_amplification': 1.5       # Negative affect multiplier
}
```

### Stress Integration Parameters
```python
stress_integration_config = {
    'challenge_bonus': 0.1,             # Challenge affect boost
    'hindrance_penalty': 0.2,           # Hindrance affect penalty
    'affect_threshold': 0.3,            # Threshold for social effects
    'coping_affect_link': 0.15          # Affect influence on coping
}
```

## Calibration and Validation

### Key Affect Metrics

1. **Affect Distribution**: Population distribution of affect levels
2. **Social Contagion Rate**: Speed of affect spread through networks
3. **Stress-Affect Sensitivity**: Responsiveness to stress events
4. **Homeostatic Stability**: Rate of return to baseline levels

### Empirical Targets

- **Baseline Distribution**: Approximately normal around 0.0
- **Response to Stress**: Appropriate sensitivity to negative events
- **Social Influence**: Realistic rates of emotional contagion
- **Recovery Patterns**: Natural return to equilibrium over time

## Integration Points

### System Interactions

**Stress System**:
- Stress events trigger immediate affect changes
- Coping outcomes influence long-term affect trajectories
- Challenge/hindrance appraisals directly modify affect

**Resilience System**:
- High affect improves resource regeneration
- Resilience provides emotional buffer against stress
- Social support requires positive affect for effectiveness

**Social Network**:
- Neighbor affect influences daily affect dynamics
- Network structure affects social contagion patterns
- Homophily emerges from similar affect states

### Feedback Loops

1. **Positive Loop**: High affect → Better coping → Higher resilience → Better affect
2. **Negative Loop**: Low affect → Poor coping → Lower resilience → Worse affect
3. **Social Amplification**: Similar neighbors reinforce existing affect states
4. **Homeostatic Regulation**: Natural tendency to return to baseline equilibrium
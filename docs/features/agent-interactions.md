# Agent Interaction Mechanisms and Impact Analysis

## Overview

Agent interactions form the core social dynamics of the mental health model. Agents interact through social networks, influencing each other's affect and resilience while creating emergent patterns of social support, emotional contagion, and network adaptation.

## Network Structure

### Grid-Based Social Network

**Implementation**: Agents are positioned on a Mesa NetworkGrid:

```python
# NetworkGrid provides spatial social structure
self.model.grid = NetworkGrid(model.grid_width, model.grid_height, torus=False)
```

**Network Properties**:
- **Topology**: 2D grid with Moore neighborhood (8 possible neighbors)
- **Boundary Conditions**: Non-torus (edges don't wrap around)
- **Connection Type**: Spatial proximity determines interaction opportunities

### Neighbor Relationships

**Neighbor Discovery**:
```python
def get_neighbors(agent):
    """Get all adjacent agents within Moore neighborhood."""
    neighbors = list(self.model.grid.get_neighbors(
        agent.pos, include_center=False
    ))
    return neighbors
```

**Network Statistics**:
- **Mean Degree**: Configurable (default: 6 neighbors)
- **Clustering**: Natural clustering from spatial structure
- **Homophily**: Emerges from similar stress/affect states

## Interaction Process

### Interaction Initiation

**Random Partner Selection**:
```python
def interact(self):
    """Interact with a randomly selected neighbor."""
    neighbors = self.get_neighbors()

    if not neighbors:
        return  # No neighbors to interact with

    # Select random interaction partner
    partner = self._rng.choice(neighbors)

    # Process interaction using utility functions
    process_interaction(self, partner)
```

**Interaction Frequency**: Approximately 50% of daily subevents are social interactions.

### Mutual Influence Mechanism

**Bidirectional Affect Changes**:
```python
def process_interaction(self_affect, partner_affect, self_resilience, partner_resilience, config):
    # Compute mutual influences
    self_influence, partner_influence = compute_mutual_influence(
        self_affect, partner_affect, config
    )

    # Apply asymmetric negative weighting
    if self_influence < 0:
        self_influence *= 1.5  # Negative influence amplified
    if partner_influence < 0:
        partner_influence *= 1.5  # Negative influence amplified

    # Update affect states
    new_self_affect = self_affect + self_influence
    new_partner_affect = partner_affect + partner_influence

    # Update resilience states (threshold-based)
    if abs(partner_affect) > config.affect_threshold:
        resilience_influence = config.resilience_influence * partner_affect
        new_self_resilience = self_resilience + resilience_influence
        new_partner_resilience = partner_resilience + resilience_influence

    return new_self_affect, new_partner_affect, new_self_resilience, new_partner_resilience
```

## Social Influence Dynamics

### Affect Contagion

**Core Mechanism**: Affect spreads through social interactions:

```python
def compute_social_influence(self_affect, partner_affect, config):
    """Compute affect change due to social interaction."""
    # Influence proportional to partner's affect and base rate
    influence = config.influence_rate * sign(partner_affect)
    return influence
```

**Asymmetric Effects**:
- **Positive Affect**: Spreads at base rate
- **Negative Affect**: Spreads at 150% strength (negative bias)

### Resilience Influence

**Threshold-Based Mechanism**:
```python
def compute_resilience_influence(partner_affect, config):
    """Compute resilience change due to partner's affect."""
    # Only applies when partner's affect exceeds threshold
    if abs(partner_affect) > config.affect_threshold:
        return config.resilience_influence * partner_affect
    return 0.0
```

**Interpretation**: Strong emotional states from neighbors can influence resilience, but only when affect is sufficiently intense.

## Network Adaptation Mechanisms

### Stress-Based Rewiring

**Trigger Condition**:
```python
def should_adapt_network(agent):
    """Check if agent should adapt network connections."""
    stress_breach_count = getattr(agent, 'stress_breach_count', 0)
    adaptation_threshold = 3  # Adapt after 3 stress breaches

    return stress_breach_count >= adaptation_threshold
```

**Rewiring Process**:
```python
def adapt_network(self, current_neighbors):
    """Adapt network connections based on stress patterns."""
    for neighbor in current_neighbors:
        if self._rng.random() < rewire_prob:
            # Calculate connection quality metrics
            affect_similarity = 1.0 - abs(self.affect - neighbor.affect)
            resilience_similarity = 1.0 - abs(self.resilience - neighbor.resilience)
            support_effectiveness = self.get_support_effectiveness(neighbor)

            # Decide whether to keep connection
            keep_prob = (affect_similarity * homophily_strength +
                        support_effectiveness * (1.0 - homophily_strength))

            if self._rng.random() > keep_prob:
                self.rewire_to_similar_agent(current_neighbors)
```

### Homophily-Based Connection

**Similarity Calculation**:
```python
def compute_similarity(self, other_agent):
    """Compute similarity between agents for homophily."""
    affect_similarity = 1.0 - abs(self.affect - other_agent.affect)
    resilience_similarity = 1.0 - abs(self.resilience - other_agent.resilience)
    stress_similarity = 1.0 - abs(self.current_stress - other_agent.current_stress)

    return (affect_similarity + resilience_similarity + stress_similarity) / 3.0
```

**Connection Preferences**:
- **Similar Stress Levels**: Agents prefer connections with similar stress experiences
- **Similar Affect States**: Emotional similarity drives connection preferences
- **Support Effectiveness**: Historical success of support influences connection strength

## Social Support Dynamics

### Support Provision

**Support Opportunity**:
```python
def provide_social_support(self, neighbor_affects):
    """Determine if agent receives social support."""
    if not neighbor_affects:
        return False

    # Social support occurs probabilistically based on neighbor states
    avg_neighbor_affect = mean(neighbor_affects)
    support_probability = 0.3 + 0.2 * max(0.0, avg_neighbor_affect)

    return self._rng.random() < support_probability
```

**Support Effectiveness**:
```python
def get_support_effectiveness(self, neighbor):
    """Calculate effectiveness of neighbor as support provider."""
    # Based on neighbor's current state and historical support
    neighbor_fitness = (neighbor.resilience + (1.0 + neighbor.affect) / 2.0) / 2.0
    base_effectiveness = min(1.0, neighbor_fitness + 0.2)

    # Could be enhanced with historical tracking
    return base_effectiveness
```

### Support Impact on Resilience

**Resilience Boost**:
```python
if received_social_support:
    support_boost = social_support_rate * support_effectiveness
    self.resilience = min(1.0, self.resilience + support_boost)
```

**Parameters**:
- `social_support_rate`: Base resilience gain from support (default: 0.1)
- `support_effectiveness`: Quality multiplier (0.0-1.0)

## Interaction Impact Analysis

### Affect Contagion Patterns

**Emergent Phenomena**:
1. **Emotional Clustering**: Similar affect states cluster spatially
2. **Contagion Waves**: Negative affect can spread through neighborhoods
3. **Resilience Pockets**: High-resilience areas resist negative contagion

**Mathematical Model**:
```
ΔA_i = Σ_{j∈neighbors} (α × (A_j - A_i)) + β × H(A_i)
```

Where:
- `ΔA_i`: Affect change for agent i
- `α`: Social influence rate
- `A_j`: Neighbor j's affect
- `β`: Homeostatic rate
- `H(A_i)`: Homeostatic function

### Network Structure Effects

**Grid Topology Impact**:
- **Local Influence**: Interactions limited to immediate neighbors
- **Spatial Correlation**: Affect patterns show spatial autocorrelation
- **Boundary Effects**: Edge agents have fewer social connections

**Network Density Effects**:
- **High Density**: Faster affect spread, more social support opportunities
- **Low Density**: Slower contagion, fewer support resources

## Coping Social Influence

### Social Influence on Coping

**Mechanism**: Neighbor affect states influence coping success:

```python
def compute_coping_probability(challenge, hindrance, neighbor_affects, config):
    # Base probability
    base_prob = config.base_coping_probability

    # Challenge/hindrance effects
    challenge_effect = config.challenge_bonus * challenge
    hindrance_effect = -config.hindrance_penalty * hindrance

    # Social influence on coping
    social_effect = 0.0
    if neighbor_affects:
        avg_neighbor_affect = mean(neighbor_affects)
        social_effect = config.social_influence_factor * avg_neighbor_affect

    coping_prob = base_prob + challenge_effect + hindrance_effect + social_effect

    return clamp(coping_prob, 0.0, 1.0)
```

**Social Influence Parameters**:
- `social_influence_factor`: Strength of social influence on coping (default: 0.15)
- `base_coping_probability`: Baseline coping success rate (default: 0.6)

### Collective Coping Patterns

**Emergent Effects**:
1. **Social Buffering**: Positive neighbor affect improves coping success
2. **Stress Contagion**: Negative neighbor affect reduces coping capacity
3. **Resilience Clusters**: Groups with high resilience support each other

## Resource Allocation in Social Context

### Social Support Resource Cost

**Resource Consumption for Interactions**:
```python
# Social interactions may consume resources
if interaction_outcome == 'support_provided':
    resource_cost = config.social_cost
    self.resources = max(0.0, self.resources - resource_cost)
```

**Resource Investment in Network**:
```python
# Agents may invest resources to maintain social connections
if maintain_social_network:
    network_cost = config.network_maintenance_cost
    self.resources = max(0.0, self.resources - network_cost)
```

## Adaptation and Learning

### Network Learning Mechanisms

**Support Effectiveness Tracking**:
```python
def update_support_history(self, partner, support_successful):
    """Track historical support effectiveness."""
    if not hasattr(self, 'support_history'):
        self.support_history = {}

    partner_id = id(partner)
    if partner_id not in self.support_history:
        self.support_history[partner_id] = []

    self.support_history[partner_id].append(support_successful)

    # Keep only recent history
    if len(self.support_history[partner_id]) > 10:
        self.support_history[partner_id] = self.support_history[partner_id][-10:]
```

**Adaptive Connection Strategies**:
```python
def adapt_connection_strategy(self):
    """Adapt connection preferences based on experience."""
    # Analyze support history
    successful_supports = sum(self.support_history.values())
    total_supports = len(self.support_history)

    if total_supports > 0:
        success_rate = successful_supports / total_supports

        # Adjust homophily preferences based on success
        if success_rate > 0.7:
            self.homophily_preference += 0.1  # Prefer similar agents more
        elif success_rate < 0.3:
            self.homophily_preference -= 0.1  # Prefer diverse agents more
```

## Configuration Parameters

### Core Interaction Parameters
```python
interaction_config = {
    'influence_rate': 0.1,              # Base social influence rate
    'resilience_influence': 0.1,        # Resilience influence rate
    'max_neighbors': 8,                 # Maximum neighbor connections
    'affect_threshold': 0.3,            # Threshold for resilience influence
    'negative_amplification': 1.5       # Negative affect multiplier
}
```

### Network Adaptation Parameters
```python
adaptation_config = {
    'rewire_probability': 0.01,         # Probability of considering rewiring
    'adaptation_threshold': 3,          # Stress events before adaptation
    'homophily_strength': 0.7,          # Weight of similarity in connections
    'support_memory_length': 10         # Historical support tracking length
}
```

### Social Support Parameters
```python
support_config = {
    'base_support_probability': 0.3,    # Base probability of receiving support
    'affect_support_bonus': 0.2,        # Bonus from positive neighbor affect
    'support_effectiveness_base': 0.2,  # Base support quality
    'social_cost': 0.05                 # Resource cost of social interaction
}
```

## Impact Assessment

### Population-Level Effects

**Emergent Patterns**:
1. **Emotional Clustering**: Similar affect states form spatial clusters
2. **Support Networks**: Effective supporters become central nodes
3. **Stress Gradients**: Stress levels show spatial gradients
4. **Resilience Communities**: High-resilience areas emerge and persist

### Intervention Effects

**Network-Based Interventions**:
- **Targeted Support**: Interventions in low-resilience areas
- **Connection Enhancement**: Programs that improve social connectivity
- **Support Training**: Programs that improve support effectiveness

### Validation Metrics

**Network Metrics**:
- **Clustering Coefficient**: Measure of local clustering
- **Average Path Length**: Measure of social connectivity
- **Homophily Index**: Measure of connection preferences
- **Support Network Density**: Measure of support availability

**Behavioral Metrics**:
- **Interaction Frequency**: Rate of social interactions
- **Support Success Rate**: Effectiveness of provided support
- **Network Adaptation Rate**: Frequency of connection changes
- **Emotional Contagion Speed**: Rate of affect spread through network

## Integration with Other Systems

### Stress System Integration

**Stress → Network**:
- High stress triggers network adaptation
- Stress patterns influence connection preferences
- Failed coping may lead to social withdrawal

**Network → Stress**:
- Social support improves coping success
- Neighbor affect influences stress response
- Network structure affects stress propagation

### Resilience System Integration

**Resilience → Network**:
- High resilience agents become better support providers
- Resilience patterns influence network adaptation
- Successful coping strengthens social ties

**Network → Resilience**:
- Social support provides direct resilience boost
- Network structure affects resilience distribution
- Support effectiveness influences resilience trajectories

### Affect System Integration

**Affect → Network**:
- Similar affect states drive homophily
- Extreme affect may trigger network adaptation
- Affect influences support effectiveness

**Network → Affect**:
- Social contagion spreads affect through network
- Support interactions change affect states
- Network structure influences affect dynamics
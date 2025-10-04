# Resource Management: Protective Factors and Resource Allocation

## Overview

Resource management forms the core constraint mechanism in the agent-based mental health model, implementing the principle that individuals have limited psychological and physical resources to allocate across various protective strategies. This system draws from conservation of resources theory and realistic decision-making under constraints.

## Theoretical Background

### Conservation of Resources (COR) Theory

**Core Principle**: "Individuals strive to retain, protect, and build resources and that what is threatening to them is the potential or actual loss of these valued resources" (Hobfoll, 1989).

**Key Tenets Applied**:
1. **Resource Loss Primacy**: Resource losses have stronger impact than equivalent gains
2. **Resource Investment**: Individuals invest resources to protect against future losses
3. **Spiral Effects**: Initial losses can create loss spirals, gains can create gain spirals

### Protective Factors Framework

**Psychological Capital**: Self-efficacy, hope, optimism, and resilience (Luthans et al., 2007)
**Social Support**: Emotional, informational, and instrumental support networks
**Family Support**: Kin-based resources and relational security
**Formal Interventions**: Professional mental health services and treatments

### Resource Allocation Theory

**Decision-Making Under Constraints**: Agents must allocate limited resources optimally across protective factors, trading off between immediate needs and long-term benefits.

**Softmax Decision Making**: Implements bounded rationality where allocation decisions become more deterministic as resources become scarce.

## Resource State Variables

### Core Resource Pool

**Resources (R ∈ [0,1])**: Finite psychological/physical resources available for allocation

```python
# Resource state tracking
self.resources = initial_resources  # Starting level (default: 0.8)
self.baseline_resources = initial_resources  # Natural equilibrium
```

**Resource Dynamics**:
- **Passive Regeneration**: Gradual return toward maximum capacity
- **Consumption**: Used for coping with stress events and maintaining protective factors
- **Investment**: Allocated to enhance protective factor efficacy

### Protective Factors

**Four Core Protective Mechanisms**:

```python
self.protective_factors = {
    'social_support': 0.5,      # Efficacy of social relationships
    'family_support': 0.5,      # Efficacy of family connections
    'formal_intervention': 0.5, # Efficacy of professional help
    'psychological_capital': 0.5 # Self-efficacy and coping skills
}
```

## Resource Regeneration Mechanisms

### Passive Regeneration

**Mathematical Foundation**:
```
ΔR_regen = γ_R × (1 - R_current)
```

Where:
- `γ_R`: Base regeneration rate (default: 0.05)
- `R_current`: Current resource level

**Interpretation**: Resources naturally regenerate toward maximum capacity, representing rest, recovery, and natural healing processes.

### Affect-Modulated Regeneration

**Affect Influence**:
```python
affect_multiplier = 1.0 + 0.2 × max(0.0, self.affect)
regenerated_resources = base_regeneration × affect_multiplier
```

**Rationale**: Positive emotional states enhance recovery and resource rebuilding, while negative states may slow regeneration.

**Psychological Foundation**: Positive affect broadens thought-action repertoires and builds personal resources (Fredrickson, 2001).

## Resource Consumption Patterns

### Coping Consumption

**Stress Event Coping**:
```python
if coped_successfully:
    resource_cost = κ × (1 + δ × hindrance)
    self.resources = max(0.0, self.resources - resource_cost)
```

**Parameters**:
- `κ`: Base resource cost (default: 0.1)
- `δ`: Hindrance scaling factor (default: 0.2)

**Rationale**: Failed coping with hindrance events requires more resources due to increased psychological demands.

### Protective Factor Maintenance

**Allocation Cost**:
```python
def compute_allocation_cost(allocated_amount):
    # Convex cost function represents diminishing returns
    cost = κ × (allocated_amount ^ γ_c)
    return cost
```

**Parameters**:
- `κ`: Cost scalar (default: 0.15)
- `γ_c`: Cost exponent (default: 1.5)

**Rationale**: Maintaining protective factors requires ongoing resource investment, with increasing marginal costs.

## Resource Allocation Mechanisms

### Softmax Decision Framework

**Mathematical Foundation**:
```python
def allocate_protective_resources(available_resources, protective_factors, temperature):
    # Convert efficacy to allocation logits
    efficacies = [pf.social_support, pf.family_support, pf.formal_intervention, pf.psychological_capital]
    logits = np.array(efficacies) / temperature

    # Softmax probabilities
    softmax_weights = np.exp(logits) / np.sum(np.exp(logits))

    # Allocate proportionally
    allocations = {
        'social_support': available_resources * softmax_weights[0],
        'family_support': available_resources * softmax_weights[1],
        'formal_intervention': available_resources * softmax_weights[2],
        'psychological_capital': available_resources * softmax_weights[3]
    }

    return allocations
```

**Temperature Effects**:
- **High Temperature (τ > 1)**: More random allocation (exploration)
- **Low Temperature (τ < 1)**: More deterministic allocation (exploitation)
- **Default**: τ = 2.0 (moderate stochasticity)

### Allocation Rationale

**Need-Based Allocation**:
```python
# Higher allocation to factors with lower current efficacy
investment_effectiveness = 1.0 - current_efficacy

# Higher allocation when overall resilience is low
need_multiplier = max(0.1, 1.0 - agent_resilience)
```

**Investment Return Model**:
```python
efficacy_increase = allocation × improvement_rate × investment_effectiveness × need_multiplier
new_efficacy = min(1.0, current_efficacy + efficacy_increase)
```

## Protective Factor Efficacy Dynamics

### Individual Factor Characteristics

#### Social Support
**Definition**: Quality and availability of emotional and instrumental support from social network

**Efficacy Parameters**:
- `α_soc`: Efficacy in reducing distress (default: 0.4)
- `ρ_soc`: Resource replenishment rate (default: 0.08)

**Theoretical Basis**: Social support buffers stress through emotional and practical assistance (Cohen & Wills, 1985).

#### Family Support
**Definition**: Support from family relationships and kinship networks

**Efficacy Parameters**:
- `α_fam`: Efficacy in reducing distress (default: 0.5)
- `ρ_fam`: Resource replenishment rate (default: 0.12)

**Theoretical Basis**: Family relationships provide stable, long-term support with higher trust and commitment.

#### Formal Intervention
**Definition**: Professional mental health services and treatments

**Efficacy Parameters**:
- `α_int`: Efficacy in reducing distress (default: 0.6)
- `ρ_int`: Resource replenishment rate (default: 0.15)

**Theoretical Basis**: Professional interventions provide specialized, evidence-based support with potentially higher efficacy.

#### Psychological Capital
**Definition**: Individual psychological resources including self-efficacy, hope, optimism, and resilience

**Efficacy Parameters**:
- `α_cap`: Efficacy in reducing distress (default: 0.35)
- `ρ_cap`: Resource replenishment rate (default: 0.04)

**Theoretical Basis**: Internal resources provide immediate availability but require personal development investment.

### Cross-Factor Interactions

**Complementary Effects**:
```python
# Combined protective effect (multiplicative rather than additive)
total_protection = 1.0 - ∏(1 - α_i × efficacy_i)
```

**Substitution Effects**:
```python
# Agents may substitute between factors based on availability
if social_support_unavailable:
    family_support_allocation += spillover_rate × social_allocation
```

## Integration with Other Systems

### Stress System Integration

**Resource Consumption During Stress**:
```python
# Successful coping consumes resources
if coped_successfully:
    resource_consumption = base_cost × (1 + hindrance_cost × hindrance)
    self.resources -= resource_consumption

# Failed coping may trigger resource conservation
if not coped_successfully:
    self._conserve_resources_for_future_stressors()
```

**Protective Factor Buffering**:
```python
# Protective factors reduce stress impact
effective_stress = appraised_stress × ∏(1 - α_i × efficacy_i)
```

### Affect System Integration

**Affect Influence on Regeneration**:
```python
# Positive affect enhances resource recovery
regeneration_multiplier = 1.0 + positive_affect_bonus × max(0, affect)

# Negative affect may increase resource consumption
if affect < 0:
    maintenance_cost_multiplier = 1.0 + abs(affect) × negative_cost_factor
```

**Resource Scarcity Affect Impact**:
```python
# Low resources may create negative affect
if self.resources < low_resource_threshold:
    scarcity_affect_impact = (low_resource_threshold - self.resources) × scarcity_rate
    self.affect = max(-1.0, self.affect - scarcity_affect_impact)
```

### Resilience System Integration

**Resilience Boost from Protective Factors**:
```python
def get_resilience_boost_from_protective_factors():
    total_boost = 0.0

    for factor, efficacy in self.protective_factors.items():
        if efficacy > 0:
            # Boost is higher when resilience is low (more needed)
            need_multiplier = max(0.1, 1.0 - self.resilience)
            boost_rate = 0.1
            total_boost += efficacy × need_multiplier × boost_rate

    return total_boost
```

**Resource Investment for Resilience**:
```python
# Agents allocate resources to build resilience
resilience_investment = available_resources × resilience_allocation_rate
resilience_boost = resilience_investment × resilience_return_rate
self.resilience = min(1.0, self.resilience + resilience_boost)
```

## Allocation Strategy Rationale

### Optimization Framework

**Objective Function**:
```
maximize E[∑(α_i × efficacy_i × ρ_i) - C(allocation_i)]
```

Where:
- `α_i`: Distress reduction efficacy
- `efficacy_i`: Current protective factor level
- `ρ_i`: Resource replenishment rate
- `C()`: Convex allocation cost function

**Constraints**:
```
∑allocation_i ≤ available_resources
0 ≤ allocation_i ≤ max_allocation_per_factor
```

### Adaptive Allocation Strategy

**Current State Dependence**:
```python
# Allocation adjusts based on current needs
for factor in protective_factors:
    current_need = 1.0 - getattr(self, f'{factor}_need_satisfaction')
    allocation_weight = current_need / sum(all_current_needs)
    allocation[factor] = available_resources × allocation_weight
```

**Investment Effectiveness**:
```python
# Higher returns when current efficacy is lower
improvement_potential = 1.0 - current_efficacy
investment_multiplier = 1.0 + improvement_bonus × improvement_potential
actual_improvement = base_improvement × investment_multiplier
```

## Dynamic Resource Constraints

### Resource Scarcity Effects

**Threshold-Based Behavior Changes**:
```python
if self.resources < high_scarcity_threshold:
    # Reduce social commitments
    self.social_activity_rate *= 0.5

if self.resources < critical_threshold:
    # Focus only on essential protective factors
    self.protective_allocation = {'psychological_capital': 1.0}
```

**Scarcity Psychology**:
- **High Resources**: Exploratory allocation, investment in multiple factors
- **Medium Resources**: Focused allocation, priority-based investment
- **Low Resources**: Conservative allocation, essential factors only

### Resource spirals

**Gain Spirals**:
```
High Resources → Investment in Protective Factors → Better Stress Coping →
Positive Affect → Enhanced Regeneration → Higher Resources
```

**Loss Spirals**:
```
Low Resources → Limited Protective Investment → Poor Stress Coping →
Negative Affect → Reduced Regeneration → Lower Resources
```

## Configuration Parameters

### Core Resource Parameters
```python
resource_config = {
    'initial_resources': 0.8,           # Starting resource level
    'base_regeneration': 0.05,          # Base regeneration rate
    'resource_cost': 0.1,               # Base cost per coping
    'allocation_rate': 0.3,             # Fraction allocated to protection
    'cost_scalar': 0.15,                # Cost function scalar
    'cost_exponent': 1.5,               # Cost function convexity
    'protective_improvement_rate': 0.5  # Rate of protective factor improvement
}
```

### Protective Factor Parameters
```python
protective_config = {
    'social_support_efficacy': 0.4,     # Social support distress reduction
    'family_support_efficacy': 0.5,     # Family support distress reduction
    'formal_intervention_efficacy': 0.6, # Professional help efficacy
    'psychological_capital_efficacy': 0.35, # Self-efficacy
    'social_replenishment': 0.08,       # Social resource return rate
    'family_replenishment': 0.12,       # Family resource return rate
    'intervention_replenishment': 0.15, # Professional resource return
    'capital_replenishment': 0.04       # Self resource return rate
}
```

### Allocation Parameters
```python
allocation_config = {
    'softmax_temperature': 2.0,         # Decision stochasticity
    'improvement_rate': 0.5,            # Base improvement rate
    'investment_effectiveness': 0.3,    # Bonus for low-efficacy factors
    'need_multiplier': 0.2,             # Need-based allocation bonus
    'max_allocation_per_factor': 0.5    # Maximum single factor allocation
}
```

### Protective Improvement Rate (`PROTECTIVE_IMPROVEMENT_RATE`)

**Parameter**: `PROTECTIVE_IMPROVEMENT_RATE` (default: 0.5, range: 0.0-1.0)

**Description**: Controls the rate at which resource investment translates into improved protective factor efficacy. This parameter determines how efficiently agents can develop and enhance their protective factors through resource allocation.

**Mathematical Effect**:
```
efficacy_increase = allocation × PROTECTIVE_IMPROVEMENT_RATE × investment_effectiveness × need_multiplier
```

**Interpretation**:
- **High values (0.3-1.0)**: Rapid improvement in protective factors, representing effective skill-building and relationship development
- **Low values (0.0-0.3)**: Slow improvement in protective factors, representing difficulty in developing coping resources
- **Research context**: This parameter can be calibrated against learning rates and skill acquisition in mental health interventions

**Usage in Model**:
```python
# In protective factor allocation and efficacy update
current_efficacy = self.protective_factors[factor]
improvement_rate = config.get('agent_parameters', 'protective_improvement_rate')
investment_effectiveness = 1.0 - current_efficacy  # Higher return when efficacy low
need_multiplier = max(0.1, 1.0 - agent_resilience)  # Higher need when resilience low

efficacy_increase = allocation * improvement_rate * investment_effectiveness * need_multiplier
new_efficacy = min(1.0, current_efficacy + efficacy_increase)
```

**Theoretical Foundation**:
- **Skill Acquisition**: Based on learning theory where practice and investment lead to improved capabilities
- **Relationship Building**: Social support efficacy improves with investment in relationship maintenance
- **Intervention Response**: Formal interventions become more effective as individuals engage with treatment

## Validation and Calibration

### Resource Dynamics Validation

**Individual Patterns**:
- **Resource Trajectories**: Realistic depletion and recovery patterns
- **Allocation Decisions**: Rational priority setting under constraints
- **Investment Returns**: Appropriate efficacy improvements

**Population Patterns**:
- **Resource Distribution**: Realistic inequality in resource availability
- **Protective Coverage**: Appropriate distribution of protective factor efficacy
- **Intervention Reach**: Realistic access to formal interventions

### Theoretical Alignment

**Conservation of Resources**:
- **Loss Primacy**: Resource losses have disproportionate impact
- **Investment Principle**: Resources allocated to prevent future losses
- **Spiral Effects**: Resource dynamics create gain/loss spirals

**Psychological Capital**:
- **State-Like**: Can change over time through investment
- **Trait-Like**: Stable individual differences in baseline levels
- **Investment**: Requires resource commitment for development

## Implementation Integration

### Code Structure

**Resource Management Components**:
- **`ResourceParams`**: Configuration dataclass for resource parameters
- **`ProtectiveFactors`**: State management for protective factor efficacy
- **`allocate_protective_resources()`**: Core allocation algorithm
- **`compute_resource_regeneration()`**: Regeneration dynamics
- **`compute_allocation_cost()`**: Cost function implementation

**Integration Points**:
- **Agent State**: Resource levels and protective factor efficacy
- **Stress Processing**: Resource consumption for coping
- **Affect Dynamics**: Affect influence on regeneration
- **Network Adaptation**: Resource constraints on social behavior

### Testing Framework

**Resource Management Tests**:
- **Unit Tests**: Individual function validation
- **Integration Tests**: Cross-system resource flow validation
- **Constraint Tests**: Behavior under resource scarcity
- **Allocation Tests**: Decision-making under different scenarios

This resource management system creates realistic constraints and trade-offs that drive adaptive behavior, investment decisions, and long-term mental health trajectories in the agent-based model.
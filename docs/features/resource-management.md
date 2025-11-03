# Resource Management: Protective Factors and Resource Allocation

_See [`.kilocode/rules/math/notation.md`](../../.kilocode/rules/math/notation.md) for symbol definitions and conventions._

## Overview

Resource management forms the core constraint mechanism in the agent-based mental health model, implementing the principle that individuals have limited psychological and physical resources to allocate across various protective strategies. This system draws from conservation of resources theory and realistic decision-making under constraints.

## Theoretical Background

### Conservation of Resources (COR) Theory

**Core Principle**: Individuals strive to retain, protect, and build resources and that what is threatening to them is the potential or actual loss of these valued resources.

**Key Tenets Applied**:
1. **Resource Loss Primacy**: Resource losses have stronger impact than equivalent gains
2. **Resource Investment**: Individuals invest resources to protect against future losses
3. **Spiral Effects**: Initial losses can create loss spirals, gains can create gain spirals

### Protective Factors Framework

**Psychological Capital**: Self-efficacy, hope, optimism, and resilience
**Social Support**: Emotional, informational, and instrumental support networks
**Family Support**: Kin-based resources and relational security
**Formal Interventions**: Professional mental health services and treatments

### Resource Allocation Theory

**Decision-Making Under Constraints**: Individuals must allocate limited resources optimally across protective factors, trading off between immediate needs and long-term benefits.

**Softmax Decision Making**: Implements bounded rationality where allocation decisions become more deterministic as resources become scarce.

## Resource State Variables

### Core Resource Pool

**Resources**: Finite psychological/physical resources available for allocation

**Resource Dynamics**:
- **Passive Regeneration**: Gradual return toward maximum capacity
- **Consumption**: Used for coping with stress events and maintaining protective factors
- **Investment**: Allocated to enhance protective factor efficacy

### Protective Factors

**Four Core Protective Mechanisms**:
- **Social Support**: Efficacy of social relationships
- **Family Support**: Efficacy of family connections
- **Formal Intervention**: Efficacy of professional help
- **Psychological Capital**: Self-efficacy and coping skills

## Resource Regeneration Mechanisms

### Passive Regeneration

**Mathematical Foundation**:
Resources naturally regenerate toward maximum capacity, representing rest, recovery, and natural healing processes.

**Resource Regeneration Equation:**

$$R' = \lambda_R \cdot (R_{\max} - R) \cdot (1 + \beta_a \cdot \max(0, A))$$

Where:
- $R' > 0$ is resource regeneration amount
- $\lambda_R \in [0,1]$ is regeneration rate
- $R_{\max} = 1$ is maximum resources
- $R \in [0,1]$ is current resources
- $\beta_a > 0$ is affect influence parameter
- $A \in [-1,1]$ is current affect

**Implementation**: [`compute_resource_regeneration()`](../../src/python/affect_utils.py#L360-L381) in `affect_utils.py`

### Affect-Modulated Regeneration

**Affect Influence**:
Positive emotional states enhance recovery and resource rebuilding, while negative states may slow regeneration.

**Psychological Foundation**: Positive affect broadens thought-action repertoires and builds personal resources.

**Affect-Modulated Regeneration:**

$$R' = \lambda_R \cdot (R_{\max} - R) \cdot (1 + \beta_a \cdot \max(0, A))$$

Where:
- $\beta_a > 0$ is affect influence parameter
- $A \in [-1,1]$ is current affect

**Implementation**: [`compute_resource_regeneration()`](../../src/python/affect_utils.py#L360-L381) in `affect_utils.py`

## Resource Consumption Patterns

### Coping Consumption

**Stress Event Coping**:
Successful coping requires resource investment, with higher costs for hindrance events due to increased psychological demands.

### Protective Factor Maintenance

**Allocation Cost**:
Maintaining protective factors requires ongoing resource investment, with increasing marginal costs representing diminishing returns.

## Resource Allocation Mechanisms

### Softmax Decision Framework

**Mathematical Foundation**:
Individuals allocate resources across protective factors using a decision-making process that balances current efficacy levels with temperature-scaled randomness.

**Softmax Allocation Equation:**

$$w_f = \frac{\exp(e_f / \beta_{\text{softmax}})}{\sum_{k \in F} \exp(e_k / \beta_{\text{softmax}})}$$

**Resource Allocation:**

$$r_f = w_f \cdot R_a$$

Where:
- $w_f \in [0,1]$ is allocation weight for factor $f$
- $e_f \in [0,1]$ is efficacy of factor $f$
- $\beta_{\text{softmax}} > 0$ is softmax temperature
- $F$ is set of protective factors
- $r_f > 0$ is resources allocated to factor $f$
- $R_a \in [0,1]$ is available resources

**Implementation**: [`allocate_protective_resources()`](../../src/python/affect_utils.py#L308-L357) in `affect_utils.py`

**Temperature Effects**:
- **High Temperature**: More random allocation (exploration)
- **Low Temperature**: More deterministic allocation (exploitation)
- **Default**: Moderate stochasticity for realistic decision-making

### Allocation Rationale

**Need-Based Allocation**:
Individuals allocate more resources to factors with lower current efficacy and when overall resilience is low, representing rational priority setting under constraints.

**Investment Return Model**:
Resource investment provides greater returns when current efficacy is lower and need is higher, reflecting realistic learning and development processes.

**Protective Factor Efficacy Update:**

$$\Delta e_f = r_f \cdot \gamma_p \cdot (1 - e_f)$$

**New Efficacy:**

$$e_f' = \min(1, e_f + \Delta e_f)$$

Where:
- $\Delta e_f > 0$ is efficacy improvement
- $\gamma_p > 0$ is improvement rate
- $e_f \in [0,1]$ is current efficacy

**Implementation**: [`allocate_protective_factors()`](../../src/python/resource_utils.py#L549-L587) in `resource_utils.py`

**Implementation**: [`allocate_protective_factors()`](../../src/python/resource_utils.py#L549-L587) method in `resource_utils.py`

## Protective Factor Efficacy Dynamics

### Individual Factor Characteristics

#### Social Support
**Definition**: Quality and availability of emotional and instrumental support from social network

**Theoretical Basis**: Social support buffers stress through emotional and practical assistance.

#### Family Support
**Definition**: Support from family relationships and kinship networks

**Theoretical Basis**: Family relationships provide stable, long-term support with higher trust and commitment.

#### Formal Intervention
**Definition**: Professional mental health services and treatments

**Theoretical Basis**: Professional interventions provide specialized, evidence-based support with potentially higher efficacy.

#### Psychological Capital
**Definition**: Individual psychological resources including self-efficacy, hope, optimism, and resilience

**Theoretical Basis**: Internal resources provide immediate availability but require personal development investment.

### Cross-Factor Interactions

**Complementary Effects**:
Different protective factors work together multiplicatively to provide comprehensive protection.

**Substitution Effects**:
Individuals may substitute between factors based on availability, representing adaptive resource allocation.

## Integration with Other Systems

### Stress System Integration

**Resource Consumption During Stress**:
Successful coping consumes resources, with higher costs for hindrance events. Failed coping may trigger resource conservation strategies.

**Protective Factor Buffering**:
Protective factors work together to reduce the impact of stress events, providing multiple layers of protection.

**Resource Consumption for Coping:**

$$R' = R - \kappa \cdot \mathbb{1}_{\mathrm{coping\ successful}}$$

Where:
- $R' \in [0,1]$ is updated resources
- $\kappa > 0$ is resource cost parameter
- $\mathbb{1}_{\mathrm{coping\ successful}}$ is indicator for successful coping

**Implementation**: [`compute_resource_depletion_with_resilience()`](../../src/python/resource_utils.py#L1-L50) in `resource_utils.py`

### Affect System Integration

**Affect Influence on Regeneration**:
Positive affect enhances resource recovery, while negative affect may increase maintenance costs and create scarcity-related negative emotions.

**Resource Scarcity Affect Impact**:
Low resources may create negative affect, representing the psychological stress of resource depletion.

**Protective Factor Resilience Boost:**

$$\Delta \mathfrak{R}_p = \sum_{f \in F} e_f \cdot (\mathfrak{R}_{\text{0}} - \mathfrak{R}_c) \cdot \theta_{\text{boost}}$$

Where:
- $\Delta \mathfrak{R}_p$ is resilience boost from protective factors
- $F = \{\mathrm{soc}, \mathrm{fam}, \mathrm{int}, \mathrm{cap}\}$ is set of protective factors
- $e_f \in [0,1]$ is efficacy of factor $f$
- $\mathfrak{R}_{\text{0}} \in [0,1]$ is baseline resilience
- $\mathfrak{R}_c \in [0,1]$ is current resilience
- $\theta_{\text{boost}} > 0$ is boost rate parameter

### Resilience System Integration

**Resilience Boost from Protective Factors**:
Protective factors provide resilience benefits, with greater boosts when resilience is low and most needed.

**Resource Investment for Resilience**:
Individuals allocate resources to build resilience, with investment returns depending on current state and need.

## Allocation Strategy Rationale

### Optimization Framework

**Objective Function**:
Individuals seek to maximize the expected return from protective factor investments while accounting for allocation costs.

**Constraints**:
Resource allocation is limited by available resources and maximum allocation per factor.

### Adaptive Allocation Strategy

**Current State Dependence**:
Allocation adjusts based on current needs, with higher allocation to factors with lower current efficacy and when overall resilience is low.

**Investment Effectiveness**:
Investment provides higher returns when current efficacy is lower, reflecting realistic learning and development processes.

## Dynamic Resource Constraints

### Resource Scarcity Effects

**Threshold-Based Behavior Changes**:
As resources become scarce, individuals adapt their behavior:
- **High Resources**: Exploratory allocation across multiple factors
- **Medium Resources**: Focused allocation based on priorities
- **Low Resources**: Conservative allocation to essential factors only

### Resource spirals

**Gain Spirals**:
High resources enable investment in protective factors, leading to better stress coping, positive affect, enhanced regeneration, and higher resources.

**Loss Spirals**:
Low resources limit protective investment, leading to poor stress coping, negative affect, reduced regeneration, and lower resources.

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

### Correlation Validation Framework

The resource management system includes validated correlations with other agent variables:

**Validated Correlation Ranges:**
- Resources ↔ Current Stress: $r \in [-0.8, 0.1]$ (negative to weak correlation)
- Resources ↔ Affect: $r \in [-0.2, 0.4]$ (weak correlation)
- Resources ↔ Resilience: $r \in [-1.0, 1.0]$ (any correlation)
- Resources ↔ PSS-10: $r \in [-0.5, 0.5]$ (variable correlation)

**Implementation**: [`test_correlation_validation.py`](../../src/python/tests/test_correlation_validation.py) validates these correlation ranges and statistical significance.

### Theoretical Alignment

**Conservation of Resources**:
- **Loss Primacy**: Resource losses have disproportionate impact
- **Investment Principle**: Resources allocated to prevent future losses
- **Spiral Effects**: Resource dynamics create gain/loss spirals

**Psychological Capital**:
- **State-Like**: Can change over time through investment
- **Trait-Like**: Stable individual differences in baseline levels
- **Investment**: Requires resource commitment for development
- **Correlation Stability**: Theoretical relationships maintained across parameter configurations

## Implementation Integration

### Code Structure

**Resource Management Components**:
- **Resource Configuration**: Parameter management for resource dynamics
- **Protective Factors**: State management for protective factor efficacy
- **Allocation Algorithms**: Core resource allocation and investment functions
- **Regeneration Systems**: Resource recovery and cost calculation mechanisms

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
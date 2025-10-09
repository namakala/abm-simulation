# Integration Overview: How All Mechanisms Work Together

_See [`_NOTATION.md`](./_NOTATION.md) for symbol definitions and conventions._

## System Architecture Overview

The agent-based model integrates six core mechanism groups that work together to simulate realistic mental health dynamics:

1. **Stress Perception** - Event appraisal and challenge/hindrance determination
2. **Resilience Dynamics** - Coping responses and resilience changes
3. **Affect Dynamics** - Emotional state changes and social influence
4. **Agent Interactions** - Social network effects and support systems
5. **Resource Management** - Protective factors and resource allocation
6. **Stress Assessment** - Empirical stress measurement and validation

## Core Integration Pathways

### Stress Event Processing Pipeline

Stress events flow through a comprehensive processing pipeline that transforms life events into psychological responses. The process begins with event occurrence and moves through appraisal, threshold evaluation, coping determination, and state updates.

**Integration Points**:
- **Stress → Affect**: Event interpretations directly influence emotional states
- **Stress → Resilience**: Coping outcomes determine resilience changes
- **Stress → Resources**: Successful coping requires resource investment
- **Stress → Network**: High stress triggers social network adaptation
- **Stress ↔ Assessment**: Bidirectional feedback between stress levels and measurement scores
- **Assessment → Agent Initialization**: Assessment scores influence initial stress thresholds
- **Assessment → Behavioral Effects**: High assessment scores may modify behavior patterns
- **Assessment → Validation**: Empirical measurements enable pattern matching with research literature

**Complete Agent State Vector:**

$$\mathbf{s}_i = \begin{bmatrix}
R_i \\
A_i \\
S_i \\
\mathbf{e}_i \\
\mathbf{n}_i \\
R_{a_i} \\
\mathbf{\Psi}_i
\end{bmatrix}$$

Where:
- $\mathbf{s}_i$ is state vector for agent $i$
- $R_i \in [0,1]$ is resilience
- $A_i \in [-1,1]$ is affect
- $S_i \in [0,1]$ is stress level
- $\mathbf{e}_i \in [0,1]^4$ is protective factor efficacy vector
- $\mathbf{n}_i$ is network connections
- $R_{a_i} \in [0,1]$ is available resources
- $\mathbf{\Psi}_i$ is PSS-10 state vector

### Daily Step Integration

Each day follows a structured sequence that integrates all mechanisms:

1. **Initialization**: Set starting conditions and gather social context
2. **Event Processing**: Generate and process daily events and interactions
3. **Effect Integration**: Combine daily challenge and hindrance effects
4. **Dynamics Update**: Apply emotional and resilience changes
5. **Resource Management**: Update resource levels and protective factors
6. **Homeostasis**: Apply natural regulatory adjustments
7. **Daily Reset**: Reset tracking variables for next day

## Cross-System Feedback Loops

### Positive Feedback Loop: Success Reinforcement

High resilience creates a positive cycle that reinforces successful coping and positive emotional states. This virtuous cycle shows how initial success can build momentum toward better mental health outcomes.

**Mechanism**:
1. High resilience improves coping success probability
2. Successful coping boosts emotional state positively
3. Positive emotions accelerate resource rebuilding
4. More resources enable better protective factor investment
5. Strong protective factors enhance resilience

### Negative Feedback Loop: Stress Degradation

Low resilience creates a negative cycle that reinforces poor coping and negative emotional states. This vicious cycle shows how initial difficulties can create momentum toward poorer mental health outcomes.

**Mechanism**:
1. Low resilience reduces coping success probability
2. Failed coping creates negative emotional states
3. Negative emotions slow resource rebuilding
4. Resource depletion limits protective factor investment
5. Weak protective factors further reduce resilience

**Positive Feedback Loop Equations:**

**Success Reinforcement:**

$$R_{t+1} = R_t + \beta_c \cdot \mathbb{1}_{\mathrm{coping\ success}} + \alpha_s \cdot \max(0, A_t)$$

**Resource Regeneration:**

$$R_a' = R_a + \lambda_R \cdot (1 - R_a) \cdot (1 + \beta_a \cdot \max(0, A_t))$$

**Protective Investment:**

$$e_f' = e_f + r_f \cdot \gamma_p \cdot (1 - e_f) \quad \forall f \in F$$

Where:
- $R_t \in [0,1]$ is current resilience
- $\beta_c > 0$ is coping success boost
- $\mathbb{1}_{\mathrm{coping\ success}}$ is coping success indicator
- $\alpha_s > 0$ is social support boost
- $A_t \in [-1,1]$ is current affect
- $R_a \in [0,1]$ is available resources
- $\lambda_R \in [0,1]$ is base regeneration rate
- $\beta_a > 0$ is affect influence on regeneration
- $e_f \in [0,1]$ is protective factor efficacy
- $r_f > 0$ is resources allocated to factor $f$
- $\gamma_p > 0$ is protective improvement rate
- $F$ is set of protective factors

**Negative Feedback Loop Equations:**

**Stress Degradation:**

$$R_{t+1} = R_t - \beta_f \cdot \mathbb{1}_{\mathrm{coping\ failure}} - \delta_o \cdot \mathbb{1}_{\mathrm{overload}}$$

**Resource Depletion:**

$$R_a' = R_a - \kappa \cdot \mathbb{1}_{\mathrm{coping\ success}} - c_m \cdot \sum e_f$$

Where:
- $\beta_f > 0$ is coping failure penalty
- $\mathbb{1}_{\mathrm{coping\ failure}}$ is coping failure indicator
- $\delta_o > 0$ is overload penalty
- $\mathbb{1}_{\mathrm{overload}}$ is overload indicator
- $\kappa > 0$ is coping resource cost
- $c_m > 0$ is maintenance cost

### Social Amplification Loop

Positive social connections create a self-reinforcing cycle where supportive relationships enhance coping success and emotional well-being, leading to stronger social connections.

**Mechanism**:
1. Positive social connections provide support
2. Social support improves coping success
3. Successful coping creates positive emotional states
4. Similar emotional patterns drive social network adaptation
5. Network adaptation connects people with similar experiences

## Temporal Integration Patterns

### Within-Day Dynamics

**Event Sequencing**:
Each day includes multiple events that can be either social interactions or stress events. The order of events matters, as social interactions can buffer the effects of subsequent stress events.

**Accumulation Effects**:
- **Daily Challenge/Hindrance**: Averaged across all stress events in the day
- **Social Influence**: Accumulated from all social interactions
- **Resource Consumption**: Summed across all coping attempts

### Day-to-Day Dynamics

**Carryover Effects**:
Ongoing hindrance events are tracked across days, with cumulative effects when hindrances persist over time. This represents how chronic stress can build up and create overload effects.

**Homeostatic Regulation**:
All systems have a natural tendency to return to baseline equilibrium levels over time, representing psychological adaptation and recovery processes.

## Network Integration Effects

### Spatial Stress Propagation

Stress can spread through local social networks, creating geographic patterns of stress clustering. High-stress areas tend to form around people experiencing chronic difficulties, while low-stress areas form around those with effective coping strategies.

### Support Network Formation

Effective supporters naturally attract more social connections, creating hubs of support within the network. Network adaptation based on support history helps maintain effective support relationships.

## Resource Allocation Integration

### Protective Factor Investment

Individuals allocate limited resources across different protective factors based on current needs and effectiveness. Investment returns depend on current state, with higher returns when protective factors are most needed.

**Cross-System Resource Flow**:
- **Stress Events**: Require resource investment for coping
- **Social Interactions**: May require resources for support provision
- **Regeneration**: Influenced by emotional state
- **Investment**: Allocates resources to build protective factors

## Configuration System Integration

### Unified Parameter Management

All mechanisms use the same configuration system, ensuring consistency across all components. This includes parameters for agent behavior, stress processing, emotional dynamics, resilience processes, network structure, and resource management.

## Validation Integration

### Multi-Level Validation

**Individual Agent Validation**:
- **Stress Response**: Appropriate sensitivity to challenge and hindrance
- **Coping Success**: Realistic success rates given event characteristics
- **Affect Dynamics**: Proper response to social and stress influences
- **Resource Management**: Sustainable resource levels over time

**Population-Level Validation**:
- **Stress Distribution**: Realistic distribution of stress levels
- **Network Structure**: Appropriate clustering and connectivity
- **Intervention Effects**: Measurable impact of protective factors
- **Emergent Patterns**: Realistic mental health outcome distributions

**Literature Alignment**:
- **Assessment Scores**: Match empirical stress scale distributions
- **Recovery Times**: Align with psychological recovery research
- **Social Effects**: Consistent with social influence studies
- **Intervention Efficacy**: Match mental health intervention research

## Research Pipeline Integration

### Parameter Sweep Integration

The system supports comprehensive sensitivity analysis across all integrated parameters, enabling researchers to understand which factors most influence outcomes and validate against research targets.

### Output Integration

The system provides both individual and population-level metrics for comprehensive analysis, including stress patterns, emotional clusters, resilience distributions, network structures, and intervention effects.

## Intervention Integration

### Multi-Tiered Intervention Effects

**Universal Interventions**:
- **Population-wide**: Affect all individuals equally
- **Mechanism**: Modify base parameters for the entire population
- **Integration**: Applied through the configuration system

**Selective Interventions**:
- **Targeted Groups**: Applied to specific subgroups
- **Mechanism**: Condition-based parameter modification
- **Integration**: Runtime parameter adjustment based on individual characteristics

**Indicated Interventions**:
- **Individual Treatment**: Applied to individuals meeting specific criteria
- **Mechanism**: State-based intervention triggering
- **Integration**: Real-time intervention based on current stress or resilience levels

## Mathematical Integration Framework

### State Vector Integration

**Complete Agent State**:
The model tracks a comprehensive state for each individual including emotional state, resilience capacity, stress level, protective factors, network connections, available resources, and stress assessment scores.

**State Transition Function**:
All mechanisms work together through an integrated function that combines external events, social interactions, and configuration parameters to determine state changes.

**State Transition Function:**

$$\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{e}_t, \mathbf{i}_t, \mathbf{p})$$

Where:
- $\mathbf{s}_{t+1}$ is next state vector
- $f$ is state transition function
- $\mathbf{s}_t$ is current state vector
- $\mathbf{e}_t$ is external events at time $t$
- $\mathbf{i}_t$ is social interactions at time $t$
- $\mathbf{p}$ is model parameters

**Implementation**: [`step()`](src/python/agent.py:208) method in `agent.py`

### Equilibrium Analysis

**Homeostatic Equilibria**:
The system tends toward equilibrium points where all mechanisms balance each other. The nature of these equilibria depends on parameter settings, with social effects potentially creating multiple possible equilibrium states.

**Equilibrium Condition:**

$$\mathbf{s}^* = f(\mathbf{s}^*, \mathbf{0}, \mathbf{0}, \mathbf{p})$$

**Stability Analysis:**

**Basin Stability:**

$$S_b = \frac{1}{N} \sum_{i=1}^N \mathbb{1}_{\mathbf{s}_i(t) \in B_\epsilon(\mathbf{s}^*)} \quad \forall t > t_0$$

Where:
- $\mathbf{s}^*$ is equilibrium state
- $B_\epsilon(\mathbf{s}^*)$ is $\epsilon$-ball around equilibrium
- $S_b \in [0,1]$ is basin stability measure

## Implementation Integration

### Code Organization Integration

**Modular Design**:
- **Stress Processing**: Event perception and appraisal mechanisms
- **Affect Processing**: Emotional dynamics and social influence
- **Agent Coordination**: Integration orchestration and state management
- **Population Management**: Group-level coordination and network management
- **Configuration Management**: Unified parameter management across all systems

**Dependency Flow**:
The main agent process integrates all mechanisms in sequence, ensuring proper coordination and state consistency across all systems.

### Testing Integration

**Comprehensive Test Coverage**:
- **Unit Tests**: Individual mechanism functions
- **Integration Tests**: Cross-mechanism interactions
- **Configuration Tests**: Parameter validation and type checking
- **Mechanism Tests**: Specific behavioral validations
- **Validation Tests**: Literature alignment and pattern matching

## Future Integration Extensions

### Planned Enhancements

**Advanced Analysis Integration**:
- **Statistical Analysis**: Population-level pattern analysis
- **Visualization**: Dynamic network and state visualization
- **Calibration**: Parameter calibration routines

**Enhanced Storage Integration**:
- **Large-Scale Storage**: Parameter sweep results and simulation outputs
- **Query Interface**: Efficient data extraction for analysis
- **Performance Optimization**: Batch processing capabilities

**Advanced Network Features**:
- **Dynamic Network Growth**: Population changes over time
- **Multi-Scale Networks**: Hierarchical social structures
- **Temporal Network Evolution**: Time-varying connection patterns

This integration framework ensures that all mechanisms work together cohesively while maintaining the modularity and testability required for rigorous research validation.
# Agent Interaction Mechanisms and Impact Analysis

_See [`.kilocode/rules/math/notation.md`](../../.kilocode/rules/math/notation.md) for symbol definitions and conventions._

## Overview

Agent interactions form the core social dynamics of the mental health model. Individuals interact through social networks, influencing each other's emotional states and resilience while creating emergent patterns of social support, emotional contagion, and network adaptation.

## Network Structure

### Watts-Strogatz Small-World Network

Individuals are connected through a Watts-Strogatz small-world network that reflects realistic social relationships. The network uses a ring lattice base topology with random rewiring, creating realistic patterns of social clustering and connectivity.

**Network Properties**:
- **Topology**: Watts-Strogatz small-world network with configurable rewiring probability
- **Clustering**: High clustering coefficient reflecting social circles
- **Path Length**: Short average path length enabling distant connections
- **Connection Type**: Based on social relationships and random opportunities

### Neighbor Relationships

Individuals discover and interact with others through their network connections. The small-world structure naturally creates clusters of similar individuals while allowing connections across the entire population, enabling realistic patterns of social influence and information flow.

## Interaction Process

### Interaction Initiation

Individuals randomly select interaction partners from their neighbors and engage in mutual exchanges that influence both participants' emotional states and resilience.

### Mutual Influence Mechanism

Social interactions create bidirectional changes where both individuals affect each other's emotional state. The model recognizes that negative emotional states tend to have stronger influence than positive ones.

**Mutual Influence Equations:**

**Basic Influence:**

$$\Delta A_i = \alpha_p \cdot \mathrm{sign}(A_j)$$

**Asymmetric Weighting:**

$$\Delta A_i' = \Delta A_i \cdot \begin{cases}
1.5 & \text{if } \Delta A_i < 0 \\
1.0 & \text{if } \Delta A_i \geq 0
\end{cases}$$

**Mutual Updates:**

$$A_i' = A_i + \Delta A_i'$$
$$A_j' = A_j + \Delta A_j'$$

Where:
- $\Delta A_i$ is affect change for agent $i$
- $\alpha_p \in [0,1]$ is peer influence rate
- $A_i, A_j \in [-1,1]$ are affect values
- $\mathrm{sign}(x)$ returns sign of $x$

**Implementation**: [`process_interaction()`](src/python/affect_utils.py:120) in `affect_utils.py`

## Social Influence Dynamics

### Affect Contagion

Emotional states spread through social interactions, with the extent of influence depending on the strength of the emotional states and the nature of the relationship.

**Asymmetric Effects**:
- **Positive Affect**: Spreads at normal rate
- **Negative Affect**: Spreads more strongly, reflecting how negative interactions can be more impactful

### Resilience Influence

Strong emotional states from social connections can influence resilience, but only when emotions are sufficiently intense to trigger meaningful change.

## Network Adaptation Mechanisms

### Stress-Based Rewiring

When individuals experience repeated stress, they may adapt their social connections to better suit their needs. This represents how people naturally adjust their social networks during difficult periods.

**Trigger Condition**:
Individuals begin adapting their networks after experiencing multiple stressful events, reflecting a threshold beyond which people seek to change their social environment.

**Rewiring Process**:
The adaptation process considers both similarity to others and the quality of support received, balancing between connecting with similar people versus connecting with helpful people.

### Homophily-Based Connection

Individuals tend to form connections with others who have similar experiences and emotional states. This creates natural clustering of people with comparable stress levels and coping abilities.

**Connection Preferences**:
- **Similar Stress Levels**: People prefer connections with others having similar stress experiences
- **Similar Emotional States**: Emotional similarity drives connection preferences
- **Support Effectiveness**: History of helpful interactions influences connection strength

**Network Adaptation Trigger:**

$$\mathrm{trigger\ adaptation} = \begin{cases}
1 & \text{if } c_{\text{breach}} \geq \eta_{\text{adapt}} \\
0 & \text{otherwise}
\end{cases}$$

**Connection Similarity:**

$$s_{ij} = 1 - \frac{|A_i - A_j| + |R_i - R_j|}{2}$$

Where:
- $c_{\text{breach}} \in \mathbb{N}$ is stress breach count
- $\eta_{\text{adapt}} \in \mathbb{N}$ is adaptation threshold
- $s_{ij} \in [0,1]$ is similarity between agents $i,j$
- $A_i, A_j \in [-1,1]$ are affect values
- $R_i, R_j \in [0,1]$ are resilience values

**Connection Retention Probability:**

$$p_{\mathrm{keep}} = s_{ij} \cdot \delta_{\text{homophily}} + e_s \cdot (1 - \delta_{\text{homophily}})$$

**Rewiring Decision:**

$$\mathrm{rewire} = \begin{cases}
1 & \text{if } U \sim \mathcal{U}(0,1) > p_{\mathrm{keep}} \\
0 & \text{otherwise}
\end{cases}$$

Where:
- $p_{\mathrm{keep}} \in [0,1]$ is probability of keeping connection
- $\delta_{\text{homophily}} \in [0,1]$ is homophily strength
- $e_s \in [0,1]$ is support effectiveness
- $U$ is uniform random variable

## Social Support Dynamics

### Support Provision

Social support occurs when individuals provide emotional or practical help to others in their network. The likelihood of receiving support depends on the emotional state of neighbors and their ability to provide help.

**Support Effectiveness**:
The quality of support depends on the current state and capabilities of the person providing help, with more resilient individuals generally being more effective supporters.

**Support Effectiveness Calculation:**

$$e_s = \frac{R_j + (1 + A_j)/2}{2} + 0.2$$

Where:
- $e_s \in [0,1]$ is support effectiveness
- $R_j \in [0,1]$ is neighbor's resilience
- $A_j \in [-1,1]$ is neighbor's affect

### Support Impact on Resilience

Receiving social support provides a direct boost to resilience, representing how supportive relationships can help people better cope with stress.

**Support Exchange Detection:**

$$\mathrm{support\ exchange} = \begin{cases}
1 & \text{if } |\Delta A_i| > 0.05 \lor |\Delta R_i| > 0.05 \lor |\Delta A_j| > 0.05 \lor |\Delta R_j| > 0.05 \\
0 & \text{otherwise}
\end{cases}$$

Where:
- $\Delta A_i, \Delta A_j$ are affect changes for agents $i,j$
- $\Delta R_i, \Delta R_j$ are resilience changes for agents $i,j$

**Implementation**: [`interact()`](src/python/agent.py:359) method in `agent.py`

## Interaction Impact Analysis

### Affect Contagion Patterns

**Emergent Phenomena**:
1. **Emotional Clustering**: Similar emotional states naturally group together through social connections
2. **Contagion Patterns**: Negative emotions can spread through both local clusters and global shortcuts
3. **Resilience Hubs**: Well-connected, high-resilience individuals resist and buffer negative emotional contagion

### Network Structure Effects

**Small-World Topology Impact**:
- **Local Clustering**: Strong local connections within social circles
- **Global Connectivity**: Short paths enable influence across the entire population
- **Clustering Effects**: High clustering coefficient creates emotional clusters

**Network Density Effects**:
- **High Density**: More connections per person, faster emotional spread
- **Low Density**: Fewer connections, slower but more selective influence

## Coping Social Influence

### Social Influence on Coping

The emotional state of social connections influences an individual's ability to cope with stress. Positive social environments tend to improve coping success, while negative environments can hinder it.

### Collective Coping Patterns

**Emergent Effects**:
1. **Social Buffering**: Positive social environments improve coping success
2. **Stress Contagion**: Negative social environments reduce coping capacity
3. **Resilience Clusters**: Groups with high resilience support each other

## Resource Allocation in Social Context

### Social Support Resource Cost

Providing social support requires personal resources, representing the energy and time invested in helping others. Maintaining social connections also requires ongoing investment.

## Adaptation and Learning

### Network Learning Mechanisms

Individuals track the effectiveness of their social connections over time and adapt their preferences based on experience. Successful support experiences lead to stronger preferences for similar connections, while unsuccessful experiences encourage seeking different types of relationships.

## Impact Assessment

### Population-Level Effects

**Emergent Patterns**:
1. **Emotional Clustering**: Similar emotional states form connected clusters through small-world structure
2. **Support Networks**: Effective supporters become central hubs in the social network
3. **Stress Propagation**: Stress can spread through both local clusters and global shortcuts
4. **Resilience Communities**: High-resilience groups emerge and persist through network connections

### Intervention Effects

**Network-Based Interventions**:
- **Targeted Support**: Programs focused on low-resilience clusters
- **Connection Enhancement**: Programs that improve social network connectivity
- **Support Training**: Programs that improve support effectiveness within social circles

### Validation Metrics

**Network Metrics**:
- **Clustering Coefficient**: Measure of local social grouping in small-world structure
- **Average Path Length**: Measure of global connectivity in small-world network
- **Homophily Index**: Measure of connection preferences for similar individuals
- **Support Network Density**: Measure of support availability through social connections

**Behavioral Metrics**:
- **Interaction Frequency**: Rate of social interactions within network structure
- **Support Success Rate**: Effectiveness of provided support through social connections
- **Network Adaptation Rate**: Frequency of connection changes in response to stress
- **Emotional Contagion Speed**: Rate of emotional spread through small-world network

## Integration with Other Systems

### Stress System Integration

**Stress → Network**:
- High stress triggers network adaptation
- Stress patterns influence connection preferences
- Failed coping may lead to social withdrawal

**Network → Stress**:
- Social support improves coping success
- Neighbor emotions influence stress response
- Small-world network structure affects stress propagation patterns

### Resilience System Integration

**Resilience → Network**:
- High resilience individuals become better support providers
- Resilience patterns influence network adaptation
- Successful coping strengthens social ties

**Network → Resilience**:
- Social support provides direct resilience boost
- Network structure affects resilience distribution
- Support effectiveness influences resilience trajectories

### Affect System Integration

**Affect → Network**:
- Similar emotional states drive homophily in small-world structure
- Extreme emotions may trigger network adaptation
- Emotional state influences support effectiveness

**Network → Affect**:
- Social contagion spreads emotions through small-world connections
- Support interactions change emotional states
- Small-world network structure influences emotional dynamics and clustering
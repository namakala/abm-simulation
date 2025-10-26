# Affect Dynamics and Influencing Factors

_See [`.kilocode/rules/math/notation.md`](../../.kilocode/rules/math/notation.md) for symbol definitions and conventions._

## Overview

Affect represents an individual's current emotional state, ranging from very negative to very positive. The model examines how various factors influence these emotional states, including social connections, stress events, and natural tendencies to return to an individual's baseline emotional level.

## Core Affect State

### State Variables

- **Affect Level**: Current emotional state ranging from very negative to very positive
- **Baseline Affect**: An individual's natural emotional equilibrium point
- **Daily Initial Affect**: Emotional state at the beginning of each day for tracking purposes

## Social Influence Mechanisms

### Peer Influence

Individuals influence each other's emotional states through social interactions. When people interact, their emotional states can spread to others, with the extent of influence depending on the number of connections and the strength of the emotional states involved.

**Social Influence Equation:**

$$\Delta A_p = \frac{1}{k} \sum_{j=1}^{k} \alpha_p \cdot (A_j - A_t) \cdot \begin{cases}
1.5 & \text{if } A_j - A_t < 0 \\
1.0 & \text{if } A_j - A_t \geq 0
\end{cases}$$

Where:
- $\Delta A_p$ is aggregated peer influence effect
- $k$ is number of influencing neighbors (limited by $k_{\text{influence}}$)
- $\alpha_p \in [0,1]$ is peer influence rate
- $A_j, A_t \in [-1,1]$ are neighbor and current affect values

**Implementation**: [`compute_peer_influence()`](../../src/python/affect_utils.py#L957-L994) in `affect_utils.py`

### Interaction Effects

Social interactions create mutual influence where both individuals affect each other's emotional state. The model recognizes that negative emotional states tend to have a stronger impact than positive ones, reflecting how negative interactions can be more memorable and influential.

**Mutual Influence with Asymmetric Effects:**

$$\Delta A_i = \alpha_p \cdot (A_j - A_i) \cdot \begin{cases}
1.5 & \text{if } A_j - A_i < 0 \\
1.0 & \text{if } A_j - A_i \geq 0
\end{cases}$$

**Resilience Influence:**

$$\Delta R_i = \lambda_{\text{res,interact}} \cdot A_j \cdot \mathbb{1}_{|A_j| > \eta_{\text{affect}}}$$

Where:
- $\lambda_{\text{res,interact}} > 0$ is resilience influence rate
- $\eta_{\text{affect}} > 0$ is affect threshold for resilience influence
- $\mathbb{1}$ is indicator function

**Implementation**: [`process_interaction()`](../../src/python/affect_utils.py#L139-L201) in `affect_utils.py`

## Stress Event Impact on Affect

### Challenge-Hindrance Effects

Stress events influence emotional states differently based on their nature and coping outcomes:

- **Successful Coping**: Overcoming challenging events tends to improve emotional state
- **Failed Coping**: Being overwhelmed by hindrance events tends to worsen emotional state

### Event Appraisal Effects

The way individuals interpret stress events affects their emotional response. Challenging events can provide motivation when emotional state is low, while hindrance events tend to create stronger negative emotional impact when someone is already struggling.

## Homeostatic Mechanisms

### Daily Homeostasis

Emotional states naturally tend to return to an individual's baseline level over time. This represents the psychological tendency for emotions to stabilize around a person's natural equilibrium point.

**Homeostasis Effect:**

$$\Delta A_h = \lambda_{\text{affect}} \cdot (A_{\text{0}} - A_c)$$

Where:
- $\Delta A_h$ is homeostatic affect change
- $\lambda_{\text{affect}} \in [0,1]$ is homeostatic rate
- $A_{\text{0}} \in [-1,1]$ is baseline affect
- $A_c \in [-1,1]$ is current affect

**Implementation**: [`compute_homeostasis_effect()`](../../src/python/affect_utils.py#L1030-L1061) in `affect_utils.py`

### Baseline Affect Dynamics

Each person has a stable baseline affect that represents their natural emotional balance. This baseline remains relatively constant and serves as the target that emotional dynamics pull toward.

**Event Appraisal Effect:**

$$\Delta A_e = \alpha_e \cdot \chi \cdot (1 - A_c) - \alpha_e \cdot \zeta \cdot \max(0.1, A_c + 1)$$

Where:
- $\Delta A_e$ is event appraisal affect change
- $\alpha_e \in [0,1]$ is event appraisal rate
- $\chi \in [0,1]$ is challenge component
- $\zeta \in [0,1]$ is hindrance component
- $A_c \in [-1,1]$ is current affect

**Implementation**: [`compute_event_appraisal_effect()`](../../src/python/affect_utils.py#L996-L1027) in `affect_utils.py`

## Integrated Affect Dynamics

### Complete Update Process

Daily emotional changes result from multiple factors working together:

- Social influence from connections
- Emotional impact of stress events
- Natural tendency to return to baseline

These factors combine to determine how emotional state changes throughout each day.

**Integrated Affect Update:**

$$A_{t+1} = A_t + \Delta A_p + \Delta A_e + \Delta A_h$$

**Final Affect Clamping:**

$$A_{t+1} = \mathrm{clamp}(A_{t+1}, -1, 1)$$

Where:
- $A_t \in [-1,1]$ is current affect
- $\Delta A_p$ is peer influence effect
- $\Delta A_e$ is event appraisal effect
- $\Delta A_h$ is homeostasis effect

**Implementation**: [`update_affect_dynamics()`](../../src/python/affect_utils.py#L1149-L1186) in `affect_utils.py`

**Aggregated Peer Influence:**

$$\Delta A_p = \frac{1}{k} \sum_{j=1}^{k} \alpha_p \cdot (A_j - A_t)$$

Where:
- $k$ is number of neighbors (limited by $\max(1, k_{\text{influence}})$)
- $A_j \in [-1,1]$ is neighbor $j$'s affect

## Resilience-Affect Interactions

### Bidirectional Influence

Emotional state and resilience influence each other through integrated mechanisms:

- **Positive affect** enhances resource regeneration and coping efficiency
- **High resilience** provides emotional buffer against stress and improves social support effectiveness
- **Social optimization**: Recent social interactions boost resilience through optimized resource allocation

### Threshold Effects

Some resilience effects only occur when emotional states exceed certain thresholds, representing how intense emotions can trigger different coping responses.

**Social Resilience Optimization:**

$$\Delta R_{\text{soc}} = f(\text{daily interactions}, \text{daily support exchanges}, \text{resources}, R_{\text{0}}, \mathbf{e})$$

Where:
- $\Delta R_{\text{soc}}$ is resilience change from social optimization
- $f$ integrates social benefit with resource allocation
- $\mathbf{e}$ is protective factor efficacy vector

**Implementation**: [`integrate_social_resilience_optimization()`](../../src/python/affect_utils.py#L690-L745) in `affect_utils.py`

## Social Network Effects

### Neighbor Selection

Individuals interact with others in their social network based on Watts-Strogatz small-world connections, creating clusters of similar emotional states and patterns of emotional contagion.

### Interaction Frequency

Each day includes multiple opportunities for social interactions mixed with stress events, with social interactions making up about half of daily experiences.

## Integration Points

### System Interactions

**Stress System**:
- Stress events immediately change emotional states
- Coping outcomes influence long-term emotional patterns
- Event interpretations directly modify affect

**Resilience System**:
- Positive emotions improve resource recovery
- High resilience provides emotional protection against stress
- Social support effectiveness depends on emotional state

**Social Network**:
- Neighbor emotions influence daily emotional dynamics
- Network structure affects how emotions spread
- Similar emotional states lead to stronger connections

### Feedback Loops

The model includes several interconnected patterns:

1. **Positive Loop**: Good mood → Better coping → Higher resilience → Better mood
2. **Negative Loop**: Poor mood → Worse coping → Lower resilience → Poorer mood
3. **Social Amplification**: Similar social connections reinforce existing emotional states
4. **Homeostatic Regulation**: Natural tendency to return to baseline emotional equilibrium
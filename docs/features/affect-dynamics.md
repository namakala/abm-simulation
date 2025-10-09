# Affect Dynamics and Influencing Factors

_See [`_NOTATION.md`](./_NOTATION.md) for symbol definitions and conventions._

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

### Interaction Effects

Social interactions create mutual influence where both individuals affect each other's emotional state. The model recognizes that negative emotional states tend to have a stronger impact than positive ones, reflecting how negative interactions can be more memorable and influential.

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

### Baseline Affect Dynamics

Each person has a stable baseline affect that represents their natural emotional balance. This baseline remains relatively constant and serves as the target that emotional dynamics pull toward.

## Integrated Affect Dynamics

### Complete Update Process

Daily emotional changes result from multiple factors working together:

- Social influence from connections
- Emotional impact of stress events
- Natural tendency to return to baseline

These factors combine to determine how emotional state changes throughout each day.

## Resilience-Affect Interactions

### Bidirectional Influence

Emotional state and resilience influence each other:

- **Positive affect** tends to improve resource recovery
- **High resilience** provides emotional buffer against stress

### Threshold Effects

Some resilience effects only occur when emotional states exceed certain thresholds, representing how intense emotions can trigger different coping responses.

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
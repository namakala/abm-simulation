# Step-by-Step Calculations for Resilience, Stress, and Affect

_See [`_NOTATION.md`](./_NOTATION.md) for symbol definitions and conventions._

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
At the beginning of each day, the system captures initial values for affect and resilience, gathers information about neighbor emotional states, and initializes tracking variables for daily events.

### Step 2: Subevent Generation and Execution

**Subevent Count Determination**:
Each day includes a variable number of subevents (typically 7-8) that can be either social interactions or stress events, with the sequence randomized for each day.

**Action Processing**:
The system processes each subevent, tracking social interactions and accumulating challenge/hindrance effects from stress events.

### Step 3: Daily Challenge/Hindrance Integration

**Normalization by Event Count**:
Daily challenge and hindrance values are averaged across all stress events in the day to create representative daily values.

**Interpretation**: Daily challenge/hindrance represent average event appraisal across all stress events in the day.

### Step 4: Affect Dynamics Calculation

**Integrated Affect Update**:
Affect changes result from multiple components working together:

- **Peer Influence**: Social influence from neighbors
- **Event Appraisal**: Emotional impact of stress events
- **Homeostasis**: Natural tendency to return to baseline

These components combine to determine daily affect changes.

**Detailed Component Calculations**:

#### Peer Influence
Social influence from neighbors affects emotional state, with the extent depending on the number of influential neighbors and the strength of their emotional states.

#### Event Appraisal Effect
Stress events influence emotional state based on their challenge/hindrance nature, with challenging events potentially motivating and hindrance events potentially demotivating.

#### Homeostasis Effect
Emotional states naturally tend to return to baseline levels over time, representing psychological adaptation and equilibrium-seeking tendencies.

### Step 5: Resilience Dynamics Calculation

**Integrated Resilience Update**:
Resilience changes result from multiple components working together:

- **Coping Success Effect**: Impact of successful stress management
- **Social Support Effect**: Benefits from supportive relationships
- **Overload Effect**: Consequences of cumulative hindrance events

These components combine to determine daily resilience changes.

**Detailed Component Calculations**:

#### Coping Success Effect
Coping outcomes influence resilience based on challenge/hindrance characteristics, with successful coping building resilience and failed coping depleting it.

#### Social Support Effect
Social support received during interactions provides a boost to resilience, representing how supportive relationships help build coping capacity.

#### Overload Effect
When hindrance events accumulate beyond a threshold, they create an overload effect that reduces resilience, representing how chronic stress can overwhelm coping capacity.

### Step 6: Resource Dynamics Integration

**Resource Regeneration**:
Resources naturally regenerate toward maximum capacity, with positive affect enhancing the regeneration process. Successful coping consumes resources, representing the energy invested in stress management.

**Resource Regeneration Function**:
Resources regenerate linearly toward maximum capacity, representing natural recovery and rest processes.

### Step 7: Protective Factor Management

**Resource Allocation to Protective Factors**:
Individuals allocate available resources across protective factors using a decision-making process that considers current efficacy levels and allocation temperature.

**Efficacy Updates**:
Resource investment improves protective factor efficacy, with higher returns when current efficacy is lower, representing realistic learning and development processes.

### Step 8: Homeostatic Adjustment

**Homeostatic Pull to Baseline**:
All systems have a natural tendency to return to baseline equilibrium levels, representing psychological adaptation and recovery processes.

**Application to Both Systems**:
Both affect and resilience are adjusted toward their baseline levels, with the rate of adjustment determining how quickly equilibrium is restored.

### Step 9: Daily Reset and Tracking

**End-of-Day Processing**:
At the end of each day, the system applies daily resets to affect and stress levels, stores daily summaries for analysis, and resets tracking variables for the next day.

**Stress Decay**:
Stress levels naturally decay over time when no new stress events occur, representing psychological adaptation and forgetting of past stressors.

### Stress Decay Rate

**Parameter**: `STRESS_DECAY_RATE` (default: 0.05, range: 0.0-1.0)

**Description**: Controls the rate at which stress levels decay over time when no new stress events occur, representing natural recovery processes and psychological adaptation.

**Interpretation**:
- **High values**: Rapid stress decay, representing effective natural recovery
- **Low values**: Slow stress decay, representing persistent stress effects
- **Research context**: Calibrated against psychological research on stress recovery

**Theoretical Foundation**:
- **Natural Recovery**: Psychological tendency to return to baseline stress levels
- **Adaptation Process**: How individuals habituate to ongoing stressors
- **Memory Effects**: Fading of stressful memories and their emotional impact

**Integration with Other Systems**:
- **Stress Events**: New events add to current stress before decay
- **Consecutive Hindrances**: Decay affects hindrance tracking
- **Network Adaptation**: Persistent stress may trigger network changes

### Step 10: Stress Assessment Score Computation and Agent State Integration

**Assessment Integration Steps**:

The stress assessment integration occurs at the end of each simulation step and serves as both a measurement tool and a validation mechanism.

**Detailed Assessment Update Process**:

1. **Stress Dimension Input**: Current stress controllability and overload values
2. **Dimension Score Generation**: Create correlated dimension scores
3. **Item Response Generation**: Generate individual assessment item responses
4. **Total Score Calculation**: Sum all items to get total assessment score
5. **Reverse Mapping**: Update stress dimensions based on assessment responses
6. **History Tracking**: Store assessment trajectory for analysis and validation

**Assessment Dimension Score Generation**:
Dimension scores are generated using correlated distributions that reflect the empirical structure of stress responses.

**Example Dimension Score Generation**:
For an individual with moderate stress levels, the system generates slightly correlated dimension scores that reflect realistic stress response patterns.

**Assessment Item Response Generation**:
Individual item responses are generated based on dimension scores and factor loadings, implementing the empirical structure of stress assessment scales.

**Stress Level Update from Assessment**:
Assessment responses create a feedback loop that influences underlying stress dimensions, enabling empirical calibration and validation.

**Example Assessment Integration in Agent Step**:
The assessment integration occurs as part of the daily step process, ensuring continuous measurement and feedback.

**Parameters**:
- `correlation`: 0.3 (correlation between stress dimensions)
- `controllability_sd`: 1.0 (variability in controllability dimension)
- `overload_sd`: 1.0 (variability in overload dimension)

## Complete Integration Example

### Example: High-Challenge Event Day

**Scenario**: Individual experiences a high-challenge, low-hindrance event with positive social connections

**Step-by-Step Calculations**:

1. **Event Generation**:
   - High controllability, Low overload

2. **Appraisal**:
   - High challenge, Low hindrance

3. **Threshold Evaluation**:
   - Event exceeds stress threshold, triggering stress response

4. **Social Interaction**:
   - Positive neighbor emotional states provide social influence

5. **Affect Update**:
   - Combined effects result in positive affect change

6. **Resilience Update**:
   - Social support provides resilience boost

7. **Resource Update**:
   - Regeneration enhanced by positive affect

### Example: High-Hindrance Event Day

**Scenario**: Individual experiences a high-hindrance, low-challenge event with negative social connections

**Step-by-Step Calculations**:

1. **Event Generation**:
   - Low controllability, High overload

2. **Appraisal**:
   - Low challenge, High hindrance

3. **Threshold Evaluation**:
   - Event exceeds stress threshold, triggering stress response

4. **Coping Determination**:
   - Low coping success probability due to hindrance and negative social influence

5. **State Updates**:
   - Combined effects result in negative affect and resilience changes

6. **Final Values**:
   - Overall negative impact on mental health state

## Mathematical Integration Summary

### Core Integration Equations

**Affect Dynamics**:
Affect changes result from peer influence, event appraisal, and homeostatic processes.

**Resilience Dynamics**:
Resilience changes result from coping success, social support, overload effects, and challenge/hindrance outcomes.

**Stress Processing**:
Stress levels change based on event characteristics and natural decay processes.

### Parameter Integration

All calculations use the unified configuration system, ensuring consistent parameter usage across all calculation steps and maintaining research reproducibility requirements.
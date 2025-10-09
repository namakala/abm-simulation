# Data Collection Implementation: Agent and Model-Level Metrics

_See [`.kilocode/rules/math/notation.md`](../../.kilocode/rules/math/notation.md) for symbol definitions and conventions._

## Overview

This document provides an overview of the data collection system that tracks both individual experiences and population patterns. The system captures detailed information about how individuals respond to stress and social influences, as well as broader trends across the entire population.

## Data Collection Architecture

### Collection Framework

The system uses a structured approach to collect data efficiently, focusing on key indicators of mental health and social dynamics. Data is gathered once per day during the simulation process.

**Key Benefits:**
- **Performance Optimized**: Efficient data collection methods
- **Memory Efficient**: Streamlined storage approach
- **Research-Ready**: Comprehensive metrics for mental health research
- **Standardized Access**: Consistent data access patterns

### Collection Frequency

Data is collected once per simulation day, capturing the state of all individuals and population-level patterns at regular intervals.

## Agent-Level Variables

Agent-level variables capture individual experiences, emotional states, and coping patterns. Each person is recorded once per time step, enabling analysis of individual differences and personal trajectories.

### Core State Variables

| Variable | Range | Description |
|----------|-------|-------------|
| **Perceived Stress** | 0-40 | Individual stress scale score |
| **Resilience** | 0-1 | Current capacity to adapt and recover |
| **Affect** | -1 to +1 | Current emotional state |
| **Resources** | 0-1 | Available psychological and physical resources |

### Stress Processing Variables

| Variable | Range | Description |
|----------|-------|-------------|
| **Current Stress** | 0-1 | Current stress level |
| **Stress Controllability** | 0-1 | Perceived control over stress |
| **Stress Overload** | 0-1 | Perceived burden from stress |
| **Consecutive Hindrances** | 0+ | Count of ongoing hindrance events |

### Operational Definitions

#### Perceived Stress
**Definition**: An individual's subjective experience of stress as measured by a standard scale.

**Measurement**:
- **Scale**: 0-40 based on responses to 10 statements
- **Components**: Includes both controllability and overload dimensions
- **Integration**: Updated daily based on current stress experiences

#### Resilience
**Definition**: An individual's capacity to adapt and recover from stress events.

**Measurement**:
- **Scale**: 0-1 representing capacity level
- **Baseline**: Individual's natural equilibrium point
- **Dynamics**: Changes based on coping outcomes and social support
- **Homeostasis**: Natural tendency to return to baseline level

#### Affect
**Definition**: Current emotional state ranging from negative to positive.

**Measurement**:
- **Scale**: -1 to +1 representing emotional valence
- **Baseline**: Individual's natural equilibrium point
- **Influences**: Social connections, stress events, and natural regulation
- **Social Effects**: Neighbor emotions influence daily emotional patterns

#### Resources
**Definition**: Available psychological and physical resources for coping with stress.

**Measurement**:
- **Scale**: 0-1 representing available resources
- **Regeneration**: Natural rebuilding toward maximum capacity
- **Consumption**: Used for coping and maintaining protective factors
- **Affect Influence**: Positive emotions enhance rebuilding rate

## Model-Level Variables

Model-level variables capture population patterns, aggregated statistics, and system-wide trends. These are calculated once per time step from individual data.

### Primary Outcome Measures

| Variable | Range | Description |
|----------|-------|-------------|
| **Average Perceived Stress** | 0-40 | Population average stress score |
| **Average Resilience** | 0-1 | Population average resilience capacity |
| **Average Affect** | -1 to +1 | Population average emotional state |
| **Coping Success Rate** | 0-1 | Population success rate in coping with stress |

### Resource and Stress Metrics

| Variable | Range | Description |
|----------|-------|-------------|
| **Average Resources** | 0-1 | Population average resource levels |
| **Average Stress** | 0-1 | Population average stress levels |
| **Social Support Rate** | 0-1 | Rate of supportive interactions |
| **Stress Events** | 0+ | Total stress events per day |

### Network and Social Metrics

| Variable | Range | Description |
|----------|-------|-------------|
| **Network Density** | 0-1 | Social network connectivity |
| **Stress Prevalence** | 0-1 | Proportion experiencing high stress |
| **Low Resilience Count** | 0+ | Number with low resilience |
| **High Resilience Count** | 0+ | Number with high resilience |

### Challenge-Hindrance Appraisal Metrics

| Variable | Range | Description |
|----------|-------|-------------|
| **Average Challenge** | 0-1 | Average challenge appraisal |
| **Average Hindrance** | 0-1 | Average hindrance appraisal |
| **Challenge-Hindrance Ratio** | -1 to +1 | Balance between challenge and hindrance |
| **Average Consecutive Hindrances** | 0+ | Average ongoing hindrance events |

### Daily Activity Statistics

| Variable | Range | Description |
|----------|-------|-------------|
| **Total Stress Events** | 0+ | Total stress events across population |
| **Successful Coping** | 0+ | Total successful coping instances |
| **Social Interactions** | 0+ | Total social interactions |
| **Support Exchanges** | 0+ | Total supportive exchanges |

## Operational Definitions and Measurement Details

### Population Averages

**Average Perceived Stress**:
- **Definition**: Population-level stress as measured by standard scale
- **Interpretation**: Higher values indicate greater population stress
- **Research Use**: Primary measure for evaluating intervention effectiveness

**Average Resilience**:
- **Definition**: Average resilience capacity across the population
- **Interpretation**: Higher values indicate greater population resilience
- **Research Use**: Key indicator of mental health promotion success

**Average Affect**:
- **Definition**: Average emotional state across the population
- **Interpretation**: Positive values indicate generally positive population mood
- **Research Use**: Indicator of overall population mental health

**Population Average Equations:**

$$\bar{\Psi} = \frac{1}{N} \sum_{i=1}^N \Psi_i$$

$$\bar{R} = \frac{1}{N} \sum_{i=1}^N R_i$$

$$\bar{A} = \frac{1}{N} \sum_{i=1}^N A_i$$

Where:
- $\bar{\Psi}$ is average PSS-10 score
- $\bar{R}$ is average resilience
- $\bar{A}$ is average affect
- $N$ is population size
- $\Psi_i, R_i, A_i$ are individual agent values

**Implementation**: [`get_avg_pss10()`](src/python/model.py:574), [`get_avg_resilience()`](src/python/model.py:597), [`get_avg_affect()`](src/python/model.py:610) in `model.py`

### Stress Processing Metrics

**Coping Success Rate**:
- **Definition**: Proportion of stress events successfully managed
- **Interpretation**: Higher rates indicate better population coping capacity
- **Research Use**: Measure of stress management effectiveness

**Stress Prevalence**:
- **Definition**: Proportion of population experiencing high stress
- **Interpretation**: Higher values indicate greater stress burden
- **Research Use**: Public health indicator for targeting interventions

### Social Network Metrics

**Social Support Rate**:
- **Definition**: Rate of meaningful supportive interactions
- **Interpretation**: Higher rates indicate more effective social support networks
- **Research Use**: Measure of social capital and support system effectiveness

**Network Density**:
- **Definition**: Connectivity of the social network
- **Interpretation**: Higher values indicate more interconnected social network
- **Research Use**: Indicator of social cohesion and information flow

**Network Metrics Equations:**

**Social Support Rate:**

$$r_s = \frac{n_{se}}{n_{si}}$$

**Network Density:**

$$\rho_n = \frac{2 \cdot n_e}{N \cdot (N-1)}$$

Where:
- $r_s \in [0,1]$ is social support rate
- $n_{se}$ is number of support exchanges
- $n_{si}$ is number of social interactions
- $\rho_n \in [0,1]$ is network density
- $n_e$ is number of edges in network
- $N$ is number of nodes (agents)

**Implementation**: [`_calculate_social_support_rate()`](src/python/model.py:263), [`_calculate_network_density()`](src/python/model.py:270) in `model.py`

**Stress Prevalence:**

$$p_s = \frac{1}{N} \sum_{i=1}^N \mathbb{1}_{\Psi_i \geq \theta_\Psi}$$

Where:
- $p_s \in [0,1]$ is stress prevalence
- $\mathbb{1}$ is indicator function
- $\Psi_i$ is PSS-10 score for agent $i$
- $\theta_\Psi$ is PSS-10 stress threshold

### Challenge-Hindrance Metrics

**Challenge-Hindrance Ratio**:
- **Definition**: Balance between challenge and hindrance stress events
- **Interpretation**: Positive values indicate more challenge-dominant stress, negative values indicate more hindrance-dominant stress
- **Research Use**: Indicator of stress type distribution in population

**Resilience Distribution:**

**Low Resilience Count:**

$$n_{R_l} = \sum_{i=1}^N \mathbb{1}_{R_i < 0.3}$$

**High Resilience Count:**

$$n_{R_h} = \sum_{i=1}^N \mathbb{1}_{R_i > 0.7}$$

**Medium Resilience Count:**

$$n_{R_m} = N - n_{R_l} - n_{R_h}$$

Where:
- $n_{R_l}, n_{R_m}, n_{R_h} \in \mathbb{N}$ are resilience category counts
- $\mathbb{1}$ is indicator function
- $R_i \in [0,1]$ is resilience for agent $i$

**Challenge-Hindrance Metrics:**

**Average Challenge:**

$$\bar{\chi} = \frac{1}{N \cdot n_d} \sum_{i=1}^N \sum_{j=1}^{n_d} \chi_{ij}$$

**Challenge-Hindrance Ratio:**

$$r_{\chi\zeta} = \frac{\bar{\chi} - \bar{\zeta}}{\bar{\chi} + \bar{\zeta}} \quad \text{if } \bar{\chi} + \bar{\zeta} > 0 \text{ else } 0$$

Where:
- $\bar{\chi}, \bar{\zeta} \in [0,1]$ are average challenge/hindrance
- $n_d$ is days in simulation
- $\chi_{ij}, \zeta_{ij} \in [0,1]$ are challenge/hindrance for agent $i$ on day $j$

## Data Access Patterns

### Direct Access

Researchers can access both individual and population data through standardized methods that provide time series information for analysis.

### Convenience Methods

The system provides simplified access methods for common research tasks, including population summaries and individual trajectories.

### Data Structure

**Population Data**: Organized by time with all population-level metrics
**Individual Data**: Organized by time and individual identifier with personal metrics

## Research Applications

### Individual-Level Analysis

Researchers can track individual patterns over time, identify at-risk individuals, and analyze personal trajectories of resilience and stress management.

### Population-Level Analysis

The data enables analysis of population trends, statistical relationships between different factors, and evaluation of overall system behavior.

### Intervention Evaluation

Data can be used to compare different scenarios, calculate effect sizes, and evaluate the impact of various intervention approaches.

## Data Export and Persistence

### Export Options

Data can be exported in standard formats for further analysis, with options for both population and individual-level data.

### Export Verification

The system includes checks to ensure data integrity during export and storage processes.

## Validation and Quality Assurance

### Data Integrity Checks

The system includes comprehensive validation:

1. **Completeness Checks**: Ensuring all data points are captured
2. **Range Validation**: Verifying data falls within expected ranges
3. **Consistency Checks**: Ensuring different calculation methods agree
4. **Continuity Validation**: Verifying sequential data collection

### Performance Validation

- **Scale Testing**: Validated with various population sizes and time periods
- **Stability Monitoring**: Memory and performance tracking during extended runs
- **Export Testing**: Large dataset export functionality verified

### Integration Testing

Comprehensive testing covers:
- Complete simulation runs with data collection
- Multiple scenarios with different configurations
- Export and storage functionality
- Error handling and edge cases
- Performance validation for large-scale simulations

## Configuration Integration

All data collection uses the unified configuration system, ensuring consistent parameters across all metrics and maintaining research reproducibility.

## Future Extensions

### Planned Enhancements

1. **Advanced Analysis**: Integration with statistical software for deeper analysis
2. **Enhanced Storage**: Large-scale data storage and querying capabilities
3. **Advanced Network Metrics**: Temporal network evolution tracking
4. **Custom Metrics**: User-defined research indicators

### Extension Points

The system is designed to be expandable, allowing researchers to add custom metrics as research needs evolve.

This data collection system supports comprehensive research into mental health promotion and cost-effectiveness by providing detailed, validated metrics at both individual and population levels.
# Data Collection Implementation: Agent and Model-Level Metrics

## Overview

This document provides a comprehensive overview of the data collection system implemented using Mesa's DataCollector framework. The system collects both **agent-level variables** (individual agent states and behaviors) and **model-level variables** (population statistics and aggregated metrics) for research analysis and validation.

## Data Collection Architecture

### Mesa DataCollector Framework

The implementation uses Mesa's built-in `DataCollector` class for efficient, standardized data collection:

```python
self.datacollector = DataCollector(
    model_reporters=model_reporters,  # Population-level metrics (20+ indicators)
    agent_reporters=agent_reporters   # Individual-level metrics (8+ per agent)
)
```

**Key Benefits:**
- **Performance Optimized**: Leverages Mesa's optimized data collection mechanisms
- **Memory Efficient**: Reduced memory footprint through optimized storage
- **Research-Ready**: Comprehensive metrics for mental health research and analysis
- **Standardized Access**: Consistent data access patterns via DataFrame outputs

### Collection Frequency

Data is collected **once per simulation day** during the `model.step()` method:

```python
def step(self):
    # Execute all agent steps
    self.agents.shuffle_do("step")

    # Single line collection of all metrics
    self.datacollector.collect(self)

    # Continue with network adaptation and other model-level processes
```

## Agent-Level Variables

Agent-level variables capture individual agent states, behaviors, and trajectories. Each agent is recorded once per time step, enabling longitudinal analysis of individual differences.

### Core State Variables

| Variable | Data Type | Range | Description | Measurement Method |
|----------|-----------|-------|-------------|-------------------|
| **pss10** | Float | [0, 40] | Individual Perceived Stress Scale-10 score | Sum of 10 PSS-10 item responses (0-4 scale each) |
| **resilience** | Float | [0, 1] | Current resilience capacity | Agent's internal resilience state variable |
| **affect** | Float | [-1, 1] | Current emotional state | Agent's internal affect state variable (-1=very negative, +1=very positive) |
| **resources** | Float | [0, 1] | Available psychological/physical resources | Agent's internal resource state variable |

### Stress Processing Variables

| Variable | Data Type | Range | Description | Measurement Method |
|----------|-----------|-------|-------------|-------------------|
| **current_stress** | Float | [0, 1] | Current stress level | Agent's internal stress state variable |
| **stress_controllability** | Float | [0, 1] | Perceived controllability of stress | Agent's controllability dimension from PSS-10 |
| **stress_overload** | Float | [0, 1] | Perceived overload from stress | Agent's overload dimension from PSS-10 |
| **consecutive_hindrances** | Integer | [0, ∞) | Count of consecutive hindrance events | Cumulative count of predominantly hindrance stress events |

### Operational Definitions

#### PSS-10 Score (`pss10`)
**Operational Definition**: The Perceived Stress Scale-10 total score representing an individual's subjective perception of stress over the past month.

**Measurement**:
- **Scale**: 0-40 (sum of 10 items)
- **Items**: 10 statements rated 0-4 (0=Never, 4=Very Often)
- **Components**:
  - Items 1, 2, 3, 6, 9, 10: Overload dimension (direct scoring)
  - Items 4, 5, 7, 8: Controllability dimension (reverse scored)
- **Integration**: Updated daily based on current stress state using bifactor model

#### Resilience (`resilience`)
**Operational Definition**: An agent's capacity to adapt and recover from stress events, representing psychological resilience resources.

**Measurement**:
- **Scale**: 0-1 (normalized)
- **Baseline**: Agent-specific natural equilibrium point
- **Dynamics**: Changes based on coping outcomes, social support, and protective factors
- **Homeostasis**: Pulls toward baseline level over time

#### Affect (`affect`)
**Operational Definition**: Current emotional state on a valence continuum from negative to positive emotions.

**Measurement**:
- **Scale**: -1 to +1 (-1=very negative, +1=very positive)
- **Baseline**: Agent-specific natural equilibrium point
- **Influences**: Peer influence, stress events, homeostasis
- **Social Effects**: Neighbor affect influences daily dynamics

#### Resources (`resources`)
**Operational Definition**: Available psychological and physical resources for coping with stress and maintaining protective factors.

**Measurement**:
- **Scale**: 0-1 (normalized)
- **Regeneration**: Passive regeneration toward maximum capacity
- **Consumption**: Used for successful coping and protective factor maintenance
- **Affect Influence**: Positive affect enhances regeneration rate

## Model-Level Variables

Model-level variables capture population statistics, aggregated metrics, and system-wide patterns. These are computed once per time step from agent data.

### Primary Outcome Measures

| Variable | Data Type | Range | Description | Measurement Method |
|----------|-----------|-------|-------------|-------------------|
| **avg_pss10** | Float | [0, 40] | Population average PSS-10 score | Mean of all agents' PSS-10 scores |
| **avg_resilience** | Float | [0, 1] | Population average resilience | Mean of all agents' resilience levels |
| **avg_affect** | Float | [-1, 1] | Population average affect | Mean of all agents' affect levels |
| **coping_success_rate** | Float | [0, 1] | Population coping success rate | Proportion of successful coping attempts across all agents |

### Resource and Stress Metrics

| Variable | Data Type | Range | Description | Measurement Method |
|----------|-----------|-------|-------------|-------------------|
| **avg_resources** | Float | [0, 1] | Population average resources | Mean of all agents' resource levels |
| **avg_stress** | Float | [0, 1] | Population average stress | Mean of all agents' current stress levels |
| **social_support_rate** | Float | [0, 1] | Rate of social support exchanges | Social support exchanges divided by total interactions |
| **stress_events** | Integer | [0, ∞) | Total stress events per day | Count of stress events across all agents |

### Network and Social Metrics

| Variable | Data Type | Range | Description | Measurement Method |
|----------|-----------|-------|-------------|-------------------|
| **network_density** | Float | [0, 1] | Network connectivity measure | Actual connections divided by possible connections |
| **stress_prevalence** | Float | [0, 1] | Proportion with high stress | Agents with affect < -0.3 divided by total agents |
| **low_resilience** | Integer | [0, N] | Count with low resilience | Agents with resilience < 0.3 |
| **high_resilience** | Integer | [0, N] | Count with high resilience | Agents with resilience > 0.7 |

### Challenge-Hindrance Appraisal Metrics

| Variable | Data Type | Range | Description | Measurement Method |
|----------|-----------|-------|-------------|-------------------|
| **avg_challenge** | Float | [0, 1] | Average challenge appraisal | Mean challenge values from all stress events |
| **avg_hindrance** | Float | [0, 1] | Average hindrance appraisal | Mean hindrance values from all stress events |
| **challenge_hindrance_ratio** | Float | [-1, 1] | Balance between challenge and hindrance | (challenge - hindrance) / (challenge + hindrance) |
| **avg_consecutive_hindrances** | Float | [0, ∞) | Average consecutive hindrance events | Mean consecutive hindrance count across agents |

### Daily Activity Statistics

| Variable | Data Type | Range | Description | Measurement Method |
|----------|-----------|-------|-------------|-------------------|
| **total_stress_events** | Integer | [0, ∞) | Total stress events across population | Sum of stress events from all agents |
| **successful_coping** | Integer | [0, ∞) | Total successful coping instances | Count of successful coping across all agents |
| **social_interactions** | Integer | [0, ∞) | Total social interactions | Sum of daily interactions from all agents |
| **support_exchanges** | Integer | [0, ∞) | Total support exchanges | Sum of daily support exchanges from all agents |

## Operational Definitions and Measurement Details

### Population Averages

**avg_pss10**:
- **Definition**: Population-level perceived stress as measured by PSS-10
- **Calculation**: `mean([agent.pss10 for agent in model.agents])`
- **Interpretation**: Higher values indicate greater population stress levels
- **Research Use**: Primary outcome measure for intervention effectiveness

**avg_resilience**:
- **Definition**: Average resilience capacity across the population
- **Calculation**: `mean([agent.resilience for agent in model.agents])`
- **Interpretation**: Higher values indicate greater population resilience
- **Research Use**: Key indicator of mental health promotion success

**avg_affect**:
- **Definition**: Average emotional state across the population
- **Calculation**: `mean([agent.affect for agent in model.agents])`
- **Interpretation**: Positive values indicate generally positive population mood
- **Research Use**: Indicator of overall population mental health

### Stress Processing Metrics

**coping_success_rate**:
- **Definition**: Proportion of stress events successfully coped with
- **Calculation**: Total successful coping events divided by total stress events
- **Interpretation**: Higher rates indicate better population coping capacity
- **Research Use**: Measure of stress management effectiveness

**stress_prevalence**:
- **Definition**: Proportion of population experiencing high stress
- **Calculation**: Count of agents with `affect < -0.3` divided by total agents
- **Interpretation**: Higher values indicate greater stress burden in population
- **Research Use**: Public health indicator for intervention targeting

### Social Network Metrics

**social_support_rate**:
- **Definition**: Rate of meaningful social support exchanges
- **Calculation**: Number of support exchanges divided by total interactions
- **Interpretation**: Higher rates indicate more effective social support networks
- **Research Use**: Measure of social capital and support system effectiveness

**network_density**:
- **Definition**: Connectivity of the social network
- **Calculation**: Actual connections divided by possible connections in NetworkX graph
- **Interpretation**: Higher values indicate more interconnected social network
- **Research Use**: Indicator of social cohesion and information flow

### Challenge-Hindrance Metrics

**challenge_hindrance_ratio**:
- **Definition**: Balance between challenge and hindrance stress events
- **Calculation**: `(avg_challenge - avg_hindrance) / (avg_challenge + avg_hindrance)`
- **Interpretation**: Positive values indicate more challenge-dominant stress, negative values indicate more hindrance-dominant stress
- **Research Use**: Indicator of stress type distribution in population

## Data Access Patterns

### Direct DataCollector Access

```python
# Get model-level time series data
model_data = model.datacollector.get_model_vars_dataframe()

# Get agent-level time series data
agent_data = model.datacollector.get_agent_vars_dataframe()

# Get latest model data point
latest_data = model_data.iloc[-1]
current_stress = latest_data['avg_pss10']
```

### Convenience Methods

```python
# Get current population summary
summary = model.get_population_summary()

# Get time series data (same as direct access)
time_series = model.get_time_series_data()

# Get agent trajectories
agent_trajectories = model.get_agent_time_series_data()
```

### DataFrame Structure

**Model DataFrame Columns**:
- Day index and metadata
- All 20+ model-level metrics
- Time series suitable for plotting and statistical analysis

**Agent DataFrame Structure**:
```python
# MultiIndex DataFrame with:
# Level 0: Step (time step)
# Level 1: AgentID (unique agent identifier)
# Columns: All 8+ agent-level variables
```

## Research Applications

### Individual-Level Analysis

```python
# Analyze individual resilience trajectories
agent_data = model.get_agent_time_series_data()
resilience_trajectories = agent_data.pivot(
    index='Step', columns='AgentID', values='resilience'
)

# Identify at-risk individuals
final_state = agent_data.groupby('AgentID').last()
at_risk = final_state[final_state['resilience'] < 0.3]
```

### Population-Level Analysis

```python
# Analyze population trends
model_data = model.get_time_series_data()
stress_trends = model_data[['avg_pss10', 'avg_resilience']].rolling(window=7).mean()

# Statistical analysis
from scipy import stats
correlation = stats.pearsonr(model_data['avg_stress'], model_data['avg_affect'])
```

### Intervention Evaluation

```python
# Compare baseline vs intervention
baseline_data = baseline_model.get_time_series_data()
intervention_data = intervention_model.get_time_series_data()

# Calculate effect sizes
effect_size = (intervention_data['avg_pss10'].mean() - baseline_data['avg_pss10'].mean()) / baseline_data['avg_pss10'].std()
```

## Data Export and Persistence

### CSV Export

```python
# Export model data
model.export_results("simulation_results.csv")

# Export agent data
model.export_agent_data("agent_trajectories.csv")

# Custom filenames
model.export_results("custom_model_data.csv")
model.export_agent_data("custom_agent_data.csv")
```

### Export Verification

```python
# Verify export integrity
original_model_data = model.datacollector.get_model_vars_dataframe()
exported_model_data = pd.read_csv("simulation_results.csv")

# Data should match exactly (allowing for minor dtype differences)
pd.testing.assert_frame_equal(
    original_model_data.reset_index(drop=True),
    exported_model_data.reset_index(drop=True),
    check_dtype=False
)
```

## Validation and Quality Assurance

### Data Integrity Checks

The system includes comprehensive validation:

1. **Null Value Detection**: All metrics checked for null/invalid values
2. **Range Validation**: Variables verified to be within expected ranges
3. **Cross-Validation**: Consistency between different calculation methods
4. **Time Series Continuity**: Sequential data collection verification

### Performance Validation

- **Large Scale Testing**: Validated with 100+ agents and 20+ day simulations
- **Memory Stability**: Memory usage monitoring during extended runs
- **Export Performance**: Large dataset export functionality verified

### Integration Testing

Comprehensive test suite covers:
- End-to-end simulation runs with data collection
- Multi-day scenarios with different configurations
- Export and persistence functionality
- Error handling and edge cases
- Performance validation for large-scale simulations

## Configuration Integration

All data collection metrics use the unified configuration system:

```python
# DataCollector configuration is integrated with main config
config = get_config()

# Model reporters use config values for thresholds and parameters
'stress_prevalence': lambda m: sum(1 for agent in m.agents if agent.affect < -0.3) / len(m.agents)
'network_density': lambda m: m._calculate_network_density()
```

This ensures consistent parameter usage across all metrics and maintains research reproducibility requirements.

## Future Extensions

### Planned Enhancements

1. **R Integration**: Statistical analysis and visualization in R
2. **SQL Storage**: Large-scale parameter sweep storage and querying
3. **Advanced Network Metrics**: Temporal network evolution tracking
4. **Custom Metric Registration**: User-defined research metrics

### Extension Points

The DataCollector system is designed for extensibility:

```python
# Add custom model metric
model_reporters['custom_metric'] = lambda m: m.calculate_custom_metric()

# Add custom agent metric
agent_reporters['custom_agent_metric'] = lambda a: a.custom_agent_calculation()
```

This modular design supports the evolving research needs of the mental health promotion cost-effectiveness study.
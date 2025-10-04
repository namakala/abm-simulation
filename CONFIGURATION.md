# Agent-Based Mental Health Simulation - Configuration Guide

## Overview

This document provides comprehensive documentation for the `.env` configuration system used in the Agent-Based Mental Health Simulation project. The configuration system allows researchers to easily modify simulation parameters, run sensitivity analyses, and adapt the model to different research scenarios without modifying source code.

## What is the .env Configuration System?

The `.env` configuration system provides:

- **Centralized parameter management** - All simulation parameters in one location
- **Environment-specific configurations** - Different settings for development, testing, and production
- **Type-safe parameter loading** - Automatic type conversion with validation
- **Research-friendly** - Easy parameter sweeps and scenario testing
- **Version control friendly** - Parameters separate from code logic

### Key Components

- **`.env`** - Your active configuration file (customize for your needs)
- **`.env.example`** - Template with all parameters and documentation
- **`src/python/config.py`** - Python module that loads and validates configuration
- **`python-dotenv`** - Library for loading environment variables from `.env` files

## Quick Start Guide

### 1. Initial Setup

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Install dependencies:**
   ```bash
   pip install python-dotenv
   ```

3. **Customize your configuration:**
   ```bash
   # Edit .env file with your preferred settings
   nano .env  # or use your preferred editor
   ```

4. **Test your configuration:**
   ```python
   from src.python.config import get_config

   config = get_config()
   config.print_summary()
   ```

### 2. Basic Usage in Code

```python
from src.python.config import get_config

# Load configuration
config = get_config()

# Access parameters
num_agents = config.get('simulation', 'num_agents')
# or
num_agents = config.num_agents

# Use in simulation
print(f"Running simulation with {config.num_agents} agents for {config.max_days} days")
```

### 3. Environment-Specific Configuration

Create different `.env` files for different scenarios:

```bash
# Development (faster, smaller scale)
cp .env.example .env.development

# Production (larger scale, more replicates)
cp .env.example .env.production

# Load specific environment
config = get_config('.env.production')
```

## Configuration Reference

All configuration parameters are organized into logical categories. Each parameter includes its default value, valid range, and description.

### Simulation and Network Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `SIMULATION_NUM_AGENTS` | 20 | 10-10,000 | Number of agents in the simulation population |
| `SIMULATION_MAX_DAYS` | 100 | 1-1000 | Maximum simulation duration in days |
| `SIMULATION_SEED` | 42 | Any integer | Random seed for reproducible results (null for random) |
| `NETWORK_WATTS_K` | 4 | Even numbers ≥2 | Watts-Strogatz network: neighbors per node |
| `NETWORK_WATTS_P` | 0.1 | 0.0-1.0 | Watts-Strogatz network: rewiring probability |
| `NETWORK_ADAPTATION_THRESHOLD` | 3 | ≥1 | Threshold for network adaptation when stress threshold breached |
| `NETWORK_HOMOPHILY_STRENGTH` | 0.7 | 0.0-1.0 | Strength of homophily in network connections |
| `NETWORK_REWIRE_PROBABILITY` | 0.01 | 0.0-1.0 | Probability of rewiring network connections |

### Agent State and Behavior Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `AGENT_INITIAL_RESILIENCE` | 0.5 | 0.0-1.0 | Starting resilience level for all agents |
| `AGENT_INITIAL_AFFECT` | 0.0 | -1.0 to 1.0 | Initial emotional state (-1=negative, 1=positive) |
| `AGENT_INITIAL_RESOURCES` | 0.6 | 0.0-1.0 | Starting psychological resources |
| `AGENT_STRESS_PROBABILITY` | 0.5 | 0.0-1.0 | Daily probability of experiencing stress events |
| `AGENT_COPING_SUCCESS_RATE` | 0.5 | 0.0-1.0 | Success rate when testing coping mechanisms |
| `AGENT_SUBEVENTS_PER_DAY` | 3 | 1-10 | Average social interactions + stress events per day |
| `AGENT_RESOURCE_COST` | 0.1 | 0.0-1.0 | Resource consumption for successful coping |

### Coping Mechanism Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `COPING_BASE_PROBABILITY` | 0.5 | 0.0-1.0 | Base coping probability before situational modifiers |
| `COPING_SOCIAL_INFLUENCE` | 0.1 | 0.0-1.0 | Social network influence on coping outcomes |
| `COPING_CHALLENGE_BONUS` | 0.2 | 0.0-1.0 | Bonus to coping success when facing challenge events |
| `COPING_HINDRANCE_PENALTY` | 0.3 | 0.0-1.0 | Penalty to coping success when facing hindrance events |

### Stress Event Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `STRESS_CONTROLLABILITY_MEAN` | 0.5 | 0.0-1.0 | Mean controllability of stress events (Beta distribution) |
| `STRESS_OVERLOAD_MEAN` | 0.5 | 0.0-1.0 | Mean overload intensity of stress events |
| `STRESS_BETA_ALPHA` | 2.0 | >0 | Alpha parameter for Beta distribution of stress events |
| `STRESS_BETA_BETA` | 2.0 | >0 | Beta parameter for Beta distribution of stress events |

### PSS-10 Integration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `PSS10_ITEM_MEAN` | [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5] | 0.0-4.0 | Mean values for PSS-10 items (10 values) |
| `PSS10_ITEM_SD` | [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8] | >0 | Standard deviations for PSS-10 items (10 values) |
| `PSS10_LOAD_CONTROLLABILITY` | [0.2, 0.8, 0.1, 0.7, 0.6, 0.1, 0.8, 0.6, 0.7, 0.1] | 0.0-1.0 | Factor loadings for controllability dimension (10 values) |
| `PSS10_LOAD_OVERLOAD` | [0.7, 0.3, 0.8, 0.2, 0.4, 0.9, 0.2, 0.3, 0.4, 0.9] | 0.0-1.0 | Factor loadings for overload dimension (10 values) |
| `PSS10_CONTROLLABILITY_SD` | 1.0 | >0 | Standard deviation for controllability factor |
| `PSS10_OVERLOAD_SD` | 1.0 | >0 | Standard deviation for overload factor |
| `PSS10_BIFACTOR_COR` | 0.3 | -1.0 to 1.0 | Correlation between bifactor dimensions |
| `PSS10_THRESHOLD` | 27 | Any integer | Threshold score for determining stressed state |

### Appraisal and Threshold Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `APPRAISAL_OMEGA_C` | 1.0 | 0.0-5.0 | Weight for controllability in challenge/hindrance appraisal |
| `APPRAISAL_OMEGA_O` | 1.0 | 0.0-5.0 | Weight for overload in appraisal |
| `APPRAISAL_BIAS` | 0.0 | -2.0 to 2.0 | Bias term in appraisal function |
| `APPRAISAL_GAMMA` | 6.0 | 1.0-20.0 | Sigmoid steepness for challenge/hindrance classification |
| `THRESHOLD_BASE_THRESHOLD` | 0.5 | 0.0-1.0 | Base stress threshold for becoming stressed |
| `THRESHOLD_CHALLENGE_SCALE` | 0.15 | 0.0-1.0 | How much challenge increases stress threshold |
| `THRESHOLD_HINDRANCE_SCALE` | 0.25 | 0.0-1.0 | How much hindrance decreases stress threshold |
| `THRESHOLD_STRESS_THRESHOLD` | 0.3 | 0.0-1.0 | Minimum stress level to trigger coping |
| `THRESHOLD_AFFECT_THRESHOLD` | 0.3 | 0.0-1.0 | Minimum emotional change for resilience adjustment |
| `STRESS_ALPHA_CHALLENGE` | 0.8 | 0.0-2.0 | Challenge stress multiplier |
| `STRESS_ALPHA_HINDRANCE` | 1.2 | 0.0-2.0 | Hindrance stress multiplier |
| `STRESS_DELTA` | 0.2 | 0.0-1.0 | Stress computation parameter |
| `STRESS_DECAY_RATE` | 0.05 | 0.0-1.0 | Natural stress recovery rate over time |

### Social Interaction Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `INTERACTION_INFLUENCE_RATE` | 0.05 | 0.0-0.5 | Base rate of affect influence between agents |
| `INTERACTION_RESILIENCE_INFLUENCE` | 0.05 | 0.0-0.5 | How partner affect influences agent resilience |
| `INTERACTION_MAX_NEIGHBORS` | 10 | 1-50 | Maximum neighbors considered for interactions |

### Affect Dynamics Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `AFFECT_PEER_INFLUENCE_RATE` | 0.1 | 0.0-1.0 | Strength of peer influence on affect |
| `AFFECT_EVENT_APPRAISAL_RATE` | 0.15 | 0.0-1.0 | How events affect baseline affect through appraisal |
| `AFFECT_HOMEOSTATIC_RATE` | 0.5 | 0.0-1.0 | Tendency for affect to return to baseline |
| `N_INFLUENCING_NEIGHBORS` | 5 | 1-20 | Number of neighbors that can influence affect |

### Resilience Dynamics Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `RESILIENCE_COPING_SUCCESS_RATE` | 0.1 | 0.0-0.5 | Resilience change from successful coping |
| `RESILIENCE_SOCIAL_SUPPORT_RATE` | 0.08 | 0.0-0.5 | Resilience boost from social support |
| `RESILIENCE_OVERLOAD_THRESHOLD` | 3 | 1-10 | Minimum consecutive hindrances for overload effect |
| `RESILIENCE_HOMEOSTATIC_RATE` | 0.05 | 0.0-1.0 | Tendency for resilience to return to baseline |
| `RESILIENCE_BOOST_RATE` | 0.1 | 0.0-1.0 | Boost rate from protective factors |
| `N_INFLUENCING_HINDRANCE` | 3 | 1-10 | Consecutive hindrances for overload effect |

### Affect and Resilience Dynamics Parameters

#### Affect Dynamics Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `AFFECT_PEER_INFLUENCE_RATE` | 0.1 | 0.0-1.0 | Strength of peer influence on affect - controls how much neighbors' emotional states influence an agent's affect |
| `AFFECT_EVENT_APPRAISAL_RATE` | 0.15 | 0.0-1.0 | How events affect baseline affect through challenge/hindrance appraisal - determines emotional impact of stress events |
| `AFFECT_HOMEOSTASIS_RATE` | 0.05 | 0.0-0.5 | Tendency for affect to return to baseline - models emotional regulation and recovery toward neutral state |
| `N_INFLUENCING_NEIGHBORS` | 5 | 1-20 | Number of neighbors that can influence an agent's affect - limits social influence to closest connections |

#### Resilience Dynamics Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `RESILIENCE_COPING_SUCCESS_RATE` | 0.1 | 0.0-0.5 | Resilience change from successful coping - positive effect on resilience when agent successfully handles stress |
| `RESILIENCE_SOCIAL_SUPPORT_RATE` | 0.08 | 0.0-0.5 | Resilience boost from social support - improvement in resilience from receiving help from others |
| `RESILIENCE_OVERLOAD_THRESHOLD` | 3 | 1-10 | Minimum consecutive hindrances for overload effect - number of hindrance events needed to trigger overload |
| `N_INFLUENCING_HINDRANCE` | 3 | 1-10 | Consecutive hindrances for overload effect - threshold for cumulative hindrance impact on resilience |

#### Stress and Coping Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `COPING_CHALLENGE_BONUS` | 0.2 | 0.0-0.5 | Bonus to coping success when facing challenge events |
| `COPING_HINDRANCE_PENALTY` | 0.3 | 0.0-0.5 | Penalty to coping success when facing hindrance events | 
| `COPING_BASE_PROBABILITY` | 0.5 | 0.0-1.0 | Base coping ability before situational modifiers |
| `COPING_SOCIAL_INFLUENCE` | 0.1 | 0.0-0.5 | Social network influence on coping outcomes |
| `STRESS_DECAY_RATE` | 0.05 | 0.0-0.5 | Natural stress recovery rate over time |
| `THRESHOLD_STRESS_THRESHOLD` | 0.3 | 0.0-0.5 | Minimum stress level to trigger coping |
| `THRESHOLD_AFFECT_THRESHOLD` | 0.3 | 0.0-0.5 | Minimum emotional change for resilience adjustment |

#### Detailed Parameter Descriptions

**Affect Dynamics Parameters:**

- **`AFFECT_PEER_INFLUENCE_RATE`**: Controls the strength of emotional contagion between agents. Higher values mean agents' emotions are more influenced by their social network, creating stronger emotional clustering. Lower values result in more independent emotional dynamics.

- **`AFFECT_EVENT_APPRAISAL_RATE`**: Determines how strongly stress events impact an agent's emotional state through cognitive appraisal. Challenge events tend to have positive emotional effects while hindrance events have negative effects. This parameter modulates the emotional sensitivity to life events.

- **`AFFECT_HOMEOSTATIC_RATE`**: Models emotional regulation and the tendency to return to baseline emotional states. Higher values create more emotionally stable agents who recover quickly from emotional extremes. Lower values allow emotions to persist longer.

- **`N_INFLUENCING_NEIGHBORS`**: Limits the scope of social influence by specifying how many network neighbors can affect an agent's emotional state. This creates more realistic local influence patterns rather than global network effects.

**Resilience Dynamics Parameters:**

- **`RESILIENCE_COPING_SUCCESS_RATE`**: Controls how much resilience improves when an agent successfully copes with stress. This represents learning and growth from successful stress management experiences.

- **`RESILIENCE_SOCIAL_SUPPORT_RATE`**: Determines the resilience benefit from receiving social support during difficult times. This models the protective effect of social relationships on mental resilience.

- **`RESILIENCE_OVERLOAD_THRESHOLD`**: Sets the minimum number of consecutive hindrance events needed to trigger overload effects. This represents the cumulative burden threshold before resilience begins to significantly decline.

- **`RESILIENCE_HOMEOSTATIC_RATE`**: Controls the tendency for resilience to return to baseline levels over time, modeling natural recovery processes.

- **`RESILIENCE_BOOST_RATE`**: Determines how quickly protective factors can improve resilience when resources are allocated to them.

- **`N_INFLUENCING_HINDRANCE`**: Specifies the number of consecutive hindrance events that amplify overload effects. This parameter controls how cumulative stress experiences compound to affect resilience.

**Coping Mechanism Parameters:**

- **`COPING_BASE_PROBABILITY`**: Base probability of successful coping before any situational modifiers are applied.

- **`COPING_SOCIAL_INFLUENCE`**: How much social connections and support affect coping success rates.

- **`COPING_CHALLENGE_BONUS`**: Additional success probability when coping with challenge-type stressors.

- **`COPING_HINDRANCE_PENALTY`**: Reduced success probability when coping with hindrance-type stressors.

**PSS-10 Integration Parameters:**

- **`PSS10_ITEM_MEAN`**: Mean values for the 10 PSS-10 questionnaire items, used to generate realistic stress scores.

- **`PSS10_ITEM_SD`**: Standard deviations for PSS-10 items, controlling the variability of stress responses.

- **`PSS10_LOAD_CONTROLLABILITY`**: Factor loadings determining how each PSS-10 item contributes to the controllability dimension.

- **`PSS10_LOAD_OVERLOAD`**: Factor loadings determining how each PSS-10 item contributes to the overload dimension.

- **`PSS10_CONTROLLABILITY_SD`** and **`PSS10_OVERLOAD_SD`**: Standard deviations for the latent factors in the bifactor model.

- **`PSS10_BIFACTOR_COR`**: Correlation between the controllability and overload factors in the bifactor model.

- **`PSS10_THRESHOLD`**: Score threshold above which an agent is considered to be experiencing significant stress.

#### Usage Scenarios

**Scenario 1: High Social Influence Environment**
```
AFFECT_PEER_INFLUENCE_RATE=0.3
N_INFLUENCING_NEIGHBORS=10
AFFECT_HOMEOSTATIC_RATE=0.02
```
*Use case:* Modeling workplace environments where emotional states spread rapidly through social networks, with persistent emotional effects.

**Scenario 2: Trauma Recovery Focus**
```
RESILIENCE_OVERLOAD_THRESHOLD=5
RESILIENCE_COPING_SUCCESS_RATE=0.2
RESILIENCE_SOCIAL_SUPPORT_RATE=0.15
PSS10_THRESHOLD=30
```
*Use case:* Studying recovery from traumatic events where agents need multiple successful coping experiences to build resilience.

**Scenario 3: Emotional Regulation Study**
```
AFFECT_EVENT_APPRAISAL_RATE=0.3
AFFECT_HOMEOSTATIC_RATE=0.1
RESILIENCE_COPING_SUCCESS_RATE=0.05
```
*Use case:* Investigating how different emotional regulation strategies affect mental health outcomes over time.

**Scenario 4: Social Isolation Effects**
```
N_INFLUENCING_NEIGHBORS=2
AFFECT_PEER_INFLUENCE_RATE=0.05
RESILIENCE_SOCIAL_SUPPORT_RATE=0.02
```
*Use case:* Modeling the impact of limited social connections on emotional well-being and resilience development.

**Scenario 5: PSS-10 Validation Study**
```
PSS10_THRESHOLD=27
PSS10_BIFACTOR_COR=0.5
STRESS_CONTROLLABILITY_MEAN=0.3
STRESS_OVERLOAD_MEAN=0.7
```
*Use case:* Validating the PSS-10 integration by simulating populations with different stress profiles and comparing against empirical benchmarks.

**Scenario 6: Network Adaptation Research**
```
NETWORK_ADAPTATION_THRESHOLD=2
NETWORK_REWIRE_PROBABILITY=0.05
NETWORK_HOMOPHILY_STRENGTH=0.8
RESILIENCE_SOCIAL_SUPPORT_RATE=0.12
```
*Use case:* Studying how agents adapt their social networks in response to stress and how this affects resilience outcomes.

### Resource Dynamics Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `PROTECTIVE_SOCIAL_SUPPORT` | 0.5 | 0.0-1.0 | Efficacy of social support in reducing distress |
| `PROTECTIVE_FAMILY_SUPPORT` | 0.5 | 0.0-1.0 | Efficacy of family support |
| `PROTECTIVE_FORMAL_INTERVENTION` | 0.5 | 0.0-1.0 | Efficacy of professional interventions |
| `PROTECTIVE_PSYCHOLOGICAL_CAPITAL` | 0.5 | 0.0-1.0 | Efficacy of personal psychological resources |
| `PROTECTIVE_IMPROVEMENT_RATE` | 0.5 | 0.0-1.0 | Rate at which protective factors improve |
| `RESOURCE_BASE_REGENERATION` | 0.05 | ≥0 | Daily resource regeneration rate |
| `RESOURCE_ALLOCATION_COST` | 0.15 | ≥0 | Base cost of allocating resources |
| `RESOURCE_COST_EXPONENT` | 1.5 | ≥1.0 | Convexity of resource allocation cost function |

### Mathematical Utility Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `UTILITY_SOFTMAX_TEMPERATURE` | 1.0 | >0 | Temperature for softmax decision making (lower = more deterministic) |

### Output and Logging Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `LOG_LEVEL` | 'INFO' | DEBUG, INFO, WARNING, ERROR, CRITICAL | Logging verbosity level |
| `OUTPUT_RESULTS_DIR` | 'data/processed' | Any valid path | Directory for processed simulation results |
| `OUTPUT_RAW_DIR` | 'data/raw' | Any valid path | Directory for raw simulation data |
| `OUTPUT_LOGS_DIR` | 'logs' | Any valid path | Directory for log files |
| `OUTPUT_SAVE_TIME_SERIES` | True | true/false | Whether to save detailed time series data |
| `OUTPUT_SAVE_NETWORK_SNAPSHOTS` | True | true/false | Whether to save network structure snapshots |
| `OUTPUT_SAVE_SUMMARY_STATISTICS` | True | true/false | Whether to save aggregated statistics |

## Usage Examples

### Scenario 1: Small-Scale Development Testing

```bash
# .env.development
SIMULATION_NUM_AGENTS=50
SIMULATION_MAX_DAYS=30
SIMULATION_SEED=12345
LOG_LEVEL=DEBUG
OUTPUT_SAVE_TIME_SERIES=true
PSS10_THRESHOLD=27
```

### Scenario 2: Large-Scale Parameter Sweep

```bash
# .env.production
SIMULATION_NUM_AGENTS=1000
SIMULATION_MAX_DAYS=365
SIMULATION_SEED=null
LOG_LEVEL=WARNING
OUTPUT_SAVE_TIME_SERIES=false
OUTPUT_SAVE_SUMMARY_STATISTICS=true
```

### Scenario 3: High-Stress Environment Simulation

```bash
# .env.high_stress
AGENT_STRESS_PROBABILITY=0.8
STRESS_OVERLOAD_MEAN=0.8
THRESHOLD_BASE_THRESHOLD=0.3
PROTECTIVE_FORMAL_INTERVENTION=0.8
PSS10_THRESHOLD=30
```

### Scenario 4: Social Network Focus

```bash
# .env.social_network
NETWORK_WATTS_K=12
NETWORK_WATTS_P=0.05
NETWORK_ADAPTATION_THRESHOLD=2
INTERACTION_INFLUENCE_RATE=0.15
INTERACTION_MAX_NEIGHBORS=20
```

### Scenario 5: PSS-10 Integration Testing

```bash
# .env.pss10_test
SIMULATION_NUM_AGENTS=100
SIMULATION_MAX_DAYS=50
PSS10_THRESHOLD=25
PSS10_BIFACTOR_COR=0.4
STRESS_CONTROLLABILITY_MEAN=0.4
STRESS_OVERLOAD_MEAN=0.6
OUTPUT_SAVE_TIME_SERIES=true
```

### Scenario 6: Resilience Intervention Study

```bash
# .env.intervention_study
SIMULATION_NUM_AGENTS=200
SIMULATION_MAX_DAYS=180
RESILIENCE_COPING_SUCCESS_RATE=0.15
RESILIENCE_SOCIAL_SUPPORT_RATE=0.12
PROTECTIVE_FORMAL_INTERVENTION=0.7
PROTECTIVE_IMPROVEMENT_RATE=0.3
NETWORK_ADAPTATION_THRESHOLD=3
```

## Advanced Usage

### Programmatic Configuration

```python
from src.python.config import Config, ConfigurationError

# Custom configuration with validation
try:
    config = Config('.env.custom')
    config.validate()

    # Access nested parameters
    simulation_params = config.get('simulation')
    network_params = config.get('network')

    # Modify and re-validate
    config.num_agents = 500
    config.validate()

except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Parameter Sweeps

```python
import os
from src.python.config import reload_config

# Example parameter sweep
stress_levels = [0.2, 0.5, 0.8]
results = []

for stress_level in stress_levels:
    # Update environment
    os.environ['AGENT_STRESS_PROBABILITY'] = str(stress_level)

    # Reload configuration
    config = reload_config()

    # Run simulation
    results.append(run_simulation(config))

# Analyze results
analyze_sweep_results(results, stress_levels)
```

### Configuration Validation

```python
from src.python.config import get_config

config = get_config()

# Manual validation
try:
    config.validate()
    print("✓ Configuration is valid")
except ConfigurationError as e:
    print(f"✗ Configuration error: {e}")

# Check specific parameter ranges
print(f"Network k={config.network_watts_k} (should be even and ≥2)")
print(f"All stress parameters in [0,1]: OK")
```

## Best Practices

### Security

1. **Never commit `.env` files** - Add `.env` to `.gitignore`
2. **Use strong seeds for production** - Avoid default seeds in production runs
3. **Validate sensitive parameters** - Check ranges for security-critical values
4. **Environment separation** - Use different `.env` files for different environments

### Parameter Management

1. **Start with `.env.example`** - Always base custom configs on the example file
2. **Document custom changes** - Comment why parameters differ from defaults
3. **Use consistent naming** - Follow the established parameter naming conventions
4. **Version control parameters** - Track important parameter sets with meaningful names

### Performance Optimization

1. **Scale parameters appropriately** - Large populations need different settings
2. **Output management** - Disable unnecessary outputs for large runs
3. **Memory considerations** - Monitor memory usage with large agent counts
4. **Parallel execution** - Use appropriate seeds for reproducible parallel runs

### Research Workflow

1. **Hypothesis-driven configuration** - Set parameters based on research hypotheses
2. **Sensitivity analysis** - Systematically vary key parameters including PSS-10 thresholds
3. **Validation against literature** - Use parameter ranges from published studies and PSS-10 benchmarks
4. **Reproducibility** - Use fixed seeds for important results
5. **PSS-10 calibration** - Validate stress measurements against empirical PSS-10 data
6. **Network adaptation studies** - Use network parameters to study social support dynamics

## Troubleshooting

### Common Issues

#### 1. Configuration Not Loading

**Problem:** Changes to `.env` file don't take effect.

**Solutions:**
```bash
# Check file exists and has correct permissions
ls -la .env

# Verify syntax (no spaces around =)
cat .env | grep "SIMULATION_NUM_AGENTS"

# Test loading in Python
from src.python.config import get_config
config = get_config()
print(config.num_agents)
```

#### 2. Type Conversion Errors

**Problem:** `ConfigurationError` about invalid parameter types.

**Solutions:**
- Ensure numeric parameters don't have quotes: `SIMULATION_NUM_AGENTS=100` (not `"100"`)
- Check boolean values: use `true`/`false`, not `yes`/`no`
- Verify float values use decimal points, not commas

#### 3. Validation Failures

**Problem:** `ConfigurationError` during validation.

**Solutions:**
- Check parameter ranges in this documentation
- Verify network parameters: `NETWORK_WATTS_K` should be even
- Ensure all probability parameters are in [0,1] range
- Check PSS-10 array parameters have exactly 10 values each
- Verify PSS-10 factor loadings are in [0,1] range
- Ensure PSS-10 standard deviations are positive

#### 4. Missing Dependencies

**Problem:** Import errors for `python-dotenv`.

**Solution:**
```bash
pip install python-dotenv
# or
conda install python-dotenv
```

#### 5. File Path Issues

**Problem:** Output directories not created or inaccessible.

**Solutions:**
- Use relative paths from project root
- Ensure write permissions for output directories
- Create directories manually if needed:
  ```bash
  mkdir -p data/processed data/raw logs
  ```

### Debugging Configuration

```python
import logging
from src.python.config import get_config

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Load with verbose output
config = get_config()

# Print full configuration
config.print_summary()

# Check specific values
print(f"Current working directory: {os.getcwd()}")
print(f"Environment file exists: {Path('.env').exists()}")
```

### Getting Help

1. **Check the example file** - `.env.example` has comprehensive documentation
2. **Review error messages** - Configuration errors include helpful details
3. **Validate step by step** - Test configuration loading before running simulations
4. **Compare with working examples** - Use known-good configurations as reference

## Integration with Research Pipeline

### Parameter Studies

```python
# Example: Systematic parameter variation
parameters_to_study = {
    'agent_stress_probability': [0.1, 0.3, 0.5, 0.7, 0.9],
    'protective_social_support': [0.2, 0.4, 0.6, 0.8],
    'network_watts_k': [4, 8, 12, 16],
    'pss10_threshold': [25, 27, 30],
    'resilience_coping_success_rate': [0.05, 0.1, 0.15, 0.2]
}

results = run_parameter_study(parameters_to_study)
```

### PSS-10 Sensitivity Analysis

```python
# Example: PSS-10 parameter sensitivity
pss10_parameters = {
    'pss10_threshold': [20, 25, 27, 30, 35],
    'pss10_bifactor_cor': [0.1, 0.3, 0.5, 0.7],
    'stress_controllability_mean': [0.2, 0.4, 0.6, 0.8],
    'stress_overload_mean': [0.2, 0.4, 0.6, 0.8]
}

# Run analysis with multiple replicates
for params in generate_parameter_combinations(pss10_parameters):
    for replicate in range(10):
        config = update_config_with_params(params)
        results = run_simulation(config)
        save_results(results, f"pss10_study_rep_{replicate}")
```

### Batch Processing

```bash
# Run multiple configurations
for config_file in .env.*; do
    if [[ "$config_file" != ".env.example" ]]; then
        echo "Running simulation with $config_file"
        python simulate.py --config "$config_file"
    fi
done
```

### Result Management

```python
# Automatic result organization
import os
from pathlib import Path

def organize_results(config, results):
    """Organize results based on configuration parameters."""
    # Create descriptive subdirectory names
    param_string = f"agents_{config.num_agents}_days_{config.max_days}_stress_{config.agent_stress_probability}"

    output_dir = Path(config.output_results_dir) / param_string
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results with metadata
    save_results(results, output_dir, config)
```

## Contributing

When adding new configuration parameters:

1. Add to `.env.example` with comprehensive documentation
2. Update `src/python/config.py` with type conversion and validation
3. Add parameter to appropriate category in this documentation
4. Update validation rules if needed (including PSS-10 array validation)
5. Test with various parameter values and edge cases
6. Update shell utilities if parameter extraction logic needs changes
7. Add integration tests for new parameter categories

## Support

For configuration-related issues:

1. Check this documentation first
2. Review `.env.example` for parameter details
3. Test with known-good configurations
4. Validate configuration loading independently
5. Check the troubleshooting section above

---

*This configuration system supports the Agent-Based Mental Health Simulation project's goal of evaluating cost-effectiveness of mental health promotion programs through comprehensive scenario modeling and sensitivity analysis.*

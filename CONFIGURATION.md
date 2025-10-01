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

### Stress Event Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `STRESS_CONTROLLABILITY_MEAN` | 0.5 | 0.0-1.0 | Mean controllability of stress events (Beta distribution) |
| `STRESS_PREDICTABILITY_MEAN` | 0.5 | 0.0-1.0 | Mean predictability of stress events |
| `STRESS_OVERLOAD_MEAN` | 0.5 | 0.0-1.0 | Mean overload intensity of stress events |
| `STRESS_MAGNITUDE_SCALE` | 0.4 | 0.0-2.0 | Scale parameter for event magnitude distribution |

### Appraisal and Threshold Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `APPRAISAL_OMEGA_C` | 1.0 | 0.0-5.0 | Weight for controllability in challenge/hindrance appraisal |
| `APPRAISAL_OMEGA_P` | 1.0 | 0.0-5.0 | Weight for predictability in appraisal |
| `APPRAISAL_OMEGA_O` | 1.0 | 0.0-5.0 | Weight for overload in appraisal |
| `APPRAISAL_BIAS` | 0.0 | -2.0 to 2.0 | Bias term in appraisal function |
| `APPRAISAL_GAMMA` | 6.0 | 1.0-20.0 | Sigmoid steepness for challenge/hindrance classification |
| `THRESHOLD_BASE_THRESHOLD` | 0.5 | 0.0-1.0 | Base stress threshold for becoming stressed |
| `THRESHOLD_CHALLENGE_SCALE` | 0.15 | 0.0-1.0 | How much challenge increases stress threshold |
| `THRESHOLD_HINDRANCE_SCALE` | 0.25 | 0.0-1.0 | How much hindrance decreases stress threshold |
| `STRESS_ALPHA_CHALLENGE` | 0.8 | 0.0-2.0 | Challenge stress multiplier |
| `STRESS_ALPHA_HINDRANCE` | 1.2 | 0.0-2.0 | Hindrance stress multiplier |
| `STRESS_DELTA` | 0.2 | 0.0-1.0 | Stress computation parameter |

### Social Interaction Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `INTERACTION_INFLUENCE_RATE` | 0.05 | 0.0-0.5 | Base rate of affect influence between agents |
| `INTERACTION_RESILIENCE_INFLUENCE` | 0.05 | 0.0-0.5 | How partner affect influences agent resilience |
| `INTERACTION_MAX_NEIGHBORS` | 10 | 1-50 | Maximum neighbors considered for interactions |

### Resource Dynamics Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `PROTECTIVE_SOCIAL_SUPPORT` | 0.5 | 0.0-1.0 | Efficacy of social support in reducing distress |
| `PROTECTIVE_FAMILY_SUPPORT` | 0.5 | 0.0-1.0 | Efficacy of family support |
| `PROTECTIVE_FORMAL_INTERVENTION` | 0.5 | 0.0-1.0 | Efficacy of professional interventions |
| `PROTECTIVE_PSYCHOLOGICAL_CAPITAL` | 0.5 | 0.0-1.0 | Efficacy of personal psychological resources |
| `RESOURCE_BASE_REGENERATION` | 0.05 | 0.0-0.5 | Daily resource regeneration rate |
| `RESOURCE_ALLOCATION_COST` | 0.15 | 0.0-1.0 | Base cost of allocating resources |
| `RESOURCE_COST_EXPONENT` | 1.5 | 1.0-5.0 | Convexity of resource allocation cost function |

### Mathematical Utility Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `UTILITY_SOFTMAX_TEMPERATURE` | 1.0 | 0.01-50.0 | Temperature for softmax decision making (lower = more deterministic) |

### Output and Logging Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `LOG_LEVEL` | INFO | DEBUG, INFO, WARNING, ERROR, CRITICAL | Logging verbosity level |
| `OUTPUT_RESULTS_DIR` | data/processed | Any valid path | Directory for processed simulation results |
| `OUTPUT_RAW_DIR` | data/raw | Any valid path | Directory for raw simulation data |
| `OUTPUT_LOGS_DIR` | logs | Any valid path | Directory for log files |
| `OUTPUT_SAVE_TIME_SERIES` | true | true/false | Whether to save detailed time series data |
| `OUTPUT_SAVE_NETWORK_SNAPSHOTS` | true | true/false | Whether to save network structure snapshots |
| `OUTPUT_SAVE_SUMMARY_STATISTICS` | true | true/false | Whether to save aggregated statistics |

## Usage Examples

### Scenario 1: Small-Scale Development Testing

```bash
# .env.development
SIMULATION_NUM_AGENTS=50
SIMULATION_MAX_DAYS=30
SIMULATION_SEED=12345
LOG_LEVEL=DEBUG
OUTPUT_SAVE_TIME_SERIES=true
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
STRESS_MAGNITUDE_SCALE=0.8
THRESHOLD_BASE_THRESHOLD=0.3
PROTECTIVE_FORMAL_INTERVENTION=0.8
```

### Scenario 4: Social Network Focus

```bash
# .env.social_network
NETWORK_WATTS_K=12
NETWORK_WATTS_P=0.05
INTERACTION_INFLUENCE_RATE=0.15
INTERACTION_MAX_NEIGHBORS=20
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
2. **Sensitivity analysis** - Systematically vary key parameters
3. **Validation against literature** - Use parameter ranges from published studies
4. **Reproducibility** - Use fixed seeds for important results

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
    'stress_probability': [0.1, 0.3, 0.5, 0.7, 0.9],
    'social_support': [0.2, 0.4, 0.6, 0.8],
    'network_density': [4, 8, 12, 16]
}

results = run_parameter_study(parameters_to_study)
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
4. Update validation rules if needed
5. Test with various parameter values

## Support

For configuration-related issues:

1. Check this documentation first
2. Review `.env.example` for parameter details
3. Test with known-good configurations
4. Validate configuration loading independently
5. Check the troubleshooting section above

---

*This configuration system supports the Agent-Based Mental Health Simulation project's goal of evaluating cost-effectiveness of mental health promotion programs through comprehensive scenario modeling and sensitivity analysis.*
# Agent-Based Model for Mental Health Promotion Cost-Effectiveness

## Project Rationale

This agent-based model evaluates the cost-effectiveness of workplace-based mental health promotion programs by simulating realistic social networks and stress dynamics. Mental health interventions vary significantly in quality and methodology, making comparative effectiveness research challenging. This model addresses this gap by implementing a comprehensive framework that simulates individual stress responses, social influence, and resilience dynamics to compare universal, selective, and indicated mental health promotion approaches. By modeling realistic social networks and empirically-grounded stress processes, the model provides evidence-based insights for policymakers and researchers to optimize mental health intervention strategies and resource allocation.

## Implemented Features

### Core Model Components

**Stress Perception System** - Implements challenge-hindrance appraisal framework where events are evaluated based on controllability and overload characteristics, with realistic stress threshold dynamics and coping probability calculations.

**Affect Dynamics** - Models emotional state changes through social influence, stress event impacts, and homeostatic regulation, with asymmetric effects where negative emotions have stronger influence than positive ones.

**Agent Interactions** - Simulates social networks using Watts-Strogatz small-world topology, enabling realistic patterns of emotional contagion, social support, and network adaptation based on stress experiences.

**Resilience Dynamics** - Tracks individual resilience capacity with multiple influencing factors including coping outcomes, social support, protective factors, and cumulative overload effects from consecutive hindrance events.

**Resource Management** - Implements conservation of resources theory with finite psychological/physical resources allocated across protective factors (social support, family support, formal interventions, psychological capital) using softmax decision-making.

**Stress Assessment Integration** - Complete PSS-10 (Perceived Stress Scale-10) implementation with bifactor model, dimension score generation, and empirical validation testing for research-grade stress measurement.

### Data Collection & Analysis

**Comprehensive Metrics** - 20+ population-level and 8+ agent-level variables tracked daily, including PSS-10 scores, resilience trajectories, affect dynamics, social support rates, and network statistics.

**Research-Ready Output** - Mesa DataCollector integration provides efficient, standardized data access with DataFrame outputs for statistical analysis and visualization.

## Quick Start

### 1. Clone/Fork the Repository
```bash
git clone <repository-url>
cd abm-simulation
```

### 2. Set Up Environment
```bash
# Copy environment configuration
cp .env.example .env

# Edit parameters as needed
nano .env  # or your preferred editor
```

### 3. Create Conda Environment
```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate ABM

# Verify installation
python -c "import mesa, networkx, numpy, pandas; print('All dependencies installed successfully')"
```

### 4. Run Simulation
```bash
# Run the simulation
python simulate.py

# Run with custom parameters
python simulate.py --days 100 --agents 500 --seed 42
```

### 5. Run Tests
```bash
# Run all tests
python -m pytest src/python/tests

# Run with coverage report
python -m pytest src/python/tests --cov=src/python --cov-report=html

# Run specific test file
python -m pytest src/python/tests/test_agent_integration.py -v
```

## Command-Line Arguments

The `simulate.py` script supports several command-line arguments for customizing simulation runs. Below is a table of available options:

| Argument | Description | Default |
|----------|-------------|---------|
| `-h, --help` | Show this help message and exit | N/A |
| `--days` | Number of simulation days | from config |
| `--agents` | Number of agents in simulation | from config |
| `--seed` | Random seed for reproducibility | from config |
| `--env` | Path to .env configuration file | .env |
| `--output-data` | Directory for raw data output | from config |
| `--output-fig` | Directory for figure output | from config |
| `--prefix` | Prefix for output files | none |

### Examples

```bash
# Run with custom simulation parameters
python simulate.py --days 500 --agents 1000 --seed 42

# Use a custom configuration file and output directories
python simulate.py --env custom.env --output-data ./results --prefix test_

# Specify figure output directory with prefix
python simulate.py --output-fig ./figures --prefix experiment1_
```

## Configuration

The model uses a comprehensive configuration system with 50+ parameters organized into logical groups:

- **Simulation Parameters**: Population size, duration, random seed
- **Network Parameters**: Watts-Strogatz topology, adaptation thresholds
- **Agent Parameters**: Initial states, coping rates, interaction frequencies
- **Stress Parameters**: Event generation, appraisal weights, threshold dynamics
- **PSS-10 Parameters**: Factor loadings, correlations, response distributions
- **Resource Parameters**: Regeneration rates, allocation costs, protective factors

All parameters are documented in `CONFIGURATION.md` with usage scenarios and research applications.

## Project Structure

```
abm-simulation/
├── src/python/           # Core ABM implementation
│   ├── agent.py         # Person agent with stress/social dynamics
│   ├── model.py         # Main simulation model with DataCollector
│   ├── stress_utils.py  # Stress event generation and processing
│   ├── affect_utils.py  # Social interaction and affect dynamics
│   ├── config.py        # Configuration management system
│   └── tests/           # Comprehensive test suite (30+ test files)
├── src/shell/           # Configuration management utilities
├── docs/               # Documentation and research outputs
│   ├── features/       # Detailed feature documentation
│   └── _source/        # Research protocols and manuscripts
├── data/               # Simulation results and analysis
└── simulate.py         # Main simulation runner
```

## Research Applications

### Individual-Level Analysis
- Track resilience trajectories and stress response patterns
- Identify at-risk individuals and intervention responders
- Analyze coping strategy effectiveness

### Population-Level Analysis
- Evaluate intervention impact across different scenarios
- Study emergent patterns of stress propagation and resilience
- Compare cost-effectiveness of different program approaches

### Network Analysis
- Examine social influence patterns and emotional contagion
- Study support network formation and adaptation
- Analyze clustering and homophily effects

## Output and Results

The model generates comprehensive outputs for research analysis:

- **Time Series Data**: Daily population and individual metrics
- **Network Snapshots**: Social connection patterns and evolution
- **Statistical Summaries**: Population distributions and trends
- **Individual Trajectories**: Complete agent state histories

## Dependencies

- **Python 3.12+** with Mesa 3.3.0 for agent-based modeling
- **NetworkX 3.5** for social network analysis
- **NumPy 2.3.3** and **Pandas 2.3.2** for data processing
- **R 4.5.1** for statistical analysis (optional)

## Citation

If you use this model in your research, please cite:

```bibtex
@software{abm_mental_health_2024,
  title={Agent-Based Model for Mental Health Promotion Cost-Effectiveness},
  author={Aly Lamuri},
  year={2025},
  url={https://github.com/namakala/abm-simulation}
}
```

## License

This project is licensed under the CC-BY-4.0 License - see the LICENSE file for details.

## Support and Contributing

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Contributing**: See CONTRIBUTING.md for development guidelines

For research collaborations or technical questions, please open an issue or discussion on GitHub.
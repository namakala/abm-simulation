# System Architecture

## Core Components

### Agent Structure
Each agent represents an individual with the following state variables:
- **Resources** `R ∈ [0,1]` - finite psychological/physical resources
- **Distress** `D ∈ [0,1]` - current distress level  
- **Stress Threshold** `T_stress` - baseline threshold for becoming stressed
- **Network Position** - connections to other agents
- **Protective Factors** - social support, family support, formal interventions, psychological capital

### Event Processing Pipeline
1. **Event Generation** - Poisson process generates life events with:
   - Controllability `c ∈ [0,1]`
   - Predictability `p ∈ [0,1]`
   - Overload `o ∈ [0,1]`

2. **Appraisal Mechanism** - Challenge/Hindrance mapping:
   ```
   z = ωc*c + ωp*p - ωo*o + b
   challenge = σ(γ*z)  [sigmoid function]
   hindrance = 1 - challenge
   ```

3. **Threshold Evaluation** - Agent becomes stressed if:
   ```
   L > T_stress + λC*challenge - λH*hindrance
   ```

## Source Code Organization

### [`src/python/`](src/python/) - Core ABM Implementation
- `agent.py` - Person class with stress events, social interactions, and resource dynamics
- `model.py` - StressModel class using Mesa framework with NetworkGrid and comprehensive DataCollector integration
- `stress_utils.py` - Stress event generation and appraisal processing utilities
- `affect_utils.py` - Social interaction and resilience computation utilities
- `math_utils.py` - Mathematical utilities for random sampling and clamping
- `config.py` - Centralized configuration management with environment variable loading and validation
- `simulate.py` - Main simulation runner (in project root)
- `debug/` - Debugging utilities for threshold evaluation (`debug_threshold.py`) and stress processing troubleshooting (`simple_threshold_test.py`)
- **PSS-10 Integration** - Complete Perceived Stress Scale-10 implementation with bifactor model, dimension score generation, and empirical validation testing
- **DataCollector Integration** - Comprehensive Mesa DataCollector implementation with 20+ model-level metrics and 8+ agent-level variables for research analysis
- `test_*.py` - Comprehensive test suite with 22 specialized test files:
  - Unit tests: `test_math_utils.py`, `test_stress_utils.py`, `test_affect_utils.py`
  - Integration tests: `test_agent_integration.py`, `test_model_integration.py`, `test_affect_resilience_integration.py`, `test_agent_pss10_integration.py`, `test_datacollector_integration_comprehensive.py`
  - Configuration tests: `test_config_integration.py`, `test_dataclass_config.py`, `test_environment_variable_validation.py`, `test_pss10_config.py`
  - Mechanism tests: `test_stress_processing_mechanisms.py`, `test_affect_resilience_dynamics.py`, `test_homeostatic_adjustment.py`, `test_homeostatic_stabilization_integration.py`
  - Validation tests: `test_imports_and_dependencies.py`, `test_integration_new_mechanisms.py`, `test_new_mechanisms.py`
  - PSS-10 tests: `test_pss10_comprehensive.py`, `test_pss10_empirical.py`, `test_generate_pss10_dimension_scores.py`

### [`src/shell/`](src/shell/) - Configuration Management Utilities
- `extract_env.sh` - Shell script for extracting default parameter values from configuration files
- `update_env_example.sh` - Shell script to update `.env.example` value based on `.env`
- Configuration extraction and management scripts for research workflows

### Development Infrastructure
- **CI/CD Pipeline** - GitHub Actions workflow for automated testing and coverage reporting
- **Coverage Reporting** - Automated test coverage tracking with 80% minimum threshold and HTML report generation (`.coverage`, `coverage.xml` files)
- **Test Results** - Automated test result reporting with XML output (`test-results.xml`) for CI/CD pipeline integration
- **Code Quality** - Codecov integration for coverage analysis and reporting service
- **Environment Management** - Conda-based environment with automated setup and caching
- **Configuration Management** - Comprehensive `.env`-based parameter system with validation and documentation
- **Shell Utilities** - Configuration management scripts (`extract_env.sh`, `update_env_example.sh`) for automated parameter extraction and environment file synchronization

### [`docs/features/`](docs/features/) - Feature Documentation
- `affect-dynamics.md` - Emotional state changes and social influence mechanisms
- `agent-interactions.md` - Social network effects and support systems
- `integration-overview.md` - System integration and cross-mechanism interactions
- `resilience-dynamics.md` - Coping responses and resilience mechanisms
- `resource-management.md` - Protective factors and resource allocation
- `step-calculations.md` - Detailed step-by-step calculation procedures
- `stress-perception.md` - Event appraisal and challenge/hindrance determination
- `data-collection.md` - Comprehensive data collection system with 28 variable definitions and research applications

### [`src/r/`](src/r/) - Analysis and Visualization
- `analysis/` - Statistical analysis and sensitivity analysis (planned)
- `visualization/` - Plot generation and dashboard creation (planned)
- `calibration/` - ABC/SMM parameter calibration routines (planned)

### [`src/sql/`](src/sql/) - Data Management
- `schema.sql` - Database schema for parameter sweeps (planned)
- `queries/` - Standard queries for result extraction (planned)
- `procedures/` - Stored procedures for aggregation (planned)

## Data Flow

```mermaid
graph TD
    A[Parameter Sampling] --> B[Simulation Runs]
    B --> C[Raw Results]
    C --> D[Statistical Analysis]
    D --> E[Validation Against Patterns]
    E --> F[Sensitivity Analysis]
    F --> G[Publication Outputs]
    
    H[Literature Targets] --> E
    I[Empirical Data] --> E
```

## Key Design Patterns

### Resource Allocation Model
Agents make softmax decisions about allocating resources across protective factors:
- **Social Support** - efficacy `αsoc`, replenishment `ρsoc`
- **Family Support** - efficacy `αfam`, replenishment `ρfam`  
- **Formal Interventions** - efficacy `αint`, replenishment `ρint`
- **Psychological Capital** - efficacy `αcap`, replenishment `ρcap`

### Network Adaptation
Agents adapt connections based on:
- **Rewiring probability** `p_rewire` when stress threshold breached repeatedly
- **Support effectiveness** - successful help requests strengthen ties
- **Homophily** - similar stress levels attract connections

### Parameter Space Management
- **Latin Hypercube Sampling** for initial parameter exploration
- **Sobol indices** for global sensitivity analysis
- **PRCC** for monotonic parameter-output relationships

## Critical Implementation Paths

### Simulation Engine
1. Initialize agent population with network structure
2. For each time step:
   - Generate events for subset of agents
   - Process appraisals and stress responses
   - Update resource levels and protective factor allocation
   - Apply network adaptation rules
   - Record state variables and metrics

### Validation Pipeline  
1. Define pattern targets from literature review
2. Run parameter sweeps with multiple replicates
3. Compute distance metrics between simulated and target patterns
4. Use ABC/SMM to identify plausible parameter regions
5. Validate top parameter sets across different scenarios

### Output Generation
- **Time series** of agent states and population metrics
- **Network snapshots** at key time points  
- **Resilience metrics** - recovery time, basin stability, FTLE
- **Cost-effectiveness ratios** for intervention scenarios
# Technology Stack

## Core Technologies

### R Environment (Analysis & Visualization)
- **R Version:** 4.4.2 (configured via renv)
- **Package Management:** renv for reproducible environments
- **Repository:** CRAN via Posit Package Manager
- **Key Libraries:** (to be configured)
  - Statistical analysis: tidyverse, data.table
  - Network analysis: igraph, network
  - Visualization: ggplot2, plotly, shiny
  - Parameter calibration: abc, sensitivity

### Python Environment (Core ABM Implementation)
- **Primary Language:** Python 3.x (implemented)
- **Framework:** Mesa for agent-based modeling
- **Network Library:** NetworkX for social network operations
- **Configuration Management:** python-dotenv for environment variable loading and validation
- **Current Libraries:**
  - Agent-based modeling: mesa (implemented)
  - Network analysis: networkx (implemented)
  - Scientific computing: numpy (implemented)
  - Data manipulation: pandas (implemented)
  - Configuration management: python-dotenv (implemented)
  - Testing: pytest (implemented)
  - Coverage reporting: coverage.py (implemented)
  - Debug utilities: Custom debugging tools for threshold evaluation and stress processing
  - PSS-10 integration: Complete Perceived Stress Scale-10 implementation with bifactor model
- **Generated Files:**
  - `.coverage` - Coverage data file
  - `coverage.xml` - Coverage XML report
  - `test-results.xml` - Test results XML

### Shell Environment (Configuration Management)
- **Shell Scripts:** POSIX-compliant shell utilities for configuration management
- **extract_env.sh:** Automated extraction of default parameter values from Python configuration files
- **update_env_example.sh:** Automated syncing between `.env` and `.env.example`
- **Configuration Utilities:** Shell-based tools for research workflow automation and parameter management

### SQL Database (Data Management)
- **Purpose:** Large-scale parameter sweep storage
- **Schema:** Parameter configurations, simulation results, aggregated metrics
- **Usage:** Batch processing, result queries, performance optimization

## Development Environment

### R Configuration
```r
# .Rprofile automatically sources renv/activate.R
# renv.lock specifies:
# - R 4.4.2
# - CRAN repository via packagemanager.posit.co
# - renv 1.0.7 for environment management
```

### Project Structure
```
/data/
├── raw/          # Original data sources
└── processed/    # Cleaned, analysis-ready data

/docs/
├── _source/      # Source documentation (protocols, manuscripts)
├── _author.yml   # Author metadata for documentation
├── _header.tex   # LaTeX header configuration
├── _knitr-bib.yml # Knitr bibliography configuration
├── article.qmd   # Quarto manuscript source
├── article.tex   # Compiled LaTeX manuscript
├── features/     # Feature documentation
├── results/      # Generated outputs, figures, tables
└── ref.bib       # Bibliography (empty, ready for references)

/src/
├── python/       # Core ABM implementation
│   ├── agent.py, model.py, stress_utils.py, affect_utils.py, math_utils.py, resource_utils.py, config.py, simulate.py
│   ├── debug/     # Debugging utilities
│   ├── demos/     # Demonstration scripts
│   └── tests/     # Comprehensive test suite with 35+ specialized test files
├── shell/        # Configuration management utilities
├── r/           # Analysis and visualization
└── sql/         # Database schemas and queries

/renv/           # R package library and configuration
/render/         # Build outputs directory
```

## Technical Constraints

### Performance Requirements
- Support for large-scale parameter sweeps (1000+ parameter sets)
- Multiple stochastic replicates per parameter configuration
- Network simulations with 100-10,000 agents
- Time series data across simulation periods

### Reproducibility Standards
- Version-controlled parameter configurations
- Deterministic random number generation with seeds
- Complete dependency specification via renv
- Documented calibration and validation procedures

### Integration Points
- Python → SQL: Simulation results storage
- SQL → R: Data extraction for analysis
- R → Docs: Automated report generation
- Cross-language parameter sharing via JSON/CSV

## Development Workflow

### Setup Process
1. R environment activation via renv
2. Python virtual environment creation
3. Database initialization with schema
4. Parameter configuration validation

### Execution Pipeline
1. **Python:** Run ABM simulations, store results in SQL (planned)
2. **SQL:** Aggregate and query simulation data (planned)
3. **R:** Statistical analysis, visualization, reporting (planned)
4. **Integration:** Cross-validate results across languages (planned)

## Tool Usage Patterns

### Version Control
- Git for source code management
- renv.lock for R dependency tracking
- requirements.txt for Python dependencies (to be created)

### Documentation
- Markdown for technical documentation
- R Markdown for analysis reports
- LaTeX/Word for manuscript preparation
- BibTeX for reference management

### Quality Assurance
- **Unit testing frameworks** for each language with comprehensive coverage
- **Integration testing** for component interactions
- **Configuration validation** with type checking and range validation
- **Parameter validation routines** with comprehensive error handling
- **CI/CD Pipeline** - GitHub Actions workflow (`coverage-test.yml`) for automated testing and coverage reporting
- **Coverage Reporting** - Automated test coverage tracking with 80% minimum threshold and HTML report generation (`.coverage`, `coverage.xml` files)
- **Codecov Integration** - External coverage analysis and reporting service
- **Test Results Integration** - Automated test result reporting with XML output (`test-results.xml`) for CI/CD pipeline integration
- **Debug Tools** - Threshold evaluation and stress processing debugging utilities for troubleshooting
- **Comprehensive Testing Suite** - 30 specialized test files covering unit tests, integration tests, configuration validation, environment variable validation, dataclass validation, dependency testing, stress processing mechanisms, affect dynamics, resilience mechanisms, homeostatic adjustment, new mechanism validation, daily reset functionality, interaction tracking, and comprehensive PSS-10 testing
- **Sensitivity analysis** for robustness checks (planned)
- **Pattern matching** for model validation (planned)
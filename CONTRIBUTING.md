# Contributing to the Mental Health ABM Project

Thank you for your interest in contributing to this agent-based model for mental health promotion cost-effectiveness research! This document provides guidelines for contributing to the project.

## Development Workflow

### 1. Clone/Fork the Repository

Start by forking the repository to your GitHub account:

```bash
# Clone your fork locally
git clone https://github.com/yourusername/abm-simulation.git
cd abm-simulation

# Add the original repository as upstream
git remote add upstream https://github.com/originalusername/abm-simulation.git
```

### 2. Create an Issue for Features/Bugs

**For all contributions, start by creating a GitHub issue:**

#### Feature Requests
- Create a new issue with the label `enhancement` or type `feature`
- Clearly describe the proposed feature and its rationale
- Explain how it fits into the existing model architecture
- If you'd like to implement it yourself, comment on the issue with your implementation plan

#### Bug Reports
- Create a new issue with the label or type `bug`
- Include detailed reproduction steps
- Provide expected vs actual behavior
- Include relevant configuration parameters and error messages

#### Implementation Plans
When contributing code:
- Reference the existing issue in your pull request
- Explain your implementation approach
- Note any breaking changes or dependencies

## Configuration Management

### Adding New .env Parameters

The project uses a centralized configuration system. **Never edit `.env` or `.env.example` directly.**

#### Step-by-Step Process:

1. **Modify `src/python/config.py`** - This is the single source of truth for all parameters:
   ```python
   # Add new parameter to appropriate section
   'new_section': {
       'new_parameter': 0.5,  # Add default value and documentation
       'parameter_description': 'Description of what this parameter does'
   }
   ```

2. **Run the extraction script** to update `.env`:
   ```bash
   # Extract current parameter values from config.py
   bash src/shell/extract_env.sh
   ```

3. **Update `.env.example`** with the new parameter:
   ```bash
   # Update .env.example with new parameter and description
   bash src/shell/update_env_example.sh
   ```

4. **Test your changes**:
   ```bash
   # Verify the configuration loads correctly
   python -c "from src.python.config import get_config; print(get_config().get('new_section', 'new_parameter'))"

   # Run tests to ensure no regressions
   python -m pytest src/python/tests/test_config_integration.py -v
   ```

## Pull Request Process

### 1. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-number-description
```

### 2. Make Your Changes

- Follow the existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

```bash
# Stage your changes
git add .

# Write a clear commit message
git commit -m "Add: brief description of changes

- Detailed explanation of what was changed
- Why the change was made
- Any breaking changes or migration notes
- References to related issues"

# Example for a new feature
git commit -m "Add: social influence parameter for coping probability

- Added SOCIAL_INFLUENCE_FACTOR parameter to config.py
- Modified compute_coping_probability() in affect_utils.py
- Neighbor affect now influences individual coping success rates
- No breaking changes - uses default value of 0.3
- Closes #123"
```

### 4. Push and Create Pull Request

```bash
# Push your branch to GitHub
git push origin feature/your-feature-name

# Create a pull request through GitHub interface
```

### 5. Pull Request Requirements

**All pull requests must include:**

#### Detailed Description
- **What**: Clear explanation of what the change does
- **Why**: Rationale for the change and problem it solves
- **How**: Technical implementation details
- **Testing**: How the change was tested

#### Breaking Changes
If your PR includes breaking changes:
- Clearly state what breaks and why
- Provide migration instructions
- Update version numbers if needed

#### Configuration Changes
If your PR adds new parameters:
- Confirm you've followed the configuration management process above
- Include the updated `.env.example` in your PR
- Document the new parameters in `CONFIGURATION.md`

## Code Standards

### Python Code Style
- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Maximum line length: 100 characters
- Use descriptive variable and function names

### Testing Requirements

**All pull requests must include comprehensive tests. New features will not be accepted without proper test coverage.**

#### For New Features:
- **Unit tests** for every new function and class (required)
- **Integration tests** for cross-module interactions (required)
- **Test coverage** must meet or exceed 90% for modified files
- **Edge case testing** for boundary conditions and error scenarios
- **Performance benchmarks** for computationally intensive features

#### For Bug Fixes:
- **Regression tests** to prevent the bug from reoccurring (required)
- **Unit tests** for the specific fix (required)
- **Integration tests** if the fix affects multiple components (required)

#### For Configuration Changes:
- **Parameter validation tests** for new configuration options (required)
- **Type checking tests** for configuration value conversion (required)
- **Boundary tests** for parameter range validation (required)

**Test Coverage Requirements:**
- Minimum overall coverage: 85%
- Minimum coverage for new/modified files: 90%
- All new utility functions must have 100% coverage
- Use `python -m pytest --cov=src/python --cov-report=html --cov-fail-under=85` to verify

### Documentation
- Update docstrings for modified functions
- Add feature documentation in `docs/features/` for significant changes
- Update `CONFIGURATION.md` for new parameters

## Development Environment

### Setting Up for Development

```bash
# Create development environment
conda env create -f environment.yml
conda activate ABM

# Install development dependencies
pip install pre-commit black isort mypy

# Set up pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

### Running Tests During Development

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest src/python/tests/test_*.py -k "integration"  # Integration tests only
python -m pytest src/python/tests/test_*.py -k "config"      # Configuration tests only

# Run with coverage (required for PRs)
python -m pytest --cov=src/python --cov-report=html --cov-fail-under=85

# Check specific file coverage (for new features)
python -m pytest --cov=src.python.new_module --cov-report=html --cov-fail-under=90

# Run performance benchmarks
python -m pytest src/python/tests/ -k "benchmark" --benchmark-only

# Run only new tests you've added
python -m pytest src/python/tests/test_your_new_feature.py -v
```

## Review Process

### What Reviewers Look For

1. **Code Quality**: Following style guidelines, clear logic, proper error handling
2. **Testing**: **Comprehensive test coverage (85%+ required), meaningful test cases, edge case coverage**
3. **Documentation**: Updated docstrings, parameter documentation
4. **Integration**: No breaking changes, proper configuration management
5. **Performance**: No significant performance regressions

**PRs without adequate testing will be rejected.** All new features must include:
- Unit tests for every new function
- Integration tests for cross-module changes
- Test coverage verification with `--cov-fail-under=85`
- Edge case and error condition testing

### Responding to Review Feedback

- Address all reviewer comments
- Update your PR with requested changes
- Re-request review when ready
- Be open to alternative approaches

## Release Process

### Version Management
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `src/python/config.py`
- Tag releases with `vX.Y.Z` format

### Release Checklist
- [ ] All tests pass with 85%+ coverage
- [ ] New features include comprehensive unit and integration tests
- [ ] Documentation updated
- [ ] Configuration files updated
- [ ] Breaking changes documented
- [ ] Performance benchmarks pass
- [ ] Test coverage report generated and reviewed

## Getting Help

### Resources
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use for questions and ideas
- **Documentation**: Check `docs/` folder for detailed information
- **Configuration**: See `CONFIGURATION.md` for parameter details

### Asking Questions
When asking for help:
- Reference specific files and line numbers when possible
- Include relevant configuration parameters
- Describe expected vs actual behavior
- Mention what you've already tried

## Acknowledgments

Thank you for contributing to this research project! Your contributions help improve mental health promotion strategies and support evidence-based policymaking.

---

*This contributing guide is adapted from best practices for academic software development and research reproducibility.*
# Agent Baseline Initialization Processes

_See [`.kilocode/rules/math/notation.md`](../../.kilocode/rules/math/notation.md) for symbol definitions and conventions._

## Overview

Agent baseline initialization establishes the starting state for each individual in the simulation, setting initial values for resilience, affect, resources, protective factors, PSS-10 dimensions, and stress levels. This process uses configurable parameters and random sampling to create realistic population variation while ensuring reproducibility through seeded random number generation.

## Agent State Variables

Each agent represents an individual with the following baseline state variables:

- **Baseline Resilience** $R_{\text{0}} \in [0,1]$: Natural equilibrium point for resilience capacity
- **Baseline Affect** $A_{\text{0}} \in [-1,1]$: Natural equilibrium point for emotional state
- **Initial Resources** $R \in [0,1]$: Available psychological and physical resources
- **Protective Factors** $\mathbf{e} \in [0,1]^4$: Efficacy levels for social support, family support, formal interventions, and psychological capital
- **PSS-10 Dimensions** $c_\Psi, o_\Psi \in [0,1]$: Controllability and overload stress dimensions
- **Initial Stress Level** $S \in [0,1]$: Starting stress level derived from PSS-10

## Initialization Process

### Step 1: Configuration Loading

The initialization process begins by loading parameters from the unified configuration system.

**Implementation**: [`__init__()`](src/python/agent.py:68) in `agent.py`

### Step 2: Random Number Generator Setup

A reproducible random number generator is created for each agent.

**Implementation**: [`create_rng()`](src/python/math_utils.py:45) in `math_utils.py`

### Step 3: Baseline Resilience Initialization

Baseline resilience is initialized using a sigmoid transformation to ensure values remain in [0,1]:

$$R_{\text{0}} = \sigma\left(\frac{X - \mu_{R,\text{init}}}{\sigma_{R,\text{init}}}\right)$$

Where:
- $X \sim \mathcal{N}(\mu_{R,\text{init}}, \sigma_{R,\text{init}}^2)$ is a normal random variable
- $\sigma(x) = \frac{1}{1+e^{-x}}$ is the sigmoid function
- $\mu_{R,\text{init}} = 0.5$ (default mean)
- $\sigma_{R,\text{init}} = 0.2$ (default standard deviation)

**Implementation**: [`sigmoid_transform()`](src/python/math_utils.py:78) in `math_utils.py`

### Step 4: Baseline Affect Initialization

Baseline affect is initialized using a tanh transformation to ensure values remain in [-1,1]:

$$A_{\text{0}} = \tanh\left(\frac{X - \mu_{A,\text{init}}}{\sigma_{A,\text{init}}}\right)$$

Where:
- $X \sim \mathcal{N}(\mu_{A,\text{init}}, \sigma_{A,\text{init}}^2)$ is a normal random variable
- $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ is the hyperbolic tangent function
- $\mu_{A,\text{init}} = 0.0$ (default mean)
- $\sigma_{A,\text{init}} = 0.5$ (default standard deviation)

**Implementation**: [`tanh_transform()`](src/python/math_utils.py:95) in `math_utils.py`

### Step 5: Resources Initialization

Initial resources are initialized using the same sigmoid transformation as resilience:

$$R = \sigma\left(\frac{X - \mu_{\text{Res,init}}}{\sigma_{\text{Res,init}}}\right)$$

Where:
- $X \sim \mathcal{N}(\mu_{\text{Res,init}}, \sigma_{\text{Res,init}}^2)$
- $\mu_{\text{Res,init}} = 0.6$ (default mean)
- $\sigma_{\text{Res,init}} = 0.2$ (default standard deviation)

### Step 6: Protective Factors Initialization

Protective factors are initialized to neutral values (0.5) representing average efficacy levels:

$$\mathbf{e} = [0.5, 0.5, 0.5, 0.5]$$

For:
- Social support
- Family support
- Formal interventions
- Psychological capital

### Step 7: PSS-10 Initialization

PSS-10 state is initialized through a comprehensive process:

1. **Dimension Score Generation**:
   $$c_\Psi, o_\Psi \sim \mathcal{N}\left(\begin{bmatrix} \mu_c \\ \mu_o \end{bmatrix}, \begin{bmatrix} \sigma_c^2 & \rho_\Psi \sigma_c \sigma_o \\ \rho_\Psi \sigma_c \sigma_o & \sigma_o^2 \end{bmatrix}\right)$$

2. **Item Response Generation**:
   $$\Psi_i = \mathrm{clamp}\left(\mathrm{round}\left(\mu_{\Psi,i} + (\lambda_{c,\Psi,i}(1-c_\Psi) + \lambda_{o,\Psi,i} o_\Psi - 0.5) \cdot 0.5 + \epsilon\right), 0, 4\right)$$

3. **Total Score Calculation**:
   $$\Psi = \sum_{i=1}^{10} \begin{cases} \Psi_i & \text{if } i \notin \{4,5,7,8\} \\ 4 - \Psi_i & \text{if } i \in \{4,5,7,8\} \end{cases}$$

**Implementation**: [`_initialize_pss10_scores()`](src/python/agent.py:168) in `agent.py`

### Step 8: Stress Level Initialization

Initial stress level is computed from PSS-10 score:

$$S = \frac{\Psi - \Psi_{\min}}{\Psi_{\max} - \Psi_{\min}} \cdot (1 - c_\Psi) + \frac{\Psi - \Psi_{\min}}{\Psi_{\max} - \Psi_{\min}} \cdot o_\Psi$$

Where:
- $\Psi_{\min} = 0$, $\Psi_{\max} = 40$
- $c_\Psi, o_\Psi$ are PSS-10 dimension scores

**Implementation**: [`_initialize_stress_from_pss10()`](src/python/agent.py:720) in `agent.py`

## Population Variation

The initialization process creates realistic population variation through:

1. **Individual Differences**: Each agent samples from normal distributions with configurable means and standard deviations
2. **Reproducible Randomness**: Seeded random number generation ensures reproducible simulations
3. **Parameter Sensitivity**: All parameters are configurable through environment variables for sensitivity analysis

## Integration with Configuration System

All initialization parameters are managed through the unified configuration system, allowing for:

- **Parameter Sweeps**: Systematic variation of initialization parameters
- **Scenario Analysis**: Different baseline conditions for different research scenarios
- **Validation**: Alignment with empirical population distributions

## Validation and Testing

The initialization process is validated through:

1. **Range Validation**: Ensuring all values fall within expected ranges
2. **Distribution Testing**: Verifying that sampled values match intended distributions
3. **Integration Testing**: Confirming proper interaction with other systems
4. **Reproducibility Testing**: Ensuring consistent results with same seeds

**Implementation**: [`test_agent_initialization.py`](src/python/tests/test_agent_initialization.py) in `tests/`

This initialization framework ensures that agents start with realistic and configurable baseline states that support comprehensive mental health simulation and research validation.
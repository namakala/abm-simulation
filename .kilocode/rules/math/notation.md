# Mathematical Notation Reference

_See [`notation.md`](notation.md) for symbol definitions and conventions._

## Overview

This document provides the authoritative reference for all mathematical notation used in the agent-based mental health model. All feature documentation must use symbols exactly as defined here and include a reference to this file.

## General Mathematical Symbols

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\mathbb{R}$ | Real numbers | $\mathbb{R}$ | - |
| $\mathbb{N}$ | Natural numbers | $\mathbb{N}$ | - |
| $\in$ | Element of | $x \in [0,1]$ | - |
| $\sim$ | Distributed as | $x \sim \mathcal{N}(0,1)$ | - |
| $\propto$ | Proportional to | $y \propto x$ | - |
| $\sum$ | Summation | $\sum_{i=1}^n x_i$ | - |
| $\prod$ | Product | $\prod_{i=1}^n x_i$ | - |
| $\int$ | Integral | $\int_0^1 f(x) dx$ | - |
| $\frac{d}{dx}$ | Derivative | $\frac{d}{dx} f(x)$ | - |
| $\partial$ | Partial derivative | $\partial f / \partial x$ | - |
| $\nabla$ | Gradient | $\nabla f$ | - |
| $\Delta$ | Change/difference | $\Delta x = x_2 - x_1$ | - |
| $\approx$ | Approximately equal | $x \approx 0.5$ | - |
| $\equiv$ | Identical to | $f(x) \equiv g(x)$ | - |
| $\Rightarrow$ | Implies | $x > 0 \Rightarrow y > 0$ | - |
| $\Leftrightarrow$ | If and only if | $x > 0 \Leftrightarrow y > 0$ | - |
| $\forall$ | For all | $\forall x \in \mathbb{R}$ | - |
| $\exists$ | There exists | $\exists x > 0$ | - |
| $\therefore$ | Therefore | $x > 0 \therefore y > 0$ | - |
| $\because$ | Because | $x > 0 \because y > 0$ | - |
| $\emptyset$ | Empty set | $\emptyset$ | - |
| $\subseteq$ | Subset of | $A \subseteq B$ | - |
| $\cap$ | Intersection | $A \cap B$ | - |
| $\cup$ | Union | $A \cup B$ | - |
| $\setminus$ | Set difference | $A \setminus B$ | - |
| $\times$ | Cartesian product | $A \times B$ | - |
| $\otimes$ | Tensor product | $A \otimes B$ | - |
| $\oplus$ | Direct sum | $A \oplus B$ | - |

## Probability and Statistics

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\mathbb{E}[X]$ | Expected value | $\mathbb{E}[X] = \sum x p(x)$ | - |
| $\mathbb{V}(X)$ | Variance | $\mathbb{V}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$ | - |
| $\sigma^2$ | Variance | $\sigma^2 = \mathbb{V}(X)$ | - |
| $\sigma$ | Standard deviation | $\sigma = \sqrt{\mathbb{V}(X)}$ | - |
| $\rho$ | Correlation coefficient | $\rho \in [-1,1]$ | - |
| $\mathcal{N}(\mu,\sigma^2)$ | Normal distribution | $X \sim \mathcal{N}(0,1)$ | - |
| $\mathcal{U}(a,b)$ | Uniform distribution | $X \sim \mathcal{U}(0,1)$ | - |
| $\mathcal{B}(\alpha,\beta)$ | Beta distribution | $X \sim \mathcal{B}(2,2)$ | - |
| $\mathcal{P}(\lambda)$ | Poisson distribution | $X \sim \mathcal{P}(5)$ | - |
| $\sigma(x)$ | Sigmoid function | $\sigma(x) = \frac{1}{1+e^{-x}}$ | - |
| $\mathrm{softmax}(\mathbf{x})$ | Softmax function | $\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | - |
| $\mathrm{clamp}(x,a,b)$ | Clamping function | $\mathrm{clamp}(x,0,1) = \max(0,\min(1,x))$ | - |

## Core Model Variables

### Agent State Variables

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $R$ | Resources | $R \in [0,1]$ | - |
| $D$ | Distress | $D \in [0,1]$ | - |
| $\eta_{\text{stress}}$ | Stress threshold | $\eta_{\text{stress}} \in [0,1]$ | - |
| $A$ | Affect | $A \in [-1,1]$ | - |
| $A_{\text{baseline}}$ | Baseline affect | $A_{\text{baseline}} \in [-1,1]$ | - |
| $R_{\text{baseline}}$ | Baseline resilience | $R_{\text{baseline}} \in [0,1]$ | - |
| $L$ | Appraised stress load | $L \in [0,1]$ | - |
| $S$ | Current stress level | $S \in [0,1]$ | - |

### Stress Processing Variables

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $c$ | Controllability | $c \in [0,1]$ | - |
| $o$ | Overload | $o \in [0,1]$ | - |
| $z$ | Weighted combination | $z = \omega_c c - \omega_o o + b$ | - |
| $\chi$ | Challenge component | $\chi \in [0,1]$ | - |
| $\zeta$ | Hindrance component | $\zeta \in [0,1]$ | - |
| $s$ | Event magnitude | $s \in [0,1]$ | - |
| $\eta$ | Threshold (general) | $\eta \in [0,1]$ | - |
| $\eta_{\mathrm{eff}}$ | Effective threshold | $\eta_{\mathrm{eff}} = \eta_{\text{stress}} + \chi \cdot \eta_{\chi} - \zeta \cdot \eta_{\zeta}$ | - |

### PSS-10 Assessment Variables

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\Psi$ | Total PSS-10 score | $\Psi \in [0,40]$ | - |
| $c_\Psi$ | PSS-10 controllability dimension | $c_\Psi \in [0,1]$ | - |
| $o_\Psi$ | PSS-10 overload dimension | $o_\Psi \in [0,1]$ | - |
| $\Psi_i$ | PSS-10 item response for item $i$ | $\Psi_i \in \{0,1,2,3,4\}$ | - |
| $\lambda_{ij}$ | Factor loading for item $j$ on dimension $i$ | $\lambda_{ij} \in [0,1]$ | - |
| $\rho_\Psi$ | PSS-10 dimension correlation | $\rho_\Psi \in [-1,1]$ | 0.3 |

## Parameter Constants

### Stress Event Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\lambda_{\text{shock}}$ | Shock arrival rate | $\lambda_{\text{shock}} > 0$ | - |
| $\alpha_s$ | Beta distribution shape parameter | $\alpha_s > 0$ | 2.0 |
| $\beta_s$ | Beta distribution shape parameter | $\beta_s > 0$ | 2.0 |

### Appraisal Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\omega_c$ | Controllability weight | $\omega_c \in \mathbb{R}$ | 1.0 |
| $\omega_o$ | Overload weight | $\omega_o \in \mathbb{R}$ | 1.0 |
| $b$ | Bias term | $b \in \mathbb{R}$ | 0.0 |
| $\gamma$ | Sigmoid steepness | $\gamma > 0$ | 6.0 |
| $\eta_{\chi}$ | Challenge threshold modifier | $\eta_{\chi} > 0$ | 0.8 |
| $\eta_{\zeta}$ | Hindrance threshold modifier | $\eta_{\zeta} > 0$ | 1.2 |

### Resource Dynamics Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\lambda_R$ | Resource regeneration rate | $\lambda_R \in [0,1]$ | - |
| $\kappa$ | Cost scalar | $\kappa > 0$ | - |
| $\gamma_c$ | Cost function exponent | $\gamma_c > 0$ | - |

### Protective Factor Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\epsilon_{\text{soc}}$ | Social support efficacy | $\epsilon_{\text{soc}} > 0$ | - |
| $\epsilon_{\text{fam}}$ | Family support efficacy | $\epsilon_{\text{fam}} > 0$ | - |
| $\epsilon_{\text{int}}$ | Formal intervention efficacy | $\epsilon_{\text{int}} > 0$ | - |
| $\epsilon_{\text{cap}}$ | Psychological capital efficacy | $\epsilon_{\text{cap}} > 0$ | - |
| $\upsilon_{\text{soc}}$ | Social support replenishment | $\upsilon_{\text{soc}} > 0$ | - |
| $\upsilon_{\text{fam}}$ | Family support replenishment | $\upsilon_{\text{fam}} > 0$ | - |
| $\upsilon_{\text{int}}$ | Formal intervention replenishment | $\upsilon_{\text{int}} > 0$ | - |
| $\upsilon_{\text{cap}}$ | Psychological capital replenishment | $\upsilon_{\text{cap}} > 0$ | - |

### Simulation Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $T_{\max}$ | Maximum simulation days | $T_{\max} \in \mathbb{N}$ | 100 |
| $S_{\text{seed}}$ | Random seed | $S_{\text{seed}} \in \mathbb{N}$ | 42 |

### Behavioral Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\beta$ | Softmax temperature | $\beta > 0$ | - |
| $\eta_{\text{adapt}}$ | Adaptation threshold | $\eta_{\text{adapt}} > 0$ | - |
| $\lambda_{\text{adapt}}$ | Learning rate | $\lambda_{\text{adapt}} \in [0,1]$ | - |
| $p_{\text{rewire}}$ | Rewiring probability | $p_{\text{rewire}} \in [0,1]$ | 0.01 |
| $p_b$ | Base coping probability | $p_b \in [0,1]$ | 0.5 |
| $\theta_{\chi}$ | Challenge bonus parameter | $\theta_{\chi} > 0$ | 0.2 |
| $\theta_{\zeta}$ | Hindrance penalty parameter | $\theta_{\zeta} > 0$ | 0.3 |
| $\delta_{\text{soc}}$ | Social influence factor | $\delta_{\text{soc}} \in [0,1]$ | 0.1 |
| $\lambda_{\text{appraise}}$ | Event appraisal rate | $\lambda_{\text{appraise}} \in [0,1]$ | 0.15 |
| $\theta_{\text{boost}}$ | Boost rate parameter | $\theta_{\text{boost}} > 0$ | 0.1 |
| $\theta_{\text{boost\|cope}}$ | Coping success boost | $\theta_{\text{boost\|cope}} > 0$ | 0.1 |

### Output Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $L_{\text{level}}$ | Logging level | $L_{\text{level}} \in \{\text{DEBUG, INFO, WARNING, ERROR}\}$ | INFO |
| $D_{\text{results}}$ | Results output directory | $D_{\text{results}} \in \mathbb{S}$ | data/processed |
| $D_{\text{raw}}$ | Raw data output directory | $D_{\text{raw}} \in \mathbb{S}$ | data/raw |
| $D_{\text{logs}}$ | Logs output directory | $D_{\text{logs}} \in \mathbb{S}$ | logs |
| $F_{\text{ts}}$ | Save time series flag | $F_{\text{ts}} \in \{\text{true, false}\}$ | true |
| $F_{\text{net}}$ | Save network snapshots flag | $F_{\text{net}} \in \{\text{true, false}\}$ | true |
| $F_{\text{sum}}$ | Save summary statistics flag | $F_{\text{sum}} \in \{\text{true, false}\}$ | true |

### Network Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $N$ | Network size | $N \in \mathbb{N}$ | - |
| $WS_k$ | Mean degree | $k \in \mathbb{N}$ | - |
| $WS_p$ | Rewiring probability | $WS_p \in [0,1]$ | 0.1 |
| $C$ | Clustering coefficient | $C \in [0,1]$ | - |
| $L$ | Average path length | $L > 0$ | - |

### Homeostatic Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\lambda_{\text{affect}}$ | Affect homeostasis rate | $\lambda_{\text{affect}} \in [0,1]$ | - |
| $\lambda_{\text{resilience}}$ | Resilience homeostasis rate | $\lambda_{\text{resilience}} \in [0,1]$ | - |
| $\delta_{\text{stress}}$ | Stress decay rate | $\delta_{\text{stress}} \in [0,1]$ | 0.05 |

## Model-Level Variables

### Population Statistics

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\bar{R}$ | Average resources | $\bar{R} \in [0,1]$ | - |
| $\bar{D}$ | Average distress | $\bar{D} \in [0,1]$ | - |
| $\bar{A}$ | Average affect | $\bar{A} \in [-1,1]$ | - |
| $\bar{S}$ | Average stress | $\bar{S} \in [0,1]$ | - |
| $\bar{\Psi}$ | Average PSS-10 score | $\bar{\Psi} \in [0,40]$ | - |

### Network Statistics

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\rho_{\text{network}}$ | Network density | $\rho_{\text{network}} \in [0,1]$ | - |
| $C_{\text{global}}$ | Global clustering coefficient | $C_{\text{global}} \in [0,1]$ | - |
| $L_{\text{avg}}$ | Average path length | $L_{\text{avg}} > 0$ | - |

### Intervention Metrics

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\text{CER}$ | Cost-effectiveness ratio | $\text{CER} > 0$ | - |
| $\text{QALY}$ | Quality-adjusted life years | $\text{QALY} > 0$ | - |
| $\text{ICER}$ | Incremental cost-effectiveness ratio | $\text{ICER} \in \mathbb{R}$ | - |

## Function Definitions

### Stress Appraisal Functions

| Function | Definition | Description |
|----------|------------|-------------|
| $W(c,o)$ | $W(c,o) = \omega_c c - \omega_o o + b$ | Weight function for event appraisal |
| $\sigma(z)$ | $\sigma(z) = \frac{1}{1+e^{-z}}$ | Sigmoid activation function |
| $L(s,c,o)$ | $L(s,c,o) = s \cdot (1 + \delta \cdot (\zeta - \chi))$ | Appraised stress load |

### Resource Dynamics Functions

| Function | Definition | Description |
|----------|------------|-------------|
| $R'(t)$ | $R'(t) = \lambda_R (R_{\max} - R(t))$ | Resource regeneration |
| $\Kappa(s)$ | $\Kappa(s) = \kappa s^{\gamma_c}$ | Coping cost function |
| $\mathrm{softmax}(\mathbf{x})$ | $\mathrm{softmax}(x_i) = \frac{e^{x_i / \beta}}{\sum_j e^{x_j / \beta}}$ | Softmax allocation |

### Network Functions

| Function | Definition | Description |
|----------|------------|-------------|
| $d(i,j)$ | $d(i,j) = $ shortest path length | Graph distance |
| $C_i$ | $C_i = \frac{2e_i}{k_i(k_i-1)}$ | Local clustering coefficient |
| $P(k)$ | $P(k) = $ degree distribution | Degree probability |

## Equation Formatting Conventions

### LaTeX/MathJax Guidelines

1. **Inline equations**: Use single `$` delimiters for inline math: `$x = y + z$`
2. **Display equations**: Use double `$$` delimiters for centered equations:
   ```latex
   $$
   \sigma(z) = \frac{1}{1+e^{-z}}
   $$
   ```
3. **Equation arrays**: Use `\begin{align}...\end{align}` for multiple equations:
   ```latex
   \begin{align}
   x &= y + z \\
   a &= b + c
   \end{align}
   ```
4. **Matrices**: Use `\begin{matrix}...\end{matrix}` or `\begin{pmatrix}...\end{pmatrix}`:
   ```latex
   \begin{pmatrix}
   a & b \\
   c & d
   \end{pmatrix}
   ```

### Symbol Reference Format

When referencing symbols in text, use the format: `$\symbol$ (see [_NOTATION.md](./_NOTATION.md))`

Example: The stress threshold $\eta_{\text{stress}}$ (see [_NOTATION.md](./_NOTATION.md)) determines when agents become stressed.

## Validation Metrics

### Pattern Matching Targets

| Metric | Symbol | Description | Typical Range |
|--------|--------|-------------|---------------|
| Recovery time | $\tau$ | Time to return to baseline | 1-30 days |
| Basin stability | $S_{\text{basin}}$ | Proportion maintaining resilience | 0.4-0.8 |
| FTLE | $\sigma_{\text{FTLE}}$ | Finite-time Lyapunov exponent | -1 to 1 |
| Area under curve | $\text{AUC}$ | Recovery trajectory area | 0-1 |

## Configuration Integration

All parameters are configurable through the unified configuration system. See [`CONFIGURATION.md`](../CONFIGURATION.md) for complete parameter documentation and environment variable mappings.

## Version History

- **v1.0**: Initial notation standardization

This notation reference serves as the single source of truth for all mathematical representations in the agent-based mental health model.
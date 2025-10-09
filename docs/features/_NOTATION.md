# Mathematical Notation Reference

_See [`_NOTATION.md`](./_NOTATION.md) for symbol definitions and conventions._

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
| $\text{softmax}(x)$ | Softmax function | $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | - |
| $\text{clamp}(x,a,b)$ | Clamping function | $\text{clamp}(x,0,1) = \max(0,\min(1,x))$ | - |

## Core Model Variables

### Agent State Variables

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $R$ | Resources | $R \in [0,1]$ | - |
| $D$ | Distress | $D \in [0,1]$ | - |
| $T_{\text{stress}}$ | Stress threshold | $T_{\text{stress}} \in [0,1]$ | - |
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
| $\text{challenge}$ | Challenge component | $\text{challenge} \in [0,1]$ | - |
| $\text{hindrance}$ | Hindrance component | $\text{hindrance} \in [0,1]$ | - |
| $s$ | Event magnitude | $s \in [0,1]$ | - |
| $T^{\text{eff}}$ | Effective threshold | $T^{\text{eff}} = T_{\text{stress}} + \lambda_C \cdot \text{challenge} - \lambda_H \cdot \text{hindrance}$ | - |

### PSS-10 Assessment Variables

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\text{PSS-10}$ | Total PSS-10 score | $\text{PSS-10} \in [0,40]$ | - |
| $c_{\text{PSS}}$ | PSS-10 controllability dimension | $c_{\text{PSS}} \in [0,1]$ | - |
| $o_{\text{PSS}}$ | PSS-10 overload dimension | $o_{\text{PSS}} \in [0,1]$ | - |
| $r_{ij}$ | PSS-10 item response | $r_{ij} \in \{0,1,2,3,4\}$ | - |
| $\lambda_{ij}$ | Factor loading for item $j$ on dimension $i$ | $\lambda_{ij} \in [0,1]$ | - |
| $\rho_{\text{PSS}}$ | PSS-10 dimension correlation | $\rho_{\text{PSS}} \in [-1,1]$ | 0.3 |

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
| $\lambda_C$ | Challenge threshold modifier | $\lambda_C \in \mathbb{R}$ | - |
| $\lambda_H$ | Hindrance threshold modifier | $\lambda_H \in \mathbb{R}$ | - |

### Resource Dynamics Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\gamma_R$ | Resource regeneration rate | $\gamma_R \in [0,1]$ | - |
| $\kappa$ | Cost scalar | $\kappa > 0$ | - |
| $\gamma_c$ | Cost function exponent | $\gamma_c > 0$ | - |

### Protective Factor Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\alpha_{\text{soc}}$ | Social support efficacy | $\alpha_{\text{soc}} > 0$ | - |
| $\alpha_{\text{fam}}$ | Family support efficacy | $\alpha_{\text{fam}} > 0$ | - |
| $\alpha_{\text{int}}$ | Formal intervention efficacy | $\alpha_{\text{int}} > 0$ | - |
| $\alpha_{\text{cap}}$ | Psychological capital efficacy | $\alpha_{\text{cap}} > 0$ | - |
| $\rho_{\text{soc}}$ | Social support replenishment | $\rho_{\text{soc}} > 0$ | - |
| $\rho_{\text{fam}}$ | Family support replenishment | $\rho_{\text{fam}} > 0$ | - |
| $\rho_{\text{int}}$ | Formal intervention replenishment | $\rho_{\text{int}} > 0$ | - |
| $\rho_{\text{cap}}$ | Psychological capital replenishment | $\rho_{\text{cap}} > 0$ | - |

### Behavioral Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\beta$ | Softmax temperature | $\beta > 0$ | - |
| $T^{\text{adapt}}$ | Adaptation threshold | $T^{\text{adapt}} > 0$ | - |
| $\eta_{\text{adapt}}$ | Learning rate | $\eta_{\text{adapt}} \in [0,1]$ | - |
| $p_{\text{rewire}}$ | Rewiring probability | $p_{\text{rewire}} \in [0,1]$ | - |

### Network Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $N$ | Network size | $N \in \mathbb{N}$ | - |
| $k$ | Mean degree | $k \in \mathbb{N}$ | - |
| $p_{\text{rewire}}$ | Rewiring probability | $p_{\text{rewire}} \in [0,1]$ | - |
| $C$ | Clustering coefficient | $C \in [0,1]$ | - |
| $L$ | Average path length | $L > 0$ | - |

### Homeostatic Parameters

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\gamma_{\text{affect}}$ | Affect homeostasis rate | $\gamma_{\text{affect}} \in [0,1]$ | - |
| $\gamma_{\text{resilience}}$ | Resilience homeostasis rate | $\gamma_{\text{resilience}} \in [0,1]$ | - |
| $\gamma_{\text{stress}}$ | Stress decay rate | $\gamma_{\text{stress}} \in [0,1]$ | 0.05 |

## Model-Level Variables

### Population Statistics

| Symbol | Meaning/Description | Example/Range | Defaults |
|--------|-------------------|---------------|----------|
| $\bar{R}$ | Average resources | $\bar{R} \in [0,1]$ | - |
| $\bar{D}$ | Average distress | $\bar{D} \in [0,1]$ | - |
| $\bar{A}$ | Average affect | $\bar{A} \in [-1,1]$ | - |
| $\bar{S}$ | Average stress | $\bar{S} \in [0,1]$ | - |
| $\text{PSS-10}_{\text{avg}}$ | Average PSS-10 score | $\text{PSS-10}_{\text{avg}} \in [0,40]$ | - |

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
| $L(s,c,o)$ | $L(s,c,o) = s \cdot (1 + \delta \cdot (\text{hindrance} - \text{challenge}))$ | Appraised stress load |

### Resource Dynamics Functions

| Function | Definition | Description |
|----------|------------|-------------|
| $R'(t)$ | $R'(t) = \gamma_R (R_{\max} - R(t))$ | Resource regeneration |
| $C(s)$ | $C(s) = \kappa s^{\gamma_c}$ | Coping cost function |
| $\text{softmax}(\mathbf{x})$ | $\text{softmax}(x_i) = \frac{e^{x_i / \beta}}{\sum_j e^{x_j / \beta}}$ | Softmax allocation |

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

Example: The stress threshold $T_{\text{stress}}$ (see [_NOTATION.md](./_NOTATION.md)) determines when agents become stressed.

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
---
title: Mathematical Notation Reference
description: Core symbols and equations for stress processing, resource dynamics, and parameters
date: 2025-06-04
---

# Agent State Variables

$R \in [0,1]$ — resources. $D \in [0,1]$ — distress. $A \in [-1,1]$ — affect.
$\eta_{\text{stress}} \in [0,1]$ — stress threshold. $\mathfrak{R} \in [0,1]$ — resilience.

# Stress Processing

Event attributes: controllability $c \in [0,1]$, overload $o \in [0,1]$.

Weight function: $z = \omega_c c - \omega_o o + b$

Challenge: $\chi = \sigma(\gamma z)$ where $\sigma(x) = \frac{1}{1+e^{-x}}$
Hindrance: $\zeta = 1 - \chi$

Appraised load: $L = s \cdot (1 + \delta(\zeta - \chi))$

Agent stressed when: $L > \eta_{\text{stress}} + \lambda_\chi \chi - \lambda_\zeta \zeta$

# PSS-10 Bifactor Model

Total score $\Psi \in [0,40]$ from 10 items. Orthogonal general factor + specific factors $(c_\Psi, o_\Psi)$. Items $\Psi_i \in \{0,1,2,3,4\}$ with factor loadings $\lambda_{ij}$.

# Resource Dynamics

Regeneration: $R'(t) = \lambda_R (R_{\max} - R(t))$
Softmax allocation: $\mathrm{softmax}(x_i) = e^{x_i / \beta} / \sum_j e^{x_j / \beta}$
Cost function: $\Kappa(s) = \kappa s^{\gamma_c}$

# Key Parameters

| Symbol | Meaning | Default |
|--------|---------|---------|
| $\omega_c, \omega_o$ | Appraisal weights | 1.0 |
| $\gamma$ | Sigmoid steepness | 6.0 |
| $\lambda_\chi, \lambda_\zeta$ | Threshold modifiers | 0.15, 0.25 |
| $\eta_{\text{stress}}$ | Base stress threshold | 0.3 |
| $\lambda_R$ | Resource regen rate | 0.05 |
| $\beta$ | Softmax temperature | 1.0 |

Full parameter reference: `@src/python/config.py`

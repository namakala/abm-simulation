---
title: Model Architecture — Detailed Design
description: Agent structure, event processing pipeline, PSS-10 integration, network adaptation
date: 2025-06-04
---

# Agent State

Each agent maintains:
- **Resources** $R \in [0,1]$ — finite psychological capacity
- **Distress** $D \in [0,1]$ — current stress level
- **Affect** $A \in [-1,1]$ — emotional valence (negative/positive)
- **Resilience** $\mathfrak{R} \in [0,1]$ — capacity to recover
- **Stress threshold** $\eta_{\text{stress}}$ — sensitivity to events
- **Protective factors** — social support, family support, formal intervention, psych capital (each $[0,1]$)

# Event Processing Pipeline

1. **Generate event.** Poisson process yields event with controllability $c$, overload $o$, magnitude $s$.
2. **Appraise event.** Weight $z = \omega_c c - \omega_o o + b$, sigmoid to challenge/hindrance.
3. **Compute load.** $L = s \cdot (1 + \delta(\zeta - \chi))$.
4. **Evaluate threshold.** If $L > \eta_{\text{eff}}$ → agent is stressed.
5. **Coping check.** Probability based on resilience, affect, and social support.
6. **Update state.** Distress, affect, resources updated based on outcome.

# Social Network

Watts-Strogatz small-world with configurable $k$ (mean degree) and $p$ (rewiring). Agents interact with neighbors for social support and affect contagion. Network adapts: nodes rewire when stress threshold repeatedly breached.

# PSS-10 Integration

Bifactor model with general stress factor and 2 specific factors (controllability, overload). Generates 3 composite scores per agent from 10 item responses. Used for threshold evaluation and empirical validation.

# Resource Allocation

Agent allocates resources to protective factors via softmax. Each factor has efficacy $\alpha$ and replenishment $\rho$. Decision stochasticity controlled by temperature $\beta$.

See `@docs/ADR/008-model-architecture.md` for rationale.

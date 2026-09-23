---
title: Design and Architecture of the Agent-Based Simulation Approach
subtitle: Simulation Strategy Document
author: Aly Lamuri
---

# Agent Initialization

Each agent is initialised with baseline values representing natural equilibrium
points using mathematical transformations ensuring proper statistical distributions.

{{< include design/_init-baseline.md >}}

{{< include design/_init-state.md >}}

# Daily Simulation Loop

The orchestrator is `Person.step()` which builds agent state, runs both loops,
and writes back state.

Reference: [src/python/agent.py:L601-L850](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/agent.py#L601-L850)

The number of subevents per day is drawn from a Poisson distribution
($\lambda = 3$), ensuring at least one daily subevent. Each subevent is
randomly assigned as either a stress event or a social interaction.

```
n_subevents ← max(Poisson(λ), 1)
actions ← random sequence of ["stress", "interact"] of length n_subevents
```

{{< include design/_subevent-loop.md >}}

{{< include design/_daily-consolidation.md >}}

# Model Orchestration

The simulation uses Mesa's agent-based modelling framework with dual-class
architecture separating agent behaviours (`Person`) from model orchestration
(`StressModel`).

{{< include design/_orchestration.md >}}

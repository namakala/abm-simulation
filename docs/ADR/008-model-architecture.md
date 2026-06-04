---
title: Model Architecture & Design
description: Challenge/hindrance appraisal, Watts-Strogatz networks, PSS-10, modular utility design
status: accepted
date: 2025-06-04
---

# Context

ABM simulating workplace mental health promotion cost-effectiveness. Needed core model design: agent structure, stress perception, network topology, code organization, and configuration strategy.

# Decision

- **Agent Structure:** Each agent has resources R, distress D, stress threshold, affect A, resilience R, protective factors (social/family/intervention/capital)
- **Stress Perception:** Challenge/hindrance appraisal — sigmoid mapping of controllability/overload to binary challenge/hindrance polarity. See `@docs/agents/ARCHITECTURE_MODEL.md`
- **Network:** Watts-Strogatz small-world — configurable mean degree and rewiring probability, realistic clustering
- **PSS-10:** Bifactor model with orthogonal general stress factor and specific factors (controllability, overload)
- **Code Organization:** Utility-modular design — separate files for stress, affect, resources, math, config, visualization
- **Configuration:** .env-based system via python-dotenv with 50+ parameters, type conversion, validation
- **Resource Allocation:** Softmax decision-making across protective factors with regeneration dynamics

Alternatives considered: fully custom event model (rejected — reusable Mesa scheduler), monolithic agent class (rejected — testability issue), fixed parameters (rejected — sweep analysis requirement).

# Impact

Modular utilities enable independent testing. Challenge/hindrance maps established stress theory. Small-world networks produce realistic social dynamics. .env-config enables systematic parameter sweeps and sensitivity analysis. See ADR-001 for framework rationale.

---
title: Interaction Phase
description: Extract interaction into atomic phase isolated from PF feedback
date: 2026-06-04
---

# Overview

Extract social interaction -> affect/resilience/resource exchange pipeline into a standalone phase function. Strictly isolated from protective factor efficacy updates: interaction only directly changes affect, resilience, and resources.

# Goals

- Theory-based tests assert all interaction predictions
- `process_interaction()` is a pure function with (self_state, partner_state, config, rng) -> PhaseOutput
- No PF efficacy modification inside interaction phase
- Refactored agent.py and affect_utils.py keep helpers

# Implementation Steps

- [ ] 1. Write `test_interaction_theory.py`:
  - Affect convergence: after interaction, |self_A - partner_A| decreases
  - Negativity bias: negative influence 1.5x stronger than positive (same |delta|)
  - Resource exchange: transfer bounded by available resources of both agents
  - Support exchange detection: >0.05 change in any dimension -> detected; all <0.05 -> not detected
  - No PF efficacy is modified by interaction outcomes
  - Empty neighbors: returns zero-effect result
- [ ] 2. Create `phases/interaction.py`:
  - `process_interaction()`: mutual affect adjustment -> resilience influence -> social resource exchange -> support detection
  - Returns PhaseOutput with state_delta (affect, resilience, resources) and observation (affect_delta, resilience_delta, resource_transfer, support_exchange flag)
- [ ] 3. Move from `affect_utils.py` and `agent.py` into phase function
- [ ] 4. Run: `pytest src/python/tests/test_interaction_theory.py -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Negativity bias 1.5x may be too strong | Med | Med | Parameterize the bias factor in config |
| Partner state mutation | Med | High | Phase function takes BOTH states, returns BOTH deltas; model applies both |

# UAT

1. All theory tests pass
2. Phase function takes two AgentState inputs (self + partner)
3. No protective factor keys in returned state_delta

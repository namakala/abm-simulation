---
title: Interaction Phase
description: Extract social interaction with win-win/lose-lose resource exchange state machine
date: 2026-06-05
---

# Overview

Extract social interaction, mutual affect/resilience convergence, and resource exchange from agent.py:382-504 into an event-driven phase. The resource exchange uses a win-win/lose-lose state machine based on both agents' stressed flags and support detection from convergence magnitude. Takes two AgentState inputs (self + partner) and returns two PhaseOutputs.

# Goals

- Theory-based tests assert all interaction predictions
- `process_interaction(self_state, partner_state, config, rng) -> (self_delta, partner_delta)`
- 5-scenario state machine for resource exchange based on (stressed, stressed, support)
- Support detection from affect/resilience convergence magnitude (> threshold)
- Always converge affect and resilience (current behavior preserved)
- Social resource exchange logic moved from resource_utils.py into this phase

# Implementation Steps

- [ ] 1. Write `test_interaction_theory.py`:
  - Affect convergence: |self_A - partner_A| decreases after interaction
  - Negativity bias: negative influence 1.5x stronger than positive (same |delta|)
  - No PF efficacy modification (no protective_factors in state delta)
  - Empty neighbors: returns zero-effect result
  - Resource state machine: test all 5 scenarios (win-win, lose-lose, win, lose, both non-stressed boost PF)
  - Support detection: >0.05 convergence change -> support_occurred=True
- [ ] 2. Create `phases/interaction.py`:
  - `process_interaction()`: affect convergence -> resilience convergence -> support detection -> resource exchange state machine
  - State machine: match (self_stressed, partner_stressed, support_occurred):
    - (T,T,T) -> both resources +boost
    - (T,T,F) -> both resources -cost
    - (T,F,T) -> self +boost, partner unchanged
    - (T,F,F) -> self -cost, partner unchanged
    - (F,F,_) -> both PF social_support +small_boost, resources unchanged
  - Returns (self_delta, partner_delta) with affect, resilience, resources, protective_factors keys
- [ ] 3. Extract from agent.py:382-504 and process_social_resource_exchange from resource_utils.py
- [ ] 4. Hardcoded constants (support_threshold=0.05, boost=0.10, etc.) noted for Plan 007
- [ ] 5. Run: `pytest src/python/tests/test_interaction_theory.py -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Partner state mutation | Med | High | Phase takes BOTH states, returns BOTH deltas; orchestrator applies both |
| State machine complex — 5 scenarios difficult to cover | High | Med | Write one test per scenario with explicit input/output table |

# UAT

1. All theory tests pass
2. Phase function takes two AgentState inputs and returns two PhaseOutputs
3. No protective factor keys in resource-allocation-only scenarios
4. PF social_support boost happens only in non-stressed-non-stressed scenario

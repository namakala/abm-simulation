---
title: Resilience Activation Phase
description: Extract coping, resilience delta, resource cost, PF allocation into event-driven phase
date: 2026-06-05
---

# Overview

Extract coping outcome, resilience/affect/stress changes, resource cost, PSS-10 generation, and PF allocation from agent.py:572-709 into an event-driven phase. Called only when is_stressed=True. Receives state already updated by Plan 002. This is the largest phase — it handles the full coping pipeline.

# Goals

- Theory-based tests assert all resilience activation predictions
- `process_resilience_activation()` is a pure function with (state, config, rng) -> PhaseOutput
- Called per-event, only when is_stressed
- PSS-10 generation from stress dimensions is included here
- Resource cost, reward, and penalty are included here (hardcoded constants noted for Plan 007)

# Implementation Steps

- [ ] 1. Write `test_resilience_activation_theory.py`:
  - Higher challenge -> higher coping prob; higher hindrance -> lower
  - Positive neighbor affect -> higher coping prob; negative -> lower
  - Successful coping -> DeltaR >= 0; failed -> DeltaR <= 0
  - Asymmetry: ch_gain > ch_loss; hi_loss > hi_gain
  - Overload: DeltaR_o = -0.4 * min(h_c / eta, 1.0)
  - Homeostasis: R > R0 -> pull down; R < R0 -> pull up
  - Resource cost scales with resilience and event difficulty
  - Resource reward = base_cost * 0.75 after successful coping (constant, Plan 007 externalizes)
  - Resource penalty = base_cost * 0.10 after failed coping (constant, Plan 007 externalizes)
  - PF allocation fraction = resources * 0.30 after successful coping (constant, Plan 007 externalizes)
  - Stress dimensions updated from event outcome; PSS-10 generated from updated dims
- [ ] 2. Create `phases/resilience_activation.py`:
  - `process_resilience_activation()`: neighbor affects -> coping prob -> coping outcome -> affect/resilience/stress deltas -> stress dim update -> PSS-10 gen -> resource cost -> PF allocation -> hindrance tracking
  - Returns PhaseOutput with state_delta (affect, resilience, current_stress, stress_controllability, stress_overload, resources, protective_factors, consecutive_hindrances, stress_breach_count, pss10, pss10_responses, stressed)
- [ ] 3. Extract from agent.py:572-709 and affect_utils.py coping/resilience functions
- [ ] 4. Hardcoded constants in affect_utils.py noted for Plan 007 (not changed now)
- [ ] 5. Run: `pytest src/python/tests/test_resilience_activation_theory.py -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| State delta has 12 keys — easy to miss one | High | High | Line-by-line cross-reference of agent.py:572-709 for all self.* assignments |
| Overlap with Plan 006 stress buffering (both modify resilience) | Med | High | Plan 003 handles coping-induced resilience change; Plan 006 handles PF boost. Keys documented. |

# UAT

1. All theory tests pass
2. Phase function returns correct per-component DeltaR breakdown in observation
3. For same seed and same stress event -> same coping outcome and resource change as current code

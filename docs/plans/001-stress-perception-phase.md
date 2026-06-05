---
title: Stress Perception Phase
description: Extract event generation, appraisal, threshold into event-driven phase
date: 2026-06-05
---

# Overview

Extract the event generation, appraisal, threshold, and is_stressed classification from agent.py:517-560 into a standalone event-driven phase function. This is the first phase called per stress event. Non-stressed events return early without coping. PSS-10 generation is NOT in this phase — it happens in Plan 002.

# Goals

- Theory-based tests assert all stress perception predictions
- `process_stress_perception()` is a pure function with (state, config, rng) -> PhaseOutput
- Called per-event inside subevent loop (not once per day)
- Stress dimension update for non-stressed path handled (confirmed no-op with is_stressful=False)

# Implementation Steps

- [ ] 1. Write `test_stress_perception_theory.py`:
  - Appraisal monotonicity: higher c -> higher ch, lower hi; higher o -> higher hi, lower ch
  - Complementarity: ch + hi = 1 within 1e-10
  - Threshold: higher ch -> higher eta_eff (harder to trigger stress)
  - Stress classification: low-c + high-o -> stressed; high-c + low-o -> not stressed
  - Non-stressed path: is_stressful=False -> update_stress_dimensions returns input values
  - Event sampling: controllability/overload in [0,1]
- [ ] 2. Create `phases/stress_perception.py`:
  - `process_stress_perception()`: event gen -> appraisal -> threshold -> is_stressed
  - If not stressed: call update_stress_dimensions(is_stressful=False), return early
  - Returns PhaseOutput with state_delta (challenge, hindrance, is_stressed, event attrs, stress dimensions) and observation (event attrs, appraisal, threshold)
- [ ] 3. Extract code from agent.py:517-560 (stressful_event top through non-stressed return)
- [ ] 4. Keep helpers in stress_utils.py: generate_stress_event, apply_weights, evaluate_stress_threshold, update_stress_dimensions_from_event
- [ ] 5. Hardcoded constants in stress_utils.py noted for Plan 006 parameterization (not changed now)
- [ ] 6. Run: `pytest src/python/tests/test_stress_perception_theory.py -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Non-stressed path no-op misunderstood | Med | High | Test verifies update_stress_dimensions(is_stressful=False) returns inputs unchanged |
| PSS-10 generation not in this phase | Low | Low | Document clearly; Plan 002 generates PSS-10 after coping |

# UAT

1. All theory tests pass
2. Phase function returns correct state_delta keys for both stressed and non-stressed paths
3. For same seed, same event -> same challenge/hindrance/is_stressed as current code

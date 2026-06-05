---
title: Resource Allocation Phase
description: Extract resource regeneration, softmax allocation, efficacy update into daily phase
date: 2026-06-05
---

# Overview

Extract resource regeneration, softmax-based PF allocation, and efficacy updates from agent.py:296-318 and resource_utils.py into a standalone daily phase. Strictly isolated from interaction concerns: no social resource exchange (moved to Plan 005). Runs once after the event loop.

# Goals

- Theory-based tests assert all resource allocation predictions
- `process_resource_allocation()` is a pure function with (state, config, rng) -> PhaseOutput
- No interaction variables referenced (no social exchange here)
- Social resource exchange removed — moved to Plan 005
- Resource regeneration multipliers are hardcoded constants (noted for Plan 007)

# Implementation Steps

- [ ] 1. Write `test_resource_allocation_theory.py`:
  - Softmax: higher e_f -> higher w_f (monotonic)
  - Temperature: high T -> uniform w_f; low T -> winner-take-most
  - Diminishing returns: higher e_f -> smaller Delta e_f per unit r_f
  - Regeneration: R near 0 -> high R'; R = 1 -> R' = 0; positive A boosts R'
  - R always in [0,1]; sum(r_f) = available R (conservation)
  - No interaction variables referenced
  - Resource regeneration multipliers: affect 0.50x, resilience 0.30x (constants, Plan 007 externalizes)
- [ ] 2. Create `phases/resource_allocation.py`:
  - `process_resource_allocation()`: resource regeneration -> softmax allocation weights -> PF efficacy updates -> resource depletion
  - Returns PhaseOutput with state_delta (resources, protective_factors) and observation (allocation_weights per PF, efficacies before/after, regeneration_amount)
- [ ] 3. Extract from agent.py:296-318 and resource_utils.py (excluding social exchange functions)
- [ ] 4. Hardcoded constants in resource_utils.py noted for Plan 007 (not changed now)
- [ ] 5. Run: `pytest src/python/tests/test_resource_allocation_theory.py -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Social resource exchange accidentally kept | Low | High | Verify no process_social_resource_exchange import in this phase |
| Resource regeneration constants differ from current behavior | Low | Med | Cross-reference agent.py:299-307 for exact multiplier values |

# UAT

1. All theory tests pass
2. Phase function returns per-factor allocation breakdown
3. No interaction functions referenced in this phase (grep for social, interact, partner)

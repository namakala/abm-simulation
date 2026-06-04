---
title: Resource Allocation Phase
description: Extract resource allocation into atomic phase isolated from interaction
date: 2026-06-04
---

# Overview

Extract resource regeneration -> softmax allocation -> efficacy update pipeline into a standalone phase function. Strictly isolated from interaction concerns: only tests PF allocation dynamics.

# Goals

- Theory-based tests assert all resource allocation predictions
- `process_resource_allocation()` is a pure function with (state, config, rng) -> PhaseOutput
- No interaction code touches this phase
- Refactored resource_utils.py keeps helpers

# Implementation Steps

- [ ] 1. Write `test_resource_allocation_theory.py`:
  - Softmax: higher e_f -> higher w_f (monotonic)
  - Temperature: high T -> uniform w_f; low T -> winner-take-most
  - Diminishing returns: higher e_f -> smaller Delta e_f per unit r_f
  - Regeneration: R near 0 -> high R'; R = 1 -> R' = 0; positive A boosts R'
  - R always in [0,1]; sum(r_f) = available R (conservation)
  - No interaction variables referenced
  - COR loss primacy: net effect of stress event on R is negative
- [ ] 2. Create `phases/resource_allocation.py`:
  - `process_resource_allocation()`: regeneration -> softmax allocation to each PF -> efficacy updates -> resource depletion if coping
  - Returns PhaseOutput with state_delta (resources, protective_factors) and observation (allocation_weights per PF, efficacies before/after, regeneration_amount)
- [ ] 3. Move from `resource_utils.py` and `affect_utils.py` into phase function
- [ ] 4. Run: `pytest src/python/tests/test_resource_allocation_theory.py -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Softmax temperature affects realism | Med | High | Default T must produce non-trivial multi-factor allocation |
| Resource cost double-counted | Low | High | Phase only handles allocation; cost already applied in WP 1 |

# UAT

1. All theory tests pass
2. Phase function returns per-factor allocation breakdown
3. No interaction functions referenced in this phase

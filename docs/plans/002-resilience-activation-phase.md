---
title: Resilience Activation Phase
description: Extract resilience activation into atomic phase with theory-based tests
date: 2026-06-04
---

# Overview

Extract coping outcome -> resilience update pipeline from agent.py/affect_utils.py into a standalone phase function. TDD: write theory-based tests first.

# Goals

- Theory-based tests assert all resilience activation predictions
- `process_resilience_activation()` is a pure function with (state, config, rng) -> PhaseOutput
- Refactored affect_utils.py keeps only helpers
- All tests pass

# Implementation Steps

- [ ] 1. Write `test_resilience_activation_theory.py`:
  - Higher challenge -> higher coping probability; higher hindrance -> lower
  - Positive neighbor affect -> higher coping probability; negative -> lower
  - Successful coping -> DeltaR >= 0; failed coping -> DeltaR <= 0
  - Asymmetry: ch_gain(0.3*ch) > ch_loss(0.1*ch); hi_loss(0.4*hi) > hi_gain(0.1*hi)
  - Overload: DeltaR_o = -0.4 * min(h_c / eta, 1.0) when h_c >= eta
  - Homeostasis: R > R0 -> pull down; R < R0 -> pull up; R = R0 -> no change
  - Protective factor boost: larger when R_t far below R_0
- [ ] 2. Create `phases/resilience_activation.py`:
  - `process_resilience_activation()`: coping probability -> outcome -> resilience delta (ch/hi effect, overload, protective boost, social support, homeostasis)
  - Returns PhaseOutput with state_delta (resilience) and observation (coping_prob, coped_successfully, each DeltaR component)
- [ ] 3. Move from `affect_utils.py` into phase function
- [ ] 4. Run: `pytest src/python/tests/test_resilience_activation_theory.py -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Overload formula changed from original | Low | High | Verify new formula -0.4*min(h_c/eta,1.0) matches intent |
| Homeostasis interacts with other phases | Med | Med | Test in isolation first; integration test in WP 6 catches conflicts |

# UAT

1. All theory tests pass
2. Phase function returns per-component DeltaR breakdown in observation
3. Old resilience dynamics still work

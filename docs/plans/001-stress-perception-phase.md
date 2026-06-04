---
title: Stress Perception Phase
description: Extract stress perception into atomic phase with theory-based tests
date: 2026-06-04
---

# Overview

Extract the event generation -> appraisal -> threshold -> PSS-10 pipeline from agent.py into a standalone phase function. TDD: write theory-based tests first, then implement.

# Goals

- Theory-based tests assert all stress perception predictions
- `process_stress_perception()` is a pure function with (state, config, rng) -> PhaseOutput
- Refactored stress_utils.py keeps only general helpers
- All tests pass

# Implementation Steps

- [ ] 1. Write `test_stress_perception_theory.py`:
  - Appraisal monotonicity: higher c -> higher ch, lower hi; higher o -> higher hi, lower ch
  - Complementarity: ch + hi = 1 within 1e-10
  - Threshold: higher ch -> higher eta_eff (harder to trigger stress)
  - PSS-10 in [0,40]; PSS-10 vs resilience r in [-0.6, -0.1]
  - Stress classification: low-c + high-o -> stressed; high-c + low-o -> not stressed
- [ ] 2. Create `phases/stress_perception.py`:
  - `process_stress_perception()`: event gen -> appraisal -> stress load -> threshold -> PSS-10 update
  - Returns PhaseOutput with state_delta (stress vars, PSS-10) and observation (event attrs, appraisal, threshold, PSS-10 items)
- [ ] 3. Move extracted code from `stress_utils.py` into phase function
- [ ] 4. Keep in stress_utils.py: general helpers called by phase (e.g., `generate_stress_event`, `apply_weights`, `evaluate_stress_threshold`)
- [ ] 5. Run: `pytest src/python/tests/test_stress_perception_theory.py -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PSS-10 correlation range fails | Med | High | Tune test bounds from empirical sim runs; document observed range |
| Circular import with config | Low | Med | Config stays as imported singleton, not passed through |

# UAT

1. All theory tests pass
2. Phase function returns correct observation keys
3. Old code still works (no regression)

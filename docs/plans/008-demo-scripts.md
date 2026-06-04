---
title: Demo Scripts
description: Create 5 standalone demo scripts with correlogram visualizations
date: 2026-06-04
---

# Overview

Create one demo script per phase. Each demo runs its phase function across systematic input variations and produces a correlogram (scatter matrix with histograms on diagonal, pairwise scatter + Pearson r on off-diagonal). Resource allocation demo covers multiple parameter scenarios.

# Goals

- 5 demo scripts in src/python/demos/
- Each generates a correlogram saved to docs/figures/
- Resource allocation demo shows 3+ parameter scenarios side-by-side
- Each demo prints statistical interpretation

# Implementation Steps

- [ ] 1. `demo_stress_perception.py`:
  - Sweep controllability [0,1] x 20 and overload [0,1] x 20 (400 events)
  - Collect: c, o, ch, hi, stress_load, threshold, PSS-10
  - Correlogram: 6x6 matrix with annotated r values
  - Interpretation: monotonicity, complementarity, threshold dynamics
- [ ] 2. `demo_resilience_activation.py`:
  - Vary ch/hi across [0,1], 3 neighbor affect levels (-0.5, 0, +0.5)
  - Collect: coping_prob, coped_successfully, each DeltaR component
  - Correlogram: 8x8 matrix; overlay success/failure scatter
  - Interpretation: asymmetry, social influence, overload effect
- [ ] 3. `demo_resource_allocation.py` (multiple scenarios):
  - Scenario A: High R (0.8), default T -> balanced allocation
  - Scenario B: Low R (0.2), default T -> concentrated on highest e_f
  - Scenario C: High R (0.8), high T -> near-uniform allocation
  - Scenario D: Starting efficacies varied (one very low, one high)
  - Correlogram per scenario: 5x5 (4 PFs + resource level)
  - Time series: PF efficacy trajectories across repeated allocations
- [ ] 4. `demo_interaction.py`:
  - Vary partner_A [-1, 1] x 20, self_A [-1, 1] x 5
  - Collect: self/partner affect_delta, resilience_delta, resource_transfer
  - Correlogram: 8x8; color by support_exchange flag
  - Interpretation: convergence, negativity bias, resource exchange
- [ ] 5. `demo_stress_buffering.py`:
  - Controlled mediation: vary stress input [0,1], measure resource -> buffering
  - Path plot: annotated diagram with a, b, c', a*b coefficients
  - Correlogram: stress x resources x social_support x buffering_strength
- [ ] 6. Save all figures to `docs/figures/<phase>_correlogram.png`
- [ ] 7. Run each demo: `python src/python/demos/demo_<phase>.py`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| matplotlib output formatting | Med | Low | Use seaborn pairgrid with corr annotations |
| Demo uses unpublished phase API | Med | Med | Freeze phase signatures before demos |

# UAT

1. Each demo runs standalone and produces a correlogram
2. Resource allocation demo shows 4 distinct scenarios
3. All figures render correctly in docs/figures/

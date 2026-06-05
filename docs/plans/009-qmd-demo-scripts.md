---
title: QMD Demo Scripts
description: Create 5 standalone Quarto demo documents with inline correlogram figures
date: 2026-06-05
---

# Overview

Convert the Python demo scripts to Quarto (.qmd) documents. Each demo runs its phase function across systematic input variations and produces inline correlogram figures. Narrative markdown explains the theory, the code, and the observed patterns. No `demo_` prefix since files are already in src/python/demos/. pixi task added for quarto rendering.

# Goals

- 5 .qmd files in src/python/demos/ (no demo_ prefix)
- Each generates inline correlogram figures with annotated Pearson r values
- Resource allocation demo shows 4 parameter scenarios
- Narrative explains theory predictions and how they map to code
- pixi task: `pixi run quarto <file>` renders a single demo

# Implementation Steps

- [ ] 1. Add quarto to pixi.toml dependencies and add task: `quarto = "quarto render"`
- [ ] 2. Write `src/python/demos/stress_perception.qmd`:
  - Sweep c [0,1] x 20 and o [0,1] x 20 (400 events)
  - Collect: c, o, ch, hi, stress_load, threshold, is_stressed
  - Correlogram: 6x6 with annotated r; narrative: monotonicity, complementarity, classification
- [ ] 3. Write `src/python/demos/resilience_activation.qmd`:
  - Vary ch/hi across [0,1], 3 neighbor affect levels
  - Collect: coping_prob, coped_successfully, each DeltaR component, resource cost/reward
  - Correlogram: 8x8; overlay success/failure scatter
- [ ] 4. Write `src/python/demos/resource_allocation.qmd` (4 scenarios):
  - High R + default T, Low R + default T, High R + high T, varied starting efficacies
  - Correlogram per scenario + time series of PF efficacy trajectories
- [ ] 5. Write `src/python/demos/interaction.qmd`:
  - Vary partner_A [-1,1] x 20, self_A [-1,1] x 5
  - Collect: affect/resilience/resource deltas, support flag
  - Correlogram: 8x8, color by support_exchange; test all 5 state machine scenarios
- [ ] 6. Write `src/python/demos/stress_buffering.qmd`:
  - Controlled mediation: vary stress input [0,1], measure resource -> buffering
  - Annotated path plot with a, b, c', a*b coefficients
- [ ] 7. Verify each renders: `pixi run quarto src/python/demos/<name>.qmd`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| quarto not available in environment | Med | Low | Add to pixi dependencies |
| Demo uses unpublished phase API | Low | Med | Demos written AFTER Plan 007 orchestrator is stable |

# UAT

1. `pixi run quarto src/python/demos/stress_perception.qmd` produces HTML with inline correlogram
2. All 5 .qmd files render without errors
3. Resource allocation demo shows 4 distinct scenarios
4. No demo_ prefix on file names

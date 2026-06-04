---
title: Integrated Population Tests
description: Validate full simulation produces theoretically-expected population patterns
date: 2026-06-04
---

# Overview

Test the fully assembled simulation (post-WP 6) against theoretical predictions at the population level. Requires researching and documenting precise reference correlation ranges before writing tests.

# Goals

- Research and document theoretical reference values for all key variable pairs
- Write population-level theory tests that assert correlations against references
- Multi-seed stability validation
- All phase-level tests (WP 1-5) + integration pass before this runs

# Implementation Steps

- [ ] 1. Research reference values (document in test as constants):
  - PSS-10 general population norm: mean 13-15, SD 6-8
  - Correlation reference table (populate ranges before writing tests):

  | Pair | Expected r | Citation Needed |
  |------|-----------|----------------|
  | PSS-10 vs Resilience | [-0.6, -0.1] | confirmed |
  | PSS-10 vs Stress | [0.2, 0.9] | existing |
  | PSS-10 vs Affect | [?, ?] | research |
  | PSS-10 vs Resources | [?, ?] | research |
  | Resources vs Stress | [-0.8, 0.1] | existing |
  | Coping vs Challenge | [0.1, 0.6] | theory |
  | Coping vs Hindrance | [-0.6, -0.1] | theory |
  | Interaction vs Resilience | [0.0, 0.4] | theory |
  | PSS-10 population mean | 13-15 (SD 6-8) | norm |

- [ ] 2. Write `test_integrated_population_theory.py`:
  - Agent-level cross-sectional correlations (final epoch, N=200, days=100)
  - Population-level time-series correlations (model_data means)
  - Mediation at population: stress -> resource -> buffering with Sobel test
  - Multi-seed stability: 10 seeds, r_std < 0.3
  - PSS-10 distribution: mean in [13,15], SD in [6,8]
  - Resources within [0,1] invariant for all agents at all steps
- [ ] 3. If tests fail: calibrate parameters (not phase logic) to bring correlations into range
- [ ] 4. Run: `pytest src/python/tests/test_integrated_population_theory.py -v --timeout=120`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Population correlations weaker than theory | Med | High | First run with current model to get empirical baseline; adjust ranges realistically |
| Tests take too long (N=200, days=100) | High | Low | Mark as slow; run separately from unit tests |

# UAT

1. All integrated population tests pass
2. Reference values documented in test file header
3. Parameter calibration notes (if any) recorded

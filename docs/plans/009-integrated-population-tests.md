---
title: Integrated Population Tests
description: Validate full simulation against 11 empirical correlations with calibration step
date: 2026-06-05
---

# Overview

Test the fully assembled simulation (post-Plan 008) against empirically-sourced correlation targets. Includes a calibration step to bring PSS-10 population mean from ~24.8 to the literature norm of 13-15. Adds src/python/calibration/ for batch tuning. References 11 correlations with source citations and 95% CIs.

# Goals

- PSS-10 mean calibrated to [13,15] (SD [6,8]) — literature norm (Cohen 1983)
- All 11 empirical correlations asserted with 95% CI bounds
- Multi-seed stability (10 seeds, r_std < 0.3)
- Sensitivity analysis documented (assumptions set to 0, activated one-by-one)
- Existing test_correlation_validation.py consolidated into this suite

# Empirical Reference Table

| Pair | r target | 95% CI | Source |
|------|---------|--------|--------|
| PSS-10 vs Resilience | -0.57 | [-0.645, -0.485] | Thomas & Zolkoski 2020 |
| PSS-10 vs Stress | 0.20-0.39 | [0.09, 0.55] | Cohen, Kamarck & Mermelstein 1983 |
| PSS-10 vs Affect | -0.176 | [-0.267, -0.082] | Acoba 2024 |
| PSS-10 vs Resources | -0.180 | [-0.270, -0.086] | Acoba 2024 |
| Resources vs Stress | -0.180 | [-0.270, -0.086] | Acoba 2024 |
| Coping vs Challenge | 0.35 | [0.242, 0.449] | Thomas & Zolkoski 2020 |
| Coping vs Hindrance | -0.20 | [-0.248, -0.151] | Chmitorz et al. 2018 |
| Interaction vs Resilience | 0.249 | [0.158, 0.336] | Acoba 2024 |
| Resilience vs Affect PA | gamma=0.70 | — | Montero-Marin et al. 2015 |
| Resilience vs Affect NA | gamma=-0.35 | — | Montero-Marin et al. 2015 |
| Affect vs Resources | 0.249 | [0.158, 0.336] | Acoba 2024 |

# Implementation Steps

- [ ] 1. Create `src/python/calibration/` with __init__.py and AGENTS.md
- [ ] 2. Write `calibrate_pss10_population.py`: runs N=200, D=100, 10 seeds; measures PSS-10 mean; adjusts PSS-10 item params to hit [13,15]; re-runs until calibrated
- [ ] 3. Add pixi task: `calibrate = "python src/python/calibration/run_full_calibration.py"`
- [ ] 4. Write `test_integrated_population_theory.py`:
  - PSS-10 distribution: mean in [13,15], SD in [6,8]
  - All 11 correlation targets with 95% CI bounds
  - Multi-seed stability: 10 seeds, r_std < 0.3
  - Resources in [0,1] invariant at all steps
  - Mediation at population: stress -> resource -> buffering (Sobel test)
- [ ] 5. Consolidate existing test_correlation_validation.py: move overlapping tests, archive unique coverage
- [ ] 6. Mark new test as `@pytest.mark.slow`; add pixi task `test-slow` to run it
- [ ] 7. Sensitivity analysis: write script that sets each ASSUMPTION_* to 0 and measures delta from baseline correlations. Output: which assumptions matter for which correlations.
- [ ] 8. Run: `pytest src/python/tests/test_integrated_population_theory.py -v --timeout=120`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PSS-10 mean calibration changes other correlations | High | High | After calibration, verify ALL 11 targets still hold; iterate if not |
| Tests too slow (N=200, D=100, 10 seeds) | High | Low | Mark as slow; separate from unit tests |
| Sensitivity analysis is large | Med | Low | Document as separate script, not a test; run manually |

# UAT

1. PSS-10 mean in [13,15] after calibration
2. All 11 correlation targets met with 95% CI bounds
3. Multi-seed stability: r_std < 0.3 across 10 seeds
4. Sensitivity script runs and produces correlation delta table

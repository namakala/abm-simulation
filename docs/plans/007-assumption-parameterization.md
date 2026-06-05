---
title: Assumption Parameterization
description: Externalize all ~84 hardcoded constants as ASSUMPTION_* env vars, harmonize 5 discrepancies
date: 2026-06-05
---

# Overview

Single pass over all five phase functions to replace hardcoded numeric constants with config references. Every constant becomes an `ASSUMPTION_*` env var under grouped sections in .env.example. Five data discrepancies (same concept, different values across files) are harmonized to a single source of truth. Behavioral equivalence preserved by using current default values.

# Goals

- All ~84 hardcoded constants replaced by ASSUMPTION_* env var references
- 5 discrepancies harmonized (single value per concept across all files)
- Config hierarchy: AssumptionConfig dataclass with sub-groups
- .env.example updated with # Assumption sections
- All tests pass with unchanged default values
- Sensitivity analysis enabled: set any ASSUMPTION_* to 0 to disable the mechanism

# Implementation Steps

- [ ] 1. Design config hierarchy in config.py:
  - `AssumptionCopingConfig`: resource_reward (0.75), resource_penalty (0.10), pf_allocation_fraction (0.30), affect_improvement_scale (0.2), affect_deterioration_scale (0.4), resilience_improvement_scale (0.1), resilience_deterioration_scale (0.2), challenge_success_resilience (0.3), challenge_failure_resilience (-0.1), hindrance_success_resilience (0.1), hindrance_failure_resilience (-0.4), success_stress_reduction (0.2), failure_stress_increase (0.3), success_affect_change (0.1), failure_affect_change (-0.2)
  - `AssumptionStressConfig`: controllability_challenge_weight (0.10), controllability_hindrance_weight (0.05), overload_challenge_weight (0.05), overload_hindrance_weight (0.10), baseline_controllability (0.5), baseline_overload (0.5), controllability_homeostasis_rate (0.05), overload_homeostasis_rate (0.05), event_intensity_challenge_weight (0.7), event_intensity_hindrance_weight (1.3), failed_coping_intensity_multiplier (1.5), stress_intensity_decay_rate (0.8), new_intensity_weight (0.2), momentum_increase_rate (0.1), momentum_decrease_rate (0.05), momentum_zero_threshold (0.01), momentum_decay_factor (0.9), pss10_estimation_base (10), pss10_controllability_max_effect (8), pss10_overload_max_effect (12), pss10_estimation_variance (3)
  - `AssumptionResourceConfig`: resilience_efficiency_factor (0.3), min_resource_threshold (0.05), coping_difficulty_scale (0.5), min_cost_floor (0.3), failed_coping_cost_penalty (1.3), max_efficiency_gain (0.5), social_resilience_boost_factor (0.1), support_exchange_benefit_weight (0.2), challenge_resilience_bonus_factor (0.2), hindrance_resilience_bonus_factor (0.1), overload_allocation_penalty_rate (0.1), stress_improvement_effectiveness (0.1), social_resource_boost_factor (0.1), preservable_allocation_fraction (0.1), social_support_allocation_boost (0.3), affect_regeneration_multiplier (0.5), resilience_regeneration_multiplier (0.3)
  - `AssumptionSocialConfig`: support_exchange_threshold (0.05), social_support_probability (0.3), social_support_exchange_boost (0.1)
  - `AssumptionBufferingConfig`: resilience_low_threshold (0.3), resilience_high_threshold (0.7), volatility_beta_alpha (1), volatility_beta_beta (1), initial_protective_factor_values (0.5)
- [ ] 2. Harmonize 5 discrepancies:
  - resilience_efficiency_factor: pick 0.3 (affect_utils value, used in resource cost calculation)
  - resilience_allocation_bonus: pick 0.2 (more conservative than 0.5)
  - min_cost_floor: pick 0.3 (more conservative floor)
  - failed_coping_cost_penalty: pick 1.3 (larger penalty for failed coping)
  - event_difficulty formula: extract to shared function, not constant
- [ ] 3. Update .env.example with assumption sections
- [ ] 4. Replace hardcoded constants in stress_utils.py, affect_utils.py, resource_utils.py, agent.py, model.py with config references
- [ ] 5. Verify: all defaults unchanged -> all existing tests pass
- [ ] 6. Run: `pytest src/python/tests/ -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Missed constant changes behavior | Med | High | grep for remaining float constants in source; diff behavioral output before/after |
| Harmonizing discrepancies changes behavior | Med | Med | Document each discrepancy choice; test behavioral equivalence with baseline |
| Config hierarchy too deep | Low | Low | Flat ASSUMPTION_* env vars with dotted access in config dataclass |

# UAT

1. `pytest src/python/tests/ -v` passes (same defaults -> same behavior)
2. Every hardcoded constant replaced: `grep -r '\* 0\.[0-9]' src/python/` returns only loop indices and benchmark values
3. `python -c "from src.python.config import get_config; c = get_config(); print(c.assumptions)"` shows all 84 vars with correct defaults
4. .env.example has complete `# ── Assumption: ` sections

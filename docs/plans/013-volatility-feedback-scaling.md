---
title: Volatility, Feedback, Homeostatic Scaling
description: Formalize volatility param, relocate PSS-10 feedback, extract homeostatic rate scaling
date: 2026-06-05
---

# Overview

Cover 3 medium-severity gaps. The volatility parameter exists (agent.py:174) but is hardcoded and not in AgentState. PSS-10 feedback and validate_theoretical_correlations calls in stressful_event() will be orphaned when Plan 008 removes that method. Homeostatic rate scaling (agent.py:336-342) is inline and not extracted.

# Goals

- `volatility` in AgentState TypedDict and config (parameterized as assumption)
- PSS-10 feedback loop relocated to consolidation phase; validation moved to test code
- `scale_homeostatic_rate()` extracted as pure function in orchestrator
- All existing tests pass -- no behavioral regression

# Implementation Steps

## Step 1: Formalize Volatility Parameter

- [ ] 1a. Add `volatility: float` field to `AgentState` TypedDict in `phases/interfaces.py` (created by Plan 000)
- [ ] 1b. Add `ASSUMPTION_STRESS_VOLATILITY_ALPHA` (default `1.0`) and `ASSUMPTION_STRESS_VOLATILITY_BETA` (default `1.0`) to `AssumptionStressConfig` in `config.py` (Plan 007)
- [ ] 1c. Update `initialize_volatility()` (Plan 012, initialization.py) to read alpha/beta from config instead of hardcoded `beta(1, 1)`
- [ ] 1d. Ensure `volatility` is written back in `_write_back_state()` (Plan 008)
- [ ] 1e. Write test: volatility mean ~0.5 with alpha=1, beta=1; volatility mean ~0.8 with alpha=4, beta=1
- [ ] 1f. Update `ARCHITECTURE_MODEL.md` agent state table to include `volatility` with range [0,1]

## Step 2: Relocate PSS-10 Feedback and Validation

- [ ] 2a. Move `update_stress_dimensions_from_pss10_feedback()` call from `agent.py:stressful_event()` into `process_pss10_consolidation()` (orchestrator-internal phase from Plan 008)
- [ ] 2b. Move `validate_theoretical_correlations()` call from `stressful_event()` to `test_correlation_validation.py` -- it is a runtime assertion, not production control flow
- [ ] 2c. Write test: PSS-10 feedback loop correctly updates stress dimensions from generated responses
- [ ] 2d. Remove `@pytest.mark.flaky` from `test_resilience_affect_positive_correlation` (post-relocation correlation should be stable)

## Step 3: Extract Homeostatic Rate Scaling

- [ ] 3a. Extract `scale_homeostatic_rate(base_rate: float, resources: float, current_stress: float) -> float` as pure function in a utility module
  - Resources modulate: `rate * (0.5 + 0.5 * resources)` (low resources = slower homeostasis)
  - Stress modulates: `rate * (1.0 + current_stress)` (high stress = faster pull to baseline)
- [ ] 3b. Call `scale_homeostatic_rate()` inside `process_affect_dynamics()` and `process_daily_reset()` (Plan 008 orchestrator-internal phases)
- [ ] 3c. Write test: resources=0 -> rate halved; resources=1 -> rate unchanged; stress=1 -> rate doubled; stress=0 -> rate unchanged

## Step 4: Verification

- [ ] 4. Run: `pytest src/python/tests/ -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Volatility config changes population variance | Med | Med | Default values alpha=1, beta=1 produce same Beta(1,1) as before |
| Moving feedback logic changes PSS-10 timing | Med | High | Feedback occurs in same consolidation step; verify same seed output |
| Homeostatic scaling formula not documented in math docs | Low | Low | Add formula to step-calculations.md and MATH_NOTATION.md |

# UAT

1. `python -c "from src.python.phases.interfaces import AgentState; assert 'volatility' in AgentState.__annotations__"`
2. `grep -r "validate_theoretical_correlations" src/python/agent.py` returns nothing
3. `grep -r "scale_homeostatic_rate" src/python/` returns exactly 1 definition (in utility module)
4. `pytest src/python/tests/ -v` passes

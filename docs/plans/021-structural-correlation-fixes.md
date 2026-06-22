# Implementation Plan — 7 Structural Fixes

## Overview
Fix structural issues causing 21 xfail correlation tests. No parametric changes — only mechanism design changes.

## Files Modified (source)
- `src/python/affect_utils.py` — Fix 3 (stress→affect pathway)
- `src/python/agent.py` — Fix 3, 4, 5, 2a, 6 (orchestration + consolidation)
- `src/python/model.py` — Fix 1 (resource floor)
- `src/python/stress_utils.py` — Fix 4 (rename compute_stress_from_pss10→compute_stress_from_dimensions)

## Files Modified (test — name-only)
- `test_correlation_validation.py` — rename import
- `test_pss10_empirical.py` — rename import
- `test_resilience_activation_theory.py` — rename import
- `test_stress_utils.py` — rename import
- `test_complete_stress_processing_loop.py` — rename import/patch
- `test_agent_pss10_integration.py` — rename import
- `test_daily_reset_functionality.py` — rename import

## Implementation Order
1. Fix 4: Rename `compute_stress_from_pss10` → `compute_stress_from_dimensions` in source + test imports
2. Fix 6: Replace `append_daily_pss10_score` call with direct `scores.append()` in agent.py subevent loop
3. Fix 5 + Fix 4 cadence: Modify `process_pss10_consolidation` — add dynamic resilience coupling, compute current_stress daily with α=0.30
4. Fix 3: Add stress→affect pathway in affect_utils.py + agent.py
5. Fix 2a: Remove duplicate resource regeneration from process_affect_dynamics
6. Fix 1: Add soft resource floor in model.py step()
7. Remove xfail markers from tests that now pass
8. Fix test_pss10_comprehensive.py stale defaults

## UAT
- `pixi run test` — all tests pass with 0 failures
- Previously xfailed tests are removed from xfail and pass

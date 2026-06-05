---
title: Pre-Extraction Fixes & Harmonization
description: Fix 3 latent bugs, add missing init, consolidate duplicate dataclasses, harmonize 5 constant discrepancies before phase extraction
date: 2026-06-05
---

# Overview

Clean the codebase before any phase extraction. Three bugs would propagate into phase functions if extracted as-is. Five constant discrepancies (same concept, different values across files) must be harmonized to a single source of truth. This plan affects `agent.py`, `stress_utils.py`, `affect_utils.py`, and `resource_utils.py` only — no new files.

# Goals

- Fix 3 bugs that would corrupt phase extraction
- Add `stress_breach_count` to `Person.__init__`
- Consolidate duplicate `ResourceOptimizationConfig` dataclass
- Harmonize 5 constant discrepancies across files
- Extract shared overload formula `ch*0.7 + hi*1.3` to a named function
- `pytest src/python/tests/ -v` passes with same output as before

# Implementation Steps

## Step 1: Fix 3 Bugs

- [ ] 1a. `_update_recent_stress_intensity(str_utils.py:957-992)` — accepts `recent_stress_intensity` and `stress_momentum` as parameters but ignores them; resets to zero each call. Fix: thread the parameters through instead of hardcoding `0.0`.
- [ ] 1b. `determine_coping_outcome_and_psychological_impact(affect_utils.py:867-928)` — line 898 uses `np.random.random()` instead of injected `rng.random()`. Fix: replace with `rng.random()` (callers already pass `rng`).
- [ ] 1c. `validate_theoretical_correlations(stress_utils.py:1096-1177)` — line 1164 `if pss10_responses := {}` uses walrus operator that creates a falsy empty dict, shadowing the parameter. Fix: replace with proper `if pss10_responses:` using the parameter directly. (Also remove the dead dict literal `{}` on line 1164.)

## Step 2: Fix Missing Initializations

- [ ] 2a. Add `self.stress_breach_count = 0` to `Person.__init__` (currently set via `getattr` with default 0 in `stressful_event()` line 681).
- [ ] 2b. Add `self._adapted_network = False` to `Person.__init__` (referenced in `Model._apply_network_adaptation()` line 321).

## Step 3: Consolidate Duplicate Dataclass

- [ ] 3a. Remove `ResourceOptimizationConfig` from `affect_utils.py:559-567` (import it from `resource_utils` instead).
- [ ] 3b. Update `affect_utils.py` imports to use `from src.python.resource_utils import ResourceOptimizationConfig`.
- [ ] 3c. Update `resource_utils.py:ResourceOptimizationConfig` to include all fields used by `affect_utils.py`.
- [ ] 3d. Fix `resource_utils.py:allocate_resilience_optimized_resources` line 291: `config.get("utility", "softmax_temperature")` — `ResourceOptimizationConfig` is a dataclass, not a dict. Replace with `get_config().get("utility", "softmax_temperature")`.

## Step 4: Harmonize 5 Constant Discrepancies

Pick the more conservative value for each:

| Concept | Old File 1 | Old File 2 | New Value |
|---------|-----------|-----------|-----------|
| `resilience_efficiency_factor` | 0.3 (affect_utils) | 0.15 (resource_utils) | **0.3** |
| `failed_coping_cost_penalty` | 1.3 (affect_utils`compute_resource_depletion`) | 1.1 (resource_utils`compute_resource_depletion`) | **1.3** |
| `min_cost_floor` | 0.3 (affect_utils max() arg) | 0.1 (resource_utils max() arg) | **0.3** |
| `challenge_resilience_bonus` | 0.2 (both) | 0.2 (both) | No change |
| `hindrance_resilience_bonus` | 0.1 (both) | 0.1 (both) | No change |

- [ ] 4a. Set `resource_utils.py:ResourceOptimizationConfig.resilience_efficiency_factor` to `0.3`.
- [ ] 4b. Set `resource_utils.py:compute_resource_depletion_with_resilience` penalty to `1.3`.
- [ ] 4c. Set `resource_utils.py:compute_resource_depletion_with_resilience` min cost floor to `0.3`.
- [ ] 4d. Update `affect_utils.py:compute_resource_depletion_with_resilience` to match after consolidation.

## Step 5: Extract Shared Overload Formula

- [ ] 5a. Add `compute_event_difficulty(challenge, hindrance)` to `stress_utils.py` returning `challenge * 0.7 + hindrance * 1.3`.
- [ ] 5b. Replace inline formula in `_update_recent_stress_intensity` (stress_utils.py:972) with call to shared function.
- [ ] 5c. Replace inline formula in `compute_resilience_optimized_resource_cost` (affect_utils.py:596 / resource_utils.py:172) with call to shared function.

# Verification

- [ ] 6. Run `pytest src/python/tests/ -v` — all tests pass, no behavioral change.
- [ ] 7. Confirm no `np.random` calls remain in `affect_utils.py` (should use injected `rng`).
- [ ] 8. Confirm `stress_breach_count` and `_adapted_network` appear in `Person.__init__`.

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Harmonizing constant changes numerical output | Med | Med | Verify test suite passes; diff before/after for same seed |
| Consolidating dataclass breaks import | Low | High | Grep for all `ResourceOptimizationConfig()` usages before/after |
| Removing duplicate function leaves callers orphaned | Low | High | Check both files have identical `compute_resource_depletion_with_resilience` signatures |

# UAT

1. `pytest src/python/tests/ -v` passes (100%)
2. `grep -r "np\.random\.random()" src/python/affect_utils.py` returns nothing
3. `grep -r "class ResourceOptimizationConfig" src/python/` returns exactly 1 match
4. `grep -r "challenge \* 0\.7 \+ hindrance \* 1\.3" src/python/` returns 0 (replaced by function call)

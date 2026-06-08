---
title: Network Adaptation, Init, DataCollector
description: Implement network rewiring, extract agent init, audit DataCollector reporters
date: 2026-06-05
---

# Overview

Cover 3 high-severity gaps identified in the ADR/architecture cross-reference. Network adaptation (ADR-008) is currently a stub -- implement actual topology rewiring. Agent initialization (agent.py:79-174) is untested inline code -- extract to pure functions. DataCollector reporters (30 lambdas) may break after phase refactoring -- audit and fix.

# Goals

- `_apply_network_adaptation()` modifies network topology (rewiring, homophily)
- Agent init extracted to testable `initialization.py` functions
- DataCollector reporters updated for post-phase state variable names
- All existing tests pass -- no behavioral regression

# Implementation Steps

## Step 1: Implement Network Adaptation

- [ ] 1a. Create `src/python/network_utils.py` with:
  - `build_watts_strogatz_network(N, k, p, rng)` -- extract from model.py:91
  - `compute_connection_similarity(state_i, state_j)` -- homophily score from affect, resilience
  - `compute_connection_retention_probability(similarity, support_effectiveness, homophily_strength)`
  - `apply_stress_adaptation(G, agents, config, rng)` -- rewires edges when stress_breach_count >= threshold
- [ ] 1b. Rewrite `model.py:_apply_network_adaptation()` to call `apply_stress_adaptation()` with actual rewiring logic
- [ ] 1c. Write `test_network_adaptation_theory.py`:
  - Watts-Strogatz correct topology (clustering, path length) for given k, p, N
  - Rewiring triggers only when breach_count >= config threshold
  - Rewired edges favor similar agents (homophily test)
  - Support effectiveness increases retention probability
  - Clustering preserved within 10% of original after adaptation
  - No edges created to self
- [ ] 1d. Add `ASSUMPTION_NETWORK_ADAPTATION_SIMILARITY_STRESS_WEIGHT`, `ASSUMPTION_NETWORK_ADAPTATION_SIMILARITY_AFFECT_WEIGHT`, `ASSUMPTION_NETWORK_ADAPTATION_SIMILARITY_RESILIENCE_WEIGHT` to config

## Step 2: Extract Agent Initialization

- [ ] 2a. Create `src/python/initialization.py` with pure functions:
  - `initialize_baseline_resilience(rng: Generator, config: dict) -> float`
  - `initialize_baseline_affect(rng: Generator, config: dict) -> float`
  - `initialize_resources(rng: Generator, config: dict) -> float`
  - `initialize_protective_factors() -> Dict[str, float]` (constant 0.5 each)
  - `initialize_volatility(rng: Generator, config: dict) -> float` (Beta distribution, configurable alpha/beta)
  - `initialize_pss10_state(rng, config) -> PSS10State` (responses, dims, total)
  - `compute_initial_stress(pss10_score, controllability, overload) -> float`
- [ ] 2b. Rewrite `Person.__init__()` to call these functions instead of inline logic
- [ ] 2c. Write `test_agent_initialization_theory.py`:
  - Each output in correct range: resilience [0,1], affect [-1,1], resources [0,1], volatility [0,1]
  - Distribution moments match config (mean, std via repeated draws)
  - Same seed -> same initialization (reproducibility)
  - PSS-10 total in [0,40] with correct reverse scoring for items 4,5,7,8
  - Protective factors all 0.5

## Step 3: Audit and Update DataCollector

- [ ] 3a. Create `src/python/reporters.py` extracting all reporter lambdas from model.py:126-201 as named functions
- [ ] 3b. Audit each reporter against phase-state variable names (post-Plans 002-008):
  - Model reporters (22): `avg_pss10`, `avg_resilience`, `avg_affect`, `coping_success_rate`, `avg_resources`, `avg_stress`, `social_support_rate`, `stress_events`, `network_density`, `stress_prevalence`, `low_resilience`, `high_resilience`, `avg_challenge`, `avg_hindrance`, `challenge_hindrance_ratio`, `avg_consecutive_hindrances`, `total_stress_events`, `successful_coping`, `social_interactions`, `support_exchanges`, `total_interactions`, `social_support_exchanges`
  - Agent reporters (8): `pss10`, `resilience`, `affect`, `resources`, `current_stress`, `stress_controllability`, `stress_overload`, `consecutive_hindrances`
- [ ] 3c. Fix any lambda references that would break after Plan 008 orchestrator refactoring
- [ ] 3d. Write `test_datacollector_integration.py`:
  - All reporters return correct types and ranges
  - DataCollector produces DataFrame with all expected columns
  - Same seed -> same collected data before and after refactoring

## Step 4: Verification

- [ ] 4. Run: `pytest src/python/tests/ -v` -- all tests pass

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Network adaptation changes emergent behavior | High | High | Document as new feature; compare pre/post correlation targets |
| Init extraction changes numerical output | Med | High | Same seed -> same init; verify with __init__ before/after diff |
| DataCollector lambdas have implicit state dependencies | Low | High | Extract to named functions with explicit state parameter |

# UAT

1. `model._apply_network_adaptation()` rewires at least one edge when all agents breach threshold
2. `initialization.initialize_baseline_resilience(rng, config)` returns float in [0,1]
3. `pytest src/python/tests/test_datacollector_integration.py -v` passes
4. `python simulate.py --seed 42` produces same output before and after changes

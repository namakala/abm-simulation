---
title: Model Architecture — Detailed Design
description: Agent state variables, phase pipeline decomposition, assumption parameter reference
date: 2025-06-04
---

# Phase Pipeline Decomposition

| Phase | Frequency | Inputs (from config) | Outputs (state_delta keys) | State Vars Modified |
|-------|-----------|----------------------|----------------------------|---------------------|
| stress_perception | event_driven | omega_c, omega_o, bias, gamma, base_threshold, challenge_scale, hindrance_scale | challenge, hindrance, is_stressed, event_controllability, event_overload, stress_controllability, stress_overload, recent_stress_intensity, stress_momentum | transient event fields + stress dimensions |
| resilience_activation | event_driven | neighbor_affects, base_resource_cost | affect, resilience, current_stress, resources, protective_factors, stress_controllability, stress_overload, consecutive_hindrances, stress_breach_count, pss10, pss10_responses, stressed | core traits + stress + PSS-10 |
| interaction | event_driven | influence_rate, resilience_influence | affect, resilience, current_stress, resources, protective_factors, daily_interactions, daily_support_exchanges | social metrics |
| resource_allocation | daily | (none; uses assumptions) | resources, protective_factors | resources + PF |
| stress_buffering | daily | (none; uses assumptions) | current_stress, protective_factors | stress decay |

# State Variable Table

| Key | Range | Description | Modified By |
|-----|-------|-------------|-------------|
| baseline_resilience / resilience | [0, 1] | Core coping capacity | resilience_activation |
| baseline_affect / affect | [-1, 1] | Emotional valence | resilience_activation, interaction |
| resources | [0, 1] | Finite psychological capacity | resilience_activation, resource_allocation |
| protective_factors | dict[0,1]^4 | Social support, family, formal, psych capital | resilience_activation, resource_allocation, stress_buffering |
| current_stress | [0, 1] | Current distress level | resilience_activation, stress_buffering |
| stress_controllability | [0, 1] | Perceived control over stress | stress_perception, resilience_activation |
| stress_overload | [0, 1] | Perceived demands exceeding capacity | stress_perception, resilience_activation |
| recent_stress_intensity | float | Decaying intensity accumulator | stress_perception |
| stress_momentum | float | Stress change momentum | stress_perception |
| pss10 | 0-40 | PSS-10 total score | resilience_activation |
| pss10_responses | dict | 10 item responses (0-4 each) | resilience_activation |
| consecutive_hindrances | int | Consecutive hindrance event count | resilience_activation |
| stress_breach_count | int | Total threshold breaches | resilience_activation |
| daily_interactions | int | Interactions today | interaction |
| daily_support_exchanges | int | Support exchanges today | interaction |
| volatility | float | Personality trait (inherited) | (read-only) |

# Assumption Parameters

All tunable parameters live in `src/python/assumption_config.py` with ASSUMPTION_* env overrides.

| Parameter | Default | Description | Used By |
|-----------|---------|-------------|---------|
| ASSUMPTION_COPING_SOCIAL_SUPPORT_FACTOR | 0.30 | Weight of social support in coping | resilience_activation |
| ASSUMPTION_COPING_SUPPORT_BOOST_FACTOR | 0.15 | Additional support from interactions | resilience_activation |
| ASSUMPTION_RESILIENCE_COPING_FACTOR | 0.20 | Resilience contribution to coping | resilience_activation |
| ASSUMPTION_RESOURCE_PENALTY | 0.05 | Failed coping resource cost | resilience_activation |
| ASSUMPTION_FAILED_COPING_COST_PENALTY | 0.10 | Failed coping affect penalty | resilience_activation |
| ASSUMPTION_AFFECT_DETERIORATION_SCALE | 0.30 | Hindrance affect deterioration | resilience_activation |
| ASSUMPTION_AFFECT_REGENERATION_MULTIPLIER | 0.50 | Daily affect regeneration rate | stress_buffering |
| ASSUMPTION_RESILIENCE_IMPROVEMENT_SCALE | 0.10 | Successful coping resilience gain | resilience_activation |
| ASSUMPTION_PF_ALLOCATION_FRACTION | 0.05 | Daily PF allocation budget | resource_allocation |
| ASSUMPTION_CONTROLLABILITY_HOMEOSTASIS_RATE | 0.02 | Controllability drift toward 0.5 | stress_perception |
| ASSUMPTION_OVERLOAD_HOMEOSTASIS_RATE | 0.02 | Overload drift toward 0.5 | stress_perception |
| ASSUMPTION_STRESS_DECAY_RATE | 0.08 | Daily stress decay rate | stress_buffering |

See `@docs/ADR/008-model-architecture.md` for rationale.

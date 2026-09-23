## Daily Consolidation

Runs once per day after the subevent loop. Applies homeostatic dynamics,
resource regeneration, stress buffering, PSS-10 consolidation, and daily
reset of tracking variables.

## Affect Dynamics

Applies daily affect and resilience dynamics including peer influence, event
appraisal effects, homeostasis, protective factor boosts, and social
resilience optimisation.

**Algorithm:**

```
FUNCTION process_affect_dynamics(state, config, rng):
    // 1. Affect dynamics (homeostasis + peer influence + event appraisal)
    new_affect ← update_affect_dynamics(
        current_affect, baseline_affect, neighbor_affects,
        daily_challenge, daily_hindrance, current_stress, resources
    )
    
    // 2. Resilience dynamics + PF boost
    new_resilience ← update_resilience_dynamics(
        current_resilience, consecutive_hindrances
    )
    protective_boost ← get_resilience_boost(protective_factors, baseline, current)
    new_resilience ← min(1, new_resilience + protective_boost)
    
    // 3. Social resilience optimisation
    new_resilience ← integrate_social_resilience_optimization(...)
    
    // 4. Interaction-frequency boost
    new_resilience += daily_interactions × 0.005
    
    // 5. Consecutive hindrance decay
    new_consecutive_hindrances ← max(0, hindrances - decay_rate)
    
    // 6. Homeostatic adjustment
    new_affect ← compute_homeostatic_adjustment(baseline, new_affect, rate)
    new_resilience ← compute_homeostatic_adjustment(baseline, new_resilience, rate)
    
    RETURN PhaseOutput(state_delta, observation)
```

Reference: [src/python/agent.py:L67-L210](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/agent.py#L67-L210)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `state` | Current agent state | — |
| `config` | `neighbor_affects`, `daily_challenge`, `daily_hindrance`, `stress_decay_rate` | — |
| `rng` | Seeded RNG | — |

Returns: `PhaseOutput` with `state_delta` (affect, resilience, resources,
consecutive_hindrances) and `observation`.

## Resource Allocation

Regenerates resources and allocates to protective factors via softmax.
Resource regeneration is linear in the deficit, modulated by affect and
resilience. Allocation uses softmax with temperature for bounded-rational
distribution.

**Algorithm:**

```
FUNCTION run_resource_allocation(state, config, rng):
    // 1. Resource regeneration
    affect_mult ← 1 + 0.5 × max(0, affect)
    resil_mult ← 1 + 0.3 × resilience
    regeneration ← base_rate × (1 - resources) × affect_mult × resil_mult
    
    // 2. Softmax allocation across protective factors
    available ← resources + regeneration
    spendable ← available × (1 - preservable_fraction)
    weights ← softmax(efficacies / temperature)
    allocations ← spendable × weights
    
    // 3. PF efficacy updates (diminishing returns)
    FOR each factor f:
        Δe_f ← allocations[f] × rate × (1 - e_f) × efficiency_gain
        e_f ← min(1, e_f + Δe_f)
    
    // 4. Remaining resources
    new_resources ← preserved + (spendable - total_allocated)
    
    RETURN PhaseOutput(state_delta: {resources, protective_factors}, observation)
```

Reference: [src/python/phases/resource_allocation.py:L141-L219](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/resource_allocation.py#L141-L219)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `state` | Current agent state (`resources`, `affect`, `resilience`, `protective_factors`) | — |
| `config` | `base_regeneration`, `softmax_temperature`, `protective_improvement_rate` | — |
| `rng` | Seeded RNG | — |

Returns: `PhaseOutput` with `state_delta` (resources, protective_factors)
and `observation` (regeneration_amount, allocation_weights).

## Stress Buffering

Applies protective-factor resilience boost and resource mediation of stress
buffering using Baron & Kenny mediation framework: a-path (stress $\to$
resources), b-path (resources $\to$ buffering), c'-path (stress $\to$
buffering $|$ resources).

**Algorithm:**

```
FUNCTION run_stress_buffering(state, config, rng):
    // 1. PF boost to resilience
    resilience_need ← max(0, baseline - current)
    pf_boost ← SUM(efficacy × need × rate) for each factor
    pf_boost ← min(pf_boost, resilience_need)
    new_resilience ← current + pf_boost
    
    // 2. Resource mediation (a-path: stress → resources)
    effective_stress ← stress × (1 + 0.2 × overload)
    resource_depletion ← a_coefficient × effective_stress
    new_resources ← resources + resource_depletion
    
    // 3. Buffering strength (b-path + c'-path)
    buffering ← max(0, b × new_resources + c' × stress)
    
    RETURN PhaseOutput(state_delta: {resilience, resources}, observation)
```

Reference: [src/python/phases/stress_buffering.py:L42-L148](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/stress_buffering.py#L42-L148)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `state` | Current agent state (`resilience`, `baseline_resilience`, `protective_factors`, `current_stress`, `resources`) | — |
| `config` | `boost_rate`, `a_coefficient`, `b_coefficient`, `c_prime_coefficient` | — |
| `rng` | Seeded RNG | — |

Returns: `PhaseOutput` with `state_delta` (resilience, resources) and
`observation` (pf_boost, buffering_strength, indirect_effect).

## PSS-10 Consolidation

Consolidates daily PSS-10 scores, updates stress level via exponential
smoothing, and applies post-hoc adjustments for resilience coupling,
resource coupling, and mood-congruent appraisal.

**Algorithm:**

```
FUNCTION process_pss10_consolidation(state, config, rng):
    // 1. Update stress dimensions from PSS-10 feedback
    controllability, overload ← update_stress_dimensions_from_pss10_feedback(...)
    
    // 2. Compute current_stress from dimensions
    new_stress ← compute_stress_from_dimensions(controllability, overload, ...)
    current_stress ← 0.5 × new_stress + 0.5 × current_stress    // smoothing
    
    // 3. Consolidate PSS-10 score
    IF daily_scores not empty:
        consolidated ← mean(daily_scores)
    ELSE:
        consolidated ← regenerate_from_dimensions(...)
    
    // 4. Exponential smoothing across days
    smoothed ← α × consolidated + (1 - α) × prev_smoothed
    
    // 5. Post-hoc adjustments
    final ← smoothed + bias - 0.5 × coupling × (resilience - 0.5)
    final ← 20 + (final - 20) × 1.2                  // variance stretch
    final += (0.5 - resources) × 12.0                  // resource penalty
    stressed ← (final ≥ threshold)
    
    RETURN PhaseOutput(state_delta: {pss10, current_stress, stressed, ...})
```

Reference: [src/python/agent.py:L212-L348](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/agent.py#L212-L348)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `state` | Current agent state (`daily_pss10_scores`, `current_stress`, `pss10_smoothed`) | — |
| `config` | `pss10_threshold` | — |
| `rng` | Seeded RNG | — |

Returns: `PhaseOutput` with `state_delta` (pss10, pss10_smoothed,
current_stress, stress_controllability, stress_overload, stressed,
daily_pss10_scores cleared).

## Daily Reset

Clears daily tracking counters, applies affect reset toward baseline,
and decays stress and consecutive hindrances.

**Algorithm:**

```
FUNCTION process_daily_reset(state, config, rng):
    // 1. Affect reset toward baseline
    scaled_rate ← scale_homeostatic_rate(affect_rate, resources, stress)
    new_affect ← compute_daily_affect_reset(affect, baseline_affect, scaled_rate)
    
    // 2. Stress decay
    new_stress ← compute_stress_decay(stress, scaled_rate)
    
    // 3. Hindrance decay
    new_consecutive_hindrances ← max(0, hindrances - 0.05)
    
    // 4. Clear daily counters
    daily_interactions ← 0
    daily_support_exchanges ← 0
    daily_stress_events ← []
    daily_pss10_scores ← []
    
    RETURN PhaseOutput(state_delta: {affect, stress, counters cleared})
```

Reference: [src/python/agent.py:L350-L412](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/agent.py#L350-L412)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `state` | Current agent state | — |
| `config` | `current_day` (int) | — |
| `rng` | Seeded RNG | — |

Returns: `PhaseOutput` with `state_delta` (affect, stress, counters cleared)
and `observation` (stress_summary).

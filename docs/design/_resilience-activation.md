## Resilience Activation

### Purpose

Determine coping outcome and update resilience after a stressed event.
Implements Lazarus secondary appraisal and coping response. Runs the full
coping pipeline: neighbor influence, coping probability, coping outcome,
affect/resilience/stress changes, PSS-10 generation, resource cost, and
protective factor allocation.

- **Frequency:** event_driven
- **Inputs:** challenge, hindrance (from stress perception), affect, resilience, resources, protective_factors
- **Outputs:** state_delta: affect, resilience, current_stress, resources, protective_factors, pss10, stressed (12 keys)
- **Observation:** coped_successfully, coping_probability, resource_cost

### Algorithm

```
FUNCTION run_resilience_activation(state, config, rng):
    // Step 1: Coping outcome
    new_affect, new_resilience, new_stress, coped ←
        determine_coping_outcome_and_psychological_impact(
            current_affect, current_resilience, current_stress,
            challenge, hindrance, neighbor_affects, rng,
            config, social_support_efficacy, support_boost
        )

    // Step 2: Update stress dimensions from event
    updated_controllability, updated_overload, updated_intensity, updated_momentum ←
        update_stress_dimensions_from_event(
            current_controllability, current_overload,
            challenge, hindrance, coped, is_stressful=True,
            volatility, recent_stress_intensity, stress_momentum, resilience
        )

    // Step 3: Generate PSS-10 from updated dimensions
    pss10_data ← generate_pss10_from_stress_dimensions(
        updated_controllability, updated_overload,
        updated_intensity, updated_momentum,
        new_affect, current_resources, new_resilience, rng
    )

    // Step 4: Resource cost and depletion
    optimized_cost ← compute_resilience_optimized_resource_cost(
        base_cost, new_resilience, challenge, hindrance, config
    )
    new_resources ← compute_resource_depletion_with_resilience(
        current_resources, optimized_cost, new_resilience, coped, is_stressed=True, config
    )

    // Step 5: Track consecutive hindrances
    new_consecutive_hindrances ← consecutive_hindrances + 1.0 IF hindrance > challenge ELSE 0.0

    // Step 6: Increment stress breach count
    new_stress_breach_count ← stress_breach_count + 1

    // Step 7: PF allocation + resource reward (if coped)
    new_protective_factors ← dict(protective_factors)
    IF coped:
        reward ← 0.03
        new_resources ← clamp(new_resources + reward, 0.0, 1.0)
        allocations ← allocate_protective_factors(
            new_resources × pf_allocation_fraction,
            new_resilience, baseline_resilience, protective_factors, rng
        )
        new_protective_factors ← update_protective_factors_with_allocation(
            protective_factors, allocations, new_resilience
        )
        new_resources ← clamp(new_resources - sum(allocations.values()), 0.0, 1.0)

    // Build PhaseOutput
    state_delta ← {
        affect: new_affect, resilience: new_resilience,
        current_stress: new_stress,
        stress_controllability: updated_controllability,
        stress_overload: updated_overload,
        resources: new_resources,
        protective_factors: new_protective_factors,
        consecutive_hindrances: new_consecutive_hindrances,
        stress_breach_count: new_stress_breach_count,
        pss10: pss10_data.pss10_score,
        pss10_responses: pss10_data.pss10_responses,
        stressed: pss10_data.stressed
    }

    observation ← {
        coped_successfully: coped,
        coping_probability, resilience_effect,
        delta_stress: new_stress - current_stress,
        delta_affect: new_affect - current_affect,
        resource_cost: optimized_cost,
        resource_reward: 0.03 IF coped ELSE None
    }

    RETURN PhaseOutput(state_delta, observation)
```

### Parameters

| Name | Description | Default | Source |
|------|-------------|---------|--------|
| `base_resource_cost` | Cost per coping attempt | 0.1 | config |
| `pf_allocation_fraction` | Resources allocated to PF on success | 0.15 | assumption |
| `resource_reward` | Flat reward for successful coping | 0.03 | assumption |

Reference: [src/python/phases/resilience_activation.py:L42-L252](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/resilience_activation.py#L42-L252)

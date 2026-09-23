## Stress Perception

Transforms raw stress events into appraised challenge/hindrance scores and
determines whether the event exceeds the agent's adaptive threshold.

**Algorithm:**

```
FUNCTION run_stress_perception(state, config, rng):
    event ← generate_stress_event(rng)
    weights ← AppraisalWeights(omega_c, omega_o, bias, gamma)
    challenge, hindrance ← apply_weights(event, weights)
    appraised_stress ← compute_appraised_stress(event, challenge, hindrance)
    is_stressed ← evaluate_stress_threshold(appraised_stress, challenge, hindrance)

    updated_dims ← update_stress_dimensions(
        current_controllability, current_overload,
        challenge, hindrance, volatility, resilience
    )
    
    RETURN PhaseOutput(
        state_delta: {challenge, hindrance, is_stressed, updated_dims},
        observation: {event attrs, appraisal values}
    )
```

Reference: [src/python/phases/stress_perception.py:L28-L130](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/stress_perception.py#L28-L130)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `state` | Current agent state | — |
| `config` | Phase configuration (`omega_c`, `omega_o`, `bias`, `gamma`, `delta`, thresholds) | — |
| `rng` | Seeded RNG | — |

Returns: `PhaseOutput` with `state_delta` (challenge, hindrance, is_stressed,
updated stress dimensions) and `observation` (appraisal values, threshold).

## Resilience Activation

Determines coping outcome and updates resilience after a stressed event.
Implements the full coping pipeline: neighbour influence, coping probability,
coping outcome, affect/resilience/stress changes, PSS-10 generation, resource
cost, and protective factor allocation.

**Algorithm:**

```
FUNCTION run_resilience_activation(state, config, rng):
    // Step 1: Coping outcome
    coped ← determine_coping_outcome(
        challenge, hindrance, neighbor_affects,
        resilience, social_support_efficacy, support_boost
    )
    
    // Step 2: Update stress dimensions
    updated_dims ← update_stress_dimensions_from_event(...)
    
    // Step 3: Generate PSS-10 from updated dimensions
    pss10_data ← generate_pss10_from_stress_dimensions(updated_dims, ...)
    
    // Step 5: Resource cost and depletion
    cost ← compute_resilience_optimized_resource_cost(base_cost, resilience)
    new_resources ← compute_resource_depletion(resources, cost, resilience)
    
    // Step 9: PF allocation (if coped successfully)
    IF coped:
        allocations ← allocate_protective_factors(resources × fraction, ...)
        new_resources ← resources - sum(allocations)
    
    RETURN PhaseOutput(state_delta: {12 keys}, observation: {coping info})
```

Reference: [src/python/phases/resilience_activation.py:L42-L252](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/resilience_activation.py#L42-L252)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `state` | Current agent state (must include `challenge`, `hindrance`) | — |
| `config` | `neighbor_affects` (list[float]), `base_resource_cost` (float) | — |
| `rng` | Seeded RNG | — |

Returns: `PhaseOutput` with `state_delta` (12 keys: affect, resilience,
current_stress, resources, protective_factors, pss10, etc.) and `observation`
(coped_successfully, coping_probability, resource_cost).

## Social Interaction

Dyadic interaction between two agents. Converges affect and resilience with
negativity bias (negative influence 1.5x stronger), detects support from
convergence magnitude, and applies a win-win/lose-lose resource exchange
state machine.

**Algorithm:**

```
FUNCTION process_interaction(self_state, partner_state, config, rng):
    // Affect convergence with negativity bias
    ΔA_self ← influence_rate × partner_affect
    ΔA_partner ← influence_rate × self_affect
    IF ΔA_self < 0: ΔA_self ×= 1.5    // negativity bias
    IF ΔA_partner < 0: ΔA_partner ×= 1.5
    
    // Resilience convergence
    ΔR_self ← resilience_influence × partner_affect
    ΔR_partner ← resilience_influence × self_affect
    
    // Support detection
    total_convergence ← |ΔA_self| + |ΔA_partner| + |ΔR_self| + |ΔR_partner|
    support_occurred ← (total_convergence > threshold)
    
    // Resource exchange state machine
    IF self_stressed AND partner_stressed:
        IF support: both +boost
        ELSE: both -cost
    ELIF self_stressed AND NOT partner_stressed:
        IF support: self +boost
        ELSE: self -cost
    ELIF NOT self_stressed AND NOT partner_stressed:
        both PF social_support +small_boost
    
    RETURN (self_output, partner_output)
```

Reference: [src/python/phases/interaction.py:L48-L190](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/interaction.py#L48-L190)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `self_state` | Calling agent state | — |
| `partner_state` | Partner agent state | — |
| `config` | `influence_rate`, `resilience_influence`, `support_threshold`, `boost`, `cost` | — |
| `rng` | Seeded RNG | — |

Returns: `Tuple[PhaseOutput, PhaseOutput]` with `state_delta` (affect,
resilience, resources changes) and `observation` (support_occurred).

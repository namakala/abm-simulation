## Affect Dynamics

### Purpose

Applies daily affect and resilience dynamics including peer influence, event
appraisal effects, homeostasis, protective factor boosts, and social
resilience optimisation. Runs once per day after the subevent loop.

- **Frequency:** daily
- **Inputs:** affect, baseline_affect, resilience, resources, current_stress, neighbor_affects
- **Outputs:** state_delta: affect, resilience, consecutive_hindrances
- **Observation:** neighbor_affects_summary, protective_boost

### Algorithm

```
FUNCTION process_affect_dynamics(state, config, rng):
    // 1. Affect dynamics (homeostasis + peer influence + event appraisal)
    new_affect ← update_affect_dynamics(
        current_affect, baseline_affect, neighbor_affects,
        daily_challenge, daily_hindrance,
        current_stress, resources
    )

    // 2. Resilience dynamics + PF boost
    new_resilience ← update_resilience_dynamics(
        current_resilience, consecutive_hindrances
    )
    protective_boost ← get_resilience_boost(protective_factors, baseline, current)
    new_resilience ← min(1, new_resilience + protective_boost)

    // 3. Social resilience optimisation
    new_resilience ← integrate_social_resilience_optimization(
        new_resilience, daily_interactions, daily_support_exchanges,
        resources, baseline_resilience, protective_factors, rng
    )

    // 4. Interaction-frequency boost
    new_resilience += daily_interactions × 0.005

    // 5. Consecutive hindrance decay
    new_consecutive_hindrances ← max(0, hindrances - decay_rate)

    // 6. Homeostatic adjustment
    new_affect ← compute_homeostatic_adjustment(baseline, new_affect, rate)
    new_resilience ← compute_homeostatic_adjustment(baseline, new_resilience, rate)

    RETURN PhaseOutput(state_delta, observation)
```

### Parameters

| Name | Description | Default | Source |
|------|-------------|---------|--------|
| `affect_homeostatic_rate` | Affect return rate | 0.5 | assumption |
| `resilience_homeostatic_rate` | Resilience return rate | 0.5 | assumption |
| `interaction_boost_rate` | Per-interaction resilience boost | 0.005 | assumption |
| `hindrance_decay_rate` | Daily hindrance decay | 0.05 | assumption |

Reference: [src/python/agent.py:L67-L210](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/agent.py#L67-L210)

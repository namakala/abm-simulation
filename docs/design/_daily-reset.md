## Daily Reset

### Purpose

Clears daily tracking counters, applies affect reset toward baseline,
and decays stress and consecutive hindrances. Prepares agent state for
the next day's events.

- **Frequency:** daily
- **Inputs:** affect, baseline_affect, current_stress, consecutive_hindrances, daily counters
- **Outputs:** state_delta: affect, stress, counters cleared
- **Observation:** stress_summary

### Algorithm

```
FUNCTION process_daily_reset(state, config, rng):
    // 1. Affect reset toward baseline
    scaled_rate ← scale_homeostatic_rate(affect_rate, resources, stress)
    new_affect ← compute_daily_affect_reset(affect, baseline_affect, scaled_rate)

    // 2. Stress decay
    new_stress ← compute_stress_decay(stress, scaled_rate)

    // 3. Hindrance decay
    new_consecutive_hindrances ← max(0, hindrances - 0.05)

    // 4. Stress summary observation
    stress_summary ← {
        avg_stress: mean(daily_stress_events.stress_level),
        max_stress: max(daily_stress_events.stress_level),
        num_events: len(daily_stress_events),
        coping_success_rate: mean(daily_stress_events.coped_successfully)
    }

    // 5. Clear daily counters
    daily_interactions ← 0
    daily_support_exchanges ← 0
    daily_stress_events ← []
    daily_pss10_scores ← []

    RETURN PhaseOutput(state_delta: {affect, current_stress,
        daily_interactions, daily_support_exchanges,
        daily_stress_events, daily_pss10_scores,
        consecutive_hindrances, last_reset_day: current_day},
        observation: {stress_summary})
```

### Parameters

| Name | Description | Default | Source |
|------|-------------|---------|--------|
| `affect_homeostatic_rate` | Affect return rate | 0.5 | assumption |
| `hindrance_decay_rate` | Daily hindrance decay | 0.05 | assumption |

Reference: [src/python/agent.py:L350-L412](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/agent.py#L350-L412)

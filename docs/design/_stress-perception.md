## Stress Perception

### Purpose

Transform raw stress events into appraised challenge/hindrance scores and
determine whether the event exceeds the agent's adaptive threshold. Implements
Lazarus primary appraisal.

- **Frequency:** event_driven
- **Inputs:** stress_controllability, stress_overload, volatility, recent_stress_intensity, stress_momentum
- **Outputs:** state_delta: challenge, hindrance, is_stressed, event attrs, updated dimensions
- **Observation:** event attrs, appraisal values, threshold values

### Algorithm

```
FUNCTION run_stress_perception(state, config, rng):
    // 1. Generate stress event
    event ← generate_stress_event(rng)

    // 2. Build appraisal weights
    weights ← AppraisalWeights(omega_c, omega_o, bias, gamma)

    // 3. Compute challenge / hindrance
    challenge, hindrance ← apply_weights(event, weights)

    // 4. Compute appraised stress load
    appraised_stress ← compute_appraised_stress(event, challenge, hindrance, delta)

    // 5. Evaluate stress threshold
    threshold_params ← ThresholdParams(base_threshold, challenge_scale, hindrance_scale)
    is_stressed ← evaluate_stress_threshold(appraised_stress, challenge, hindrance, threshold_params)

    // 6. Update stress dimensions
    updated_controllability, updated_overload, new_intensity, new_momentum ←
        update_stress_dimensions_from_event(
            current_controllability, current_overload,
            challenge, hindrance, coped_successfully=True,
            is_stressful=is_stressed, volatility,
            recent_stress_intensity, stress_momentum, resilience
        )

    // 7. Build PhaseOutput
    state_delta ← {
        challenge, hindrance, is_stressed,
        event_controllability: event.controllability,
        event_overload: event.overload,
        stress_controllability: updated_controllability,
        stress_overload: updated_overload,
        recent_stress_intensity: new_intensity,
        stress_momentum: new_momentum
    }

    observation ← {
        event_controllability, event_overload,
        challenge, hindrance, appraised_stress,
        effective_threshold, is_stressed
    }

    RETURN PhaseOutput(state_delta, observation)
```

### Parameters

| Name | Description | Default | Source |
|------|-------------|---------|--------|
| `omega_c` | Controllability weight in appraisal | 1.0 | config |
| `omega_o` | Overload weight in appraisal | 1.0 | config |
| `bias` | Appraisal bias term | 0.0 | config |
| `gamma` | Sigmoid steepness in appraisal | 6.0 | config |
| `delta` | Polarity effect strength | 0.2 | config |
| `base_threshold` | Base stress threshold | 0.5 | config |
| `challenge_scale` | Challenge threshold scale | 0.15 | config |
| `hindrance_scale` | Hindrance threshold scale | 0.25 | config |

Reference: [src/python/phases/stress_perception.py:L28-L130](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/stress_perception.py#L28-L130)

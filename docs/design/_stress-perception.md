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
    // Generate and appraise stress event
    event ← generate_stress_event(rng)
    weights ← {omega_c, omega_o, bias, gamma}
    challenge, hindrance ← apply_weights(event, weights)

    // Compute stress load
    stress_load ← appraised_stress(event, challenge,
        hindrance, config.delta)

    // Evaluate threshold
    threshold ← config.base_threshold
        + config.challenge_scale × challenge
        + config.hindrance_scale × hindrance
    is_stressed ← stress_load ≥ threshold

    // Update stress dimensions
    c, o, inten, mom ← update_dimensions(
        state.controllability, state.overload,
        challenge, hindrance, is_stressed,
        config.volatility, state.stress_intensity,
        state.stress_momentum, state.resilience
    )

    // Build output
    delta ← {
        challenge, hindrance, is_stressed,
        event_controllability: event.controllability,
        event_overload: event.overload,
        stress_controllability: c,
        stress_overload: o,
        stress_intensity: inten,
        stress_momentum: mom
    }
    obs ← {
        event_controllability: event.controllability,
        event_overload: event.overload,
        challenge, hindrance, stress_load,
        threshold, is_stressed
    }
    RETURN {delta, obs}
```

### Parameters

```{=latex}
\begin{tabularx}{\textwidth}{p{4.5cm}Xp{2cm}}
\toprule
\textbf{Name} & \textbf{Description} & \textbf{Default} \\
\midrule
\trow{omega\_c}{Controllability weight in appraisal}{1.0}
\trow{omega\_o}{Overload weight in appraisal}{1.0}
\trow{bias}{Appraisal bias term}{0.0}
\trow{gamma}{Sigmoid steepness in appraisal}{6.0}
\trow{delta}{Polarity effect strength}{0.2}
\trow{base\_threshold}{Base stress threshold}{0.5}
\trow{challenge\_scale}{Challenge threshold scale}{0.15}
\trow{hindrance\_scale}{Hindrance threshold scale}{0.25}
\bottomrule
\end{tabularx}
```

Reference: [src/python/phases/stress_perception.py:L28-L130](https://github.com/namakala/abm-simulation/blob/84ff9b37fb0c569bef9f122e789d9c23a89348ec/src/python/phases/stress_perception.py#L28-L130)

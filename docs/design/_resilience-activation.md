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
    // Determine coping outcome
    a, r, s, coped ← coping_outcome(
        state.affect, state.resilience,
        state.stress, state.challenge,
        state.hindrance, state.neighbors,
        config, rng
    )

    // Update stress dimensions
    c, o, inten, mom ← update_dimensions(
        state.controllability, state.overload,
        state.challenge, state.hindrance,
        coped, config.volatility,
        state.stress_intensity,
        state.stress_momentum, state.resilience
    )

    // Generate PSS-10
    pss10 ← generate_pss10(c, o, inten, mom,
        a, state.resources, r, rng)

    // Resource cost and depletion
    cost ← resource_cost(config.base_cost,
        r, state.challenge, state.hindrance)
    res ← depletion(state.resources, cost, r,
        coped, config)

    // Track consecutive hindrances
    hindrances ← state.consecutive_hindrances
    IF state.hindrance > state.challenge:
        hindrances ← hindrances + 1
    ELSE:
        hindrances ← 0

    // PF allocation if coped
    pf ← copy(state.protective_factors)
    IF coped:
        res ← clamp(res + 0.03, 0, 1)
        alloc ← allocate_pf(
            res × config.pf_alloc_fraction,
            r, state.baseline_resilience, pf, rng
        )
        pf ← update_pf(pf, alloc, r)
        res ← clamp(res - sum(alloc), 0, 1)

    // Build output
    delta ← {
        affect: a, resilience: r,
        stress: s,
        controllability: c, overload: o,
        resources: res, protective_factors: pf,
        consecutive_hindrances: hindrances,
        stress_breach_count: state.breach_count + 1,
        pss10: pss10.score,
        pss10_responses: pss10.responses,
        stressed: pss10.stressed
    }
    obs ← {
        coped, coping_probability,
        resilience_effect, resource_cost: cost,
        delta_stress: s - state.stress,
        delta_affect: a - state.affect
    }
    RETURN {delta, obs}
```

### Parameters

```{=latex}
\begin{tabularx}{\textwidth}{p{4cm}p{1.8cm}X}
\toprule
\textbf{Name} & \textbf{Description} & \textbf{Default} \\
\midrule
\trow{base\_resource\_cost}{Cost per coping attempt}{0.1}
\trow{pf\_allocation\_fraction}{Resources allocated to PF on success}{0.15}
\trow{resource\_reward}{Flat reward for successful coping}{0.03}
\bottomrule
\end{tabularx}
```

Reference: [src/python/phases/resilience_activation.py:L42-L252](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/resilience_activation.py#L42-L252)

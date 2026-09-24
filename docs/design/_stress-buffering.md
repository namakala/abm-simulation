## Stress Buffering

### Purpose

Applies protective-factor resilience boost and resource mediation of stress
buffering using Baron & Kenny mediation framework: a-path (stress →
resources), b-path (resources → buffering), c'-path (stress → buffering
| resources).

- **Frequency:** daily
- **Inputs:** resilience, baseline_resilience, protective_factors, current_stress, resources
- **Outputs:** state_delta: resilience, resources
- **Observation:** pf_boost, buffering_strength, indirect_effect

### Algorithm

```
FUNCTION run_stress_buffering(state, config, rng):
    // PF boost to resilience
    need ← max(0, state.baseline - state.resilience)
    boost ← Σ(efficacy × need × config.rate)
    boost ← min(boost, need)
    new_r ← state.resilience + boost

    // Resource mediation (a-path)
    eff_stress ← state.stress × (1 + 0.2 × state.overload)
    depletion ← config.a_coeff × eff_stress
    new_res ← state.resources + depletion

    // Buffering strength (b-path + c'-path)
    buf ← max(0,
        config.b_coeff × new_res
        + config.c_coeff × state.stress)

    RETURN {delta: {resilience: new_r,
        resources: new_res},
        obs: {boost, buf}}
```

### Parameters

```{=latex}
\begin{tabularx}{\textwidth}{p{4.5cm}Xp{2cm}}
\toprule
\textbf{Name} & \textbf{Description} & \textbf{Default} \\
\midrule
\trow{boost\_rate}{PF resilience boost rate}{0.1}
\trow{a\_coefficient}{Stress→resources path coefficient}{0.3}
\trow{b\_coefficient}{Resources→buffering path coefficient}{0.7}
\trow{c\_prime\_coefficient}{Stress→buffering path coefficient}{1.0}
\bottomrule
\end{tabularx}
```

Reference: [src/python/phases/stress_buffering.py:L42-L148](https://github.com/namakala/abm-simulation/blob/812534b0da5d3401464acbd1e65c8e1b6eb1bd29/src/python/phases/stress_buffering.py#L42-L148)

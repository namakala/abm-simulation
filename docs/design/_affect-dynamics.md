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
FUNCTION run_affect_dynamics(state, config, rng):
    // Affect: homeostasis + peers + appraisal
    new_a ← update_affect(state.affect,
        state.baseline_affect, state.neighbors,
        state.challenge, state.hindrance,
        state.stress, state.resources)

    // Resilience dynamics + PF boost
    new_r ← update_resilience(state.resilience,
        state.consecutive_hindrances)
    pf_boost ← get_pf_boost(state.protective_factors,
        state.baseline, new_r)
    new_r ← min(1, new_r + pf_boost)

    // Social resilience optimisation
    new_r ← social_resilience(new_r,
        state.interactions, state.support_exchanges,
        state.resources, state.baseline_resilience,
        state.protective_factors, rng)

    // Interaction-frequency boost
    new_r ← new_r + state.interactions × 0.005

    // Hindrance decay
    new_h ← max(0,
        state.consecutive_hindrances - config.decay)

    // Homeostatic adjustment
    new_a ← homeostatic(state.baseline_affect,
        new_a, config.affect_rate)
    new_r ← homeostatic(state.baseline_resilience,
        new_r, config.resilience_rate)

    RETURN {delta: {affect: new_a,
        resilience: new_r,
        consecutive_hindrances: new_h},
        obs: {}}
```

### Parameters

```{=latex}
\begin{tabularx}{\textwidth}{p{4.5cm}Xp{2cm}}
\toprule
\textbf{Name} & \textbf{Description} & \textbf{Default} \\
\midrule
\trow{affect\_homeostatic\_rate}{Affect return rate}{0.5}
\trow{resilience\_homeostatic\_rate}{Resilience return rate}{0.5}
\trow{interaction\_boost\_rate}{Per-interaction resilience boost}{0.005}
\trow{hindrance\_decay\_rate}{Daily hindrance decay}{0.05}
\bottomrule
\end{tabularx}
```

Reference: [src/python/agent.py:L67-L211](https://github.com/namakala/abm-simulation/blob/812534b0da5d3401464acbd1e65c8e1b6eb1bd29/src/python/agent.py#L67-L211)

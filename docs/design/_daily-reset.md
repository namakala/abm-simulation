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
FUNCTION run_daily_reset(state, config, rng):
    // Affect reset toward baseline
    rate ← scale_rate(config.affect_rate,
        state.resources, state.stress)
    new_a ← affect_reset(state.affect,
        state.baseline_affect, rate)

    // Stress decay
    new_s ← stress_decay(state.stress, rate)

    // Hindrance decay
    new_h ← max(0,
        state.consecutive_hindrances - 0.05)

    // Stress summary
    events ← state.stress_events
    summary ← {
        avg: mean(events.stress_level),
        max: max(events.stress_level),
        n: len(events),
        coped: mean(events.coped)
    }

    // Clear daily counters
    state.interactions ← 0
    state.support_exchanges ← 0
    state.stress_events ← []
    state.pss10_scores ← []

    RETURN {delta: {affect: new_a,
        stress: new_s,
        consecutive_hindrances: new_h,
        last_reset_day: state.day},
        obs: {summary}}
```

### Parameters

```{=latex}
\begin{tabularx}{\textwidth}{p{4.5cm}Xp{2cm}}
\toprule
\textbf{Name} & \textbf{Description} & \textbf{Default} \\
\midrule
\trow{affect\_homeostatic\_rate}{Affect return rate}{0.5}
\trow{hindrance\_decay\_rate}{Daily hindrance decay}{0.05}
\bottomrule
\end{tabularx}
```

Reference: [src/python/agent.py:L350-L450](https://github.com/namakala/abm-simulation/blob/812534b0da5d3401464acbd1e65c8e1b6eb1bd29/src/python/agent.py#L350-L450)

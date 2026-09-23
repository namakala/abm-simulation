## PSS-10 Consolidation

### Purpose

Consolidates daily PSS-10 scores, updates stress level via exponential
smoothing, and applies post-hoc adjustments for resilience coupling,
resource coupling, and mood-congruent appraisal.

- **Frequency:** daily
- **Inputs:** daily_pss10_scores, current_stress, stress_controllability, stress_overload
- **Outputs:** state_delta: pss10, pss10_smoothed, current_stress, stressed, daily_pss10_scores (cleared)
- **Observation:** avg_pss10, num_events

### Algorithm

```
FUNCTION run_pss10_consolidation(state, config, rng):
    // Update dimensions from PSS-10 feedback
    c, o ← update_dimensions_from_pss10(
        state.controllability, state.overload,
        state.pss10_responses, state.resources)

    // Compute stress from dimensions
    new_s ← stress_from_dimensions(c, o,
        state.affect, state.resources,
        state.resilience)
    s ← 0.5 × new_s + 0.5 × state.stress

    // Consolidate daily scores
    IF state.daily_scores ≠ []:
        consolidated ← mean(state.daily_scores)
    ELSE:
        consolidated ← regenerate(c, o)

    // Exponential smoothing
    α ← config.smoothing_alpha
    smoothed ← α × consolidated + (1 - α) × state.pss10_smoothed

    // Post-hoc adjustments
    final ← smoothed + config.bias
        - 0.5 × config.coupling
        × (state.resilience - 0.5)
    final ← 20 + (final - 20) × 1.2
    final ← final + (0.5 - state.resources) × 12
    stressed ← final ≥ config.threshold

    // Mood-congruent appraisal
    adj ← -state.affect × config.affect_adj

    RETURN {delta: {pss10: final + adj,
        pss10_smoothed: smoothed,
        stress: s, controllability: c,
        overload: o, stressed,
        daily_scores: []},
        obs: {}}
```

### Parameters

```{=latex}
\begin{tabularx}{\textwidth}{p{4cm}p{1.8cm}X}
\toprule
\textbf{Name} & \textbf{Description} & \textbf{Default} \\
\midrule
\trow{pss10\_smoothing\_alpha}{Exponential smoothing factor}{0.3}
\trow{pss10\_resilience\_coupling}{Resilience penalty coefficient}{3.5}
\trow{pss10\_threshold}{Clinical cutoff score}{27}
\trow{pss10\_affect\_adjustment}{Mood-congruent appraisal factor}{0.5}
\bottomrule
\end{tabularx}
```

Reference: [src/python/agent.py:L212-L348](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/agent.py#L212-L348)

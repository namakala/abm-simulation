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
FUNCTION process_pss10_consolidation(state, config, rng):
    // 1. Update stress dimensions from PSS-10 feedback
    controllability, overload ← update_stress_dimensions_from_pss10_feedback(
        current_controllability, current_overload,
        pss10_responses, current_resources
    )

    // 2. Compute current_stress from dimensions
    new_stress ← compute_stress_from_dimensions(
        controllability, overload, affect, resources, resilience
    )
    current_stress ← 0.5 × new_stress + 0.5 × current_stress

    // 3. Consolidate PSS-10 score
    IF daily_scores not empty:
        consolidated ← mean(daily_scores)
    ELSE:
        consolidated ← regenerate_from_dimensions(...)

    // 4. Exponential smoothing across days
    smoothed ← α × consolidated + (1 - α) × prev_smoothed

    // 5. Post-hoc adjustments
    final ← smoothed + bias - 0.5 × coupling × (resilience - 0.5)
    final ← 20 + (final - 20) × 1.2                  // variance stretch
    final += (0.5 - resources) × 12.0                  // resource penalty
    stressed ← (final ≥ threshold)

    // 6. Mood-congruent appraisal
    affect_adjust ← -affect × pss10_affect_adjustment

    RETURN PhaseOutput(state_delta: {pss10, pss10_smoothed, current_stress,
        stress_controllability, stress_overload, stressed, daily_pss10_scores: []})
```

### Parameters

| Name | Description | Default | Source |
|------|-------------|---------|--------|
| `pss10_smoothing_alpha` | Exponential smoothing factor | 0.3 | assumption |
| `pss10_resilience_coupling` | Resilience penalty coefficient | 3.5 | assumption |
| `pss10_threshold` | Clinical cutoff score | 27 | literature |
| `pss10_affect_adjustment` | Mood-congruent appraisal factor | 0.5 | assumption |

Reference: [src/python/agent.py:L212-L348](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/agent.py#L212-L348)

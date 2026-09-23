# Data Collection

### Purpose

Mesa's DataCollector captures model-level and agent-level metrics using
named reporter functions. Model reporters track population averages;
agent reporters track individual trajectories.

- **Frequency:** model-level (collected each step)
- **Inputs:** agent attributes, model attributes
- **Outputs:** CSV export

### Agent Reporters

```
AGENT_REPORTERS = {
    pss10, resilience, affect, resources, current_stress,
    stress_controllability, stress_overload, consecutive_hindrances,
    coping_success, challenge_appraisal, hindrance_appraisal,
    interaction_frequency, stressed, support_boost
}
```

### Model Reporters

```
MODEL_REPORTERS = {
    avg_pss10, avg_resilience, avg_affect, coping_success_rate,
    avg_resources, avg_stress, social_support_rate, network_density,
    stress_prevalence, low_resilience, high_resilience,
    avg_challenge, avg_hindrance, challenge_hindrance_ratio
}
```

### Phase-Level Instrumentation

The `_last_phase_outputs` dictionary stores the `PhaseOutput` from each
phase executed during a step. This enables phase-specific metric extraction
for the cumulative block experiment and debugging.

```python
self._last_phase_outputs = {
    "stress_perception": PhaseOutput,
    "resilience_activation": PhaseOutput,
    "interaction_self": PhaseOutput,
    "interaction_partner": PhaseOutput,
    "affect_dynamics": PhaseOutput,
    "resource_allocation": PhaseOutput,
    "stress_buffering": PhaseOutput,
    "pss10_consolidation": PhaseOutput,
    "daily_reset": PhaseOutput,
}
```

Reference: [src/python/reporters.py:L289-L333](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/reporters.py#L289-L333)

Reference: [src/python/model.py:L114-L133](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/model.py#L114-L133)

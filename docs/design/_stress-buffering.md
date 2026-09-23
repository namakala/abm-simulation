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
    // 1. PF boost to resilience
    resilience_need ← max(0, baseline - current)
    pf_boost ← SUM(efficacy × need × rate) for each factor
    pf_boost ← min(pf_boost, resilience_need)
    new_resilience ← current + pf_boost

    // 2. Resource mediation (a-path: stress → resources)
    effective_stress ← stress × (1 + 0.2 × overload)
    resource_depletion ← a_coefficient × effective_stress
    new_resources ← resources + resource_depletion

    // 3. Buffering strength (b-path + c'-path)
    buffering ← max(0, b × new_resources + c' × stress)

    RETURN PhaseOutput(state_delta: {resilience, resources}, observation)
```

### Parameters

| Name | Description | Default | Source |
|------|-------------|---------|--------|
| `boost_rate` | PF resilience boost rate | 0.1 | assumption |
| `a_coefficient` | Stress→resources path coefficient | 0.3 | assumption |
| `b_coefficient` | Resources→buffering path coefficient | 0.7 | assumption |
| `c_prime_coefficient` | Stress→buffering path coefficient | 1.0 | assumption |

Reference: [src/python/phases/stress_buffering.py:L42-L148](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/stress_buffering.py#L42-L148)

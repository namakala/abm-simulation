## Resource Allocation

### Purpose

Regenerates resources and allocates to protective factors via softmax.
Resource regeneration is linear in the deficit, modulated by affect and
resilience. Allocation uses softmax with temperature for bounded-rational
distribution across social support, family support, formal intervention,
and psychological capital.

- **Frequency:** daily
- **Inputs:** resources, affect, resilience, protective_factors
- **Outputs:** state_delta: resources, protective_factors
- **Observation:** regeneration_amount, allocation_weights

### Algorithm

```
FUNCTION run_resource_allocation(state, config, rng):
    // 1. Resource regeneration
    affect_mult ← 1 + 0.5 × max(0, affect)
    resil_mult ← 1 + 0.3 × resilience
    regeneration ← base_rate × (1 - resources) × affect_mult × resil_mult

    // 2. Softmax allocation across protective factors
    available ← resources + regeneration
    spendable ← available × (1 - preservable_fraction)
    weights ← softmax(efficacies / temperature)
    allocations ← spendable × weights

    // 3. PF efficacy updates (diminishing returns)
    FOR each factor f:
        Δe_f ← allocations[f] × rate × (1 - e_f) × efficiency_gain
        e_f ← min(1, e_f + Δe_f)

    // 4. Remaining resources
    new_resources ← preserved + (spendable - total_allocated)

    RETURN PhaseOutput(state_delta: {resources, protective_factors}, observation)
```

### Parameters

| Name | Description | Default | Source |
|------|-------------|---------|--------|
| `base_regeneration` | Daily regeneration rate | 0.5 | config |
| `preservable_fraction` | Fraction of resources preserved | 0.2 | config |
| `softmax_temperature` | Allocation temperature | 1.0 | config |
| `protective_improvement_rate` | PF efficacy update rate | 0.5 | config |

Reference: [src/python/phases/resource_allocation.py:L141-L219](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/resource_allocation.py#L141-L219)

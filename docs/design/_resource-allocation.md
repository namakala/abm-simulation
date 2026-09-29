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
    // Resource regeneration
    a_mult ← 1 + 0.5 × max(0, state.affect)
    r_mult ← 1 + 0.3 × state.resilience
    regen ← config.base_rate × (1 - state.resources)
        × a_mult × r_mult

    // Softmax allocation
    available ← state.resources + regen
    spendable ← available × (1 - config.preserve_frac)
    weights ← softmax(state.efficacies / config.temp)
    alloc ← spendable × weights

    // Update efficacies (diminishing returns)
    FOR each factor f:
        de ← alloc[f] × config.rate
            × (1 - state.efficacies[f])
        state.efficacies[f] ← min(1,
            state.efficacies[f] + de)

    // Remaining resources
    new_res ← config.preserve_frac × available
        + (spendable - total(alloc))

    RETURN {delta: {resources: new_res,
        protective_factors: state.efficacies},
        obs: {regen, alloc}}
```

### Parameters

```{=latex}
\begin{tabularx}{\textwidth}{p{4.5cm}Xp{2cm}}
\toprule
\textbf{Name} & \textbf{Description} & \textbf{Default} \\
\midrule
\trow{base\_regeneration}{Daily regeneration rate}{0.5}
\trow{preservable\_fraction}{Fraction of resources preserved}{0.2}
\trow{softmax\_temperature}{Allocation temperature}{1.0}
\trow{protective\_improvement\_rate}{PF efficacy update rate}{0.5}
\bottomrule
\end{tabularx}
```

Reference: [src/python/phases/resource_allocation.py:L141-L219](https://github.com/namakala/abm-simulation/blob/84ff9b37fb0c569bef9f122e789d9c23a89348ec/src/python/phases/resource_allocation.py#L141-L219)

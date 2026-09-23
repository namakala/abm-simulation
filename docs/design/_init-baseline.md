## Baseline Resilience

Sampled from a sigmoid-transformed normal distribution, centreing the population
near 0.5 on $[0, 1]$.

**Algorithm:**

```
FUNCTION initialize_baseline_resilience(rng, mean, std):
    X ← rng.normal(mean, std)
    R₀ ← sigmoid(X / 6)        // gamma = 6 fixed for population sampling
    RETURN clamp(R₀, 0, 1)
```

Reference: [src/python/initialization.py:L21-L37](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/initialization.py#L21-L37)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rng` | Seeded random number generator | — |
| `mean` | Mean of underlying normal | 0.0 |
| `std` | SD of underlying normal | 1.0 |

Returns: `float` $\in [0, 1]$.

## Baseline Affect

Sampled from a tanh-transformed normal distribution, mapping latent 0 to neutral
valence on $[-1, 1]$.

**Algorithm:**

```
FUNCTION initialize_baseline_affect(rng, mean, std):
    X ← rng.normal(mean, std)
    A₀ ← tanh(X)
    RETURN clamp(A₀, -1, 1)
```

Reference: [src/python/initialization.py:L39-L55](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/initialization.py#L39-L55)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rng` | Seeded random number generator | — |
| `mean` | Mean of underlying normal | 0.0 |
| `std` | SD of underlying normal | 1.0 |

Returns: `float` $\in [-1, 1]$.

## Resources

Sampled from a sigmoid-transformed normal distribution, centreing the population
near 0.5 on $[0, 1]$.

**Algorithm:**

```
FUNCTION initialize_resources(rng, mean, std):
    X ← rng.normal(mean, std)
    R₀ ← sigmoid(X / 6)        // gamma = 6 fixed for population sampling
    RETURN clamp(R₀, 0, 1)
```

Reference: [src/python/initialization.py:L57-L73](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/initialization.py#L57-L73)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rng` | Seeded random number generator | — |
| `mean` | Mean of underlying normal | 0.0 |
| `std` | SD of underlying normal | 1.0 |

Returns: `float` $\in [0, 1]$.

## Protective Factors

Four-component dictionary representing social support, family support, formal
intervention, and psychological capital. Initialised uniformly.

**Algorithm:**

```
FUNCTION initialize_protective_factors(value = 0.5):
    RETURN {
        social_support: value,
        family_support: value,
        formal_intervention: value,
        psychological_capital: value
    }
```

Reference: [src/python/initialization.py:L75-L91](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/initialization.py#L75-L91)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `value` | Initial efficacy for each factor | 0.5 |

Returns: `Dict[str, float]` with four keys.

## Volatility

Agent-specific volatility sampled from a Beta distribution, controlling
sensitivity to stress events.

**Algorithm:**

```
FUNCTION initialize_volatility(rng, alpha, beta):
    RETURN rng.beta(alpha, beta)
```

Reference: [src/python/initialization.py:L93-L109](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/initialization.py#L93-L109)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rng` | Seeded random number generator | — |
| `alpha` | Beta distribution alpha | 1.0 |
| `beta` | Beta distribution beta | 1.0 |

Returns: `float` $\in [0, 1]$.

## PSS-10 State

Generates the full PSS-10 state including item responses, dimension scores
(controllability, overload), total score, and stressed classification.

**Algorithm:**

```
FUNCTION initialize_pss10_state(rng):
    items ← generate_pss10_items(rng)       // from empirical item params
    responses ← generate_item_responses(items, rng)
    score ← sum(responses)
    controllability, overload ← compute_dimensions(responses)
    stressed ← (score >= threshold)
    RETURN {responses, controllability, overload, score, stressed}
```

Reference: [src/python/initialization.py:L111-L127](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/initialization.py#L111-L127)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rng` | Seeded random number generator | — |

Returns: `Dict` with keys `pss10_responses`, `stress_controllability`,
`stress_overload`, `pss10_score`, `stressed`.

## Initial Stress

Derives initial stress level from PSS-10 dimensions using a weighted combination
of overload and inverse controllability.

**Algorithm:**

```
FUNCTION compute_initial_stress(pss10_score, controllability, overload, dampening):
    S ← (overload + (1 - controllability)) / 2 × dampening
    RETURN clamp(S, 0, 1)
```

Reference: [src/python/initialization.py:L129-L149](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/initialization.py#L129-L149)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pss10_score` | Initial PSS-10 total (0--40) | — |
| `controllability` | Stress controllability $\in [0, 1]$ | — |
| `overload` | Stress overload $\in [0, 1]$ | — |
| `dampening` | Scaling factor | 1.0 |

Returns: `float` $\in [0, 1]$.

---
title: Homeostatic Stabilization
description: Affect and resilience homeostatic adjustment mechanism — theory, implementation, and testing
status: final
date: 2026-06-15
---

# Homeostatic Stabilization

## Theoretical Basis

Homeostatic stabilization models the natural tendency of psychological variables to return to a set-point (baseline) after perturbation. This is grounded in:

- **Control theory of self-regulation** (Carver & Scheier, 1998): Behavior is organized around feedback loops that reduce discrepancies between current state and reference values.
- **Affect homeostasis** (Headey & Wearing, 1989): Individuals have baseline affect levels; events cause deviations but affect tends to revert to baseline over time.
- **Resilience as a dynamic process** (Bonanno, 2004; Norris et al., 2009): Resilience is not a fixed trait but fluctuates around a baseline in response to stressors and recovery periods.

The ABM implements two distinct homeostatic rates — one for affect, one for resilience — reflecting the theoretical distinction between:

| Variable | Homeostatic rate | Interpretation |
|----------|-----------------|----------------|
| Affect | 0.5 (faster) | Affect is more state-like and labile; it recovers quickly after emotional events. |
| Resilience | 0.3 (slower) | Resilience is more trait-like and stable; it changes more gradually in response to cumulative experience. |

This differentiation follows empirical findings that emotional states revert to baseline within hours to days, while coping resources (resilience) develop or degrade over weeks to months.

## Implementation

### Core formula

```
adjusted = final_value + rate × (initial_value - final_value)
```

Where:

- `initial_value`: Value at the start of the day (baseline)
- `final_value`: Value after all daily actions (stress events, interactions, coping)
- `rate`: Homeostatic rate in [0, 1] — higher = faster convergence toward baseline
- `adjusted`: Output value after homeostatic pull

The adjustment respects type-specific bounds:
- Affect: clamped to [-1, 1]
- Resilience: clamped to [0, 1]

### Implementation files

| File | Role |
|------|------|
| `src/python/assumption_config.py` | Defines `ASSUMPTION_AFFECT_HOMEOSTATIC_RATE` (default 0.5) and `ASSUMPTION_RESILIENCE_HOMEOSTATIC_RATE` (default 0.3) in `AssumptionStressConfig` |
| `src/python/affect_utils.py` | `AffectDynamicsConfig.homeostatic_rate` reads affect rate; `ResilienceDynamicsConfig.homeostatic_rate` reads resilience rate |
| `src/python/affect_utils.py` | `compute_homeostatic_adjustment()` — pure function implementing the formula, dispatching to the correct rate based on `value_type` |
| `src/python/affect_utils.py` | `scale_homeostatic_rate()` — modulates the base rate by current resources and stress (lower resources / higher stress → slower recovery) |
| `src/python/agent.py` | `ConsolidationAffectAndResilience` phase applies scaled homeostatic adjustment every day to both affect and resilience |
| `src/python/config.py` | Backward-compatible exposure via `config.get("affect_dynamics", "homeostatic_rate")` and `config.get("resilience_dynamics", "homeostatic_rate")` — forwards to `get_assumptions()` |

### Scaled homeostatic rate

The `scale_homeostatic_rate()` function modulates the base rate:

```
scaled_rate = base_rate × (1.0 + resource_boost - stress_penalty)
```

Where:

- `resource_boost = 0.2 × (1.0 - resources)` — agents with fewer resources recover more slowly
- `stress_penalty = 0.3 × stress` — agents under higher stress recover more slowly

This produces a clinically realistic pattern: stressed, resource-depleted individuals recover from emotional perturbations more slowly than those with abundant resources and low stress.

## Configuration

Set via environment variables:

```bash
export ASSUMPTION_AFFECT_HOMEOSTATIC_RATE=0.5    # Default: affect recovers quickly
export ASSUMPTION_RESILIENCE_HOMEOSTATIC_RATE=0.3 # Default: resilience recovers slowly
```

## Testing

### Test files

| File | Tests |
|------|-------|
| `src/python/tests/test_homeostatic_adjustment.py` | 22 unit tests: basic functionality, edge cases, value type validation, multi-day convergence, mathematical properties, configuration integration |
| `src/python/tests/test_homeostatic_stabilization_integration.py` | 12 integration tests: full agent model behavior, social network context, tunable strength, boundary conditions |
| `src/python/tests/test_affect_resilience_dynamics.py` | Tests affect dynamics with specific homeostatic rates via `AffectDynamicsConfig` |
| `src/python/tests/test_affect_resilience_integration.py` | Tests different homeostatic rate configurations in multi-agent settings |
| `src/python/tests/test_internal_phases.py` | Tests `process_affect_dynamics` phase uses `scale_homeostatic_rate` without crashing |

### Key test: `test_independent_affect_and_resilience_homeostatic_rates`

This test (in `test_homeostatic_adjustment.py`) validates the core theoretical claim:

1. Reads both rates from `get_assumptions().stress`
2. Verifies `affect_rate != resilience_rate` (they are differentiated: 0.5 vs 0.3)
3. Applies affect homeostatic adjustment with affect rate to an affect value
4. Applies resilience homeostatic adjustment with resilience rate to a resilience value
5. Verifies the adjusted values are different due to the different rates

This ensures the differentiated rates produce distinct recovery trajectories, which is the theoretical basis for affect being more labile than resilience.

### How this tests the theoretical remarks

The theoretical correlation structure of the model depends on homeostatic rates in two ways:

1. **Differentiated rates produce distinct variances**: Affect (fast rate = 0.5) stays closer to baseline at any snapshot, reducing cross-sectional variance. Resilience (slow rate = 0.3) shows more dispersion. This differential variance affects the detectability of correlations: resilience↔PSS-10 correlations benefit from resilience's wider spread, while affect↔PSS-10 correlations require larger samples.

2. **Scaled rates modulate recovery under stress**: Agents with low resources or high stress experience slower homeostasis for both affect and resilience. This creates a correlation between resources and recovery speed, which feeds into the stress→affect→resilience causal chain that the theoretical correlation tests validate.

## Theoretical Correlation Predictions

| Variable pair | Expected sign | How homeostasis contributes |
|---------------|---------------|---------------------------|
| PSS-10 ↔ affect | Negative | Faster affect recovery (rate=0.5) means affect stays near baseline, weakening PSS-10↔affect correlation at any snapshot. Stronger direct stress→affect pathways compensate. |
| PSS-10 ↔ resilience | Negative | Slower resilience recovery (rate=0.3) means resilience shows more spread, strengthening PSS-10↔resilience correlation. |
| Resilience ↔ affect | Positive | Different recovery rates produce a wider gap between the two variables, increasing correlation detectability. |
| Stress ↔ affect | Negative | Faster affect recovery dampens stress→affect coupling; requires `N ≥ 300` for significance at current parameterization. |

## References

- Bonanno, G. A. (2004). Loss, trauma, and human resilience: Have we underestimated the human capacity to thrive after extremely aversive events? *American Psychologist*, 59(1), 20–28.
- Carver, C. S., & Scheier, M. F. (1998). *On the self-regulation of behavior*. Cambridge University Press.
- Headey, B., & Wearing, A. (1989). Personality, life events, and subjective well-being: Toward a dynamic equilibrium model. *Journal of Personality and Social Psychology*, 57(4), 731–739.
- Norris, F. H., Tracy, M., & Galea, S. (2009). Looking for resilience: Understanding the longitudinal trajectories of responses to stress. *Social Science & Medicine*, 68(12), 2190–2198.

## Social Interaction

### Purpose

Dyadic interaction between two agents. Converges affect and resilience with
negativity bias (negative influence 1.5x stronger), detects support from
convergence magnitude, and applies resource exchange state machine based on
mutual stress states.

- **Frequency:** event_driven
- **Inputs:** self_state, partner_state (full AgentState)
- **Outputs:** Tuple[PhaseOutput, PhaseOutput] — delta values (not absolute)
- **Observation:** support_occurred

### Algorithm

```
FUNCTION process_interaction(self_state, partner_state, config, rng):
    // Affect convergence with negativity bias
    ΔA_self ← influence_rate × partner_affect
    ΔA_partner ← influence_rate × self_affect
    IF ΔA_self < 0: ΔA_self ×= 1.5    // negativity bias
    IF ΔA_partner < 0: ΔA_partner ×= 1.5

    // Resilience convergence
    ΔR_self ← resilience_influence × partner_affect
    ΔR_partner ← resilience_influence × self_affect

    // Support detection
    total_convergence ← |ΔA_self| + |ΔA_partner| + |ΔR_self| + |ΔR_partner|
    support_occurred ← (total_convergence > threshold)

    // Resource exchange state machine
    IF self_stressed AND partner_stressed:
        IF support: both +boost
        ELSE: both -cost
    ELIF self_stressed AND NOT partner_stressed:
        IF support: self +boost
        ELSE: self -cost
    ELIF NOT self_stressed AND NOT partner_stressed:
        both protective_factors.social_support +small_boost

    RETURN (self_output, partner_output)
```

### Parameters

| Name | Description | Default | Source |
|------|-------------|---------|--------|
| `influence_rate` | Affect convergence rate | 0.05 | config |
| `resilience_influence` | Resilience convergence rate | 0.05 | config |
| `negativity_bias` | Negative influence multiplier | 1.5 | assumption |
| `support_threshold` | Convergence threshold for support detection | 0.1 | assumption |
| `boost` | Resource boost on support exchange | 0.05 | assumption |
| `cost` | Resource cost without support | 0.03 | assumption |

Reference: [src/python/phases/interaction.py:L48-L190](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/interaction.py#L48-L190)

## Social Interaction

### Purpose

Dyadic interaction between two agents. Converges affect and resilience with
negativity bias (negative influence 1.5x stronger), detects support from
convergence magnitude, and applies resource exchange state machine based on
mutual stress states.

- **Frequency:** event_driven
- **Inputs:** self_state, partner_state (full AgentState)
- **Outputs:** Tuple[PhaseOutput, PhaseOutput]: delta values (not absolute)
- **Observation:** support_occurred

### Algorithm

```
FUNCTION interact(self_state, partner_state, config, rng):
    // Affect convergence with negativity bias
    dA_self ← config.rate × partner.affect
    dA_partner ← config.rate × self.affect
    IF dA_self < 0:
        dA_self ← dA_self × 1.5
    IF dA_partner < 0:
        dA_partner ← dA_partner × 1.5

    // Resilience convergence
    dR_self ← config.resilience_rate × partner.affect
    dR_partner ← config.resilience_rate × self.affect

    // Support detection
    total ← |dA_self| + |dA_partner|
        + |dR_self| + |dR_partner|
    support ← total > config.threshold

    // Resource exchange
    IF self.stressed AND partner.stressed:
        IF support: boost both
        ELSE: cost both
    ELSE IF self.stressed:
        IF support: boost self
        ELSE: cost self
    ELSE IF partner.stressed:
        IF support: boost partner
        ELSE: cost partner
    ELSE:
        small_boost(self, partner)

    RETURN (self_out, partner_out)
```

### Parameters

```{=latex}
\begin{tabularx}{\textwidth}{p{4.5cm}Xp{2cm}}
\toprule
\textbf{Name} & \textbf{Description} & \textbf{Default} \\
\midrule
\trow{influence\_rate}{Affect convergence rate}{0.05}
\trow{resilience\_influence}{Resilience convergence rate}{0.05}
\trow{negativity\_bias}{Negative influence multiplier}{1.5}
\trow{support\_threshold}{Convergence threshold for support detection}{0.1}
\trow{boost}{Resource boost on support exchange}{0.05}
\trow{cost}{Resource cost without support}{0.03}
\bottomrule
\end{tabularx}
```

Reference: [src/python/phases/interaction.py:L44-L182](https://github.com/namakala/abm-simulation/blob/812534b0da5d3401464acbd1e65c8e1b6eb1bd29/src/python/phases/interaction.py#L44-L182)

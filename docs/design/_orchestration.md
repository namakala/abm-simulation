# Two-Loop Orchestration

## Purpose

Explain how `Person.step()` orchestrates phases across two temporal scales
within a single simulation day: event-driven responses and daily homeostatic
consolidation.

- **Frequency:** daily (orchestrator)
- **Inputs:** AgentState at day start
- **Outputs:** AgentState at day end

## Algorithm

```
FUNCTION step(agent, config, rng):
    // Reset phase outputs
    agent.last_phase_outputs ← {}

    // Build state from agent attributes
    state ← build_state(agent)
    neighbors ← get_neighbor_affects(agent)

    // Subevent loop
    n ← sample_poisson(config.subevents_per_day)
    actions ← random_sequence("stress", "interact", n)
    shuffle(actions)

    challenge_total ← 0
    hindrance_total ← 0
    stress_count ← 0

    FOR EACH action IN actions:
        // Decay support boost
        state.support_boost ← state.support_boost × 0.9

        IF action = "stress":
            // Appraise stress event
            result ← run_stress_perception(state, config, rng)
            state ← apply_delta(state, result.delta)
            challenge_total ← challenge_total + result.delta.challenge
            hindrance_total ← hindrance_total + result.delta.hindrance
            stress_count ← stress_count + 1

            // Activate resilience if stressed
            IF state.is_stressed:
                r2 ← run_resilience_activation(
                    state, config, rng)
                state ← apply_delta(state, r2.delta)
                scores ← state.pss10_scores
                state.pss10_scores ← scores ∪ {state.pss10}

            // Record stress event
            state.stress_events ← state.stress_events ∪ {event}

        ELSE IF action = "interact":
            // Dyadic interaction
            partner ← random_neighbor(agent)
            pstate ← build_state(partner)
            out1, out2 ← interact(state, pstate, config, rng)
            state ← apply_delta(state, out1.delta)
            apply_delta(partner, out2.delta)
            write_back(partner)

            state.interactions ← state.interactions + 1
            IF out1.obs.support_occurred:
                state.support_exchanges ← state.support_exchanges + 1
                sb ← state.support_boost + 0.1
                state.support_boost ← min(1, sb)

    // Normalize daily totals
    IF stress_count > 0:
        challenge_total ← challenge_total ÷ stress_count
        hindrance_total ← hindrance_total ÷ stress_count

    // Daily consolidation
    state ← apply_delta(state,
        affect_dynamics(state, config, rng).delta)
    state ← apply_delta(state,
        resource_allocation(state, config, rng).delta)
    state ← apply_delta(state,
        stress_buffering(state, config, rng).delta)
    state ← apply_delta(state,
        pss10_consolidation(state, config, rng).delta)
    state ← apply_delta(state,
        daily_reset(state, config, rng).delta)

    // Write back
    write_back(agent, state)
```

## Parameters

```{=latex}
\begin{tabularx}{\textwidth}{p{4.5cm}Xp{2cm}}
\toprule
\textbf{Name} & \textbf{Description} & \textbf{Default} \\
\midrule
\trow{subevents\_per\_day}{Poisson rate (\(\lambda\)) for daily subevents}{3}
\trow{action\_distribution}{Probability of stress vs interact}{0.5 each}
\trow{support\_boost\_decay}{Per-subevent decay of support boost}{0.9}
\trow{support\_boost\_increment}{Per-support-exchange boost increment}{0.10}
\trow{interaction\_boost\_rate}{Per-interaction resilience boost}{0.005}
\bottomrule
\end{tabularx}
```

Reference: [src/python/agent.py:L601-L836](https://github.com/namakala/abm-simulation/blob/84ff9b37fb0c569bef9f122e789d9c23a89348ec/src/python/agent.py#L601-L836)

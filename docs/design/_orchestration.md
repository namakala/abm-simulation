# Two-Loop Orchestration

### Purpose

Explain how `Person.step()` orchestrates phases across two temporal scales
within a single simulation day: event-driven responses and daily homeostatic
consolidation.

- **Frequency:** daily (orchestrator)
- **Inputs:** AgentState at day start
- **Outputs:** AgentState at day end

### Algorithm

```
FUNCTION Person.step():
    // Step 0: Reset phase output instrumentation
    self._last_phase_outputs ← {}

    // Step 1: Build agent state
    state ← self._build_agent_state()

    // Step 2: Get shared config values
    neighbor_affects ← get_neighbor_affects(self, self.model)
    cfg ← get_config()

    // Step 3: Subevent loop (event-driven phases)
    n_subevents ← sample_poisson(lam=cfg.agent.subevents_per_day, min_value=1)
    actions ← random sequence of ["stress", "interact"] of length n_subevents
    shuffle(actions)

    daily_challenge_total ← 0.0
    daily_hindrance_total ← 0.0
    stress_event_count ← 0

    FOR EACH action IN actions:
        // Decay support_boost at each subevent (10% per subevent)
        state["support_boost"] ← state["support_boost"] × 0.9

        IF action == "stress":
            // Stress perception phase
            perception_result ← run_stress_perception(state, config, rng)
            state ← _apply_delta(state, perception_result.state_delta)

            // Accumulate daily totals
            daily_challenge_total += perception_result.state_delta.challenge
            daily_hindrance_total += perception_result.state_delta.hindrance
            stress_event_count += 1

            // Resilience activation phase (only if stressed)
            IF state["is_stressed"]:
                activation_result ← run_resilience_activation(state, config, rng)
                state ← _apply_delta(state, activation_result.state_delta)

                // Accumulate PSS-10 score
                state["daily_pss10_scores"].append(state["pss10"])

            // Track stress event for model-level reporting
            state["daily_stress_events"].append({...})

        ELSE IF action == "interact":
            // Interaction phase
            partner ← random neighbor
            partner_state ← partner._build_agent_state()
            self_output, partner_output ← process_interaction(state, partner_state, config, rng)

            // Apply delta values (interaction returns changes, not absolutes)
            state ← apply_interaction_delta(state, self_output)
            partner._write_back_state(apply_interaction_delta(partner_state, partner_output))

            state["daily_interactions"] += 1
            IF self_output.observation.support_occurred:
                state["daily_support_exchanges"] += 1
                state["support_boost"] ← min(1.0, state["support_boost"] + 0.10)

    // Normalize daily challenge/hindrance
    IF stress_event_count > 0:
        daily_challenge_total /= stress_event_count
        daily_hindrance_total /= stress_event_count

    // Step 4: Daily consolidation loop
    affect_result ← process_affect_dynamics(state, config, rng)
    state ← _apply_delta(state, affect_result.state_delta)

    resource_result ← run_resource_allocation(state, config, rng)
    state ← _apply_delta(state, resource_result.state_delta)

    buffering_result ← run_stress_buffering(state, config, rng)
    state ← _apply_delta(state, buffering_result.state_delta)

    pss10_result ← process_pss10_consolidation(state, config, rng)
    state ← _apply_delta(state, pss10_result.state_delta)

    reset_result ← process_daily_reset(state, config, rng)
    state ← _apply_delta(state, reset_result.state_delta)

    // Step 5: Write back state
    self._write_back_state(state)
```

### Parameters

| Name | Description | Default | Source |
|------|-------------|---------|--------|
| `subevents_per_day` | Poisson rate (λ) for daily subevents | 3 | config |
| `action_distribution` | Probability of stress vs interact | 0.5 each | config |
| `support_boost_decay` | Per-subevent decay of support boost | 0.9 | assumption |
| `support_boost_increment` | Per-support-exchange boost increment | 0.10 | assumption |
| `interaction_boost_rate` | Per-interaction resilience boost | 0.005 | assumption |

Reference: [src/python/agent.py:L601-L850](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/agent.py#L601-L850)

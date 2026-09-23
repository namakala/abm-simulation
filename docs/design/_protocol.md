# Phase Function Protocol

## Purpose

Define the type contract that all phase functions must satisfy, enabling
modular composition and independent testing.

- **Frequency:** N/A (interface definition)
- **Inputs:** N/A (type definitions)
- **Outputs:** N/A (type definitions)

## Definitions

**PhaseFunction Protocol:**

```
FUNCTION phase(state, config, rng) → PhaseOutput
```

Every phase function accepts (state, config, rng) and returns PhaseOutput.
This uniform signature enables sequential composition.

**PhaseOutput:**

```
PhaseOutput = {
    state_delta: {key: value}   // updates to apply to agent state
    observation: {key: value}   // logging data, no state effect
}
```

state_delta contains updates to apply via apply_delta().
observation contains metrics that do not affect simulation state.

**PhaseFrequency:**

```
PhaseFrequency = event_driven | daily
```

event_driven: runs per stress event or interaction (multiple times/day)
daily: runs once per day during consolidation loop

Reference: [src/python/phases/interfaces.py:L1-L85](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/interfaces.py#L1-L85)

# Phase Function Protocol

### Purpose

Define the type contract that all phase functions must satisfy, enabling
modular composition and independent testing.

- **Frequency:** N/A (interface definition)
- **Inputs:** N/A (type definitions)
- **Outputs:** N/A (type definitions)

### Definitions

**PhaseFunction Protocol:**

```python
class PhaseFunction(Protocol):
    def __call__(
        self,
        state: AgentState,
        config: Dict[str, Any],
        rng: Generator,
    ) -> PhaseOutput: ...
```

Every phase function accepts `(state, config, rng)` and returns `PhaseOutput`.
This uniform signature enables sequential composition in the orchestrator.

**PhaseOutput TypedDict:**

```python
class PhaseOutput(TypedDict):
    state_delta: Dict[str, Any]  # key-value pairs to apply to AgentState
    observation: Dict[str, Any]  # non-state data for logging/recording
```

`state_delta` contains updates to apply via `_apply_delta()`. `observation`
contains metrics the caller may log but does not affect simulation state.

**PhaseFrequency Enum:**

```python
PhaseFrequency = Literal["event_driven", "daily"]
```

- `event_driven`: Phase runs per stress event or interaction (multiple times/day)
- `daily`: Phase runs once per day during consolidation loop

Reference: [src/python/phases/interfaces.py:L1-L85](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/interfaces.py#L1-L85)

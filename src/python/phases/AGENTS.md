---
title: Phase Functions
description: Conventions and calling contexts for the phases/ package
---

## Overview

Each module in `phases/` implements a single phase function that transforms
a slice of agent state.  Phases are pure (injectable RNG) and composable.

## Contract

Every phase function follows:

```python
def run_phase(
    state: AgentState,
    config: dict,
    rng: Generator,
) -> PhaseOutput: ...
```

- **state**: the calling agent's current `AgentState` (read + write via state_delta).
- **config**: phase-specific parameters passed by the orchestrator.
- **rng**: a seeded `numpy.random.Generator` for reproducibility.
- **returns**: `PhaseOutput = {"state_delta": {...}, "observation": {...}}`.

## Frequency

| Frequency | Modules | Called |
|-----------|---------|--------|
| `event_driven` | `stress_perception`, `resilience_activation`, `interaction` | Once per triggering event |
| `daily` | `resource_allocation`, `stress_buffering` | Once per simulation day |

## File Structure

```
phases/
├── __init__.py              # Re-exports all run_phase + PHASE_FREQUENCY
├── AGENTS.md                # This file
├── interfaces.py            # AgentState, PhaseOutput, PhaseFunction, PhaseFrequency
├── stress_perception.py     # event_driven
├── resilience_activation.py # event_driven
├── resource_allocation.py   # daily
├── stress_buffering.py      # daily
└── interaction.py           # event_driven, dual-state
```

## Phase Dependencies

Phases may depend on `interfaces.py` only.  No imports from `agent.py`,
`model.py`, or other utility modules — those will be refactored into phases
over time.

## Adding a New Phase

1. Create `<name>.py` with `run_phase` + `PHASE_FREQUENCY`.
2. Add entry in `__init__.py` and append to `__all__`.
3. Update this file's tables.

## Calling Contexts

- **Event-driven**: invoked inside `Person.stressful_event()` or `Person.interact()`.
  The phase receives the current agent state and returns a delta to apply.
- **Daily**: invoked at end of `Person.step()` or as a batch over all agents
  in `StressModel.step()`.  Handles resource regeneration and stress decay.

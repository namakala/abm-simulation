---
title: Phase Package Setup
description: Create src/python/phases/ with shared types, interfaces, and stub phase functions
date: 2026-06-04
---

# Overview

Foundation for all refactoring. Create the phases/ package with shared type contracts that every phase function, test, and demo will reference.

# Goals

- Define AgentState TypedDict (single source of truth for agent variables)
- Define PhaseOutput = (state_delta, observation) contract
- Define PhaseFunction protocol
- Create stub phase functions with NotImplementedError
- No behavioral changes to existing code

# Implementation Steps

- [ ] 1. Create `src/python/phases/__init__.py`
- [ ] 2. Define `interfaces.py`:
  - `AgentState` TypedDict: all agent variables (resilience, affect, resources, protective_factors, PSS-10 state, counters)
  - `PhaseOutput` TypedDict: `state_delta: Dict[str, Any]`, `observation: Dict[str, Any]`
  - `PhaseFunction` Protocol: `(state: AgentState, config: dict, rng: Generator) -> PhaseOutput`
- [ ] 3. Create stub files: `stress_perception.py`, `resilience_activation.py`, `resource_allocation.py`, `stress_buffering.py`, `interaction.py`
  - Each exports its phase function with correct signature, body raises NotImplementedError
- [ ] 4. Verify `from src.python.phases import *` works
- [ ] 5. Write `phases/AGENTS.md` documenting phase conventions

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Forgetting a state variable | Med | High | Cross-reference agent.py __init__ for all state vars |
| PhaseOutput contract too rigid | Low | Med | Use Dict[str, Any] for flexibility |

# UAT

1. `python -c "from src.python.phases import *; print('OK')"` succeeds
2. Each stub function can be imported and inspected (but raises NotImplementedError on call)
3. AgentState includes all fields from Person.__init__

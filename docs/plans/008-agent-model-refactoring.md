---
title: Agent Model Refactoring
description: Compose phase functions into Person.step() with two-loop orchestrator and signature freeze
date: 2026-06-05
---

# Overview

Rewrite Person.step() as a two-loop orchestrator that calls event-driven phases inside the subevent loop and daily consolidation phases after it. This is the integration point: all 5 phases must be implemented (Plans 002-006) and parameterized (Plan 007) before this starts. The signature freeze checkpoint prevents API drift.

# Goals

- Signature freeze: tests/phase_signatures.py asserts all phase interfaces are locked
- Person.step() delegates to event-driven phase functions (loop) + daily phases (consolidation)
- Person.stressful_event() removed (decomposed into Plans 002-003)
- Person.interact() simplified to call Plan 005 phase function
- _apply_delta() is the single merge point for sequential delegation
- All existing tests pass (behavioral regression)

# Implementation Steps

- [ ] 1. Write `tests/test_phase_signatures.py` (signature freeze):
  - Every phase function importable
  - Correct PHASE_FREQUENCY per phase
  - Correct parameter count and return type
  - All state_delta keys documented in AgentState
  - Plan 005: return type is Tuple[PhaseOutput, PhaseOutput]
- [ ] 2. Add orchestrator helpers to Person:
  - `_build_agent_state()` -> AgentState (reads self.*)
  - `_apply_delta(state, delta)` -> new state (dict merge, sequential)
  - `_write_back_state(state)` -> writes self.* from state
- [ ] 3. Rewrite Person.step():
  ```
  state = _build_agent_state()
  for action in shuffled ["interact","stress",...]:
    if action == "stress":
      state = _apply_delta(state, process_stress_perception(state, ...))
      if state.is_stressed:
        state = _apply_delta(state, process_resilience_activation(state, ...))
    elif action == "interact":
      self_d, partner_d = process_interaction(state, partner_state, ...)
      state = _apply_delta(state, self_d)
      _apply_partner_delta(partner, partner_d)
  for phase in [affect_dynamics, resource_allocation, stress_buffering,
                pss10_consolidation, daily_reset]:
    state = _apply_delta(state, phase(state, ...))
  _write_back_state(state)
  ```
- [ ] 4. Remove Person.stressful_event() — fully replaced
- [ ] 5. Simplify Person.interact() — delegates to process_interaction
- [ ] 6. Add orchestrator-internal phases (not in phases/ package):
  - process_affect_dynamics (homeostasis + peer influence from step() lines 260-285, 326-358)
  - process_pss10_consolidation (PSS-10 averaging, stress update from step() lines 363-380)
  - process_daily_reset (affect reset, stress decay, counter reset from _daily_reset() lines 747-807)
- [ ] 7. Update StressModel.step() — keep DataCollector, delegate _daily_reset to phase
- [ ] 8. Run: `pytest src/python/tests/ -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Phase API changes during Plans 001-005 | High | High | Signature freeze in step 1 prevents this |
| _apply_delta merge logic wrong | Low | High | Unit test _apply_delta with known input/output pairs |
| Subevent loop reordering changes output | Med | Med | Behavioral test: same seed -> same model output before and after |

# UAT

1. `python simulate.py` produces same output as before for same seed
2. `pytest src/python/tests/ -v` passes (100%)
3. `test_phase_signatures.py` passes (API freeze maintained)
4. Person.stressful_event() no longer exists

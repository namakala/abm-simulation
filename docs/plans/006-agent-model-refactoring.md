---
title: Agent Model Refactoring
description: Compose phase functions into Person.step() and StressModel.step()
date: 2026-06-04
---

# Overview

Rewrite Person.step() as a thin orchestrator that builds AgentState, calls each phase function, and applies state_deltas. Eliminate the monolithic stressful_event(). Ensure all existing integration tests pass.

# Goals

- Person.step() delegates entirely to phase functions
- Person.stressful_event() removed; logic in WP 1-5
- Person.interact() simplified to call interaction phase
- StressModel.step() data collection updated
- All existing tests pass

# Implementation Steps

- [ ] 1. Restructure `Person.__init__()` to store config in a way phase functions can access
- [ ] 2. Rewrite `Person.step()`:
  - Build AgentState dict from self.*
  - Call each phase function in order: stress_perception, resilience_activation, resource_allocation, stress_buffering
  - Apply returned state_deltas to self
  - Store observations in self._phase_obs for debugging
- [ ] 3. Remove `Person.stressful_event()` — call WP 1-2 phase functions directly
- [ ] 4. Simplify `Person.interact()` — delegate to interaction phase function
- [ ] 5. Update `StressModel.step()`:
  - Optionally collect phase observations
  - Keep existing DataCollector reporters
- [ ] 6. Update `_daily_reset()` if needed (likely minimal changes)
- [ ] 7. Run full test suite: `pytest src/python/tests/ -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Phase function API changes during WP 1-5 | High | High | Freeze phase signatures before starting WP 6 |
| DataCollector misses new state vars | Med | Med | Add agent reporters for any new observation fields |

# UAT

1. `python simulate.py` runs end-to-end with same behavior
2. All existing tests pass (no regression)
3. Agent data collected correctly

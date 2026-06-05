---
title: Cleanup and Documentation
description: Archive old code, update architecture docs, ensure full test suite passes
date: 2026-06-05
---

# Overview

Final cleanup after all phases implemented, parameterized, integrated, tested, and demonstrated. Archive superseded code, remove redundant tests, and update architectural documentation to reflect the two-context phase pipeline, sequential delegation, and assumption parameterization.

# Goals

- Old .py demos archived (not deleted) to src/python/demos/.archive/
- Superseded tests (test_correlation_validation.py) archived
- ARCHITECTURE.md updated with two-context phase pipeline diagram
- ARCHITECTURE_MODEL.md updated with full state variable table and assumption parameter reference
- All feature docs corrected with updated theory (PSS-10 vs Resilience r = -0.57, etc.)
- Full test suite passes, simulation runs end-to-end

# Implementation Steps

- [ ] 1. Archive old .py demos: move to src/python/demos/.archive/
  - Keep only .qmd files in src/python/demos/
  - Include: agent_initialization_demo.py, stress_processing_mechanism.py, track_daily_stress.py, etc.
- [ ] 2. Review and archive superseded tests:
  - Keep: conftest.py, test_phase_signatures.py, core utility tests (test_math_utils.py, test_config_*.py)
  - Archive: test_correlation_validation.py (replaced by Plan 008 tests)
  - Flag: tests that overlap with Plan 008 integrated tests
- [ ] 3. Update architecture docs:
  - `docs/agents/ARCHITECTURE.md`: add two-context pipeline diagram, PhaseFrequency, sequential delegation via _apply_delta
  - `docs/agents/ARCHITECTURE_MODEL.md`: add phase pipeline decomposition, state variable table with each phase that modifies it, assumption parameter reference
  - `docs/features/stress-perception.md`: correct PSS-10 vs Resilience r to -0.57 (Thomas & Zolkoski 2020)
  - `docs/features/resilience-dynamics.md`: update overload formula, add assumption param reference
  - `docs/features/resource-management.md`: clarify PF vs interaction boundary, social exchange in Plan 004
  - `docs/features/agent-interactions.md`: document win-win/lose-lose state machine, support detection
- [ ] 4. Remove Person.stressful_event() from docs and code references
- [ ] 5. Run full test suite: `pytest src/python/tests/ -v`
- [ ] 6. Run end-to-end: `python simulate.py`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Deleting a test that is still useful | Med | High | Move to .archive/ instead of delete; check coverage before archiving |
| Docs out of sync with code | Low | Med | Update docs in same session as code changes |

# UAT

1. `pytest src/python/tests/ -v` passes (100%)
2. `python simulate.py` runs without errors
3. ARCHITECTURE.md correctly describes two-context phase pipeline with diagram
4. All old demos safely archived, not deleted
5. No references to Person.stressful_event() in doc or code

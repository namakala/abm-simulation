---
title: Cleanup and Documentation
description: Remove old code, archive demos, update architecture docs
date: 2026-06-04
---

# Overview

Final cleanup after all phases are implemented, tested, and demonstrated. Archive superseded code, remove redundant tests, and update architectural documentation to reflect the new phase structure.

# Goals

- Old demos archived (not deleted)
- Superseded tests removed or marked as deprecated
- ARCHITECTURE.md updated with phase pipeline
- Feature docs updated with corrected theory
- Full test suite passes

# Implementation Steps

- [ ] 1. Archive old demos:
  - Move to `src/python/demos/.archive/`: agent_initialization_demo.py, stress_processing_mechanism.py, track_daily_stress.py, etc.
  - Keep only new phase demos in `src/python/demos/`
- [ ] 2. Review and remove/skip superseded tests:
  - Keep: conftest.py, core utility tests (test_math_utils.py, test_config_*.py)
  - Remove: tests that only stress individual functions now covered by phase tests
  - Flag: tests that overlap with WP 7 integrated tests
- [ ] 3. Update architecture docs:
  - `docs/agents/ARCHITECTURE.md` — add phase pipeline diagram, observation contract
  - `docs/agents/ARCHITECTURE_MODEL.md` — update to reflect phase decomposition
  - `docs/features/stress-perception.md` — correct PSS-10 vs Resilience r to [-0.6, -0.1]
  - `docs/features/resilience-dynamics.md` — update overload formula
  - `docs/features/resource-management.md` — clarify PF vs interaction boundary
  - `docs/features/agent-interactions.md` — document interaction does not modify PF
- [ ] 4. Run full test suite: `pytest src/python/tests/ -v`
- [ ] 5. Run simulation end-to-end: `python simulate.py`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Deleting a test that is still useful | Med | High | Move to .archive/ instead of delete; check coverage before removing |
| Docs out of sync with code | Low | Med | Update docs in same session as code changes |

# UAT

1. `pytest src/python/tests/ -v` passes (100%)
2. `python simulate.py` runs without errors
3. ARCHITECTURE.md correctly describes phase pipeline
4. All old demos safely archived, not deleted

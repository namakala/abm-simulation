---
title: CI/CD, Lint Gates, TypedDict Fix
description: Update GitHub Actions, add lint/typecheck gates, fix _adapted_network in AgentState
date: 2026-06-05
---

# Overview

Cover 3 low-severity gaps. CI/CD workflows need updating for new pixi tasks (calibrate, quarto, slow tests). Linting and type-checking gates are missing from plans. The `_adapted_network` flag added by Plan 001 is not in the AgentState TypedDict from Plan 000.

# Goals

- GitHub Actions runs fast tests on PR, slow tests on schedule, validates quarto rendering
- `pixi run lint` and `pixi run typecheck` pass after all refactoring
- `adapted_network` field in AgentState TypedDict
- Full test suite + lint + typecheck pass in CI

# Implementation Steps

## Step 1: Update GitHub Actions

- [ ] 1a. Add `pytest -m "not slow"` as the PR trigger step (fast unit tests)
- [ ] 1b. Add `pytest -m slow` as a nightly/scheduled workflow (integration + calibration)
- [ ] 1c. Add `pixi run calibrate` step to confirm calibration script runs without error
- [ ] 1d. Add `pixi run quarto src/python/demos/` step to confirm all .qmd files render
- [ ] 1e. Add `pixi run lint` and `pixi run typecheck` as PR checks

## Step 2: Add Lint and TypeCheck Tasks

- [ ] 2a. Add to `pixi.toml`:
  ```
  [tasks]
  lint = "ruff check src/python/"
  typecheck = "mypy src/python/"
  ```
- [ ] 2b. Fix any ruff violations that accumulated during phase extraction
- [ ] 2c. Fix any mypy type errors in the new phase functions and extracted code
- [ ] 2d. Add lint and typecheck as verification steps in Plan 011

## Step 3: Fix _adapted_network in AgentState

- [ ] 3a. Add `adapted_network: bool` to `AgentState` TypedDict in `phases/interfaces.py`
- [ ] 3b. Update `_write_back_state()` in `Person` (Plan 008) to write back `adapted_network` to `self._adapted_network`
- [ ] 3c. Update `_build_agent_state()` to read `self._adapted_network`
- [ ] 3d. Update `ARCHITECTURE_MODEL.md` agent state table to include `_adapted_network`

## Step 4: Final Verification

- [ ] 4. Run: `pixi run lint && pixi run typecheck && pytest src/python/tests/ -v`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CI changes break PR workflow | Low | High | Test workflow locally with `act` or dry-run commit |
| mypy finds many pre-existing violations | Med | Med | Silo by file; fix only files touched by Plans 012-014 |
| Quarto not installed in CI runner | Med | Low | Add `pixi add quarto` to deps or skip in PR workflow |

# UAT

1. `pixi run lint` exits 0
2. `pixi run typecheck` exits 0
3. `pytest src/python/tests/ -v` passes
4. `python -c "from src.python.phases.interfaces import AgentState; assert 'adapted_network' in AgentState.__annotations__"`
5. `.github/workflows/` has `test-slow` and `render-demos` jobs

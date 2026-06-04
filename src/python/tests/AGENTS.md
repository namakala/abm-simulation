---
title: Python Test Suite
description: Scope-specific instructions for editing tests in src/python/tests/
---

## Organization

One test file per module + integration tests. All under `src/python/tests/`.

| Pattern | Examples |
|---------|----------|
| `test_<module>.py` | `test_stress_utils.py`, `test_affect_utils.py` |
| `test_<feature>.py` | `test_daily_reset_functionality.py`, `test_homeostatic_adjustment.py` |
| `test_*_integration.py` | `test_agent_integration.py`, `test_model_integration.py` |

## Running Tests

```bash
pytest src/python/tests/                          # all tests
pytest src/python/tests/ -v                       # verbose
pytest src/python/tests/ -k "stress"              # filtered by keyword
pytest src/python/tests/test_stress_utils.py      # single file
```

## Fixtures

Shared fixtures in `conftest.py`. Provides: config, RNG, agent instances, stress events, threshold params, appraisal weights, protective factors, interaction configs.

Use `@pytest.fixture` — no `conftest.py` imports needed in test files.

## Conventions

- **Naming**: `test_<behavior>` — describes what is being verified
- **One assertion per test** preferred (separate concerns)
- **Test behavior, not implementation** — mock at boundaries only
- **Pure functions**: inject RNG for deterministic tests

## Reference

- Python module docs: `@src/python/AGENTS.md`
- Coding standards: `@docs/agents/STANDARDS.md`

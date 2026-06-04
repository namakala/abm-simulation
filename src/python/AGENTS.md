---
title: Python Simulation Module
description: Scope-specific instructions for editing the ABM simulation core in src/python/
---

## Overview

Agent-based model for mental health research. Utility-first modular layout. Mesa framework. Config-driven via `.env`.

## Architecture

```
config.py ──► agent.py ──► model.py
  │              │
  ▼              ▼
*_utils.py ──► pure functions (injectable RNG/deps)
```

- **`agent.py`**: `Person` agent — stress events, social interactions, affect, resources, PSS-10
- **`model.py`**: `StressModel` (Mesa) — NetworkGrid, DataCollector, daily step orchestration
- **`config.py`**: `Config` dataclass — `.env` loading, validation, type conversion
- **utility modules**: stateless pure functions with dependency injection

## Module Map

| File | Purpose |
|------|---------|
| `agent.py` | Person agent: stress/social/affect dynamics (808 lines) |
| `model.py` | StressModel: Mesa ABM orchestration (861 lines) |
| `config.py` | .env config parsing and validation (717 lines) |
| `stress_utils.py` | Event gen, appraisal, threshold, PSS-10 mapping (1177 lines) |
| `affect_utils.py` | Social influence, affect dynamics, recovery (1209 lines) |
| `resource_utils.py` | Resource allocation, protective factors (738 lines) |
| `math_utils.py` | Sampling, clamping, distributions (508 lines) |
| `analysis_utils.py` | Stats computation, correlation analysis (388 lines) |
| `visualization_utils.py` | Population plots, distribution plots (376 lines) |

## Conventions

- **Imports**: `from src.python.X import Y` — absolute, no relative imports
- **Order**: standard library → third-party → local, alphabetical within groups
- **Functions**: under 50 lines, pure where possible, injectable RNG
- **Types**: all function signatures typed. No `Any`, no implicit casts.
- **Errors**: explicit error handling. No bare `except:`.

## Testing

```bash
pytest src/python/tests/          # all tests
pytest src/python/tests/ -v -k stress   # filtered
```

Tests at `@src/python/tests/AGENTS.md`. Shared fixtures in `conftest.py`.

## Running

```bash
python simulate.py            # CLI runner (project root)
python src/python/demos/*.py  # exploration scripts
```

Demos at `@src/python/demos/AGENTS.md`.

## Debugging

Debug scripts at `@src/python/debug/AGENTS.md` — isolating threshold/stress pipeline issues.

## Reference

- Project standards: `@docs/agents/STANDARDS.md`
- Known pitfalls: `@docs/agents/KNOWN_PITFALLS.md`
- Architecture: `@docs/agents/ARCHITECTURE.md`

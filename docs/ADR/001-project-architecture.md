---
title: Project Architecture
description: Python 3.12, Mesa 3.3.0, NetworkX, pixi, pytest, GitHub Actions
status: accepted
date: 2025-06-04
---

# Context

New data analysis project for agent-based modeling of mental health promotion cost-effectiveness. Needed to decide runtime, framework, testing approach, and CI/CD pipeline.

# Decision

- **Runtime:** Python 3.12.11 — enables rapid development with rich scientific ecosystem
- **ABM Framework:** Mesa 3.3.0 — established ABM patterns, built-in DataCollector, NetworkGrid
- **Graph Library:** NetworkX 3.5 — graph operations, Watts-Strogatz generation, network metrics
- **Data Stack:** NumPy 2.3.3 + Pandas 2.3.2 — numerical computing and tabular data
- **Env Manager:** pixi — conda-compatible, deterministic lockfile, fast
- **Testing:** pytest with 85% coverage threshold, markers for unit/integration/config/slow
- **CI/CD:** GitHub Actions — automated testing on PR via `prefix-dev/setup-pixi`
- **Doc Generation:** Quarto for manuscript pipeline (see ADR-007)

Alternatives considered: custom ABM (rejected — reinvention without benefit), pure Python without Mesa (rejected — would lose DataCollector, scheduler, grid).

# Impact

Python + Mesa accelerate development with battle-tested patterns. pixi lockfile ensures fully reproducible envs. GitHub Actions enforce quality gates on every PR. Switching to ruff (see ADR-004) replaces three tools while maintaining standards.

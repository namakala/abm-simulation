---
title: System Architecture
description: Agent-based model for mental health promotion cost-effectiveness — tech stack, data flow, design principles
status: stable
date: 2025-06-04
---

# Overview

ABM simulating workplace mental health promotion programs. Agents experience stress events with challenge/hindrance appraisal, allocate resources across protective factors, interact via small-world social networks, and adapt over time. Output: time series, network snapshots, cost-effectiveness ratios for universal/selective/indicated programs.

# Tech Stack

| Layer | Choice |
|-------|--------|
| Runtime | Python 3.12.11 |
| ABM Framework | Mesa 3.3.0 |
| Graph Library | NetworkX 3.5 |
| Data Processing | NumPy 2.3.3 + Pandas 2.3.2 |
| Visualization | matplotlib + seaborn |
| Env Manager | pixi |
| Lint/Format | ruff |
| Type Checker | mypy |
| Testing | pytest (85% cov threshold) |
| CI/CD | GitHub Actions |
| Manuscript | Quarto |

# Design Principles

- **Modular utility design.** Each mechanism (stress, affect, resources, math) in separate file. Enables independent testing and reuse.
- **Explicit I/O boundaries.** Data validation at ingestion. Idempotent transformations. No hidden state.
- **Configuration over hardcode.** All tunable parameters in .env with type conversion and validation. Enables systematic sweeps.
- **Deterministic simulation.** Seeded RNG for full reproducibility across replicates.

# Data Flow

Client / Runner → Model (Mesa scheduler) → Agent step (event gen → appraisal → threshold eval → resource allocation → network update) → DataCollector → DataFrame → Statistical Analysis → Output (figures, tables, manuscripts).

# Key Decisions

| Decision | ADR | Summary |
|----------|-----|---------|
| Project Architecture | `@docs/ADR/001-project-architecture.md` | Python + Mesa + pixi + pytest + GHA |
| Environment & Tooling | `@docs/ADR/004-environment-tooling.md` | pixi, ruff, mise, pytest plugins |
| Data Processing Pipeline | `@docs/ADR/007-data-processing-pipeline.md` | Quarto, mpl+seaborn, pandas+numpy |
| Model Architecture | `@docs/ADR/008-model-architecture.md` | Agent structure, stress appraisal, networks |

# Subsystems

Detailed model design: `@docs/agents/ARCHITECTURE_MODEL.md`
Mathematical notation reference: `@docs/agents/MATH_NOTATION.md`

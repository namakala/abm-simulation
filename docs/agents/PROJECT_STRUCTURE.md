---
title: Project Structure
description: Directory tree and key files for the ABM simulation project
date: 2025-06-04
---

# Overview

Utility-first modular layout. Core ABM in `src/python/` with separate files per mechanism. Tests co-located in `src/python/tests/`. Docs in `docs/` with feature docs, agents, ADRs, and manuscript.

# Directory Map

```
abm-simulation/
├── pixi.toml                  # Project manifest + dependencies
├── pixi.lock                  # Reproducible environment lockfile
├── simulate.py                # CLI runner
├── .env                       # Active configuration
├── .env.example               # Documented configuration template
├── src/
│   ├── python/
│   │   ├── agent.py           # Person agent with stress/social dynamics
│   │   ├── model.py           # StressModel (Mesa) with DataCollector
│   │   ├── stress_utils.py    # Event gen, appraisal, threshold eval
│   │   ├── affect_utils.py    # Social interaction, affect dynamics
│   │   ├── resource_utils.py  # Protective factor allocation
│   │   ├── math_utils.py      # Sampling, clamping utilities
│   │   ├── config.py          # .env parsing, validation
│   │   ├── analysis_utils.py  # Post-simulation analysis
│   │   ├── visualization_utils.py
│   │   ├── tests/             # 35+ test files
│   │   ├── demos/             # Exploration scripts
│   │   └── debug/             # Debugging utilities
│   ├── shell/                 # extract_env.sh, update_env_example.sh
│   ├── r/                     # R analysis pipeline (planned)
│   └── sql/                   # Schema and queries (planned)
├── docs/
│   ├── agents/                # Agent-facing docs (this directory)
│   ├── ADR/                   # Architecture Decision Records
│   ├── features/              # Feature documentation (8 files)
│   ├── templates/             # Doc generation templates
│   ├── article.qmd            # Manuscript source
│   └── ref.bib                # Bibliography
├── data/
│   ├── raw/                   # Raw simulation outputs
│   └── processed/             # Cleaned analysis data
└── .github/workflows/         # CI/CD
```

# Key Files

| Path | Purpose |
|------|---------|
| `@AGENTS.md` | Root agent instructions |
| `@pixi.toml` | Env + deps + tasks |
| `@simulate.py` | CLI entry point |
| `@.env.example` | Parameter template |

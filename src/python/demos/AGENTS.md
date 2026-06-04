---
title: Python Demo Scripts
description: Scope-specific instructions for demo/exploration scripts in src/python/demos/
---

## Purpose

Exploration, validation, and debugging scripts. Not part of the core library. May contain rough or experimental code not covered by tests.

## Scripts

| Script | Purpose |
|--------|---------|
| `agent_initialization_demo.py` | Agent init with realistic variation, distribution checks |
| `agent_correlation_analysis.py` | Correlation analysis of agent-level variables |
| `agent_diversity_demo.py` | Comprehensive diversity demo (stats, trajectories, correlations) |
| `analyze.py` | Quick EDA on model.csv / agent.csv |
| `parameter_sweep_correlations.py` | Latin Hypercube sweep for correlation stability |
| `population_analysis.py` | Population variation analysis verification |
| `population_correlation_analysis.py` | Population-level correlation analysis |
| `stress_pipeline_debug_demo.py` | Stress processing pipeline + PSS-10 correlation debug |
| `stress_processing_mechanism.py` | Complete stress pipeline: appraisal, social, reset, decay |
| `track_daily_stress.py` | Inline stress tracking per agent per day |

## Running

```bash
python src/python/demos/<name>.py
```

Run from project root. Most demos expect `.env` to exist with valid config.

## Reference

- Python module docs: `@src/python/AGENTS.md`

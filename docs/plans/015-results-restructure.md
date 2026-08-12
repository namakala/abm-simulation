---
title: Results Section Restructure
description: Restructure results draft for reading flow and reconcile all numbers to the stage-7 pipeline
date: 2026-08-11
---

# Overview

The Results section (docs/draft, assembled by docs/article.qmd) has four structural problems: no subheadings, the time-series figure is narrated before it appears (orphaned in `_results_final_state.qmd`, defined later in `_results_homeostasis.qmd`), the narrative order mismatches the research question, and two conflicting "full model" number sets exist (stale base run vs stage-7 pipeline). This plan restructures the section into five headed blocks and makes the stage-7 run the single source of truth for figures, tables, and narrative numbers.

# Goals

- Results reads as five headed blocks in order: baseline -> dynamics/homeostasis -> heterogeneity -> mechanism build-up -> emergent end state
- Every number in Results derives from `data/output/stage7_model.csv` and `stage7_agent.csv`; no hardcoded model stats remain
- Figures regenerated from stage-7 data; each figure/table defined and referenced once
- stage-7 sub-tables folded into the main model-level and agent-level tables
- Each draft `.qmd` file stays under the 100-line documentation limit
- `docs/article.qmd` renders to PDF without errors or undefined references

# Implementation Steps

## WP1: Data pipeline — export figures and stats from the stage-7 run

- [ ] 1.1 Add `export_results_assets(model_csv, agent_csv, output_dir, figures_dir)` to `src/python/demos/cumulative_blocks.py`: renders three figures from stage-7 exports via `create_visualization_report` (Step==1 slice, final Step slice) and `create_time_series_visualization` (model df), and writes `stage7_results_stats.csv` (population-level and agent-step-level mean/SD/CV/range, final-day agent stats, initial and final pss10-stress correlation)
- [ ] 1.2 Call `export_results_assets` from `run_cumulative_demo()` after the stage-7 export; figures go to `docs/figures/full_model_initial_population.pdf`, `full_model_final_population.pdf`, `full_model_time_series.pdf`
- [ ] 1.3 Add `src/python/tests/test_cumulative_blocks_export.py` with synthetic dataframes (no full simulation) asserting the stats CSV contents and figure file creation under a tmp dir
- [ ] 1.4 Run `pixi run cumulative-blocks` and `pixi run test-unit`; both must pass

## WP2: Restructure the Results draft fragments

- [ ] 2.1 Create `docs/draft/_results_setup.qmd` with an R setup chunk: `find_data_dir()` + load `stage7_model.csv` (m), `stage7_agent.csv` (a), `stage7_results_stats.csv` (stats), and compute named narrative strings (mean +/- SD, CVs, ranges) used by inline R in later fragments
- [ ] 2.2 `_results.qmd`: add roadmap paragraph, include `_results_setup.qmd` first, then include the five fragments in the agreed order
- [ ] 2.3 `_results_initial_state.qmd`: add `## Baseline: initial population state` heading; retarget figure to `full_model_initial_population.pdf`; verify claims against step-1 data (pss10-stress correlation)
- [ ] 2.4 `_results_homeostasis.qmd`: add `## Homeostatic stability of population dynamics` heading; move the time-series narrative here next to `fig-time-series` (`full_model_time_series.pdf`); replace hardcoded `tbl-model-level` LaTeX with an R `kable` chunk computed from `m`; update narrative numbers via inline R; drop the heterogeneity CV sentences (moved to 2.5)
- [ ] 2.5 `_results_individual_dynamics.qmd`: add `## Individual heterogeneity within population stability` heading; replace hardcoded `tbl-agent-level` LaTeX with an R `kable` chunk computed from `a`; add the CV-contrast narrative (values from data) here only
- [ ] 2.6 `_results_cumulative_block.qmd`: remove the "Full model (stage 7)" subsection (both kables, folding their content into the main tables); add a transition sentence pointing to the main results; keep stage composition + stage metrics
- [ ] 2.7 `_results_final_state.qmd`: add `## Emergent associations at the end state` heading; remove the time-series paragraph; retarget figure to `full_model_final_population.pdf`
- [ ] 2.8 Delete `docs/figures/base_*.pdf` and the two stale `*_not_available.txt` placeholders
- [ ] 2.9 Add `pixi run article` task (`src/shell/article-render.sh` mirroring `quarto-render.sh` but rendering `docs/article.qmd`) for reproducible rendering

## WP3: Render and verify the article

- [ ] 3.1 Render: `pixi run article` must exit 0
- [ ] 3.2 Verify float order and numbering in the PDF/article.tex: Fig1 initial, Fig2 time-series, Table1 model-level, Table2 agent-level, Table3 stage composition, Table4 stage metrics, Fig3 final population
- [ ] 3.3 Grep `article.tex` for duplicate `\label{fig-time-series}` and for "??" undefined references; both must be absent
- [ ] 3.4 Cross-check every PDF number against `data/output/*.csv`

# Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `cumulative-blocks` run is slow | Med | Med | Run once; export logic unit-tested with synthetic data |
| R chunks / inline R fail during render | Med | High | Small shared setup chunk; render each file individually before full article |
| New data changes narrative claims (CVs, correlations) | Med | Med | Rewrite narrative from computed values; check Discussion consistency |
| Figures visually differ from old base figures | Low | Med | Inspect rendered PDFs; reuse existing `style_config` defaults |

# UAT

1. `pixi run cumulative-blocks` exits 0; `docs/figures/full_model_*.pdf` and `data/output/stage7_results_stats.csv` exist; `docs/figures/base_*` gone
2. `pixi run test-unit` passes, including `test_cumulative_blocks_export.py`
3. `pixi run article` exits 0 and produces `docs/agent-based-model-psychological-resilience.pdf`
4. PDF Results section has five subheaded blocks in the agreed order with a roadmap paragraph
5. Float numbering follows Fig1 -> Fig2 -> T1 -> T2 -> T3 -> T4 -> Fig3; each float referenced once in narrative order
6. No "??" in the PDF; `article.tex` has a single `\label{fig-time-series}`
7. All numeric values in the PDF trace to `data/output/*.csv`
8. Each `docs/draft/*.qmd` file is 100 lines or fewer

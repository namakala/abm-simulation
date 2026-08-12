# TODO

## WP-1: Data pipeline — export figures and stats from the stage-7 run

- [x] 1.1 Add `export_results_assets()` to `src/python/demos/cumulative_blocks.py`
- [x] 1.2 Wire into `run_cumulative_demo()`; figures to `docs/figures/full_model_*.pdf`
- [x] 1.3 Add `src/python/tests/test_cumulative_blocks_export.py` (synthetic data, tmp dirs)
- [x] 1.4 `pixi run cumulative-blocks` + `pixi run test-unit` pass

## WP-2: Restructure the Results draft fragments

- [ ] 2.1 Create `docs/draft/_results_setup.qmd` (R setup chunk, named narrative strings)
- [ ] 2.2 `_results.qmd`: roadmap paragraph, setup include, fragments in agreed order
- [ ] 2.3 `_results_initial_state.qmd`: heading, retarget figure, verify vs step-1 data
- [ ] 2.4 `_results_homeostasis.qmd`: heading, time-series narrative, R `kable` table, inline-R numbers
- [ ] 2.5 `_results_individual_dynamics.qmd`: heading, R `kable` table, CV contrast here only
- [ ] 2.6 `_results_cumulative_block.qmd`: remove stage-7 subsection, transition sentence
- [ ] 2.7 `_results_final_state.qmd`: heading, remove time-series paragraph, retarget figure
- [ ] 2.8 Delete `docs/figures/base_*.pdf` and stale `*_not_available.txt`
- [ ] 2.9 Add `pixi run article` task + `src/shell/article-render.sh`

## WP-3: Render and verify the article

- [ ] 3.1 `pixi run article` exits 0
- [ ] 3.2 Float order: Fig1 -> Fig2 -> T1 -> T2 -> T3 -> T4 -> Fig3
- [ ] 3.3 No duplicate `\label{fig-time-series}`; no "??" refs
- [ ] 3.4 All PDF numbers trace to `data/output/*.csv`

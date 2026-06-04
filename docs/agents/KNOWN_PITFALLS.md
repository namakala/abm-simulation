---
title: Known Pitfalls
description: Common mistakes and antipatterns to avoid — Python, data analysis, ABM-specific
date: 2025-06-04
---

# Python Pitfalls

- **Mutable default args.** Use `None` and initialize inside function.
- **`is` vs `==`.** `is` checks identity, not equality. Use `==` for value comparison.
- **Circular imports.** Structure dependency graph: config → utils → agent → model. No cycles.
- **Bare `except:`.** Catches `KeyboardInterrupt`, `SystemExit`. Use `except Exception:`.
- **Function-level imports.** Banned — all imports at module level only.

# Data Analysis Pitfalls

- **Out-of-order notebook execution.** Quarto mitigates via sequential rendering.
- **Mutating while iterating.** Copy DataFrames before mutation. Use `.loc` for assignment.
- **Silent SettingWithCopyWarning.** Use `.copy()` explicitly when chaining.
- **Memory blowup.** Load data in chunks for large parameter sweep results.
- **Lost seed state.** Use single `RandomState` instance, not global `np.random`.

# ABM / Modeling Pitfalls

- **N+1 queries in DataCollector.** Collect per-agent data in bulk, not per-agent-loop.
- **Emergent behavior surprise.** Always run multiple replicates before interpreting results.
- **Calibration overfitting.** Validate on held-out patterns or alternative network topologies.
- **Configuration drift.** .env and .env.example must stay in sync. Run `pixi run extract-env` after adding parameters.
- **Ignoring degenerate behavior.** Test boundary conditions: all resources zero, infinite stress, empty network.

---
title: Environment & Tooling
description: pixi for env, ruff for lint/format, mise for version mgmt, pytest for testing
status: accepted
date: 2025-06-04
---

# Context

Python project needing deterministic environments, fast linting/formatting, Python version management, and robust test infrastructure.

# Decision

- **Env Manager:** pixi (keep existing) — conda-compatible, Rust-based, `pixi.lock` for reproduction
- **Linter/Formatter:** ruff — replaces flake8 + black + isort with single 10-100x faster tool
- **Version Manager:** mise — Rust-based polyglot, single `mise.toml` config, manages Python alongside potential future runtimes
- **Testing:** pytest with plugins: pytest-cov (coverage), pytest-xdist (parallel), pytest-mock, pytest-benchmark, hypothesis, faker
- **Static Analysis:** mypy for type checking, bandit for security scanning

Alternatives considered: uv (faster resolver but pip-only, not conda-compatible), poetry (mature but slower), pyenv (Python-only, no polyglot), fnm/nvm (Node-only).

# Impact

ruff unifies lint+format at much higher speed — fewer deps, faster CI. pixi stays for conda-compat needed by scientific deps (r-base, mesa with compiled extensions). mise one config for all runtimes.

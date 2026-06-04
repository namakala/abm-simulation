---
title: Coding & Documentation Standards
description: Code conventions, Python import rules, writing style, documentation guidelines
date: 2025-06-04
---

# Code Conventions

- **Type hints required.** All function signatures typed. No `Any`, no implicit `as` casts.
- **Functions under 50 lines.** Split when justified by complexity.
- **Explicit error handling.** No bare `except:` or silent `throw`.
- **Tests alongside code.** Test behavior, not implementation. One assertion per test preferred.
- **Prefer readability.** One-liners not goals. Descriptive variable names.
- **Naming:** `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants.

# Python Imports

Order: standard library → third-party → local. Alphabetical within each group. One import per line. Use absolute imports from project root (`from src.python.X import Y`). No relative imports. No function-level imports.

# Writing Style (Documentation)

- Use plain English, no emojis. Academic tone for manuscripts, instructional for agent docs.
- Cite code paths with line numbers for feature docs: `@src/python/stress_utils.py:42`
- Standardize lexicon with `.env` parameter names. Do not substitute defined terms.
- Mathematical equations require formal LaTeX and symbol definitions.

# Academic Writing

- No bold or italics. Markdown headers for structure.
- Cite from `@docs/ref.bib`. Be descriptive and objective.
- Use `@docs/features/` files as source of truth for model mechanisms.
- Mathematical notation must reference `@docs/agents/MATH_NOTATION.md`.

# Config & CLI

- All tunable values in `.env`. No hardcoded magic numbers in source.
- Simulation CLI args override .env values. Documented in `@README.md:79`.

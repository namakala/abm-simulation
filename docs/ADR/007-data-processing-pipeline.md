---
title: Data Processing Pipeline
description: Quarto manuscripts, matplotlib+seaborn figures, pandas+numpy processing
status: accepted
date: 2025-06-04
---

# Context

Research project producing protocol documents, scientific manuscripts, and figures. Needed notebook environment, visualization strategy, and data processing tools.

# Decision

- **Notebook/Manuscript:** Quarto — single-source authoring, multi-format output (PDF/HTML/DOCX), VS Code native, supports LaTeX and bibliography management
- **Visualization:** matplotlib + seaborn — publication-quality static figures, total control over axes/typography, already installed stack
- **Data Processing:** pandas + numpy — widest ecosystem, integration with Mesa DataCollector DataFrames, already in pixi.toml

Alternatives considered: Jupyter Lab (largest ecosystem but weaker multi-format output), Marimo (reactive but less mature for publications), plotly (interactive but not publication-grade), polars (faster but less ecosystem).

# Impact

Quarto enables one-source-to-many-formats pipeline for manuscripts. matplotlib+seaborn produce journal-ready figures. pandas DataFrames align directly with Mesa DataCollector output, minimizing transformation overhead.

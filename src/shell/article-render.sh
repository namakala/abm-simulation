#!/usr/bin/env bash
# Render the article PDF (docs/article.qmd) reproducibly via the pixi env.
# Mirrors src/shell/quarto-render.sh but renders the manuscript article.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIXIPY="$PROJECT_ROOT/.pixi/envs/default"

# QUARTO_PYTHON must be the absolute path to the python binary
export QUARTO_PYTHON="$PIXIPY/bin/python"

cd "$PROJECT_ROOT/docs" || exit 1

if quarto render article.qmd; then
    echo "Article rendered: docs/agent-based-model-psychological-resilience.pdf"
    exit 0
else
    echo "Article render failed" >&2
    exit 1
fi

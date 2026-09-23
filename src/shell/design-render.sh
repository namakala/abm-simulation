#!/usr/bin/env bash
# Render the design document (docs/design.md) to PDF via the pixi env.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIXIPY="$PROJECT_ROOT/.pixi/envs/default"

# QUARTO_PYTHON must be the absolute path to the python binary
export QUARTO_PYTHON="$PIXIPY/bin/python"

cd "$PROJECT_ROOT/docs" || exit 1

if quarto render design.md; then
    echo "Design document rendered: docs/design.pdf"
    exit 0
else
    echo "Design document render failed" >&2
    exit 1
fi

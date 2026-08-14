#!/usr/bin/env bash
# Don't use -e so we can continue after individual demo failures

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEMOS_DIR="$PROJECT_ROOT/src/python/demos"
PIXIPY="$PROJECT_ROOT/.pixi/envs/default"

# QUARTO_PYTHON must be the absolute path to the python binary
export QUARTO_PYTHON="$PIXIPY/bin/python"

failed=0
for f in "$DEMOS_DIR"/*.qmd; do
    name=$(basename "$f")
    echo "Rendering $name..."
    if quarto render "$f" > /tmp/quarto-$name.log 2>&1; then
        echo "  OK: $name"
    else
        echo "  WARN: $name failed — see /tmp/quarto-$name.log"
        failed=$((failed + 1))
    fi
done

echo ""
if [ $failed -gt 0 ]; then
    echo "$failed demo(s) had errors — see /tmp/quarto-*.log"
else
    echo "All demos rendered successfully"
fi
exit $failed

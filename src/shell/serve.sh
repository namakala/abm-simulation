#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEMOS_DIR="$SCRIPT_DIR/../python/demos"

for f in "$DEMOS_DIR"/*.qmd; do
  quarto render "$f"
done

python -m http.server 9000 --directory "$DEMOS_DIR"

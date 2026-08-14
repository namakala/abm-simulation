#!/bin/sh
# POSIX: extract KEY=DEFAULT from _get_env_value(...), _get_env_array(...),
#        or _env_float(...) calls in Python config files.
# Usage: ./extract_env.sh [file1 file2 ...]
#   Defaults to src/python/config.py src/python/assumption_config.py if no args.

if [ $# -eq 0 ]; then
    awk -f "$(dirname "$0")/extract_env.awk" \
        "src/python/config.py" \
        "src/python/assumption_config.py"
else
    awk -f "$(dirname "$0")/extract_env.awk" "$@"
fi

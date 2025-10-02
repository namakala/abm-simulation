#!/bin/sh
# POSIX: extract KEY=DEFAULT from _get_env_value(...) calls
# Usage: ./extract_env.sh [file]
file=${1:-src/python/config.py}

awk -f - "$file" <<'AWK'
/\._get_env_value/ {
    s = $0
    p = index(s, "._get_env_value")
    if (p == 0) next
    rest = substr(s, p)
    # match the first (...) pair that contains no other parentheses
    if (match(rest, /\([^()]*\)/)) {
        args = substr(rest, RSTART+1, RLENGTH-2)
        n = split(args, a, ",")

        # trim spaces and surrounding single/double quotes from each arg
        for (i = 1; i <= n; i++) {
            gsub(/^[ \t]+|[ \t]+$/, "", a[i])
            gsub(/^["']+|["']+$/, "", a[i])
        }

        key = (n >= 1 ? a[1] : "")
        val = (n >= 3 ? a[3] : "")

        # final trim (just in case) and print
        gsub(/^[ \t]+|[ \t]+$/, "", key)
        gsub(/^[ \t]+|[ \t]+$/, "", val)
        if (key != "") print key "=" val
    }
}
AWK

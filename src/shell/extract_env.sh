#!/bin/sh
# POSIX: extract KEY=DEFAULT from _get_env_value(...) or _get_env_array(...) calls
# Usage: ./extract_env.sh [file]
file=${1:-src/python/config.py}

awk -f - "$file" <<'AWK'
# helper functions
function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
function strip_quotes(s) { gsub(/^["']+|["']+$/, "", s); return s }

# bracket-aware split
function split_args(s, a,    i, c, level, n) {
    n = 1
    level = 0
    a[1] = ""
    for (i = 1; i <= length(s); i++) {
        c = substr(s, i, 1)
        if (c == "[" || c == "(") level++
        if (c == "]" || c == ")") level--
        if (c == "," && level == 0) {
            n++
            a[n] = ""
        } else {
            a[n] = a[n] c
        }
    }
    return n
}

# main processing
/\._get_env_(value|array)/ {
    s = $0
    if (match(s, /\._get_env_(value|array)/)) {
        p = RSTART
        rest = substr(s, p)

        if (match(rest, /\((.*)\)/, m)) {
            args = m[1]
            n = split_args(args, a)

            key = (n >= 1 ? trim(strip_quotes(a[1])) : "")
            val = (n >= 3 ? trim(a[3]) : "")

            if (key != "") print key "=" val
        }
    }
}
AWK

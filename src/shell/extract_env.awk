# POSIX awk: extract KEY=DEFAULT from Python config patterns.
# Handles two patterns:
#   1. self._get_env_value("KEY", type, default)  — config.py
#   2. self._get_env_array("KEY", type, [default]) — config.py
#   3. _env_float("KEY", default)                  — assumption_config.py
#
# Usage: awk -f extract_env.awk file1.py file2.py ...

function trim(s) {
    gsub(/^[ \t]+|[ \t]+$/, "", s)
    return s
}

function strip_quotes(s) {
    gsub(/^["\x27]+|["\x27]+$/, "", s)
    return s
}

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

function emit_single_line(rest, vpos,    args, n, a, key, val) {
    # rest starts at the opening paren
    # Find matching closing paren, handling nesting
    level = 0
    end = 0
    for (i = 1; i <= length(rest); i++) {
        c = substr(rest, i, 1)
        if (c == "(") level++
        if (c == ")") {
            level--
            if (level == 0) { end = i; break }
        }
    }
    if (end == 0) return  # no closing paren on this line
    args = substr(rest, 2, end - 2)
    n = split_args(args, a)
    key = (n >= 1 ? strip_quotes(trim(a[1])) : "")
    val = (n >= vpos ? strip_quotes(trim(a[vpos])) : "")
    if (key != "") print key "=" val
}

function emit_multi_line(text, vpos,    args, n, a, key, val) {
    # text is the full multi-line function call including newlines
    # Find opening paren
    p = index(text, "(")
    if (p == 0) return
    rest = substr(text, p)
    # Find closing paren at top level
    level = 0
    end = 0
    for (i = 1; i <= length(rest); i++) {
        c = substr(rest, i, 1)
        if (c == "(") level++
        if (c == ")") {
            level--
            if (level == 0) { end = i; break }
        }
    }
    if (end == 0) return
    args = substr(rest, 2, end - 2)
    # Normalize whitespace: collapse newlines/spaces/tabs into single spaces
    gsub(/[\n\r\t]+/, " ", args)
    gsub(/  +/, " ", args)
    args = trim(args)
    n = split_args(args, a)
    key = (n >= 1 ? strip_quotes(trim(a[1])) : "")
    val = (n >= vpos ? strip_quotes(trim(a[vpos])) : "")
    if (key != "") print key "=" val
}

# State for multi-line calls
in_call == 1 {
    buf = buf "\n" $0
    # Check if closing paren at level 0 is reached
    level = paren_level
    closed = 0
    for (i = 1; i <= length($0); i++) {
        c = substr($0, i, 1)
        if (c == "(") level++
        if (c == ")") {
            level--
            if (level == 0) { closed = 1; break }
        }
    }
    if (closed) {
        emit_multi_line(buf, val_pos)
        in_call = 0
    } else {
        paren_level = level
    }
    next
}

{
    s = $0

    # Skip Python function definitions (e.g. `def _env_float(key: str, ...):`)
    # so the signature is not misparsed as a config call.
    if (match(s, /^[ 	]*def[ 	]+/)) next

    # Pattern 1 & 2: self._get_env_value / self._get_env_array
    if (match(s, /\.[ \t]*_get_env_(value|array)\(/)) {
        rest = substr(s, RSTART + RLENGTH - 1)  # rest from '('
        # Check if closing paren is on this line
        has_close = 0
        level = 0
        for (i = 1; i <= length(rest); i++) {
            c = substr(rest, i, 1)
            if (c == "(") level++
            if (c == ")") {
                level--
                if (level == 0) { has_close = 1; break }
            }
        }
        if (has_close) {
            emit_single_line(rest, 3)
        } else {
            in_call = 1
            buf = s
            paren_level = level
            val_pos = 3
        }
        next
    }

    # Pattern 3: _env_float
    if (match(s, /_env_float\(/)) {
        rest = substr(s, RSTART + RLENGTH - 1)  # rest from '('
        has_close = 0
        level = 0
        for (i = 1; i <= length(rest); i++) {
            c = substr(rest, i, 1)
            if (c == "(") level++
            if (c == ")") {
                level--
                if (level == 0) { has_close = 1; break }
            }
        }
        if (has_close) {
            emit_single_line(rest, 2)
        } else {
            in_call = 1
            buf = s
            paren_level = level
            val_pos = 2
        }
        next
    }
}

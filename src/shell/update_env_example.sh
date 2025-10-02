#!/bin/sh

envfile=".env"
examplefile=".env.example"

awk -F= '
    # First pass: read .env and store all KEY=value
    FNR==NR && /^[A-Za-z_][A-Za-z0-9_]*=/ {
        vals[$1]=$0
        next
    }

    # Second pass: process .env.example
    /^[A-Za-z_][A-Za-z0-9_]*=/ {
        seen[$1]=1
        if ($1 in vals) print vals[$1]
        next
    }

    # Preserve comments and empty lines
    { print }
    
    END {
        # Add any variables from .env not already in .env.example
        for (k in vals)
            if (!(k in seen))
                print vals[k]
    }
' "$envfile" "$examplefile" > "$examplefile.tmp" && mv "$examplefile.tmp" "$examplefile"

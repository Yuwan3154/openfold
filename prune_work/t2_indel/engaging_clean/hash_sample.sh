#!/bin/bash
# READ-ONLY: sha256 of the files named in <listfile> (relative to <root>). Usage: hash_sample.sh <root> <listfile> <out>
set -u
cd "$1" || exit 1
n=0; miss=0
: > "$3"
while IFS= read -r p; do
  if [ -f "$p" ]; then sha256sum -- "$p" >> "$3"; n=$((n+1)); else echo "MISSING $p" >> "$3"; miss=$((miss+1)); fi
done < "$2"
echo "HASHED $1 n=$n missing=$miss -> $3"

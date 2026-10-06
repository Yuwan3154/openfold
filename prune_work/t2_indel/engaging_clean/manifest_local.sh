#!/bin/bash
# READ-ONLY manifest of a directory tree: relative path <TAB> size <TAB> mtime, gzipped. Usage: manifest_local.sh <root> <out.tsv.gz>
set -u
[ -d "$1" ] || { echo "ABSENT $1"; exit 1; }
find "$1" -type f -printf '%P\t%s\t%T@\n' | gzip > "$2"
n=$(zcat "$2" | wc -l); b=$(zcat "$2" | awk -F'\t' '{s+=$2} END {printf "%.0f", s}')
echo "MANIFEST $1 files=$n bytes=$b"

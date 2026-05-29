"""Take the first 1000 entries from pdb30_200513.tsv whose sequence length is in [50, 128],
write to pdb30_200513_1k.tsv and create the empty block_data_cache_bf16 dir.

Args:
  --src_tsv  /home/jupyter-chenxi/data/pdb30_200513/pdb30_200513.tsv
  --dst_dir  /home/jupyter-chenxi/data/pdb30_200513_1k
"""

import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--src_tsv",
    default="/home/jupyter-chenxi/data/pdb30_200513/pdb30_200513.tsv",
)
parser.add_argument(
    "--dst_dir",
    default="/home/jupyter-chenxi/data/pdb30_200513_1k",
)
parser.add_argument("--n", type=int, default=1000)
parser.add_argument("--min_len", type=int, default=50)
parser.add_argument("--max_len", type=int, default=128)
args = parser.parse_args()

dst_dir = Path(args.dst_dir)
dst_dir.mkdir(parents=True, exist_ok=True)
cache_dir = dst_dir / "block_data_cache_bf16"
cache_dir.mkdir(parents=True, exist_ok=True)
out_tsv = dst_dir / "pdb30_200513_1k.tsv"

n_kept = 0
n_seen = 0
n_too_short = 0
n_too_long = 0
with open(args.src_tsv, "r") as fin, open(out_tsv, "w") as fout:
    for line in fin:
        n_seen += 1
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        seq_id, seq = parts
        L = len(seq)
        if L < args.min_len:
            n_too_short += 1
            continue
        if L > args.max_len:
            n_too_long += 1
            continue
        fout.write(f"{seq_id}\t{seq}\n")
        n_kept += 1
        if n_kept >= args.n:
            break

print(f"Wrote {n_kept} / {args.n} entries to {out_tsv}")
print(f"  scanned: {n_seen}, too_short(<{args.min_len}): {n_too_short}, too_long(>{args.max_len}): {n_too_long}")
print(f"Created empty cache dir: {cache_dir}")

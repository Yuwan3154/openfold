#!/usr/bin/env python3
"""
Plot ground-truth length vs. final TM-score from hallucination outputs.
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot length vs final TM-score")
    parser.add_argument(
        "--outputs_dir",
        type=Path,
        required=True,
        help="Base outputs directory containing *_meta.json files",
    )
    parser.add_argument(
        "--blocks",
        type=str,
        default="10-37",
        help="Straight-through block range in output names (e.g. 10-37)",
    )
    parser.add_argument(
        "--pdb_ids",
        type=str,
        default="",
        help="Optional comma-separated PDB ids to include (e.g. 1a0o,1a34)",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the plot image",
    )
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.expanduser()
    output_path = args.output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdb_filter = {p.strip().lower() for p in args.pdb_ids.split(",") if p.strip()}

    pattern = f"**/*_st-blocks-{args.blocks}*_meta.json"
    meta_paths = sorted(outputs_dir.glob(pattern))
    if not meta_paths:
        raise ValueError(f"No meta files found under {outputs_dir} with pattern {pattern}")

    lengths: List[int] = []
    tms: List[float] = []

    for meta_path in meta_paths:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        pdb_id = str(meta.get("pdb_id", "")).lower()
        if pdb_filter and pdb_id not in pdb_filter:
            continue
        tm_score = meta.get("usalign", {}).get("tm_score", None)
        if tm_score is None:
            continue
        lengths.append(int(meta["ground_truth_length"]))
        tms.append(float(tm_score))

    if not lengths:
        raise ValueError("No valid TM-score entries found after filtering")

    order = np.argsort(np.array(lengths))
    lengths_np = np.array(lengths, dtype=np.float64)[order]
    tms_np = np.array(tms, dtype=np.float64)[order]

    plt.figure(figsize=(6, 4))
    plt.scatter(lengths_np, tms_np)
    plt.xlabel("Ground-truth length")
    plt.ylabel("Final TM-score")
    plt.title("Length vs. final TM-score")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot to {output_path}", flush=True)


if __name__ == "__main__":
    main()

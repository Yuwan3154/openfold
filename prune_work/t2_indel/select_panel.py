"""Pick the 24-chain indel pilot panel: source (easy/hard) x length bin x native SSE class, 1 chain per cell.

Eligible = every residue-to-residue CA gap <= 4.5 A (no unresolved internal stretch) and >= 50 resolved residues.
Length = RESOLVED residue count (n_parsed), not the val list's sequence length (they differ by a median
of 13 residues). SSE class = terciles of (helix_frac - strand_frac) over the eligible chains (low-helix includes strand-rich AND coil-rich chains). One chain
per cell drawn with numpy default_rng(seed) from the cell's pdb_chain-sorted members.
Run: python select_panel.py --stats panel_stats.csv --out panel.csv [--seed 0]
"""
import argparse

import numpy as np
import pandas as pd

LEN_BINS = [0, 100, 200, 300, 10_000]
LEN_LABELS = ["<=100", "101-200", "201-300", ">=301"]
SSE_LABELS = ["low-helix", "mid-helix", "high-helix"]
MIN_RESOLVED = 50  # the val lists were built with length >= 50; applied here to the RESOLVED count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    d = pd.read_csv(a.stats)
    n0 = len(d)
    d = d[(d.status == "ok") & (d.n_ca_breaks == 0) & (d.n_parsed >= MIN_RESOLVED)].copy()
    print(f"eligible (no CA gaps, >= {MIN_RESOLVED} resolved): {len(d)} of {n0}")
    d["dh"] = d.helix_frac - d.strand_frac
    cuts = d.dh.quantile([1 / 3, 2 / 3]).values
    print("SSE tercile cuts on helix-strand fraction:", cuts.round(3))
    d["sse_class"] = pd.cut(d.dh, [-9, cuts[0], cuts[1], 9], labels=SSE_LABELS, include_lowest=True)
    d["len_bin"] = pd.cut(d.n_parsed, LEN_BINS, labels=LEN_LABELS)
    d["key"] = d.pdb + "_" + d.chain
    d = d.sort_values("key")
    cells = d.groupby(["source", "len_bin", "sse_class"], observed=False)
    counts = cells.size().unstack("sse_class")
    print("members per cell:\n", counts.to_string())
    assert (cells.size() > 0).all(), "empty cell"
    rng = np.random.default_rng(a.seed)
    picks = [g.iloc[rng.integers(len(g))] for _, g in cells]
    out = pd.DataFrame(picks)[["key", "pdb", "chain", "source", "len_bin", "sse_class", "n_parsed", "span",
                               "helix_frac", "strand_frac", "best_tm_to_train"]]
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))
    print("rows:", len(out))


if __name__ == "__main__":
    main()

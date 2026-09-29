"""SSE agreement with the native for the three template sources, on ONE chain population.

natural / promoted: source_sse.py rows; synthetic: pool_sse.py rows restricted to the same chains.
Agreement is over residue pairs (natural: USalign-aligned pairs; promoted: index pairs on the crop;
synthetic: every residue). Reported per TM bin: templates, chains, Q3, and native-label -> template-label
fractions, each pooled over residue pairs.

Run: python compare_sources.py --natural source_sse_natural.csv --promoted source_sse_promoted.csv \
     --synthetic pool_full_slim.csv.gz --out-prefix out/sources
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
T = [f"n_{s}to{t}" for s in "HEC" for t in "HEC"]
SRC_C = {"natural": "#2A78D6", "promoted": "#8E4FB8", "synthetic": "#E07A2E"}


def table(d):
    d = d.copy()
    d["tmbin"] = pd.cut(d.tm, BINS, right=False)
    g = d.groupby(["source", "tmbin"], observed=True)
    out = g.agg(templates=("tm", "size"), chains=("chain", "nunique"), **{c: (c, "sum") for c in T})
    pairs = out[T].sum(1)
    out["q3"] = (out.n_HtoH + out.n_EtoE + out.n_CtoC) / pairs
    for s in "HEC":
        tot = out[[f"n_{s}to{t}" for t in "HEC"]].sum(1)
        for t in "HEC":
            if s != t:
                out[f"{s}->{t}"] = out[f"n_{s}to{t}"] / tot
    return out.drop(columns=T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--natural", required=True)
    p.add_argument("--promoted", required=True)
    p.add_argument("--synthetic", required=True)
    p.add_argument("--out-prefix", required=True)
    a = p.parse_args()
    nat = pd.read_csv(a.natural)
    print("natural status:", nat.status.value_counts().to_dict())
    nat = nat[nat.status == "ok"]
    pro = pd.read_csv(a.promoted)
    chains = set(nat.chain) | set(pro.chain)
    syn = pd.read_csv(a.synthetic)
    syn = syn[syn.chain.isin(chains)].assign(source="synthetic")
    d = pd.concat([nat[["source", "chain", "tm"] + T], pro[["source", "chain", "tm"] + T],
                   syn[["source", "chain", "tm"] + T]], ignore_index=True)
    t = table(d)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string())
    t.to_csv(f"{a.out_prefix}_by_tm.csv")
    print("natural coverage (aligned pairs / native length): median",
          round(float(nat.coverage.median()), 3), "IQR", nat.coverage.quantile([.25, .75]).round(3).tolist())

    share = t.templates / t.groupby(level=0).templates.transform("sum")
    fig, ax = plt.subplots(1, 5, figsize=(19, 3.8))
    panels = [(None, "share of the source's templates"), ("q3", "Q3 vs native"), ("E->C", "strand → coil"),
              ("E->H", "strand → helix"), ("C->H", "coil → helix")]
    for src, g in t.groupby(level=0):
        x = [iv.mid if iv.right <= 1 else (iv.left + 1) / 2 for iv in g.index.get_level_values(1)]
        for k, (col, lab) in enumerate(panels):
            y = share.loc[src] if col is None else g[col]
            ax[k].plot(x, y, "o-", color=SRC_C[src], label=src)
            ax[k].set_title(lab, fontsize=10)
            ax[k].set_xlabel("TM to native (bin midpoint)")
            ax[k].set_xlim(0, 1)
    ax[0].legend(frameon=False)
    for k in (0, 2, 3, 4):
        ax[k].set_ylim(bottom=0)
    fig.suptitle("SSE agreement with the native by template source (same chains)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{a.out_prefix}_by_tm.png", dpi=140)


if __name__ == "__main__":
    main()

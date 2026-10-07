"""Refold-pilot / control summaries from the scored csvs (refold_scores_<m>.csv, refold_ctl_scores_<m>.csv, refold_dctl_scores.csv,
refold_temp_scores.csv): per backbone mean/max paired TM, pLDDT, share above 0.5/0.7, rank correlations. Run in the folder holding them."""
import glob
import os

import numpy as np
import pandas as pd


def main():
    for f in sorted(glob.glob("refold_*scores*.csv")):
        d = pd.read_csv(f)
        d["kind"] = np.where(d.seq_kind == 0, "first", "later")
        print(f"\n== {f} ({len(d)} predictions)")
        keys = [k for k in ("chain", "arm", "temp") if k in d.columns and d[k].notna().any()]
        g = d.groupby(keys).agg(n=("tm_paired", "size"), tm=("tm_paired", "mean"), tm_max=("tm_paired", "max"), plddt=("plddt", "mean"))
        print(g.round(3).to_string())
        print("share TM>0.5 %.3f  >0.7 %.3f ; spearman(plddt,TM) %.3f" % ((d.tm_paired > 0.5).mean(), (d.tm_paired > 0.7).mean(),
              d[["plddt", "tm_paired"]].corr(method="spearman").iloc[0, 1]))


if __name__ == "__main__":
    main()

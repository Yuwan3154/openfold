"""Panel-level comparison of stage B modes (RAW 141). Each mode dir holds <shard>/<id>.metrics.csv (24 panel chains x 64 variants = 1,536 templates, same edits / same sequences in every mode).
Per mode: mean tm_native (sequence-independent TM to the native, native-length-normalised), share of templates in the TM window / passing the bond gate / the loop gate / the break gate / all.
Paired against a reference mode (same chain, same draw; modes with the same noise seed share the initial noisy state): mean and mean-absolute difference of tm_native, share of templates whose
pass_all flag differs. Seed spread = the same comparison between two runs of the same mode with different noise seeds. Run: python t2_panel_compare.py DIR REF MODE [MODE ...]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

root, ref, modes = sys.argv[1], sys.argv[2], sys.argv[3:]


def load(m):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(root, m, "*", "*.metrics.csv")))])
    return d.set_index(["chain", "draw"]).sort_index()


R = load(ref)
rows = []
for m in [ref] + modes:
    d = load(m)
    j = d.join(R, rsuffix="_ref", how="inner")
    rows.append(dict(mode=m, n=len(d), tm_mean=d.tm_native.mean(), in_tm=d.pass_tm.mean(), bond=d.pass_bond.mean(), loop=d.pass_loop.mean(), brk=d.pass_break.mean(), all=d.pass_all.mean(),
                     d_tm_mean=(j.tm_native - j.tm_native_ref).mean(), abs_d_tm=(j.tm_native - j.tm_native_ref).abs().mean(), flip_all=(j.pass_all != j.pass_all_ref).mean(),
                     ks=ks_2samp(d.tm_native, R.tm_native).statistic))
pd.set_option("display.width", 220)
print(f"reference = {ref}; columns: n templates; mean tm_native; shares (in 0.4-0.9, bond, loop, break, all); paired d_tm_mean / abs_d_tm vs reference; flip_all = share with a different pass_all; ks = KS statistic of the tm_native distribution")
print(pd.DataFrame(rows).round(4).to_string(index=False))

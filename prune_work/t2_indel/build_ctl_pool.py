"""Pool-format file of the NO-INDEL controls (same partial-diffusion model, same rungs, native sequence), so the existing MPNN design,
refold and geometry tools run on them unchanged. Source: sweep_gly/cc89/<chain>/c<k>.npz (kind 'control'), scores from scores_gly.
Items per chain ordered (draw, rung): i = draw_index * n_rungs + rung_index. arm = 'control'; orig_idx = arange (no edit).
Run: python build_ctl_pool.py --sweep sweep_gly --inputs-dir inputs --scores scores_gly --out-root pool_ctl --draws 2
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import indel_pool as ip
from atomic_io import atomic_savez


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True)
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--draws", type=int, default=2)
    a = p.parse_args()
    sc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.scores, "*.csv")))], ignore_index=True)
    sc = sc[(sc.model == "cc89") & (sc.kind == "control")].set_index(["chain", "draw", "rewind"])
    assert sc.index.is_unique
    rows = []
    for chain in sorted(os.listdir(a.inputs_dir)):
        L = json.load(open(os.path.join(a.inputs_dir, chain, "plans.json")))["L"]
        items = []
        for k in range(a.draws):
            z = np.load(os.path.join(a.sweep, "cc89", chain, f"c{k:02d}.npz"))
            for ri, r in enumerate(z["rewind_steps"].tolist()):
                row = sc.loc[(chain, k, r)]
                items.append(dict(arm="control", model="cc89", rewind=int(r), draw=k, coords=z["coords"][ri], atom_mask=z["atom_mask"],
                                  aatype=z["aatype"], orig_idx=np.arange(L), ops=[], tm_native=row.tm_native, tm_template=row.tm_template,
                                  L_native=L))
        path = ip.shard_path(a.out_root, chain)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        atomic_savez(path, **ip.pack_chain(chain, items))
        for i, it in enumerate(items):
            rows.append(dict(chain=chain, file=os.path.relpath(path, a.out_root), i=i, arm="control", draw=it["draw"], rewind=it["rewind"],
                             L=L, tm_native=it["tm_native"]))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_root, "index.csv"), index=False)
    print(f"{len(rows)} control templates -> {a.out_root}")


if __name__ == "__main__":
    main()

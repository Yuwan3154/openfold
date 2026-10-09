"""DSSP secondary-structure distribution of pool templates vs their native chain (user 10-08: geometry gates for using the synthetic templates as-is).

Per template: DSSP (3-state, mdtraj, backbone N,CA,C,O; diagnose_indel.dssp) -> helix / strand / coil fractions, and over the SURVIVORS (orig_idx >= 0)
the Q3 agreement with the native DSSP (native.pdb of the same chain) and the native fractions for comparison. Rows are per template; the printed table
is per pool:arm. Env: protebm. Run: python pool_dssp.py --inputs-dir D/inputs --pool-roots P1 P2 ... --out f.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

import indel_pool as ip
from diagnose_indel import dssp, native_backbone

LAB = "HEC"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--pool-roots", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    nat = {}
    rows = []
    for root in a.pool_roots:
        idx = pd.read_csv(os.path.join(root, "index.csv"))
        for r in idx.itertuples():
            if r.chain not in nat:
                bb = native_backbone(os.path.join(a.inputs_dir, r.chain, "native.pdb"))
                nat[r.chain] = dssp(bb)
            t = ip.read_template(os.path.join(root, r.file), r.i)
            bb = ip.atom37_coords(t)[:, [0, 1, 2, 4]].astype(np.float64)
            ss = dssp(bb)
            surv = t["orig_idx"] >= 0
            ns = nat[r.chain]
            assert len(ns) == t["L_native"] and t["orig_idx"].max() < len(ns), (r.chain, r.i, len(ns), t["L_native"])
            q3 = float((ss[surv] == ns[t["orig_idx"][surv]]).mean())
            rows.append(dict(pool=os.path.basename(root.rstrip("/")), arm=r.arm, chain=r.chain, i=r.i, draw=r.draw, rewind=r.rewind, L=len(ss),
                             **{f"{k}_frac": float((ss == k).mean()) for k in LAB}, **{f"native_{k}_frac": float((ns == k).mean()) for k in LAB},
                             q3_survivors=q3))
    df = pd.DataFrame(rows)
    assert len(df) > 0
    df.to_csv(a.out, index=False)
    g = df.groupby(["pool", "arm"]).agg(n=("i", "size"), H=("H_frac", "mean"), E=("E_frac", "mean"), C=("C_frac", "mean"),
                                        nH=("native_H_frac", "mean"), nE=("native_E_frac", "mean"), nC=("native_C_frac", "mean"),
                                        q3=("q3_survivors", "mean")).round(3)
    print(g.to_string())


if __name__ == "__main__":
    main()

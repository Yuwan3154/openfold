"""Verify the pool against its sources, bit for bit, through the READER (indel_pool.read_template).

For every pool template: coords/atom_mask/aatype equal the source sweep npz at the pool's rewind; orig_idx equals the
plan's; residue_index contiguous; TM equals the score csv; design arrays (if present) have the right shape, valid
residue indices, no 'X', and MPNN's recovery is consistent with a direct recompute against the stored sequence.
Prints counts (sample size asserted > 0). Run: python verify_pool.py --pool-root pool --rewind 250 --arm comp ... (as build)
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import indel_pool as ip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-root", required=True)
    p.add_argument("--rewind", type=int, required=True)
    p.add_argument("--model", default="cc89")
    p.add_argument("--arm", nargs=4, action="append", metavar=("NAME", "SWEEP", "INPUTS", "SCORES"), required=True)
    p.add_argument("--design", action="store_true")
    a = p.parse_args()
    src = {n: (s, i, pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(sc, "*.csv"))])) for n, s, i, sc in a.arm}
    idx = pd.read_csv(os.path.join(a.pool_root, "index.csv"))
    n_ok = n_design = 0
    for (chain, file), g in idx.groupby(["chain", "file"]):
        path = os.path.join(a.pool_root, file)
        for r in g.itertuples():
            t = ip.read_template(path, r.i)
            sweep, inputs, sc = src[t["arm"]]
            z = np.load(os.path.join(sweep, a.model, chain, f"d{t['draw']:02d}.npz"))
            k = z["rewind_steps"].tolist().index(a.rewind)
            assert np.array_equal(t["coords"], z["coords"][k].astype(np.float32)), (chain, r.i, "coords")
            assert np.array_equal(t["atom_mask"], z["atom_mask"]) and np.array_equal(t["aatype"], z["aatype"]), (chain, r.i)
            plans = json.load(open(os.path.join(inputs, chain, "plans.json")))
            assert np.array_equal(t["orig_idx"], plans["plans"][t["draw"]]["orig_idx"]), (chain, r.i, "orig_idx")
            assert np.array_equal(t["residue_index"], np.arange(1, len(t["aatype"]) + 1))
            row = sc[(sc.chain == chain) & (sc.draw == t["draw"]) & (sc.rewind == a.rewind) & (sc.kind == "indel") & (sc.model == a.model)]
            assert len(row) == 1 and abs(float(row.tm_native.iloc[0]) - t["tm_native"]) < 1e-6, (chain, r.i, "tm")
            n_ok += 1
            if a.design:
                d = t["design_aatype"]
                L = len(t["aatype"])
                assert d.shape == (32, L) and d.min() >= 0 and d.max() <= 19
                direct = (d == t["aatype"][None]).mean(1)
                diff = np.abs(direct - t["design_recovery"])
                assert not np.isnan(diff).any() or not t["design_ok"], (chain, r.i, "NaN recovery on a design_ok template")
                assert diff.max() < 1e-3, (chain, r.i, "recovery", diff.max())  # MPNN prints seq_recovery with 4 decimals
                n_design += 1
    assert n_ok > 0 and n_ok == len(idx), (n_ok, len(idx))
    print(f"verified {n_ok} templates against their sources through the reader; {n_design} with design arrays")


if __name__ == "__main__":
    main()

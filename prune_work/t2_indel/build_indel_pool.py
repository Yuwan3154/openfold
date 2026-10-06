"""Assemble the indel pool (stage A) from the sweep outputs. t* is a CLI value (user 10-06: 250), arms/models explicit.

Sources (per arm): sweep dir with <model>/<chain>/dNN.npz (rewind_steps selects the rung), the arm's inputs dir with
plans.json (orig_idx, ops), the arm's score csvs (tm_native/tm_template per chain/draw/rewind).
Run: python build_indel_pool.py --out-root pool --rewind 250 --arm comp sweep_compF inputs_compF compF_scores \
       --arm raygun1dir sweep_rg3 inputs_rg3_full scores_rg3
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
    p.add_argument("--out-root", required=True)
    p.add_argument("--rewind", type=int, required=True)
    p.add_argument("--model", default="cc89")
    p.add_argument("--arm", nargs=4, action="append", metavar=("NAME", "SWEEP", "INPUTS", "SCORES"), required=True)
    p.add_argument("--chains", nargs="*", default=None)
    a = p.parse_args()
    arms = []
    for name, sweep, inputs, scores in a.arm:
        sc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(scores, "*.csv")))], ignore_index=True)
        sc = sc[(sc.model == a.model) & (sc.kind == "indel") & (sc.rewind == a.rewind)].set_index(["chain", "draw"])
        assert sc.index.is_unique, "duplicate score rows"
        arms.append((name, sweep, inputs, sc))
    chains = a.chains or sorted(os.listdir(arms[0][2]))
    index_rows, n_items = [], 0
    for chain in chains:
        items = []
        for name, sweep, inputs, sc in arms:
            plans = json.load(open(os.path.join(inputs, chain, "plans.json")))
            files = sorted(f for f in os.listdir(os.path.join(sweep, a.model, chain)) if f.startswith("d"))
            assert len(files) == len(plans["plans"]), (chain, name, len(files), len(plans["plans"]))
            for fn in files:
                k = int(fn[1:3])
                z = np.load(os.path.join(sweep, a.model, chain, fn))
                i = z["rewind_steps"].tolist().index(a.rewind)
                row = sc.loc[(chain, k)]
                assert int(row.L_template) == int(z["L_new"]), (chain, name, k)
                items.append(dict(arm=name, model=a.model, rewind=a.rewind, draw=k, coords=z["coords"][i],
                                  atom_mask=z["atom_mask"], aatype=z["aatype"], orig_idx=plans["plans"][k]["orig_idx"],
                                  ops=plans["plans"][k]["ops"], tm_native=row.tm_native, tm_template=row.tm_template,
                                  L_native=plans["L"]))
        pack = ip.pack_chain(chain, items)
        path = ip.shard_path(a.out_root, chain)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **pack)
        for i, it in enumerate(items):
            index_rows.append(dict(chain=chain, file=os.path.relpath(path, a.out_root), i=i, arm=it["arm"], draw=it["draw"],
                                   rewind=it["rewind"], L=len(it["aatype"]), tm_native=it["tm_native"]))
        n_items += len(items)
        print(f"{chain}: {len(items)} templates", flush=True)
    pd.DataFrame(index_rows).to_csv(os.path.join(a.out_root, "index.csv"), index=False)
    print(f"pool: {n_items} templates in {len(chains)} chains -> {a.out_root}")


if __name__ == "__main__":
    main()

"""All-pairs USalign check of the chosen 64-template sets of a random sample of chains (follow-up of RAW 252, user 10-16): how much does the top-10-proxy-pairs shortcut of t2_select_diverse.py (--usalign-top) underestimate the true maximum pair TM?
For every sampled chain (seeded draw among the chains with 64 chosen templates in the selection TSV): the 64 chosen templates (kept = 'round/draw;...') are read with t2_select_diverse.pool_of (pack + loose top-up rounds, as in the selection), every pair (2,016) is aligned by USalign with the TM normalised by the NATIVE length (-u L_native, parse_usalign), the proxy pair TM of the same pair is computed as well.
Per chain: max / 95th percentile / median pair TM (USalign, native length), number of pairs > 0.7 / 0.75 / 0.8, the max over the --top most similar PROXY pairs by USalign (the shortcut value), the proxy max, rank of the true argmax pair in the proxy ordering.
Run: python t2_usalign_allpairs.py --pack P --topup T --selection selection_all.tsv --out O.tsv --n-chains 500 --top 10 [--seed 0] [--shard k --n-shards n]
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

from t2_diversity import USALIGN, ca_of, orig_map, parse_usalign, proxy_tm, write_pdb
from t2_pack_reader import TemplatePack
from t2_select_diverse import pool_of


def all_pairs(cas, lnat):
    n = len(cas)
    maps = [orig_map(c[1]) for c in cas]
    us, px = np.zeros((n, n)), np.zeros((n, n))
    with tempfile.TemporaryDirectory() as td:
        paths = [os.path.join(td, f"t{i}.pdb") for i in range(n)]
        for p, (ca, _) in zip(paths, cas):
            write_pdb(p, ca)
        for i in range(n):
            for j in range(i + 1, n):
                out = subprocess.run([USALIGN, paths[i], paths[j], "-u", str(int(lnat))], capture_output=True, text=True, check=True).stdout
                us[i, j] = us[j, i] = parse_usalign(out, lnat)
                v = proxy_tm(cas[i], cas[j], lnat, maps[i], maps[j])
                px[i, j] = px[j, i] = 0.0 if np.isnan(v) else v
    return us, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--topup", required=True)
    ap.add_argument("--selection", required=True, help="selection TSV (columns chain, n_kept, kept)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-chains", type=int, required=True, help="chains of the random sample (user 10-16: 500)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, required=True, help="the shortcut under test: the N most similar proxy pairs (t2_select_diverse --usalign-top; 10)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()
    sel = pd.read_csv(a.selection, sep="\t")
    full = sel[sel.n_kept == 64].sort_values("chain").reset_index(drop=True)
    assert len(full) >= a.n_chains, (len(full), "chains with 64 chosen templates, fewer than --n-chains")
    pick = np.random.default_rng(a.seed).choice(len(full), size=a.n_chains, replace=False)
    sample = full.iloc[np.sort(pick)].iloc[a.shard::a.n_shards]
    tp = TemplatePack(a.pack)
    rows = []
    for r in sample.itertuples():
        want = [tuple(k.rsplit("/", 1)) for k in r.kept.split(";")]
        items = {(rd, str(d)): t for rd, d, t in pool_of(r.chain, tp, a.topup)}
        assert len(want) == r.n_kept and set(want) <= set(items), (r.chain, "chosen keys missing from the pool", sorted(set(want) - set(items))[:3])
        cas = [ca_of(items[k]) for k in want]
        lnat = items[want[0]]["L_native"]
        us, px = all_pairs(cas, lnat)
        iu = np.triu_indices(len(cas), 1)
        u, p = us[iu], px[iu]
        top = np.argsort(p)[::-1][:a.top]
        rows.append(dict(chain=r.chain, L_native=lnat, n=len(cas), us_max=u.max(), us_p95=np.percentile(u, 95), us_median=np.median(u), px_max=p.max(), shortcut_max=u[top].max(),
                         n_gt_0p7=int((u > 0.7).sum()), n_gt_0p75=int((u > 0.75).sum()), n_gt_0p8=int((u > 0.8).sum()), n_pairs=len(u), rank_of_argmax_in_proxy=int((p > p[np.argmax(u)]).sum()) + 1))
        print(f"{r.chain} L={lnat} usalign max {u.max():.3f} shortcut {u[top].max():.3f} proxy max {p.max():.3f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
    print(f"shard {a.shard}/{a.n_shards}: {len(rows)} chains -> {a.out}")


if __name__ == "__main__":
    main()

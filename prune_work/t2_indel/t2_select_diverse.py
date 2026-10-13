"""Diversity-driven selection of the final 64 templates per chain (T2_PACKED... T2_TOPUP_DIVERSITY_PLAN.md; user 10-13: max pairwise TM must NOT exceed 0.7, pair TM normalised by the native length, keep 64 chosen for diversity).
Pool of a chain = every passing template of the first production (read from the pack: TemplatePack) + the passing templates of the finished top-up rounds (loose npz under <topup>/<round>/templates/<id[1:3]>/<id>.npz). Pools larger than --max-pool are thinned to --max-pool by a seeded random draw (cost: the pair matrix is N x N).
Pair score = 'proxy TM' (t2_diversity.proxy_tm: TM-like score over the native residues both templates contain, iterative Kabsch, normalised by the native length). Selection = greedy min-max: first the template with the lowest mean pair TM, then repeatedly the candidate that minimises the maximum pair TM to the chosen set (ties: lowest mean). A pool of <= 64 keeps everything.
Per chain it writes: pool size, kept ids, proxy max / 95th percentile / median pair TM of the chosen set and of a seeded random 64-subset of the pool (the 'before'), flags max <= 0.7.
Run: python t2_select_diverse.py --pack P --topup T --out O.tsv [--shard k --n-shards n] [--max-pool 128] [--chains FILE]
"""
import argparse
import glob
import os
import zlib

import numpy as np
import pandas as pd

import indel_pool as ip
from t2_diversity import ca_of, orig_map, proxy_tm
from t2_pack_reader import TemplatePack


def pair_matrix(cas, lnat):
    n = len(cas)
    maps = [orig_map(c[1]) for c in cas]   # once per template, not once per pair
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = proxy_tm(cas[i], cas[j], lnat, maps[i], maps[j])
            M[i, j] = M[j, i] = 0.0 if np.isnan(v) else v
    return M


def select_greedy(M, k):
    """indices of k rows of the symmetric matrix M chosen to keep the maximum pairwise value small (greedy min-max)."""
    n = len(M)
    if n <= k:
        return list(range(n))
    mean = M.sum(1) / (n - 1)
    chosen = [int(np.argmin(mean))]
    cur_max = M[chosen[0]].copy()
    free = np.ones(n, bool)
    free[chosen[0]] = False
    while len(chosen) < k:
        score = np.where(free, cur_max, np.inf)
        best = np.flatnonzero(score == score.min())
        c = int(best[np.argmin(mean[best])])
        chosen.append(c)
        free[c] = False
        cur_max = np.maximum(cur_max, M[c])
    return chosen


def stats(M, idx):
    if len(idx) < 2:
        return np.nan, np.nan, np.nan
    sub = M[np.ix_(idx, idx)]
    v = sub[np.triu_indices(len(idx), 1)]
    return float(v.max()), float(np.percentile(v, 95)), float(np.median(v))


def pool_of(cid, tp, topup):
    items = []
    for r in tp.rows(cid) if cid in tp else []:
        t = tp._read(r)
        items.append((f"base:{t['rewind']}", t["draw"], t))
    for f in sorted(glob.glob(os.path.join(topup, "*", "templates", cid[1:3], cid + ".npz"))):
        rd = f.split(os.sep)[-4]
        z = np.load(f, allow_pickle=False)   # every member is read ONCE (ip.read_template re-opens the npz per template)
        n, ro, ao = int(z["n_templates"]), z["res_offsets"], z["atom_offsets"]
        am, co, oi, aat, draw, rewind, lnat = z["atom_mask"], z["coords"], z["orig_idx"], z["aatype"], z["draw"], z["rewind"], int(z["L_native"])
        for i in range(n):
            r0, r1, a0, a1 = int(ro[i]), int(ro[i + 1]), int(ao[i]), int(ao[i + 1])
            items.append((rd, int(draw[i]), dict(aatype=aat[r0:r1], atom_mask=am[r0:r1], coords=co[a0:a1], orig_idx=oi[r0:r1], draw=int(draw[i]), rewind=int(rewind[i]), L_native=lnat)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--topup", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-pool", type=int, default=128)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--chains", default=None, help="file of chain ids (default: every chain of the pack)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tp = TemplatePack(a.pack)
    chains = [l.strip() for l in open(a.chains)] if a.chains else [str(c) for c in tp.stems]
    chains = chains[a.shard::a.n_shards]
    rows = []
    for cid in chains:
        rng = np.random.default_rng([a.seed, zlib.crc32(cid.encode())])   # per chain: independent of the shard split and the chain order
        items = pool_of(cid, tp, a.topup)
        n_pool = len(items)
        if n_pool == 0:
            rows.append(dict(chain=cid, n_pool=0, n_used=0, n_kept=0, ok07=False, kept=""))   # recorded, not silently skipped
            continue
        if n_pool > a.max_pool:
            keep = sorted(rng.choice(n_pool, size=a.max_pool, replace=False))
            items = [items[i] for i in keep]
        cas = [ca_of(t) for _, _, t in items]
        lnat = items[0][2]["L_native"]
        M = pair_matrix(cas, lnat)
        chosen = select_greedy(M, a.k)
        before = sorted(rng.choice(len(items), size=min(a.k, len(items)), replace=False))
        mx, p95, med = stats(M, chosen)
        bmx, bp95, bmed = stats(M, before)
        rows.append(dict(chain=cid, L_native=lnat, n_pool=n_pool, n_used=len(items), n_kept=len(chosen), max_pair=mx, p95_pair=p95, med_pair=med, rand_max_pair=bmx, rand_p95_pair=bp95, rand_med_pair=bmed,
                         ok07=bool(mx <= 0.7) if mx == mx else False, kept=";".join(f"{items[i][0]}/{items[i][1]}" for i in chosen)))
    pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
    print(f"shard {a.shard}/{a.n_shards}: {len(rows)} chains -> {a.out}")


if __name__ == "__main__":
    main()

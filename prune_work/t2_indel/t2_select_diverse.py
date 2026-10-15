"""Diversity-driven selection of the final 64 templates per chain (T2_PACKED... T2_TOPUP_DIVERSITY_PLAN.md; user 10-13: max pairwise TM must NOT exceed 0.7, pair TM normalised by the native length, keep 64 chosen for diversity).
Pool of a chain = every passing template of the first production (read from the pack: TemplatePack) + the passing templates of the finished top-up rounds (loose npz under <topup>/<round>/templates/<id[1:3]>/<id>.npz). Pools larger than --max-pool are thinned to --max-pool by a seeded random draw (cost: the pair matrix is N x N).
Pair score = 'proxy TM' (t2_diversity.proxy_tm: TM-like score over the native residues both templates contain, iterative Kabsch, normalised by the native length). Selection = greedy min-max: first the template with the lowest mean pair TM, then repeatedly the candidate that minimises the maximum pair TM to the chosen set (ties: lowest mean). A pool of <= 64 keeps everything.
Per chain it writes: pool size, kept ids, proxy max / 95th percentile / median pair TM of the chosen set and of a seeded random 64-subset of the pool (the 'before'), flags max <= 0.7.
Pool labels are pack keys: kept = 'round/draw;...' with round 'base' (first production) or the top-up round name (P1.., R1..); a pack that already has the rounds appended supplies them itself (the loose rounds are then not read twice).
Run: python t2_select_diverse.py --pack P --topup T --out O.tsv [--shard k --n-shards n] [--max-pool 128] [--chains FILE] [--usalign-top 10]
Merge the shard TSVs into the selection layer read by t2_pack_reader.FinalSet: python t2_select_diverse.py --merge O0.tsv O1.tsv ... --final final_selection.npz
"""
import argparse
import glob
import os
import zlib

import numpy as np
import pandas as pd

import indel_pool as ip
from t2_diversity import ca_of, orig_map, proxy_tm, usalign_tm
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


def usalign_top(M, chosen, cas, lnat, n):
    """USalign TM (native-length normalised, -u) of the n most similar pairs (by proxy TM) of the chosen set; their maximum bounds the true max pair TM from below"""
    sub = M[np.ix_(chosen, chosen)]
    iu = np.triu_indices(len(chosen), 1)
    top = np.argsort(sub[iu])[::-1][:n]
    return [usalign_tm(cas[chosen[iu[0][t]]][0], cas[chosen[iu[1][t]]][0], lnat) for t in top]


def pool_of(cid, tp, topup):
    items, have = [], set()
    for r in tp.rows(cid) if cid in tp else []:
        t = tp._read(r)
        have.add(t["round"])
        items.append((t["round"], t["draw"], t))
    for f in sorted(glob.glob(os.path.join(topup, "*", "templates", cid[1:3], cid + ".npz"))):
        rd = f.split(os.sep)[-4]
        if rd in have:
            continue
        z = np.load(f, allow_pickle=False)   # every member is read ONCE (ip.read_template re-opens the npz per template)
        n, ro, ao = int(z["n_templates"]), z["res_offsets"], z["atom_offsets"]
        am, co, oi, aat, draw, rewind, lnat = z["atom_mask"], z["coords"], z["orig_idx"], z["aatype"], z["draw"], z["rewind"], int(z["L_native"])
        for i in range(n):
            r0, r1, a0, a1 = int(ro[i]), int(ro[i + 1]), int(ao[i]), int(ao[i + 1])
            items.append((rd, int(draw[i]), dict(aatype=aat[r0:r1], atom_mask=am[r0:r1], coords=co[a0:a1], orig_idx=oi[r0:r1], draw=int(draw[i]), rewind=int(rewind[i]), L_native=lnat)))
    return items


def write_final(tsvs, out):
    """selection layer from the shard TSVs: one chain per row, its chosen (round, draw) keys contiguous in round / draw"""
    df = pd.concat([pd.read_csv(t, sep="\t") for t in tsvs], ignore_index=True).sort_values("chain")
    assert df.chain.is_unique, "a chain appears in more than one TSV"
    keys = [[k.rsplit("/", 1) for k in str(v).split(";")] if isinstance(v, str) else [] for v in df.kept]
    count = np.array([len(k) for k in keys], np.int32)
    np.savez(out, chain=df.chain.to_numpy(dtype=str), first=np.concatenate([[0], np.cumsum(count, dtype=np.int64)[:-1]]), count=count,
             round=np.array([r for k in keys for r, _ in k], dtype=str), draw=np.array([int(d) for k in keys for _, d in k], np.int32))
    return len(df), int(count.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", nargs="+", help="shard TSVs to merge into --final (no selection is run)")
    ap.add_argument("--final", help="--merge: output final_selection.npz")
    ap.add_argument("--pack")
    ap.add_argument("--topup")
    ap.add_argument("--out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-pool", type=int, default=128)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--chains", default=None, help="file of chain ids (default: every chain of the pack)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--usalign-top", type=int, default=0, help="USalign (-u native length) the n most similar pairs of each chosen set (0 = off): columns usalign_max / usalign_ok07")
    a = ap.parse_args()
    if a.merge:
        assert a.final, "--merge needs --final"
        print("final selection: %d chains, %d chosen templates -> %s" % (*write_final(a.merge, a.final), a.final))
        return
    assert a.pack and a.topup and a.out, "--pack, --topup and --out are required without --merge"
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
        us = usalign_top(M, chosen, cas, lnat, a.usalign_top) if a.usalign_top and len(chosen) > 1 else []
        rows.append(dict(chain=cid, L_native=lnat, n_pool=n_pool, n_used=len(items), n_kept=len(chosen), max_pair=mx, p95_pair=p95, med_pair=med, rand_max_pair=bmx, rand_p95_pair=bp95, rand_med_pair=bmed,
                         ok07=bool(mx <= 0.7) if mx == mx else False, usalign_max=max(us) if us else np.nan, usalign_ok07=bool(max(us) <= 0.7) if us else np.nan, kept=";".join(f"{items[i][0]}/{items[i][1]}" for i in chosen)))
    pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
    print(f"shard {a.shard}/{a.n_shards}: {len(rows)} chains -> {a.out}")


if __name__ == "__main__":
    main()

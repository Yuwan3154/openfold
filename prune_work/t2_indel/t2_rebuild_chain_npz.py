"""Rebuild the per-chain npz arrays of level 1 (indel_pool.pack_chain layout) from level 2 of a T2 pack, so level 1 (chain.NNN.pack, chain.idx.npz, meta/chain.NNN.npz) can be removed (user 10-27).
A level-1 npz = one (chain, round) = the level-2 rows of that chain and round, in index order. Key -> level-2 source (r = the rows, records read by TemplatePack._read):
  aatype, atom_mask (u1 -> bool), coords, residue_index, orig_idx   concatenated record arrays (tmpl.NNN.bin)
  res_offsets, atom_offsets   cumsum of the L / A columns           n_templates   number of rows
  arm, model, rewind, draw, tm_native, tm_template, L_native        idx columns (rewind of a top-up row = the round rewind, asserted == the npz rewind at build time)
  ops_json   json.dumps of the list of per-row ops (ops_blob)       chain   the chain stem       aa_order, atom_types   indel_pool constants
  rebuild: python t2_rebuild_chain_npz.py rebuild --pack P --chain C [--round base] --out C.npz
  check  : python t2_rebuild_chain_npz.py check --pack P [--procs 16]   (needs level 1: EVERY chain.idx row, md5-checked against chain.idx, is rebuilt and compared array by array: dtype + shape + values (NaN == NaN) and bytes; exit 1 on any difference)
"""
import argparse
import hashlib
import io
import multiprocessing as mp

import numpy as np

import indel_pool as ip
from t2_pack_reader import ChainPack, TemplatePack

_G = {}


def part_rows(tp, cid, rnd):
    return [r for r in tp.rows(cid) if tp.round_of(r) == rnd] if cid in tp else []


def rebuild(tp, cid, rnd="base"):
    rows = part_rows(tp, cid, rnd)
    assert rows, (cid, rnd, "no level-2 rows for this chain and round")
    recs = [tp._read(r) for r in rows]
    assert len({t["L_native"] for t in recs}) == 1, (cid, rnd, "rows with different L_native")
    out = ip.pack_chain(cid, recs)
    out["residue_index"] = np.concatenate([t["residue_index"] for t in recs]).astype(np.int32)   # the stored values, not pack_chain's 1..L
    return out


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")


def check_shard(k):
    """every level-1 row of shard k: (n parts, n templates, n arrays compared, n arrays equal but not byte-identical, [(chain, round, differing keys)]); the reference slice must match its chain.idx md5"""
    cp, tp = _G["cp"], _G["tp"]
    n_tm = n_arr = n_nbi = 0
    bad = []
    rows = np.flatnonzero(cp.shard == k)
    for i in rows:
        cid, rnd = str(cp.stems[i]), str(cp.round[i])
        raw = cp.raw_row(i)
        if hashlib.md5(raw).hexdigest() != cp.md5[i]:
            bad.append((cid, rnd, ["<level-1 slice md5 != chain.idx md5>"]))
            continue
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        orig = {key: z[key] for key in z.files}
        n_tm += int(orig["n_templates"])
        if not part_rows(tp, cid, rnd):
            bad.append((cid, rnd, ["<no level-2 rows>"]))
            continue
        new = rebuild(tp, cid, rnd)
        diff = sorted(set(orig) ^ set(new)) + [key for key in sorted(set(orig) & set(new)) if not same(orig[key], new[key])]
        n_arr += len(set(orig) & set(new))
        n_nbi += sum(orig[key].tobytes() != new[key].tobytes() for key in set(orig) & set(new) if key not in diff)
        if diff:
            bad.append((cid, rnd, diff))
    return len(rows), n_tm, n_arr, n_nbi, bad


def check(a):
    cp, tp = ChainPack(a.pack), TemplatePack(a.pack)
    for k in tp.col._z.files:   # load every index column once in the parent; forked workers share them copy-on-write
        tp.col[k]
    _G["cp"], _G["tp"] = cp, tp
    shards = sorted(np.unique(cp.shard).tolist())
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(check_shard, shards, chunksize=1)
    n_parts, n_tm, n_arr, n_nbi = (sum(r[j] for r in res) for j in range(4))
    bad = [b for r in res for b in r[4]]
    n_rows = len(tp.col["chain"])
    rcol = tp.col["round"] if tp.has_round else np.full(n_rows, "base")
    l1 = set(zip(cp.stems.astype(str), cp.round.astype(str)))
    l2 = set(zip(tp.col["chain"].astype(str), rcol.astype(str)))
    ok_eq = not bad and n_nbi == 0 and n_parts == len(cp.stems) > 0 and n_arr > 0
    ok_cov = l1 == l2 and n_tm == n_rows > 0 and len(l1) == len(cp.stems)
    print(f"{'PASS' if ok_eq else 'FAIL'} rebuilt == level 1, array by array (dtype, shape, values; NaN == NaN; and every equal array byte-identical): {n_parts} chain npz (chain, round) of {len(cp.stems)} level-1 rows over {len(shards)} shards (each reference slice md5 == chain.idx md5), "
          f"{len(cp.row)} chains, {n_tm} templates, {n_arr} arrays compared, {len(bad)} npz differ; {n_nbi} equal arrays not byte-identical", flush=True)
    for b in bad[:20]:
        print("  DIFF", b)
    print(f"{'PASS' if ok_cov else 'FAIL'} coverage: (chain, round) level 1 {len(l1)} == level 2 {len(l2)}: {l1 == l2} ({len(l1 - l2)} only in level 1, {len(l2 - l1)} only in level 2); "
          f"templates of the level-1 npz {n_tm} == level-2 rows {n_rows}", flush=True)
    print("RESULT:", "ALL CHECKS PASSED" if ok_eq and ok_cov else "FAIL")
    raise SystemExit(0 if ok_eq and ok_cov else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rebuild", "check"])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--chain")
    ap.add_argument("--round", default="base")
    ap.add_argument("--out")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "check":
        check(a)
        return
    if not (a.chain and a.out):
        ap.error("rebuild needs --chain and --out")
    np.savez(a.out, **rebuild(TemplatePack(a.pack), a.chain, a.round))


if __name__ == "__main__":
    main()

"""Verification gates of the packed T2 template product (T2_PACKED_TEMPLATES_PLAN.md), printed with their sample sizes; exit code 1 and 'FAIL' lines if any gate fails.
  G0 shard files: md5 of every pack/bin file == MANIFEST (every shard).
  G1 level 1 round trip: for EVERY chain the pack slice md5 == idx md5 == md5 of the loose npz (fresh read); --n-chains random chains: np.load of the slice == the loose npz array by array.
  G2 level 2 round trip: --n-chains random chains: every template of the chain from TemplatePack == indel_pool.read_template of the loose npz, field by field (coords bit-identical float32).
  G3 completeness: chains in the packs == loose npz files; template rows == sum n_templates == passing rows of metrics_all (main chains without rescue + rescue rows); provenance: rewind / num_steps of every template row == the rewind recorded in the npz == metrics_all schedule of its chain.
  G4 throughput: --n-bench random (chain, template) reads, loose vs level 1 vs level 2 (ms per template, bytes read).
Run: python t2_pack_verify.py --templates T --pack P [--n-chains 1000] [--n-bench 1000]
"""
import argparse
import glob
import hashlib
import io
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import indel_pool as ip
from t2_pack_reader import ChainPack, TemplatePack

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def same_tmpl(a, b):
    """a = loose reader, b = pack reader; only the keys of the pack record are compared (design_* keys of a merged design stage are not part of it); NaN == NaN."""
    for k, v in b.items():
        if k in ("num_steps", "source"):
            continue
        w = a[k]
        if isinstance(v, np.ndarray):
            if not (v.dtype == w.dtype and np.array_equal(v, w)):
                return False
        elif not (v == w or (v != v and w != w)):
            return False
    return True


def md5_chain_shard(args):
    """worker: fresh md5 of the loose file and of the pack slice of every chain of one level-1 shard, compared with the idx md5"""
    pack, templates, k = args
    cp = ChainPack(pack)
    rows = np.flatnonzero(cp.shard == k)
    bad = 0
    for i in rows:
        cid = str(cp.stems[i])
        bad += not (md5_file(os.path.join(templates, cid[1:3], cid + ".npz")) == cp.md5[i] == hashlib.md5(cp.raw(cid)).hexdigest())
    return len(rows), bad


def md5_shard_files(args):
    pack, k = args
    return k, md5_file(os.path.join(pack, f"chain.{k:03d}.pack")), md5_file(os.path.join(pack, f"tmpl.{k:03d}.bin"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--n-chains", type=int, default=1000)
    ap.add_argument("--n-bench", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    man = json.load(open(os.path.join(a.pack, "MANIFEST.json")))
    files = sorted(f for f in glob.glob(os.path.join(a.templates, "*", "*.npz")) if not f.endswith(".tmp.npz"))
    loose = {os.path.basename(f)[:-4]: f for f in files}
    # G0
    with Pool(a.procs) as pool:
        got = pool.map(md5_shard_files, [(a.pack, k) for k in range(man["n_shards"])])
    bad = [k for k, m1, m2 in got if m1 != man["chain_shard_md5"][k] or m2 != man["tmpl_shard_md5"][k]]
    check(not bad, f"G0 shard md5 vs MANIFEST: {man['n_shards'] * 2} files, {len(bad)} differ")
    cp, tp = ChainPack(a.pack), TemplatePack(a.pack)
    # G1
    with Pool(a.procs) as pool:
        res = pool.map(md5_chain_shard, [(a.pack, a.templates, k) for k in range(man["n_shards"])])
    n_chk, nb = sum(r[0] for r in res), sum(r[1] for r in res)
    check(nb == 0 and n_chk == len(loose), f"G1 level 1 md5 (slice == idx == loose file): {n_chk} chains checked of {len(loose)} loose, {nb} differ")
    sample = [str(x) for x in rng.choice(list(loose), size=min(a.n_chains, len(loose)), replace=False)]
    ne = 0
    for cid in sample:
        z0, z1 = np.load(loose[cid], allow_pickle=False), cp.get_chain(cid)
        ne += not (set(z0.files) == set(z1.files) and all(np.array_equal(z0[k], z1[k]) for k in z0.files))
    check(ne == 0, f"G1 level 1 arrays equal: {len(sample)} random chains, {ne} differ")
    # G2
    nt = nd = 0
    for cid in sample:
        z = np.load(loose[cid], allow_pickle=False)
        for i in range(int(z["n_templates"])):
            nt += 1
            nd += not same_tmpl(ip.read_template(loose[cid], i), tp.get(cid, int(z["draw"][i])))
    check(nd == 0, f"G2 level 2 templates equal the loose reader (float32 bit-identical): {nt} templates of {len(sample)} random chains, {nd} differ")
    # G3
    check(set(cp.row) == set(loose) == set(tp.chain_row), f"G3 chain sets: loose {len(loose)}, level 1 {len(cp.row)}, level 2 {len(tp.chain_row)}")
    per_chain = {c: int(np.load(f)["n_templates"]) for c, f in loose.items()}
    n_tm_loose = sum(per_chain.values())
    n_cnt_bad = sum(per_chain[str(c)] != int(n) for c, n in zip(tp.stems, tp.count))
    check(n_cnt_bad == 0, f"G3 templates per chain (level 2 index vs loose npz): {len(per_chain)} chains, {n_cnt_bad} differ")
    m = pd.read_csv(os.path.join(a.pack, "metrics_all.csv.gz"))
    resc = set(m[m.source == "rescue"].chain)
    expect = int(m[((m.source == "main") & ~m.chain.isin(resc) | (m.source == "rescue")) & m.pass_all].shape[0])
    n_rows = len(tp.col["chain"])
    check(n_rows == n_tm_loose == expect, f"G3 template rows {n_rows} == sum n_templates {n_tm_loose} == passing rows of metrics_all {expect}")
    ok_prov, n_prov = 0, 0
    sched = m.groupby(["chain", "source"]).agg(rw=("rewind", "first"), ns=("num_steps", "first")).reset_index()
    sched = {(r.chain, r.source): (int(r.rw), int(r.ns)) for r in sched.itertuples()}
    for r in range(n_rows):
        c = str(tp.col["chain"][r])
        key = (c, str(tp.col["source"][r]))
        n_prov += 1
        ok_prov += sched.get(key) == (int(tp.col["rewind"][r]), int(tp.col["num_steps"][r]))
    check(ok_prov == n_prov, f"G3 provenance (rewind, num_steps of the template row == schedule of its chain in metrics_all): {n_prov} templates, {n_prov - ok_prov} differ")
    # G4
    pick = [(str(c), int(rng.integers(0, int(tp.count[tp.chain_row[str(c)]])))) for c in rng.choice(list(loose), size=a.n_bench)]
    t = {}
    t0 = time.perf_counter(); by0 = 0
    for c, i in pick:
        ip.read_template(loose[c], i); by0 += os.path.getsize(loose[c])
    t["loose npz"] = (time.perf_counter() - t0, by0)
    t0 = time.perf_counter(); by1 = 0
    for c, i in pick:
        cp.read_template(c, i); by1 += int(cp.lengths[cp.row[c]])
    t["level 1"] = (time.perf_counter() - t0, by1)
    t0 = time.perf_counter(); by2 = 0
    for c, i in pick:
        r = tp.rows(c)[i]; tp._read(r); by2 += int(tp.col["nbytes"][r])
    t["level 2"] = (time.perf_counter() - t0, by2)
    for k, (sec, by) in t.items():
        print(f"G4 {k}: {1000 * sec / len(pick):.2f} ms per template, {by / len(pick) / 1e6:.3f} MB read per template ({len(pick)} random templates, warm or cold cache as found)")
    print("RESULT:", "FAIL" if FAILS else "ALL GATES PASSED")
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

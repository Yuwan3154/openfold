"""Verification gates of the packed T2 template product (T2_PACKED_TEMPLATES_PLAN.md), printed with their sample sizes; exit code 1 and 'FAIL' lines if any gate fails.
  G0 shard files: md5 of every pack/bin file == MANIFEST (every shard).
  G1 level 1 round trip: for EVERY chain the pack slice md5 == idx md5 == md5 of the loose npz (fresh read); --n-chains random chains: np.load of the slice == the loose npz array by array.
  G2 level 2 round trip: --n-chains random chains: every template of the chain from TemplatePack == indel_pool.read_template of the loose npz, field by field (coords bit-identical float32).
  G3 completeness: chains in the packs == loose npz files; template rows == sum n_templates == passing rows of metrics_all (main chains without rescue + rescue rows); provenance: rewind / num_steps of every template row == the rewind recorded in the npz == metrics_all schedule of its chain.
  G4 throughput: --n-bench random (chain, template) reads, loose vs level 1 vs level 2 (ms per template, bytes read).
Appended top-up rounds (MANIFEST 'appended'; needs --topup, the loose round trees): G0 / G1 / G2 / G3 above then cover the new shards too (loose npz of a template = <topup>/<round>/templates/..), plus
  A1 every passing top-up run of metrics_all is in the pack exactly once (key chain, round, draw), none extra;  A2 per (chain, round) with metrics rows: passing rows == template rows of the index (0 for a chain without npz);
  A3 provenance: source / rewind / num_steps of the pack rows and metrics rows of each round == the manifest record of the round; manifest counts (templates, metrics rows) == the files.
Selection (--final final_selection.npz, --k 64): S1 chosen count per chain == min(k, candidates in the pack); S2 no duplicate (chain, round, draw); S3 chosen set is a subset of the pool (the pack rows of the chain);
  S4 every chosen template passes the final filter (pass_all in metrics_all, the run row of its source / round / draw).
Level-2-only mode (--level2-only; required iff MANIFEST.json has a 'level1_removed' record, user 10-27): level 1 and the loose files are gone, so --templates / --topup do not apply; every gate needing them is printed as SKIPPED with
  the reason; G0 = tmpl shards only; G3 = the level-2 index (chain_stems / first / count == the chain column) and the metrics_all counts (total and per (round, chain)); G4 = level 2 only; A1-A3 and S1-S5 unchanged; plus
  R1 (in place of G1 / G2): --n-chains random (chain, round) npz rebuilt from level 2 (t2_rebuild_chain_npz.rebuild): atom_mask sum == A of every record, one key set and one dtype per key over the sample, shapes and
  n_templates / offsets == the tmpl.idx rows, and indel_pool.read_template of the rebuilt npz == TemplatePack, field by field.
Run: python t2_pack_verify.py --templates T --pack P [--topup TOPUP] [--final final_selection.npz] [--n-chains 1000] [--n-bench 1000]
     python t2_pack_verify.py --level2-only --pack P [--final final_selection.npz --sel-tsv S] [--n-chains 1000] [--n-bench 1000]
"""
import argparse
import glob
import hashlib
import io
import json
import os
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

import indel_pool as ip
from t2_diversity import ca_of
from t2_pack_reader import ChainPack, FinalSet, TemplatePack
from t2_rebuild_chain_npz import rebuild
from t2_select_diverse import pair_matrix

FAILS = []
SKIPS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def skip(gate, why):
    print(f"SKIPPED {gate}: {why}", flush=True)
    SKIPS.append(gate)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def same_tmpl(a, b):
    """a = loose reader, b = pack reader; only the keys of the pack record are compared (design_* keys of a merged design stage are not part of it); NaN == NaN."""
    for k, v in b.items():
        if k in ("num_steps", "source", "round"):
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
    pack, items, k = args
    cp = ChainPack(pack)
    rows = np.flatnonzero(cp.shard == k)
    bad = 0
    for i in rows:
        bad += not (md5_file(items[(str(cp.round[i]), str(cp.stems[i]))]) == cp.md5[i] == hashlib.md5(cp.raw_row(i)).hexdigest())
    return len(rows), bad


def md5_shard_files(args):
    pack, k, with_chain = args
    return k, md5_file(os.path.join(pack, f"chain.{k:03d}.pack")) if with_chain else None, md5_file(os.path.join(pack, f"tmpl.{k:03d}.bin"))


def rebuilt_part(tp, cid, rnd, rows):
    """R1 of one (chain, round): (n records, n bad, {key: (dtype, ndim)}); bad = atom_mask sum != A, a shape / count off the idx rows, or read_template of the rebuilt npz != TemplatePack"""
    recs = [tp._read(r) for r in rows]
    n_bad = sum(int(t["atom_mask"].sum()) != int(tp.col["A"][r]) for t, r in zip(recs, rows))
    if n_bad:
        return len(rows), n_bad, {}
    z = rebuild(tp, cid, rnd)
    buf = io.BytesIO()
    np.savez(buf, **z)
    n, sL, sA = len(rows), int(tp.col["L"][rows].sum()), int(tp.col["A"][rows].sum())
    shapes = dict(aatype=(sL,), residue_index=(sL,), orig_idx=(sL,), atom_mask=(sL, 37), coords=(sA, 3), res_offsets=(n + 1,), atom_offsets=(n + 1,), arm=(n,), model=(n,), rewind=(n,), draw=(n,), tm_native=(n,), tm_template=(n,))
    ok = int(z["n_templates"]) == n and int(z["res_offsets"][-1]) == sL and int(z["atom_offsets"][-1]) == sA and all(z[k].shape == v for k, v in shapes.items())
    n_bad = n if not ok else sum(not same_tmpl(ip.read_template(io.BytesIO(buf.getvalue()), i), recs[i]) for i in range(n))
    return n, n_bad, {k: ("str" if v.dtype.kind == "U" else str(v.dtype), v.ndim) for k, v in z.items()}   # a unicode width varies with the content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", help="the loose base tree (required unless --level2-only)")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--topup", help="loose top-up round trees (<topup>/<round>/templates/..); required when the pack has appended rounds")
    ap.add_argument("--final", help="final_selection.npz: run the selection gates")
    ap.add_argument("--sel-tsv", help="selection TSV (columns chain, max_pair): S5 recomputes the proxy max pair TM of the chosen sets of --n-chains random chains and compares")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n-chains", type=int, default=1000)
    ap.add_argument("--n-bench", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--level2-only", action="store_true", help="level 1 and the loose files are removed (MANIFEST 'level1_removed'): run the gates level 2 supports, SKIP the others")
    a = ap.parse_args()
    if a.final and not a.sel_tsv:
        ap.error("--final needs --sel-tsv (gate S5)")
    if a.level2_only and (a.templates or a.topup):
        ap.error("--templates / --topup do not apply with --level2-only (the loose files are removed)")
    if not a.level2_only and not a.templates:
        ap.error("--templates is required (or --level2-only)")
    l2 = a.level2_only
    rng = np.random.default_rng(a.seed)
    man = json.load(open(os.path.join(a.pack, "MANIFEST.json")))
    assert ("level1_removed" in man) == l2, "--level2-only must be given if and only if MANIFEST.json has a 'level1_removed' record"
    l1_files = glob.glob(os.path.join(a.pack, "chain.*.pack")) + glob.glob(os.path.join(a.pack, "chain.idx.npz"))
    assert not (l2 and l1_files), ("MANIFEST says level 1 is removed but level-1 files exist", l1_files[:3])
    no_l1 = "level 1 removed (MANIFEST 'level1_removed')"
    rounds = [e["round"] for e in man.get("appended", [])]
    loose, items = {}, {}
    if not l2:
        files = sorted(f for f in glob.glob(os.path.join(a.templates, "*", "*.npz")) if not f.endswith(".tmp.npz"))
        loose = {os.path.basename(f)[:-4]: f for f in files}
        assert a.topup or not rounds, "--topup is required: the pack has appended rounds"
        items = {("base", c): f for c, f in loose.items()}
        for rd in rounds:
            items.update({(rd, os.path.basename(f)[:-4]): f for f in glob.glob(os.path.join(a.topup, rd, "templates", "*", "*.npz")) if not f.endswith(".tmp.npz")})
    # G0
    with Pool(a.procs) as pool:
        got = pool.map(md5_shard_files, [(a.pack, k, not l2) for k in range(man["n_shards"])])
    if l2:
        bad = [k for k, _, m2 in got if m2 != man["tmpl_shard_md5"][k]]
        check(not bad and len(got) == man["n_shards"] > 0, f"G0 tmpl shard md5 vs MANIFEST: {len(got)} files, {len(bad)} differ")
        skip("G0 chain shards", no_l1)
    else:
        bad = [k for k, m1, m2 in got if m1 != man["chain_shard_md5"][k] or m2 != man["tmpl_shard_md5"][k]]
        check(not bad, f"G0 shard md5 vs MANIFEST: {man['n_shards'] * 2} files, {len(bad)} differ")
    cp, tp = None if l2 else ChainPack(a.pack), TemplatePack(a.pack)
    n_rows = len(tp.col["chain"])
    rcol = tp.col["round"] if tp.has_round else np.full(n_rows, "base")
    if l2:
        skip("G1 level 1 md5 and arrays", no_l1 + ", loose files removed")
        skip("G2 level 2 vs the loose reader", "loose files removed (R1 compares read_template of the rebuilt npz instead)")
        parts = sorted(set(zip(tp.col["chain"].astype(str), rcol.astype(str))))
        sample = [parts[i] for i in rng.choice(len(parts), size=min(a.n_chains, len(parts)), replace=False)]
        nt = nb = 0
        sig, keysets = {}, set()
        for cid, rd in sample:
            n, b, s = rebuilt_part(tp, cid, rd, np.array([r for r in tp.rows(cid) if tp.round_of(r) == rd]))
            nt, nb = nt + n, nb + b
            keysets.add(frozenset(s))
            for k, v in s.items():
                sig.setdefault(k, set()).add(v)
        multi = sorted(k for k, v in sig.items() if len(v) > 1)
        check(nb == 0 and nt > 0 and len(sample) > 0 and len(keysets) == 1 and not multi, f"R1 level-1 npz rebuilt from level 2 (mask sum == A, shapes / n_templates / offsets == tmpl.idx, read_template of the rebuilt npz == TemplatePack): "
              f"{len(sample)} random (chain, round) of {len(parts)}, {nt} templates, {nb} differ; {len(keysets)} key sets ({len(sig)} keys), keys with > 1 dtype or ndim over the sample: {multi}")
    else:
        # G1
        with Pool(a.procs) as pool:
            res = pool.map(md5_chain_shard, [(a.pack, items, k) for k in range(man["n_shards"])])
        n_chk, nb = sum(r[0] for r in res), sum(r[1] for r in res)
        check(nb == 0 and n_chk == len(items) > 0, f"G1 level 1 md5 (slice == idx == loose file): {n_chk} chain npz checked of {len(items)} loose (base + top-up), {nb} differ")
        keys = list(items)
        sample = [keys[i] for i in rng.choice(len(keys), size=min(a.n_chains, len(keys)), replace=False)]
        ne = 0
        for rd, cid in sample:
            z0, z1 = np.load(items[(rd, cid)], allow_pickle=False), cp.get_chain(cid, rd)
            ne += not (set(z0.files) == set(z1.files) and all(np.array_equal(z0[k], z1[k]) for k in z0.files))
        check(ne == 0 and len(sample) > 0, f"G1 level 1 arrays equal: {len(sample)} random chains, {ne} differ")
        # G2
        nt = nd = 0
        for rd, cid in sample:
            z = np.load(items[(rd, cid)], allow_pickle=False)
            for i in range(int(z["n_templates"])):
                nt += 1
                nd += not same_tmpl(ip.read_template(items[(rd, cid)], i), tp.get(cid, int(z["draw"][i]), rd))
        check(nd == 0 and nt > 0, f"G2 level 2 templates equal the loose reader (float32 bit-identical): {nt} templates of {len(sample)} random chains, {nd} differ")
    # G3
    got = Counter(zip(rcol.astype(str), tp.col["chain"].astype(str)))
    if l2:
        skip("G3 chain sets and templates per (round, chain) vs loose npz / level 1", no_l1 + ", loose files removed (level-2 index and metrics_all counts checked instead)")
        cnt = tp.count.astype(np.int64)
        idx_ok = len(cnt) > 0 and len(set(tp.stems.astype(str))) == len(cnt) and int(cnt.sum()) == n_rows and np.array_equal(tp.first, np.concatenate([[0], np.cumsum(cnt)[:-1]])) and np.array_equal(np.repeat(tp.stems.astype(str), cnt), tp.col["chain"].astype(str))
        check(idx_ok, f"G3 level-2 index: {len(cnt)} chains (unique), chain_first / chain_count contiguous over {n_rows} rows and == the chain column: {idx_ok}")
    else:
        check(set(cp.row) == {c for _, c in items} == set(tp.chain_row), f"G3 chain sets: loose {len({c for _, c in items})}, level 1 {len(cp.row)}, level 2 {len(tp.chain_row)}")
        per_chain = {k: int(np.load(f)["n_templates"]) for k, f in items.items()}
        n_tm_loose = sum(per_chain.values())
        n_cnt_bad = sum(per_chain.get((rd, c)) != n for (rd, c), n in got.items()) + len(set(per_chain) - {(rd, c) for rd, c in got})
        check(n_cnt_bad == 0 and len(per_chain) > 0, f"G3 templates per (round, chain) (level 2 index vs loose npz): {len(per_chain)} chain npz, {n_cnt_bad} differ")
    m = pd.read_csv(os.path.join(a.pack, "metrics_all.csv.gz"))
    m["round"] = m["round"].fillna("base") if "round" in m else "base"
    resc = set(m[m.source == "rescue"].chain)
    stored = (m.source == "main") & ~m.chain.isin(resc) | (m.source == "rescue")
    expect = int(m[stored & m.pass_all].shape[0])
    top = m[m.source.str.startswith("topup")]
    expect += int(top.pass_all.sum())
    if l2:
        check(n_rows == expect > 0, f"G3 template rows {n_rows} == passing rows of metrics_all {expect}")
        pm = m[(stored | m.source.str.startswith("topup")) & m.pass_all]
        want = Counter(zip(pm["round"].astype(str), pm.chain.astype(str)))
        n_grp_bad = sum(want.get(k) != v for k, v in got.items()) + len(set(want) - set(got))
        check(n_grp_bad == 0 and len(got) > 0, f"G3 templates per (round, chain): level-2 index vs passing rows of metrics_all: {len(got)} (round, chain) in the index, {len(want)} with passing rows, {n_grp_bad} differ")
    else:
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
    if rounds:
        sel = np.flatnonzero(rcol != "base")
        in_pack = list(zip(tp.col["chain"][sel].astype(str), rcol[sel].astype(str), tp.col["draw"][sel].astype(int)))
        passing = top[top.pass_all]
        want = set(zip(passing.chain, passing["round"], passing.draw))
        check(len(want) > 0 and len(in_pack) == len(set(in_pack)) and set(in_pack) == want, f"A1 top-up passing templates present exactly once: {len(want)} passing runs in metrics_all, {len(in_pack)} top-up rows in the pack, "
              f"{len(in_pack) - len(set(in_pack))} duplicates, {len(want - set(in_pack))} missing, {len(set(in_pack) - want)} extra")
        gcount = Counter((c, rd) for c, rd, _ in in_pack)
        npass = top.groupby(["chain", "round"]).pass_all.sum()
        n_bad = sum(int(v) != gcount.get(k, 0) for k, v in npass.items())
        check(n_bad == 0 and len(npass) > 0, f"A2 passing metrics rows vs index template rows per (chain, round): {len(npass)} groups with metrics rows ({sum(v == 0 for v in npass)} without a passing run), {n_bad} differ")
        bad_prov = 0
        for e in man["appended"]:
            rd, rs = e["round"], np.flatnonzero(rcol == e["round"])
            mm = m[m["round"] == rd]
            same = lambda src, rw, ns: (set(src) <= {e["source"]}) and (set(rw) <= {e["rewind"]}) and (set(ns) <= {e["num_steps"]})
            bad_prov += not (same(tp.col["source"][rs].astype(str), tp.col["rewind"][rs], tp.col["num_steps"][rs]) and same(mm.source, mm.rewind, mm.num_steps)
                             and e["n_templates"] == len(rs) and e["metrics_rows"] == len(mm) and e["metrics_chains"] == mm.chain.nunique()
                             and e["chain_shard_md5"] == man["chain_shard_md5"][e["shards"][0]:e["shards"][1]] and e["tmpl_shard_md5"] == man["tmpl_shard_md5"][e["shards"][0]:e["shards"][1]])
        check(bad_prov == 0, f"A3 provenance and counts of the appended rounds vs MANIFEST: {len(rounds)} rounds {rounds}, {bad_prov} differ")
    if a.final:
        fs = FinalSet(a.pack, a.final)
        chosen = pd.DataFrame([(c, rd, d) for c in map(str, fs.stems) for rd, d in fs.keys(c)], columns=["chain", "round", "draw"])
        pool = pd.DataFrame(dict(chain=tp.col["chain"].astype(str), round=rcol.astype(str), draw=tp.col["draw"].astype(int), source=tp.col["source"].astype(str)))
        ncand = pool.groupby("chain").size()
        nsel = chosen.groupby("chain").size().reindex(ncand.index, fill_value=0)
        bad_n = int((nsel != np.minimum(ncand, a.k)).sum())
        check(bad_n == 0 and set(ncand.index) <= set(map(str, fs.stems)), f"S1 chosen count == min({a.k}, candidates): {len(ncand)} chains with candidates ({int((ncand >= a.k).sum())} with >= {a.k}), {bad_n} differ")
        check(len(chosen) > 0 and not chosen.duplicated().any(), f"S2 no duplicate (chain, round, draw): {len(chosen)} chosen templates, {int(chosen.duplicated().sum())} duplicates")
        joined = chosen.merge(pool, how="left", on=["chain", "round", "draw"], indicator=True)
        check(len(chosen) > 0 and len(joined) == len(chosen) and (joined["_merge"] == "both").all(), f"S3 chosen set is a subset of the pool: {len(chosen)} chosen, {int((joined['_merge'] != 'both').sum())} not in the pack")
        runs = m[["chain", "source", "round", "draw", "pass_all"]]
        j2 = joined.merge(runs, how="left", on=["chain", "source", "round", "draw"])
        check(len(chosen) > 0 and len(j2) == len(chosen) and (j2.pass_all == True).all(), f"S4 chosen templates pass the final filter (pass_all of their run row): {len(chosen)} chosen, {int((j2.pass_all != True).sum())} not passing, {len(j2) - len(chosen)} ambiguous run rows")
    if a.final and a.sel_tsv:
        tsv = pd.read_csv(a.sel_tsv, sep="\t").set_index("chain")
        assert tsv.index.is_unique and set(map(str, fs.stems)) <= set(tsv.index), "--sel-tsv must be the merged selection table of every chain of the final set"
        cand = [c for c in map(str, fs.stems) if len(fs.keys(c)) > 1]
        pick5 = np.random.default_rng(a.seed + 1).choice(len(cand), size=min(a.n_chains, len(cand)), replace=False)
        bad5 = 0
        for i in pick5:
            ts = [tp.get(cand[i], int(d), rd) for rd, d in fs.keys(cand[i])]
            assert len({t["L_native"] for t in ts}) == 1, (cand[i], "the chosen templates of a chain have different native lengths")
            M = pair_matrix([ca_of(t) for t in ts], int(ts[0]["L_native"]))
            bad5 += not np.isclose(M[np.triu_indices(len(ts), 1)].max(), tsv.loc[cand[i], "max_pair"], rtol=0, atol=1e-9)
        check(bad5 == 0 and len(pick5) > 0, f"S5 proxy max pair TM of the chosen set recomputed from the pack == selection TSV max_pair: {len(pick5)} random chains, {bad5} differ")
    # G4
    t = {}
    if l2:
        skip("G4 loose npz and level 1 timing", no_l1 + ", loose files removed")
        pick = [(str(c), int(rng.integers(0, len(tp.rows(str(c)))))) for c in rng.choice(tp.stems, size=a.n_bench)]
    else:
        pick = [(str(c), int(rng.integers(0, per_chain[("base", str(c))]))) for c in rng.choice(list(loose), size=a.n_bench)]
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
    print("RESULT:", "FAIL" if FAILS else "ALL GATES PASSED" + (f" (level-2-only mode; {len(SKIPS)} SKIPPED: {'; '.join(SKIPS)})" if l2 else ""))
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

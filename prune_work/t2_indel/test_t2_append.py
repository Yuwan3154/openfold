"""Unit test of t2_pack_build.py --append, the FinalSet reader and the append / selection gates of t2_pack_verify.py on synthetic chains (numpy + pandas). Run: python test_t2_append.py"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

from t2_pack_reader import ChainPack, FinalSet, TemplatePack
from test_t2_pack import fake_chain, make_tree

K = 3


def run(*args, ok=True):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    assert (r.returncode == 0) == ok, (args, r.stdout[-1500:], r.stderr[-1500:])
    return r.stdout + r.stderr


def write_round(top, rd, rewind, chains, rng):
    """chains: {id: (draws of the npz templates or None for a chain without npz, number of run rows)}; the first len(draws) runs pass"""
    os.makedirs(os.path.join(top, rd), exist_ok=True)
    open(os.path.join(top, rd, "ids_shard0.txt"), "w").write("\n".join(chains) + "\n")
    for cid, (draws, n_run) in chains.items():
        d = os.path.join(top, rd, "templates", cid[1:3])
        os.makedirs(d, exist_ok=True)
        start = 64 if rd.startswith("P") else 2000
        run_draws = np.arange(start, start + n_run)
        n_pass = 0 if draws is None else len(draws)
        pd.DataFrame(dict(chain=cid, draw=run_draws, pass_all=np.arange(n_run) < n_pass, n_ins=2, n_del=1, n_mut=3, cn_med=rng.normal(1.3, 0.1, n_run))).to_csv(os.path.join(d, cid + ".metrics.csv"), index=False)
        if draws is not None:
            z = fake_chain(cid, rng, len(draws), rewind)
            z["draw"] = run_draws[:n_pass].astype(np.int16)
            np.savez(os.path.join(d, cid + ".npz"), **z)


def snapshot(out):
    return {os.path.relpath(f, out): hashlib.md5(open(f, "rb").read()).hexdigest() for f in glob.glob(os.path.join(out, "**", "*"), recursive=True) if os.path.isfile(f)}


def strip_round(out):
    """a pack built before the 'round' column existed (the cluster pack)"""
    for f in glob.glob(os.path.join(out, "meta", "*.npz")) + [os.path.join(out, "chain.idx.npz"), os.path.join(out, "tmpl.idx.npz")]:
        z = np.load(f)
        np.savez(f, **{k: z[k] for k in z.files if k != "round"})


def scenario(legacy):
    rng = np.random.default_rng(1)
    tmp = tempfile.mkdtemp()
    t, out, top = os.path.join(tmp, "templates"), os.path.join(tmp, "pack"), os.path.join(tmp, "topup")
    make_tree(t, rng)
    old = datetime(2026, 10, 1, 12, 0).timestamp()   # 1abc_A is a full-schedule main chain (rewind 250), as in test_t2_pack
    os.utime(os.path.join(t, "ab", "1abc_A.metrics.csv"), (old, old))
    run("t2_pack_build.py", "--templates", t, "--out", out, "--shard-gb", "0.00001", "--procs", "2")
    mp = os.path.join(out, "metrics_all.csv.gz")
    mm = pd.read_csv(mp)
    mm["cn_med"] = rng.normal(1.3, 0.1, len(mm))   # 17-digit floats: the append must not change the text of the existing rows
    mm.to_csv(mp, index=False)
    if legacy:
        strip_round(out)
    write_round(top, "P1", 125, {"1abc_A": ([64, 65], 4), "5new_E": ([64, 65, 66], 5), "6non_F": (None, 4)}, rng)   # 1abc_A: base + top-up; 6non_F: metrics.csv without npz
    write_round(top, "R1", 100, {"3qrs_C": ([2000, 2001], 3), "5new_E": ([2000], 2)}, rng)   # 3qrs_C: rescued base chain; 5new_E: a second round
    base_snap = {k: v for k, v in snapshot(out).items() if k.startswith(("chain.0", "tmpl.0"))}
    n_base = json.load(open(os.path.join(out, "MANIFEST.json")))["n_shards"]
    pre = os.path.join(tmp, "pack_before")
    shutil.copytree(out, pre)
    # append P1 first, then R1 in a second call (numbering continues; the first call must not be redone)
    run("t2_pack_build.py", "--append", "--out", out, "--topup", top, "--rounds", "P1", "--shard-gb", "0.00001", "--procs", "2")
    n_mid = json.load(open(os.path.join(out, "MANIFEST.json")))["n_shards"]
    run("t2_pack_build.py", "--append", "--out", out, "--topup", top, "--rounds", "P1", "R1", "--shard-gb", "0.00001", "--procs", "2")
    man = json.load(open(os.path.join(out, "MANIFEST.json")))
    assert n_mid > n_base and man["n_shards"] > n_mid and man["n_base_shards"] == n_base and [e["round"] for e in man["appended"]] == ["P1", "R1"], man
    assert len(man["chain_shard_md5"]) == len(man["tmpl_shard_md5"]) == man["n_shards"] and man["appended"][0]["shards"] == [n_base, n_mid] and man["appended"][1]["shards"] == [n_mid, man["n_shards"]]
    assert (man["appended"][0]["n_chains"], man["appended"][0]["n_templates"], man["appended"][0]["metrics_chains"], man["appended"][0]["metrics_rows"]) == (2, 5, 3, 13), man["appended"][0]
    assert (man["appended"][1]["n_chains"], man["appended"][1]["n_templates"], man["appended"][1]["metrics_rows"]) == (2, 3, 5), man["appended"][1]
    now = snapshot(out)
    assert all(now[k] == v for k, v in base_snap.items()), "an existing shard changed"
    cp, tp = ChainPack(out), TemplatePack(out)
    assert cp.rounds("1abc_A") == ["base", "P1"] and cp.rounds("5new_E") == ["P1", "R1"] and cp.rounds("2xyz_B") == ["base"]
    assert cp.raw("1abc_A") == open(os.path.join(t, "ab", "1abc_A.npz"), "rb").read() and cp.raw("1abc_A", "P1") == open(os.path.join(top, "P1", "templates", "ab", "1abc_A.npz"), "rb").read()
    assert tp.draws("1abc_A") == [1, 4, 7, 10, 64, 65] and tp.draws("5new_E") == [64, 65, 66, 2000] and tp.draws("3qrs_C") == [1, 4, 7, 10, 2000, 2001] and "6non_F" not in tp
    a, b = tp.get("1abc_A", 64), tp.get("5new_E", 2000, "R1")
    assert (a["round"], a["source"], a["rewind"], a["num_steps"]) == ("P1", "topup-production", 125, 250) and (b["round"], b["source"], b["rewind"], b["num_steps"]) == ("R1", "topup-rewind100", 100, 250)
    c, d = tp.get("3qrs_C", 1), tp.get("3qrs_C", 2001, "R1")
    assert (c["round"], c["source"], c["rewind"]) == ("base", "rescue", 100) and d["source"] == "topup-rewind100"
    z = np.load(os.path.join(top, "P1", "templates", "ab", "1abc_A.npz"))
    assert np.array_equal(a["coords"], z["coords"][int(z["atom_offsets"][0]):int(z["atom_offsets"][1])])
    m = pd.read_csv(os.path.join(out, "metrics_all.csv.gz"))
    base_rows = m[m["round"].isna()]
    assert len(base_rows) == 24 + 6 and set(base_rows.source) == {"main", "rescue"} and len(m) == 30 + 13 + 5
    assert len(m[(m["round"] == "P1") & (m.chain == "6non_F")]) == 4 and not m[(m["round"] == "P1") & (m.chain == "6non_F")].pass_all.any()
    assert set(m[m["round"] == "R1"].source) == {"topup-rewind100"} and set(m[m["round"] == "R1"].rewind) == {100} and set(m[m["round"] == "P1"].num_steps) == {250}
    # re-run with the same inputs = no-op: every file byte-identical, nothing rebuilt
    before = snapshot(out)
    assert "nothing to append" in run("t2_pack_build.py", "--append", "--out", out, "--topup", top, "--rounds", "P1", "R1", "--procs", "2")
    assert snapshot(out) == before
    # a recorded round whose loose files changed is refused
    f = os.path.join(top, "P1", "templates", "ab", "1abc_A.npz")
    orig = open(f, "rb").read()
    np.savez(f, **fake_chain("1abc_A", rng, 3, 125))
    run("t2_pack_build.py", "--append", "--out", out, "--topup", top, "--rounds", "P1", "--procs", "2", ok=False)
    open(f, "wb").write(orig)
    assert snapshot(out) == before
    # the appended pack passes every gate (including the check that the loose trees match the shards)
    verify = ["t2_pack_verify.py", "--templates", t, "--pack", out, "--topup", top, "--n-chains", "50", "--n-bench", "5", "--procs", "2"]
    ok = run(*verify)
    assert "ALL GATES PASSED" in ok and "FAIL" not in ok and "A1 " in ok and "A2 " in ok and "A3 " in ok, ok
    # selection: the pool read from the appended pack == the pool read from the pre-append pack + the loose rounds
    run("t2_select_diverse.py", "--pack", pre, "--topup", top, "--out", os.path.join(tmp, "sel_pre.tsv"), "--k", str(K))
    shutil.copytree(os.path.join(top, "P1"), os.path.join(top, "P1test"))   # a stray shakedown directory must never enter a pool
    run("t2_select_diverse.py", "--pack", pre, "--topup", top, "--out", os.path.join(tmp, "sel_stray.tsv"), "--k", str(K))
    assert pd.read_csv(os.path.join(tmp, "sel_pre.tsv"), sep="\t").n_pool.tolist() == pd.read_csv(os.path.join(tmp, "sel_stray.tsv"), sep="\t").n_pool.tolist(), "a non-round directory entered the selection pool"
    run("t2_select_diverse.py", "--pack", out, "--topup", top, "--out", os.path.join(tmp, "sel0.tsv"), "--k", str(K), "--n-shards", "2", "--shard", "0")
    run("t2_select_diverse.py", "--pack", out, "--topup", top, "--out", os.path.join(tmp, "sel1.tsv"), "--k", str(K), "--n-shards", "2", "--shard", "1")
    pre_df = pd.read_csv(os.path.join(tmp, "sel_pre.tsv"), sep="\t").set_index("chain")
    sel = pd.concat([pd.read_csv(os.path.join(tmp, f"sel{i}.tsv"), sep="\t") for i in (0, 1)]).set_index("chain")
    assert set(sel.index) == set(tp.chain_row) and (sel.n_pool == [len(tp.rows(c)) for c in sel.index]).all()
    assert all(pre_df.loc[c].kept == sel.loc[c].kept and pre_df.loc[c].n_pool == sel.loc[c].n_pool for c in pre_df.index)   # pool = pack + loose rounds, or the appended pack alone
    assert sel.loc["1abc_A"].n_pool == 6 and sel.loc["5new_E"].n_pool == 4
    final = os.path.join(out, "final_selection.npz")
    print(run("t2_select_diverse.py", "--merge", os.path.join(tmp, "sel0.tsv"), os.path.join(tmp, "sel1.tsv"), "--final", final).strip())
    fs = FinalSet(out)
    got = {c: ts for c, ts in fs}
    assert set(got) == set(sel.index) and all(len(got[c]) == min(K, len(tp.rows(c))) for c in got), {c: len(v) for c, v in got.items()}
    assert [(x["round"], x["draw"]) for x in got["1abc_A"]] == fs.keys("1abc_A") and len(set(fs.keys("5new_E"))) == K
    assert all(x["coords"].dtype == np.float32 for ts in got.values() for x in ts)
    ok = run(*verify, "--final", final, "--k", str(K))
    assert "ALL GATES PASSED" in ok and "S1 " in ok and "S2 " in ok and "S3 " in ok and "S4 " in ok, ok
    # selection gates must fail on a duplicate, a key outside the pool, a wrong count
    keys = {c: fs.keys(c) for c in map(str, fs.stems)}
    k1 = keys["1abc_A"]
    bad = {"S2": [k1[0], k1[0], k1[2]], "S3": [k1[0], k1[1], ("P1", 9999)], "S1": k1[:2]}
    for name, ks in bad.items():
        path = os.path.join(tmp, f"bad_{name}.npz")
        allk = {**keys, "1abc_A": ks}
        cnt = np.array([len(v) for v in allk.values()], np.int32)
        np.savez(path, chain=np.array(list(allk)), first=np.concatenate([[0], np.cumsum(cnt, dtype=np.int64)[:-1]]), count=cnt, round=np.array([r for v in allk.values() for r, _ in v]), draw=np.array([d for v in allk.values() for _, d in v], np.int32))
        out_bad = run(*verify, "--final", path, "--k", str(K), ok=False)
        assert f"FAIL {name} " in out_bad, (name, out_bad)
    m2 = m.copy()
    m2.loc[(m2["round"] == "P1") & (m2.chain == "5new_E") & (m2.draw == 64), "pass_all"] = False
    m2.to_csv(os.path.join(out, "metrics_all.csv.gz"), index=False)
    out_bad = run(*verify, "--final", final, "--k", str(K), ok=False)
    assert "FAIL A1 " in out_bad and "FAIL A2 " in out_bad, out_bad
    m.to_csv(os.path.join(out, "metrics_all.csv.gz"), index=False)
    # refusals leave the pack untouched: tampered manifest md5, unfinished round (npz without metrics.csv), missing round directory, duplicate round names
    neg = os.path.join(tmp, "pack_neg")
    shutil.copytree(pre, neg)
    before = snapshot(neg)
    man_n = json.load(open(os.path.join(neg, "MANIFEST.json")))
    man_n["tmpl_shard_md5"][0] = "0" * 32
    json.dump(man_n, open(os.path.join(neg, "MANIFEST.json"), "w"))
    before = snapshot(neg)
    assert "differs from the manifest" in run("t2_pack_build.py", "--append", "--out", neg, "--topup", top, "--rounds", "P1", "--procs", "2", ok=False)
    assert snapshot(neg) == before, "a refused append (manifest md5) changed the pack"
    shutil.copy(os.path.join(pre, "MANIFEST.json"), os.path.join(neg, "MANIFEST.json"))
    before = snapshot(neg)
    assert "no loose files" in run("t2_pack_build.py", "--append", "--out", neg, "--topup", os.path.join(tmp, "nowhere"), "--rounds", "R1", "--procs", "2", ok=False)
    assert "duplicate rounds" in run("t2_pack_build.py", "--append", "--out", neg, "--topup", top, "--rounds", "P1", "P1", "--procs", "2", ok=False)
    broken = os.path.join(tmp, "topup_broken")
    shutil.copytree(top, broken)
    os.remove(os.path.join(broken, "R1", "templates", "qr", "3qrs_C.metrics.csv"))
    open(os.path.join(broken, "R1", "ids_shard1.txt"), "w").write("9zzz_Z\n")
    assert "round not complete" in run("t2_pack_build.py", "--append", "--out", neg, "--topup", broken, "--rounds", "R1", "--procs", "2", ok=False)
    os.remove(os.path.join(broken, "R1", "ids_shard1.txt"))
    assert "round not complete" in run("t2_pack_build.py", "--append", "--out", neg, "--topup", broken, "--rounds", "R1", "--procs", "2", ok=False)
    assert snapshot(neg) == before, "a refused append (empty / unfinished round) changed the pack"
    shutil.rmtree(tmp)


def main():
    for legacy in (True, False):
        scenario(legacy)
    print("test_t2_append: all checks passed")


if __name__ == "__main__":
    main()

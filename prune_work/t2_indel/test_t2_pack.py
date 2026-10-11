"""Unit test of the packed template product on synthetic chains (numpy + pandas only): build both levels, read back, compare with the loose npz, resume, and the corrupt-byte gate. Run: python test_t2_pack.py"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

import indel_pool as ip
from t2_pack_reader import ChainPack, TemplatePack


def fake_chain(cid, rng, n, rewind, tm_lo=0.45):
    items = []
    for d in range(n):
        L = int(rng.integers(30, 60))
        mask = np.zeros((L, 37), bool)
        mask[:, [0, 1, 2, 4]] = True
        mask[:, 3] = rng.random(L) < 0.9
        A = int(mask.sum())
        items.append(dict(arm="t2pipe", model="cc89", rewind=rewind, draw=d * 3 + 1, coords=rng.normal(size=(A, 3)).astype(np.float32) * 10, atom_mask=mask, aatype=rng.integers(0, 20, L).astype(np.int8),
                          residue_index_orig=np.arange(1, L + 1), orig_idx=np.where(rng.random(L) < 0.1, -1, np.arange(L)), ops=[["ins", 3, 2]], tm_native=tm_lo + 0.4 * rng.random(), tm_template=0.5, L_native=L - 2))
    return ip.pack_chain(cid, items)


def make_tree(root, rng):
    for cid, rw, resc in (("1abc_A", 250, False), ("2xyz_B", 125, False), ("3qrs_C", 100, True), ("4mno_D", 125, False)):
        d = os.path.join(root, cid[1:3])
        os.makedirs(d, exist_ok=True)
        n = 4 if cid != "4mno_D" else 1
        np.savez(os.path.join(d, cid + ".npz"), **fake_chain(cid, rng, n, rw))
        draws = np.arange(6) * 3 + 1
        m = pd.DataFrame(dict(chain=cid, draw=draws, pass_all=np.arange(6) < n, n_ins=2, n_del=1, n_mut=3))   # the first n run variants pass = the n stored templates
        m.to_csv(os.path.join(d, cid + (".rescue.metrics.csv" if resc else ".metrics.csv")), index=False)
        if resc:
            m.assign(pass_all=False).to_csv(os.path.join(d, cid + ".metrics.csv"), index=False)


def same(a, b):
    for k, v in a.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, b[k]), k
        elif k not in ("num_steps", "source"):
            assert v == b[k], (k, v, b[k])


def main():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()
    t, out = os.path.join(tmp, "templates"), os.path.join(tmp, "pack")
    make_tree(t, rng)
    old = datetime(2026, 10, 1, 12, 0).timestamp()   # a main-run chain of the full schedule: metrics.csv older than --full-step-until
    os.utime(os.path.join(t, "ab", "1abc_A.metrics.csv"), (old, old))
    cmd = [sys.executable, "t2_pack_build.py", "--templates", t, "--out", out, "--shard-gb", "0.00001", "--procs", "2"]
    subprocess.run(cmd, check=True, capture_output=True)
    cp, tp = ChainPack(out), TemplatePack(out)
    n_tm = 0
    for cid in ("1abc_A", "2xyz_B", "3qrs_C", "4mno_D"):
        loose = os.path.join(t, cid[1:3], cid + ".npz")
        assert cp.raw(cid) == open(loose, "rb").read(), cid   # level 1 is byte-identical
        z = np.load(loose)
        for i in range(int(z["n_templates"])):
            a = ip.read_template(loose, i)
            same(a, cp.read_template(cid, i))
            b = tp.get(cid, int(z["draw"][i]))
            same(a, b)
            assert b["source"] == ("rescue" if cid == "3qrs_C" else "main") and b["num_steps"] == (500 if b["rewind"] == 250 else 250)
            n_tm += 1
    assert len(tp.col["chain"]) == n_tm
    man = json.load(open(os.path.join(out, "MANIFEST.json")))
    assert man["n_chains"] == 4 and man["n_templates"] == n_tm and man["rescue_chains"] == 1, man
    m = pd.read_csv(os.path.join(out, "metrics_all.csv.gz"))
    assert set(m.source) == {"main", "rescue"} and set(m.rewind) == {100, 125, 250}, m.groupby("source").rewind.unique()
    # the clean pack must pass every gate (a passing control for the corruption test below)
    ok = subprocess.run([sys.executable, "t2_pack_verify.py", "--templates", t, "--pack", out, "--n-chains", "10", "--n-bench", "5", "--procs", "2"], capture_output=True, text=True)
    assert ok.returncode == 0 and "ALL GATES PASSED" in ok.stdout and "FAIL" not in ok.stdout, ok.stdout[-800:]
    # a meta of another plan must not be reused: change the shard size and rebuild into the same dir -> the packs change, the verify still passes
    cmd2 = [sys.executable, "t2_pack_build.py", "--templates", t, "--out", out, "--shard-gb", "1", "--procs", "2"]
    subprocess.run(cmd2, check=True, capture_output=True)
    assert json.load(open(os.path.join(out, "MANIFEST.json")))["n_shards"] == 1
    ok = subprocess.run([sys.executable, "t2_pack_verify.py", "--templates", t, "--pack", out, "--n-chains", "10", "--n-bench", "5", "--procs", "2"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout[-800:]
    subprocess.run(cmd, check=True, capture_output=True)   # back to many small shards: shard 0's meta has another plan (rebuilt), the others match (reused)
    # resume: removing one shard's meta and pack rebuilds only that shard to the same bytes
    before = open(os.path.join(out, "tmpl.000.bin"), "rb").read()
    os.remove(os.path.join(out, "meta", "tmpl.000.npz")); os.remove(os.path.join(out, "tmpl.000.bin"))
    open(os.path.join(out, "tmpl.000.bin.partial"), "wb").write(b"stale")   # a leftover .partial of a crashed run
    subprocess.run(cmd, check=True, capture_output=True)
    assert open(os.path.join(out, "tmpl.000.bin"), "rb").read() == before
    os.remove(os.path.join(out, "chain.001.pack"))   # a meta without its pack is rebuilt too
    subprocess.run(cmd, check=True, capture_output=True)
    assert os.path.isfile(os.path.join(out, "chain.001.pack"))
    # corrupt-byte gate: a flipped byte in a chain pack must be caught by the md5 check of t2_pack_verify
    p = os.path.join(out, "chain.000.pack")
    b = bytearray(open(p, "rb").read()); b[10] ^= 0xFF; open(p, "wb").write(b)
    r = subprocess.run([sys.executable, "t2_pack_verify.py", "--templates", t, "--pack", out, "--n-chains", "10", "--n-bench", "5", "--procs", "2"], capture_output=True, text=True)
    assert r.returncode != 0 and "FAIL" in r.stdout, r.stdout[-500:]
    shutil.rmtree(tmp)
    print("test_t2_pack: all checks passed")


if __name__ == "__main__":
    main()

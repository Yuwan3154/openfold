"""Unit test of t2_rebuild_chain_npz.py on a synthetic pack with appended rounds (chains spanning shards, a NaN tm_native): check passes on the clean pack, rebuild == the loose npz, and a flipped level-2 byte fails the check. Run: python test_t2_rebuild_chain_npz.py"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from t2_pack_reader import TemplatePack, layout
from t2_rebuild_chain_npz import rebuild, same
from test_t2_append import run, write_round
from test_t2_pack import make_tree


def main():
    rng = np.random.default_rng(2)
    tmp = tempfile.mkdtemp()
    t, out, top = os.path.join(tmp, "templates"), os.path.join(tmp, "pack"), os.path.join(tmp, "topup")
    make_tree(t, rng)
    f = os.path.join(t, "xy", "2xyz_B.npz")
    z = dict(np.load(f))
    z["tm_native"][1] = np.nan   # NaN must compare equal to NaN
    np.savez(f, **z)
    run("t2_pack_build.py", "--templates", t, "--out", out, "--shard-gb", "0.00001", "--procs", "2")
    mp = os.path.join(out, "metrics_all.csv.gz")
    pd.read_csv(mp).assign(cn_med=1.3).to_csv(mp, index=False)   # write_round's metrics.csv carry cn_med, as in test_t2_append
    write_round(top, "P1", 125, {"1abc_A": ([64, 65], 4), "5new_E": ([64, 65, 66], 5), "6non_F": (None, 4)}, rng)
    write_round(top, "R1", 100, {"3qrs_C": ([2000, 2001], 3), "5new_E": ([2000], 2)}, rng)
    run("t2_pack_build.py", "--append", "--out", out, "--topup", top, "--rounds", "P1", "R1", "--shard-gb", "0.00001", "--procs", "2")
    loose = {("base", os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(t, "*", "*.npz"))}
    for rd in ("P1", "R1"):
        loose.update({(rd, os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(top, rd, "templates", "*", "*.npz"))})
    n_tm = sum(int(np.load(p)["n_templates"]) for p in loose.values())
    assert len(loose) == 8 and n_tm == 13 + 5 + 3, (len(loose), n_tm)
    # every (round, chain) rebuilt from level 2 equals its loose npz array by array (keys, dtype, shape, values)
    tp = TemplatePack(out)
    for (rd, cid), p in loose.items():
        a, b = np.load(p), rebuild(tp, cid, rd)
        assert set(a.files) == set(b) and all(same(a[k], b[k]) for k in a.files), (rd, cid, [k for k in a.files if not same(a[k], b[k])])
    assert np.isnan(rebuild(tp, "2xyz_B")["tm_native"][1])
    assert not same(np.array([1], np.int16), np.array([1], np.int32)) and same(np.array([np.nan], np.float32), np.array([np.nan], np.float32)) and not same(np.array([1.0]), np.array([1.0, 1.0]))
    # rebuild mode writes an npz equal to the loose one
    o = os.path.join(tmp, "r.npz")
    run("t2_rebuild_chain_npz.py", "rebuild", "--pack", out, "--chain", "5new_E", "--round", "R1", "--out", o)
    a, b = np.load(loose[("R1", "5new_E")]), np.load(o)
    assert set(a.files) == set(b.files) and all(same(a[k], b[k]) for k in a.files)
    # check mode passes on the clean pack with the full sample
    s = run("t2_rebuild_chain_npz.py", "check", "--pack", out, "--procs", "2")
    assert "ALL CHECKS PASSED" in s and "FAIL" not in s and "8 chain npz (chain, round) of 8 level-1 rows" in s and f"{n_tm} templates" in s and "0 npz differ" in s, s
    # known-bad control 1: a flipped level-1 byte fails the md5 of the reference slice
    c1 = os.path.join(tmp, "pack_l1bad")
    shutil.copytree(out, c1)
    q = os.path.join(c1, "chain.000.pack")
    b1 = bytearray(open(q, "rb").read()); b1[10] ^= 0xFF; open(q, "wb").write(b1)
    bad1 = subprocess.run([sys.executable, "t2_rebuild_chain_npz.py", "check", "--pack", c1, "--procs", "2"], capture_output=True, text=True)
    assert bad1.returncode == 1 and "1 npz differ" in bad1.stdout and "md5 != chain.idx md5" in bad1.stdout, bad1.stdout[-1500:] + bad1.stderr[-1500:]
    # known-bad control 2: one flipped coords byte in level 2 must fail the check
    r = tp.rows("1abc_A")[0]
    p = os.path.join(out, f"tmpl.{int(tp.col['shard'][r]):03d}.bin")
    buf = bytearray(open(p, "rb").read())
    c0 = int(tp.col["off"][r]) + layout(int(tp.col["L"][r]), int(tp.col["A"][r]))["coords"]
    buf[c0 + 1] ^= 0xFF
    open(p, "wb").write(buf)
    bad = subprocess.run([sys.executable, "t2_rebuild_chain_npz.py", "check", "--pack", out, "--procs", "2"], capture_output=True, text=True)
    assert bad.returncode == 1 and "FAIL rebuilt == level 1" in bad.stdout and "1 npz differ" in bad.stdout and "'coords'" in bad.stdout, bad.stdout[-1500:] + bad.stderr[-1500:]
    shutil.rmtree(tmp)
    print("test_t2_rebuild_chain_npz: all checks passed")


if __name__ == "__main__":
    main()

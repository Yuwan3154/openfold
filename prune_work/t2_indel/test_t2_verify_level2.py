"""Unit test of t2_pack_verify.py --level2-only on a synthetic appended pack with a final selection: the full pack passes the default mode (control); after level 1 and the loose files are removed (MANIFEST 'level1_removed')
the default mode is refused, the level-2 mode passes with every level-1 / loose gate reported as SKIPPED, and a flipped level-2 byte (G0) or a corrupt atom_mask with a matching manifest md5 (R1) fails. Run: python test_t2_verify_level2.py"""
import glob
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

from t2_pack_reader import TemplatePack, layout
from test_t2_append import run, write_round
from test_t2_pack import make_tree

K = 3


def main():
    rng = np.random.default_rng(3)
    tmp = tempfile.mkdtemp()
    t, out, top = os.path.join(tmp, "templates"), os.path.join(tmp, "pack"), os.path.join(tmp, "topup")
    make_tree(t, rng)
    old = datetime(2026, 10, 1, 12, 0).timestamp()   # 1abc_A is a full-schedule main chain (rewind 250), as in test_t2_pack
    os.utime(os.path.join(t, "ab", "1abc_A.metrics.csv"), (old, old))
    run("t2_pack_build.py", "--templates", t, "--out", out, "--shard-gb", "0.00001", "--procs", "2")
    mp = os.path.join(out, "metrics_all.csv.gz")
    pd.read_csv(mp).assign(cn_med=1.3).to_csv(mp, index=False)   # write_round's metrics.csv carry cn_med, as in test_t2_append
    write_round(top, "P1", 125, {"1abc_A": ([64, 65], 4), "5new_E": ([64, 65, 66], 5), "6non_F": (None, 4)}, rng)
    write_round(top, "R1", 100, {"3qrs_C": ([2000, 2001], 3), "5new_E": ([2000], 2)}, rng)
    run("t2_pack_build.py", "--append", "--out", out, "--topup", top, "--rounds", "P1", "R1", "--shard-gb", "0.00001", "--procs", "2")
    sel, final = os.path.join(tmp, "sel.tsv"), os.path.join(out, "final_selection.npz")
    run("t2_select_diverse.py", "--pack", out, "--topup", top, "--out", sel, "--k", str(K))
    run("t2_select_diverse.py", "--merge", sel, "--final", final)
    common = ["--pack", out, "--final", final, "--sel-tsv", sel, "--k", str(K), "--n-chains", "50", "--n-bench", "5", "--procs", "2"]
    # control: the full pack passes the default mode, and --level2-only is refused without the manifest record
    s = run("t2_pack_verify.py", "--templates", t, "--topup", top, *common)
    assert "ALL GATES PASSED" in s and "FAIL" not in s and "SKIPPED" not in s and "PASS G1 " in s, s
    assert "'level1_removed' record" in run("t2_pack_verify.py", "--level2-only", *common, ok=False)
    # remove level 1 and the loose files, as on the cluster (10-27)
    l1 = glob.glob(os.path.join(out, "chain.*.pack")) + [os.path.join(out, "chain.idx.npz")] + glob.glob(os.path.join(out, "meta", "chain.*.npz"))
    assert len(l1) > 2
    for f in l1:
        os.remove(f)
    shutil.rmtree(t); shutil.rmtree(top)
    mf = os.path.join(out, "MANIFEST.json")
    man = json.load(open(mf))
    man["level1_removed"] = dict(date="2026-10-27", files=[os.path.relpath(f, out) for f in l1])
    json.dump(man, open(mf, "w"))
    assert "'level1_removed' record" in run("t2_pack_verify.py", "--templates", t, *common, ok=False)
    assert "do not apply" in run("t2_pack_verify.py", "--level2-only", "--templates", t, *common, ok=False)
    s = run("t2_pack_verify.py", "--level2-only", *common)
    n_tm = len(TemplatePack(out).col["chain"])
    assert "ALL GATES PASSED (level-2-only mode; 5 SKIPPED" in s and "FAIL" not in s, s
    for g in ("G0 chain shards", "G1 level 1", "G2 level 2 vs the loose reader", "G3 chain sets", "G4 loose npz"):
        assert f"SKIPPED {g}" in s, (g, s)
    for g in ("G0 tmpl shard md5", "R1 ", "G3 level-2 index", "G3 template rows", "G3 templates per (round, chain)", "G3 provenance", "A1 ", "A2 ", "A3 ", "S1 ", "S2 ", "S3 ", "S4 ", "S5 "):
        assert f"PASS {g}" in s, (g, s)
    assert f"8 random (chain, round) of 8, {n_tm} templates, 0 differ; 1 key sets" in s and "G4 level 2:" in s and "G4 level 1" not in s, s
    # known-bad control 0: one passing base run of 1abc_A moved to 2xyz_B keeps the total and fails the per-(round, chain) count
    m = pd.read_csv(mp)
    ma = m.copy()
    ma.loc[ma.index[(ma.chain == "1abc_A") & ma.pass_all & ma["round"].isna()][0], "pass_all"] = False
    ma.loc[ma.index[(ma.chain == "2xyz_B") & ~ma.pass_all & ma["round"].isna()][0], "pass_all"] = True
    ma.to_csv(mp, index=False)
    s = run("t2_pack_verify.py", "--level2-only", "--pack", out, "--n-chains", "50", "--n-bench", "5", "--procs", "2", ok=False)
    assert "PASS G3 template rows" in s and "FAIL G3 templates per (round, chain)" in s and "2 differ" in s, s
    m.to_csv(mp, index=False)
    # known-bad control 1: a flipped coords byte of level 2 fails G0
    good = os.path.join(tmp, "pack_good")
    shutil.copytree(out, good)
    tp = TemplatePack(out)
    r = tp.rows("1abc_A")[0]
    p = os.path.join(out, f"tmpl.{int(tp.col['shard'][r]):03d}.bin")
    buf = bytearray(open(p, "rb").read())
    lay = layout(int(tp.col["L"][r]), int(tp.col["A"][r]))
    o = int(tp.col["off"][r])
    buf[o + lay["coords"] + 1] ^= 0xFF
    open(p, "wb").write(buf)
    s = run("t2_pack_verify.py", "--level2-only", *common, ok=False)
    assert "FAIL G0 tmpl shard md5 vs MANIFEST" in s and "1 differ" in s and "RESULT: FAIL" in s, s
    # known-bad control 2: an atom_mask byte turned on (atoms != coords) with the manifest md5 updated passes G0 and fails R1
    shutil.rmtree(out); shutil.copytree(good, out)
    buf = bytearray(open(p, "rb").read())
    am = np.frombuffer(bytes(buf[o + lay["atom_mask"]:o + lay["atom_mask"] + 37 * int(tp.col["L"][r])]), np.uint8)
    buf[o + lay["atom_mask"] + int(np.flatnonzero(am == 0)[0])] = 1
    open(p, "wb").write(buf)
    man = json.load(open(mf))
    man["tmpl_shard_md5"][int(tp.col["shard"][r])] = hashlib.md5(bytes(buf)).hexdigest()
    json.dump(man, open(mf, "w"))
    s = run("t2_pack_verify.py", "--level2-only", "--pack", out, "--n-chains", "50", "--n-bench", "5", "--procs", "2", ok=False)   # without --final: S5 (ca_of) would crash on the corrupt record
    assert "PASS G0 tmpl shard md5" in s and "FAIL R1 " in s and "RESULT: FAIL" in s, s
    shutil.rmtree(tmp)
    print("test_t2_verify_level2: all checks passed")


if __name__ == "__main__":
    main()

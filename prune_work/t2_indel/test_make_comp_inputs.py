import json
import os
import subprocess
import sys

import numpy as np

from indel_edit import edit
from make_indel_inputs import SLOT, apply_plan, write_pdb


def test_bb_stage_types_and_merge_keeps_native_atoms(tmp_path):
    rng = np.random.default_rng(0)
    L = 14
    xyz = rng.normal(size=(L, 37, 3)) * 5
    mask = np.zeros((L, 37), bool)
    mask[:, [SLOT[n] for n in ("N", "CA", "C", "O", "CB")]] = True
    names = ["ALA", "SER"] * 7
    ops = [("ins", 4, 3), ("del", 9, 10)]
    nn, nx, nm, orig, n2n = apply_plan(names, xyz, mask, ops)
    src = tmp_path / "in" / "x_A"
    src.mkdir(parents=True)
    write_pdb(src / "d00.pdb", nn, nx, nm)
    write_pdb(src / "native.pdb", names, xyz, mask)
    json.dump({"native_resnames": names, "plans": [{"orig_idx": orig.tolist()}]}, open(src / "plans.json", "w"))
    me = os.path.dirname(os.path.abspath(__file__))
    run = lambda *x: subprocess.run([sys.executable, os.path.join(me, "make_comp_inputs.py"), *x], check=True,
                                    cwd=me, capture_output=True)
    run("--stage", "bb", "--inputs-dir", str(tmp_path / "in"), "--out-dir", str(tmp_path / "bb"), "--chains", "x_A", "--draws", "0")
    bb = [ln for ln in open(tmp_path / "bb" / "x_A" / "d00.pdb") if ln.startswith("ATOM")]
    assert all(ln[12:16].strip() in ("N", "CA", "C", "O") for ln in bb)
    assert all(ln.strip() for ln in open(tmp_path / "bb" / "x_A" / "d00.pdb")), "blank line in a written PDB"
    res = {}
    for ln in bb:
        res.setdefault(int(ln[22:26]), ln[17:20])
    ins_res = [j + 1 for j, o in enumerate(orig) if o < 0]
    assert all(res[j] in ("ALA", "SER") for j in ins_res)          # drawn from the chain's own composition
    # fake cg2all output = bb file with one extra CB per residue, then merge
    cg = tmp_path / "cg" / "x_A"
    cg.mkdir(parents=True)
    out = []
    for ln in bb:
        out.append(ln)
        if ln[12:16].strip() == "CA":
            out.append(ln[:12] + " CB " + ln[16:])
    open(cg / "d00.pdb", "w").write("\n".join(out) + "\nEND\n")
    run("--stage", "merge", "--inputs-dir", str(tmp_path / "in"), "--out-dir", str(tmp_path / "m"), "--cg-dir", str(tmp_path / "cg"),
        "--chains", "x_A", "--draws", "0")
    m = [ln for ln in open(tmp_path / "m" / "x_A" / "d00.pdb") if ln.startswith("ATOM")]
    g = [ln for ln in open(src / "d00.pdb") if ln.startswith("ATOM")]
    surv = [ln for ln in m if orig[int(ln[22:26]) - 1] >= 0]
    gsurv = [ln for ln in g if orig[int(ln[22:26]) - 1] >= 0]
    assert surv == gsurv                                            # survivors untouched, native atoms incl. CB
    assert any(ln[12:16].strip() == "CB" for ln in m if orig[int(ln[22:26]) - 1] < 0)

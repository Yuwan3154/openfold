"""localfold completion PDBs (lf_complete_export.py runs) -> af2fix-format npz (af2_complete.py output layout: pred_atom37 (L,37,3), plddt (L,) in 0-1, ptm, seq, orig_idx, rm),
so af2_to_inputs.py consumes them unchanged. Run: python lf_to_af2fix.py CMP_DIR TAG OUT_DIR [LF_OUT_SUBDIR=lf_out]"""
import json
import os
import sys

import numpy as np

import indel_pool as ip

cmp_dir, tag, out = sys.argv[1:4]
lf_dir = sys.argv[4] if len(sys.argv) > 4 else "lf_out"
os.makedirs(out, exist_ok=True)
names = ip.ATOM37.split()
for m in json.load(open(os.path.join(cmp_dir, f"exp_{tag}", "manifest.json"))):
    L = len(m["seq"])
    pos, pl = np.zeros((L, 37, 3), np.float32), np.zeros(L, np.float32)
    seen = np.zeros(L, bool)
    for ln in open(os.path.join(cmp_dir, lf_dir, m["name"] + ".pdb")):
        if ln.startswith("ATOM"):
            i = int(ln[22:26]) - 1
            a = ln[12:16].strip()
            if a in names:
                pos[i, names.index(a)] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
            pl[i] = float(ln[60:66]) / 100
            seen[i] = True
    assert seen.all(), m["name"]
    ptm = json.load(open(os.path.join(cmp_dir, lf_dir, m["name"] + "_summary_confidences.json")))["ptm"]
    orig = np.array(m["orig_idx"])
    np.savez(os.path.join(out, f"{m['chain']}_d{m['draw']:02d}.npz"), pred_atom37=pos, plddt=pl, ptm=np.float32(ptm), seq=np.array(m["seq"]), orig_idx=orig, rm=orig < 0)
print(len(os.listdir(out)), "files ->", out)

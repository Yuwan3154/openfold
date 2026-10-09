"""localfold AF2 completion (the af2_complete.py design on localfold): template = survivor backbone (N,CA,C,O + ideal CB) of an arm input d<k>.pdb at its positions
(inserted positions absent), residue names = the query sequence (needed: localfold ignores a template whose names do not match the query), residue numbers = positions;
query = the PDB's residue names. Writes <out>/tpl/<tag>.pdb and run scripts split over 4 GPUs. Run:
python lf_complete_export.py INPUTS_DIR TAG OUT_DIR RUN_PREFIX REMOTE_DIR   (chains = every subdir of INPUTS_DIR, draws = every d??.pdb)
"""
import glob
import json
import os
import sys

import numpy as np

from lf_t3_export import write_tpl

THREE2ONE = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
             "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"}


def read_bb(path):
    atoms, names = {}, {}
    for ln in open(path):
        if not ln.startswith("ATOM"):
            continue
        r, a = int(ln[22:26]), ln[12:16].strip()
        a = "O" if a == "OT1" else a
        names[r] = THREE2ONE[ln[17:20].strip()]
        if a in ("N", "CA", "C", "O"):
            atoms.setdefault(r, {})[a] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
    res = sorted(names)
    assert res == list(range(1, len(res) + 1)) and all(len(atoms[r]) == 4 for r in res), path
    return "".join(names[r] for r in res), np.array([[atoms[r][a] for a in ("N", "CA", "C", "O")] for r in res])


def main():
    inp, tag, out, prefix, rdir = sys.argv[1:6]
    os.makedirs(os.path.join(out, "tpl"), exist_ok=True)
    jobs, man = [], []
    for c in sorted(d for d in os.listdir(inp) if os.path.isdir(os.path.join(inp, d))):
        plans = json.load(open(os.path.join(inp, c, "plans.json")))["plans"]
        for f in sorted(glob.glob(os.path.join(inp, c, "d??.pdb"))):
            k = int(os.path.basename(f)[1:3])
            seq, bb = read_bb(f)
            orig = np.array(plans[k]["orig_idx"])
            assert len(orig) == len(seq), (c, k)
            keep = orig >= 0
            n = f"{tag}_{c}_d{k:02d}"
            write_tpl(os.path.join(out, "tpl", n + ".pdb"), bb[keep], "".join(s for s, m in zip(seq, keep) if m), resnums=np.flatnonzero(keep) + 1)
            jobs.append(f"$LF --sequence={seq} --model=model_1_ptm --recycles=3 --template={rdir}/tpl/{n}.pdb:A --out={rdir}/out/{n}.pdb 2>&1 | grep '^mean' | sed 's/^/{n} /'")
            man.append(dict(name=n, chain=c, draw=k, seq=seq, orig_idx=orig.tolist()))
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"))
    for g in range(4):
        with open(f"{prefix}{g}.sh", "w") as o:
            o.write("#!/bin/bash\nexport PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_VISIBLE_DEVICES=%d\nLF=\"$HOME/localfold_opt/cuda/af2/localfold-af2 --weights-dir=$HOME/lf_weights\"\nmkdir -p %s/out\n" % (g, rdir))
            o.write("\n".join(jobs[g::4]) + "\necho GPU_DONE\n")
    print(tag, len(jobs), "folds")


if __name__ == "__main__":
    main()

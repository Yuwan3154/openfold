"""T2 variants isolating the template residue identity: localfold aligns the template to the query by sequence and (unlike ColabDesign rm_template_seq=True)
may use template residue types. Writes, per native chain, backbone-only (N,CA,C,O) templates whose residue names are native (nat), all-ALA (ala) or a seeded shuffle of the native (shuf);
and the run script (query = native sequence, model_1_ptm, 3 recycles). Needs env LF (the localfold command) and, when running, $G (GPU id); copy OUTDIR to REMOTE_DIR/tpl. Optional 5th arg 'cb' keeps the native CB.
Run: python lf_t2_seqdep.py NATIVE_DIR OUTDIR RUN.sh REMOTE_DIR [cb]
"""
import os
import sys

import numpy as np

from lf_parity import THREE2ONE, native_seq

CB = len(sys.argv) > 5 and sys.argv[5] == "cb"  # keep the native CB (side chain atoms beyond CB still dropped)
ONE2THREE = {v: k for k, v in THREE2ONE.items()}
native, outdir, runsh, rdir = sys.argv[1:5]
os.makedirs(outdir, exist_ok=True)
lines = ["#!/bin/bash", "export PATH=/usr/local/cuda-12.6/bin:$PATH", f"mkdir -p {rdir}/out"]
for f in sorted(os.listdir(native)):
    c = f[:-4]
    seq = native_seq(os.path.join(native, f))
    rng = np.random.default_rng(7)
    names = {"nat": seq, "ala": "A" * len(seq), "shuf": "".join(rng.permutation(list(seq)))}
    atoms = [l for l in open(os.path.join(native, f)) if l.startswith("ATOM") and (l[12:16].strip() in ("N", "CA", "C", "O") or (l[12:16].strip() == "CB" and CB))]
    resid = sorted({int(l[22:26]) for l in atoms})
    assert len(resid) == len(seq), c
    for k, s in names.items():
        m = {r: ONE2THREE[s[i]] for i, r in enumerate(resid)}
        with open(os.path.join(outdir, f"{c}_{k}.pdb"), "w") as o:
            o.writelines(l[:17] + f"{m[int(l[22:26])]:>3s}" + l[20:] for l in atoms)
            o.write("END\n")
        lines.append(f"CUDA_VISIBLE_DEVICES=$G {os.environ['LF']} --sequence={seq} --model=model_1_ptm --recycles=3 --template={rdir}/tpl/{c}_{k}.pdb:A --out={rdir}/out/{c}_{k}.pdb 2>&1 | grep '^mean' | sed 's/^/{c}_{k} /'")
open(runsh, "w").write("\n".join(lines) + "\n")

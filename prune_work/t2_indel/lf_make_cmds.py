"""Write the localfold run script for the parity tests (generated artifact, copied to the A6000 node): T1 = template-free single sequence (model_1_ptm),
T2 = native structure as the template + native sequence as the query. Sequences are read from the native PDBs. Run: python lf_make_cmds.py NATIVE_DIR OUT.sh LF_BIN REMOTE_NATIVE REMOTE_OUT"""
import os
import sys

from lf_parity import native_seq

native, out, lf, rnative, rout = sys.argv[1:6]
lines = ["#!/bin/bash", "set -u", f"mkdir -p {rout}/t1 {rout}/t2", "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"]
for f in sorted(os.listdir(native)):
    c = f[:-4]
    seq = native_seq(os.path.join(native, f))
    for test, extra in (("t1", ""), ("t2", f" --template={rnative}/{c}.pdb:A")):
        lines.append(f'echo "== {test} {c} L={len(seq)}"; {lf} --sequence={seq} --model=model_1_ptm --recycles=3{extra} --out={rout}/{test}/{c}.pdb 2>&1 | tail -n 3')
open(out, "w").write("\n".join(lines) + "\n")
print(f"wrote {out} ({len(lines) - 4} commands)")

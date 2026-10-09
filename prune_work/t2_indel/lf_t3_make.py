"""Run script for the T3 re-run in a given localfold configuration: every af2compat template x query (input sequence + 4 MPNN designs), template file = the q-named one
already on the node (~/lf_t3/tpl/<set>_<chain>_t<ttt>_k<k>_q.pdb), 4 GPU shards. Run: python lf_t3_make.py T3_DIR OUT_PREFIX REMOTE_OUT LF_CMD [ENV...]"""
import glob
import os
import sys

import numpy as np

t3, prefix, rout, lf = sys.argv[1:5]
env = sys.argv[5:]
jobs = []
for d in sorted(glob.glob(os.path.join(t3, "af2compat_*/"))):
    s = os.path.basename(d.rstrip("/"))[len("af2compat_"):]
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        chain, ti = os.path.basename(f)[:-4].rsplit("_t", 1)
        for k, q in enumerate(np.load(f)["seqs"]):
            n = f"{s}_{chain}_t{ti}_k{k}"
            jobs.append(f"{lf} --sequence={q} --recycles=3 --template=$HOME/lf_t3/tpl/{n}_q.pdb:A --out={rout}/{n}.pdb 2>&1 | grep '^mean' | sed 's/^/{n} /'")
for g in range(4):
    with open(f"{prefix}{g}.sh", "w") as o:
        o.write("#!/bin/bash\nexport PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_VISIBLE_DEVICES=%d\n" % g + "".join(f"export {e}\n" for e in env) + f"mkdir -p {rout}\n")
        o.write("\n".join(jobs[g::4]) + "\necho GPU_DONE\n")
print(len(jobs), "folds")

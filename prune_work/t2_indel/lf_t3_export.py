"""T3 export for localfold: for every af2compat template (set/chain/t) write the pool backbone as a PDB (N,CA,C,O + ideal CB from N,CA,C [AF2 formula], Gly none)
with residue names = the template's INPUT sequence ('own') or = the query sequence ('q', one template per query), and a run script folding each query (seq_kind 0 = input
sequence, 1..4 = first MPNN designs, as af2_compat.py) with localfold model_1_ptm, 3 recycles, split over 4 GPUs. Queries are taken from the ColabDesign npz so both
implementations fold the identical sequences. Run: python lf_t3_export.py T3_DIR OUT_DIR RUN_PREFIX REMOTE_DIR
"""
import glob
import os
import sys

import numpy as np

import indel_pool as ip

POOL = {"control": "ctl", "esmc07": "esmc"}
THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET",
         "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def write_tpl(path, bb, names, resnums=None):
    """bb (L,4,3) N,CA,C,O in atom37 order [0,1,2,4]."""
    n, ca, c = bb[:, 0], bb[:, 1], bb[:, 2]
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca
    lines, k = [], 0
    resnums = np.arange(1, len(names) + 1) if resnums is None else resnums
    for i, nm in enumerate(names, start=1):
        for an, x in (("N", n[i - 1]), ("CA", ca[i - 1]), ("C", c[i - 1]), ("O", bb[i - 1, 3]), ("CB", cb[i - 1])):
            if an == "CB" and nm == "G":
                continue
            k += 1
            lines.append(f"ATOM  {k:5d} {an:<4s} {THREE[nm]:>3s} A{int(resnums[i - 1]):4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00          {an[0]:>2s}")
    open(path, "w").write("\n".join(lines) + "\nEND\n")


def main():
    t3, out, prefix, rdir = sys.argv[1:5]
    os.makedirs(os.path.join(out, "tpl"), exist_ok=True)
    jobs = []
    for d in sorted(glob.glob(os.path.join(t3, "af2compat_*/"))):
        s = os.path.basename(d.rstrip("/"))[len("af2compat_"):]
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            chain, ti = os.path.basename(f)[:-4].rsplit("_t", 1)
            z = np.load(f)
            t = ip.read_template(ip.shard_path(os.path.join(t3, "pool_" + POOL.get(s, s)), chain), int(ti))
            assert t["atom_mask"][:, [0, 1, 2, 4]].all(), (s, chain, ti)
            bb = ip.atom37_coords(t)[:, [0, 1, 2, 4]].astype(np.float64)
            own = "".join(ip.AA_ORDER[int(x)] for x in t["aatype"])
            assert str(z["seqs"][0]) == own and len(own) == len(bb), (s, chain, ti)
            tag = f"{s}_{chain}_t{ti}"
            write_tpl(os.path.join(out, "tpl", tag + "_own.pdb"), bb, own)
            for k, q in enumerate(z["seqs"]):
                q = str(q)
                write_tpl(os.path.join(out, "tpl", f"{tag}_k{k}_q.pdb"), bb, q)
                for mode, tp in (("own", f"{tag}_own"), ("q", f"{tag}_k{k}_q")):
                    jobs.append(f"{ '$LF' } --sequence={q} --model=model_1_ptm --recycles=3 --template={rdir}/tpl/{tp}.pdb:A --out={rdir}/out/{tag}_k{k}_{mode}.pdb 2>&1 | grep '^mean' | sed 's/^/{tag}_k{k}_{mode} /'")
    for g in range(4):
        with open(f"{prefix}{g}.sh", "w") as o:
            o.write("#!/bin/bash\nexport PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_VISIBLE_DEVICES=%d\nLF=\"$HOME/localfold_opt/cuda/af2/localfold-af2 --weights-dir=$HOME/lf_weights\"\nmkdir -p %s/out\n" % (g, rdir))
            o.write("\n".join(jobs[g::4]) + "\necho GPU_DONE\n")
    print(len(jobs), "folds")


if __name__ == "__main__":
    main()

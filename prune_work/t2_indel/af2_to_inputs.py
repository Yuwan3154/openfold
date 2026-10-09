"""AF2-completed structures -> partial-diffusion INPUT dirs (user 10-08: use the AF2/AF3-filled structure as the partial diffusion STARTING point).

For each arm: read <af2-root>/af2fix_<arm>/<chain>_d<kk>.npz (af2_complete.py: pred_atom37 (L',37,3), seq, orig_idx) and write
<out-dir>/<chain>/d<kk>.pdb with ALL atoms of the query residue types (AF2 slots that exist for that type; chain A, residues 1..L'), plus plans.json and native.pdb copied
from the arm's source inputs dir (same orig_idx / L_new, so run_indel_pd / score_indel / build_indel_pool consume it unchanged).
Env: protpardelle (residue_constants). Run: python af2_to_inputs.py --af2-dir D/af2fix_gly --src-inputs D/inputs --out-dir D/inputs_af2_gly --chains .. --draws ..
"""
import argparse
import json
import os
import shutil

import numpy as np

import indel_pool as ip
from protpardelle.common import residue_constants as rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--af2-dir", required=True)
    p.add_argument("--src-inputs", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--draws", type=int, nargs="+", required=True)
    a = p.parse_args()
    names = ip.ATOM37.split()
    n_files = 0
    for c in a.chains:
        dst = os.path.join(a.out_dir, c)
        os.makedirs(dst, exist_ok=True)
        for f in ("native.pdb", "plans.json"):
            shutil.copy(os.path.join(a.src_inputs, c, f), os.path.join(dst, f))
        plans = json.load(open(os.path.join(dst, "plans.json")))["plans"]
        for k in a.draws:
            z = np.load(os.path.join(a.af2_dir, f"{c}_d{k:02d}.npz"))
            seq, xyz = str(z["seq"]), z["pred_atom37"]
            assert np.isfinite(xyz).all() and len(seq) == xyz.shape[0] == plans[k]["L_new"], (c, k)
            assert np.array_equal(z["orig_idx"], plans[k]["orig_idx"]), (c, k, "orig_idx differs from the source plans")
            lines, n = [], 0
            for i, aa in enumerate(seq, start=1):
                three = rc.restype_1to3[aa]
                have = set(rc.residue_atoms[three])
                for j, nm in enumerate(names):
                    if nm in have:
                        n += 1
                        x = xyz[i - 1, j]
                        lines.append(f"ATOM  {n:5d} {nm:<4s} {three:>3s} A{i:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00           {nm[0]:>2s}")
            lines.append("END")
            open(os.path.join(dst, f"d{k:02d}.pdb"), "w").write("\n".join(lines) + "\n")
            n_files += 1
    print(f"wrote {n_files} PDBs -> {a.out_dir}")
    assert n_files == len(a.chains) * len(a.draws)


if __name__ == "__main__":
    main()

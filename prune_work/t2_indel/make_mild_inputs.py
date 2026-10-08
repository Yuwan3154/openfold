"""Gly-arm inputs with a MILD edit-size range (user 10-08 'test where refolding breaks'; fractions 1-5 % of L = my proposal, provisional).

Same sampler as make_indel_inputs.py (draw_plan: k 1-5 segments, uniform splits/placements, seed [global_seed, crc32(chain), draw]) but the
insertion and deletion fractions are drawn from U(frac_lo, frac_hi) instead of U(0.10, 0.30). Native structure = <inputs-dir>/<chain>/native.pdb;
survivors keep their native atoms, inserted residues are Gly backbone (rigid terminal rule). Output = same layout as make_indel_inputs.py
(native.pdb, plans.json, d<k>.pdb), numpy only. Run: python make_mild_inputs.py --inputs-dir D --out-dir D2 --chains .. --n-draws 8 --frac-lo 0.01 --frac-hi 0.05
"""
import argparse
import json
import os
import shutil

import numpy as np

from indel_edit import edit
from regen_gly_inputs import BB, atom_line, backbone, native_residues
from sample_indels import draw_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--n-draws", type=int, required=True)
    p.add_argument("--frac-lo", type=float, required=True)
    p.add_argument("--frac-hi", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    for c in a.chains:
        src, dst = os.path.join(a.inputs_dir, c), os.path.join(a.out_dir, c)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(src, "native.pdb"), os.path.join(dst, "native.pdb"))
        res = native_residues(os.path.join(src, "native.pdb"))
        nat = backbone(res)
        L = len(nat)
        names = json.load(open(os.path.join(src, "plans.json")))["native_resnames"]
        assert len(names) == L
        plans = []
        for k in range(a.n_draws):
            rec = draw_plan(L, c, k, a.seed, a.frac_lo, a.frac_hi)
            assert rec["ins"]["T"] >= 1 and rec["del"]["T"] >= 1, (c, k, "edit fraction rounds to zero residues for this chain length")
            new, orig, n2n = edit(nat, [tuple(o) for o in rec["ops"]])
            lines, n = [], 0
            for j, o in enumerate(orig, start=1):
                if o >= 0:
                    for ln in res[int(o) + 1]:
                        n += 1
                        lines.append(f"ATOM  {n:5d}{ln[11:22]}{j:4d}{ln[26:]}")
                else:
                    for b, x in zip(BB, new[j - 1]):
                        n += 1
                        lines.append(atom_line(n, b, "GLY", j, x))
            lines.append("END")
            open(os.path.join(dst, f"d{k:02d}.pdb"), "w").write("\n".join(lines) + "\n")
            rec["orig_idx"], rec["native_to_new"], rec["L_new"] = orig.tolist(), n2n.tolist(), len(orig)
            plans.append(rec)
        json.dump({"key": c, "L": L, "native_resnames": names, "plans": plans, "frac_range": [a.frac_lo, a.frac_hi]},
                  open(os.path.join(dst, "plans.json"), "w"))
        print(f"{c}: L={L} L_new range {min(p['L_new'] for p in plans)}-{max(p['L_new'] for p in plans)}", flush=True)


if __name__ == "__main__":
    main()

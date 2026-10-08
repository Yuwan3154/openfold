"""Regenerate the Gly-arm inputs under the rigid terminal rule (24f5c50) for the draws that change, copy the rest (user 10-07: regenerate only the affected draws).

Per chain: native.pdb and plans.json are copied; d<k>.pdb is REBUILT when the plan has a terminal insertion (has_term, same definition as
verify_regen_scope.py) and copied otherwise. Rebuilt file = survivors' native atom lines (renumbered) + inserted GLY backbone (N,CA,C,O) from
indel_edit.edit on the native.pdb backbone, written exactly like make_indel_inputs.write_pdb; orig_idx is asserted equal to plans.json. Env: numpy only. Run: python regen_gly_inputs.py --inputs-dir D --out-dir D2 --chains ...
"""
import argparse
import json
import os
import shutil

import numpy as np

from indel_edit import deleted_mask, edit, insertion_locations

BB = ("N", "CA", "C", "O")


def native_residues(path):
    res = {}
    for ln in open(path):
        if ln.startswith("ATOM"):
            res.setdefault(int(ln[22:26]), []).append(ln.rstrip("\n"))
    return res


def backbone(res):
    return np.array([[[float(l[30:38]), float(l[38:46]), float(l[46:54])] for b in BB for l in res[i] if l[12:16].strip() == b]
                     for i in sorted(res)])


def atom_line(n, name, resname, i, x):
    return f"ATOM  {n:5d} {name:<4s} {resname:>3s} A{i:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00           {name[0]:>2s}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chains", nargs="+", required=True)
    a = p.parse_args()
    n_rebuilt = n_copied = 0
    for c in a.chains:
        src, dst = os.path.join(a.inputs_dir, c), os.path.join(a.out_dir, c)
        os.makedirs(dst, exist_ok=True)
        for f in ("native.pdb", "plans.json"):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        res = native_residues(os.path.join(src, "native.pdb"))
        nat = backbone(res)
        plans = json.load(open(os.path.join(src, "plans.json")))["plans"]
        for k, pl in enumerate(plans):
            ops = [tuple(o) for o in pl["ops"]]
            L = len(nat)
            locs = insertion_locations(L, ops)
            has_term = 0 in locs or int((~deleted_mask(L, ops)).sum()) in locs
            out_f = os.path.join(dst, f"d{k:02d}.pdb")
            if not has_term:
                shutil.copy(os.path.join(src, f"d{k:02d}.pdb"), out_f)
                n_copied += 1
                continue
            new, orig, _ = edit(nat, ops)
            assert orig.tolist() == pl["orig_idx"], (c, k)
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
            open(out_f, "w").write("\n".join(lines) + "\n")
            n_rebuilt += 1
    print(f"rebuilt {n_rebuilt}, copied {n_copied}")
    assert n_rebuilt + n_copied == 64 * len(a.chains)


if __name__ == "__main__":
    main()

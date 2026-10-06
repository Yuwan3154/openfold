"""Composition-matched real-type arm (hypothesis test, user 10-06: is the coil bias the Gly sequence?).

Same edits as the Gly arm (plans.json from --inputs-dir). Inserted residues get types drawn i.i.d. from the chain's OWN
native residue composition (seed = crc32(chain) + draw) instead of Gly; nothing else changes.
stage 'bb'   : write backbone-only PDBs (real inserted types, survivors' native types) -> to be expanded by cg2all
stage 'merge': survivors keep their NATIVE atoms (from the Gly-arm input d<k>.pdb); inserted residues take the atoms
               cg2all built; write the final input d<k>.pdb. plans.json/native.pdb copied; plans gain 'comp_seq'.
Env: raygun or protpardelle (numpy only). Run: python make_comp_inputs.py --stage bb|merge --inputs-dir .. --out-dir .. [--cg-dir ..]
"""
import argparse
import json
import os
import shutil
import zlib

import numpy as np

from make_raygun_inputs import ONE2THREE

THREE2ONE = {v: k for k, v in ONE2THREE.items()}


def atom_lines_by_res(path):
    out = {}
    for ln in open(path):
        if ln.startswith("ATOM"):
            out.setdefault(int(ln[22:26]), []).append(ln.rstrip("\n"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["bb", "merge"])
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cg-dir", default=None)
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--draws", type=int, nargs="+", required=True)
    a = p.parse_args()
    for key in a.chains:
        src, dst = os.path.join(a.inputs_dir, key), os.path.join(a.out_dir, key)
        os.makedirs(dst, exist_ok=True)
        plans = json.load(open(os.path.join(src, "plans.json")))
        nat = [THREE2ONE[r] for r in plans["native_resnames"]]
        for k in a.draws:
            orig = np.array(plans["plans"][k]["orig_idx"])
            rng = np.random.default_rng([zlib.crc32(key.encode()), k, 77])
            ins_types = rng.choice(nat, size=int((orig < 0).sum()))
            seq, it = [], iter(ins_types)
            for o in orig:
                seq.append(nat[o] if o >= 0 else next(it))
            plans["plans"][k]["comp_seq"] = "".join(seq)
            gly = atom_lines_by_res(os.path.join(src, f"d{k:02d}.pdb"))
            if a.stage == "bb":
                lines = []
                for j, c in enumerate(seq, start=1):
                    for ln in gly[j]:
                        if ln[12:16].strip() in ("N", "CA", "C", "O"):
                            lines.append(ln[:17] + f"{ONE2THREE[c]:>3s}" + ln[20:])
                lines.append("END")
                open(os.path.join(dst, f"d{k:02d}.pdb"), "w").write("\n".join(lines) + "\n")
            else:
                cg = atom_lines_by_res(os.path.join(a.cg_dir, key, f"d{k:02d}.pdb"))
                lines = []
                for j, o in enumerate(orig, start=1):
                    src_lines = gly[j] if orig[j - 1] >= 0 else cg[j]
                    lines += [ln[:17] + f"{ONE2THREE[seq[j - 1]]:>3s}" + ln[20:] if orig[j - 1] < 0 else ln for ln in src_lines]
                lines.append("END")
                open(os.path.join(dst, f"d{k:02d}.pdb"), "w").write("\n".join(lines) + "\n")
        json.dump(plans, open(os.path.join(dst, "plans.json"), "w"))
        shutil.copy(os.path.join(src, "native.pdb"), os.path.join(dst, "native.pdb"))
        print(f"{key}: {len(a.draws)} {a.stage}", flush=True)


if __name__ == "__main__":
    main()

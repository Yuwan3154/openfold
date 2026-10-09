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
    p.add_argument("--fills", default=None, help="fills.json of esmc_fill_pilot.py: take the inserted types from it instead of the composition draw")
    p.add_argument("--fill-arm", default=None, help="arm name inside --fills (e.g. esmc_T0.7_p1.0)")
    p.add_argument("--seqs", default=None, help="esmc_fill_mut.py JSON: take the FULL new-frame sequence (inserted + mutated types) from it; mutated survivors (plans mut.new_idx) are rebuilt by cg2all")
    p.add_argument("--seq-arm", default=None, help="arm name inside --seqs (esmc_300m | esmc_600m)")
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--draws", type=int, nargs="+", required=True)
    a = p.parse_args()
    fill_of = {}
    if a.fills:
        assert a.fill_arm, "--fills needs --fill-arm"
        fill_of = {(r["chain"], r["draw"]): r["fill"] for r in json.load(open(a.fills)) if r["arm"] == a.fill_arm}
        assert fill_of, f"arm {a.fill_arm} not in {a.fills}"
    seq_of = {}
    if a.seqs:
        assert a.seq_arm and not a.fills, "--seqs needs --seq-arm and excludes --fills"
        seq_of = {(r["chain"], r["draw"]): r["seq"] for r in json.load(open(a.seqs)) if r["arm"] == a.seq_arm}
        assert seq_of, f"arm {a.seq_arm} not in {a.seqs}"
    for key in a.chains:
        src, dst = os.path.join(a.inputs_dir, key), os.path.join(a.out_dir, key)
        os.makedirs(dst, exist_ok=True)
        plans = json.load(open(os.path.join(src, "plans.json")))
        nat = [THREE2ONE[r] for r in plans["native_resnames"]]
        for k in a.draws:
            orig = np.array(plans["plans"][k]["orig_idx"])
            rng = np.random.default_rng([zlib.crc32(key.encode()), k, 77])
            ins_types = rng.choice(nat, size=int((orig < 0).sum()))
            if a.fills:
                ins_types = list(fill_of[(key, k)])
                assert len(ins_types) == int((orig < 0).sum()), (key, k, len(ins_types))
            seq, it = [], iter(ins_types)
            for o in orig:
                seq.append(nat[o] if o >= 0 else next(it))
            mut_idx = set()
            if a.seqs:
                seq = list(seq_of[(key, k)])
                assert len(seq) == len(orig), (key, k, len(seq), len(orig))
                mut_idx = set(plans["plans"][k]["mut"]["new_idx"])
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
                    native_kept = orig[j - 1] >= 0 and (j - 1) not in mut_idx
                    src_lines = gly[j] if native_kept else cg[j]
                    lines += [ln if native_kept else ln[:17] + f"{ONE2THREE[seq[j - 1]]:>3s}" + ln[20:] for ln in src_lines]
                lines.append("END")
                open(os.path.join(dst, f"d{k:02d}.pdb"), "w").write("\n".join(lines) + "\n")
        json.dump(plans, open(os.path.join(dst, "plans.json"), "w"))
        shutil.copy(os.path.join(src, "native.pdb"), os.path.join(dst, "native.pdb"))
        print(f"{key}: {len(a.draws)} {a.stage}", flush=True)


if __name__ == "__main__":
    main()

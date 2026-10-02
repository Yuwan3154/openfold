"""Raygun arm (Option 1) inputs: per chain/draw, align the Raygun sequence to the native, edit the native
BACKBONE accordingly (indel_edit), write d<k>.pdb with Raygun's residue types and BACKBONE-ONLY atoms (native
sidechains do not fit the new types; the model dummy-fills them), plans.json in the Gly arm's format so
run_indel_pd.py / score_indel.py work unchanged. native.pdb is copied from the Gly arm's inputs.
Env: raygun (numpy + Biopython). Run: python make_raygun_inputs.py --inputs-dir <gly inputs> --seqs raygun_seqs.json --out-dir <dir>
"""
import argparse
import json
import os
import shutil

import numpy as np

from indel_edit import edit
from make_indel_inputs import BB4_SLOT, write_pdb
from raygun_edit import aligner, alignment_pairs, ops_from_pairs

ONE2THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
             "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
             "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def native_backbone(pdb):
    rows = {}
    for ln in open(pdb):
        if ln.startswith("ATOM") and ln[12:16].strip() in ("N", "CA", "C", "O"):
            rows.setdefault(int(ln[22:26]), {})[ln[12:16].strip()] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
    return np.array([[r["N"], r["CA"], r["C"], r["O"]] for _, r in sorted(rows.items())])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--seqs", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    seqs = json.load(open(a.seqs))
    al = aligner()
    for key in sorted(seqs):
        src = os.path.join(a.inputs_dir, key)
        dst = os.path.join(a.out_dir, key)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(src, "native.pdb"), os.path.join(dst, "native.pdb"))
        bb = native_backbone(os.path.join(src, "native.pdb"))
        native_seq = seqs[key]["native_seq"]
        assert len(bb) == len(native_seq)
        gly = json.load(open(os.path.join(src, "plans.json")))
        plans = []
        for k, gen in enumerate(seqs[key]["seqs"]):
            pairs = alignment_pairs(native_seq, gen, al)
            ops = ops_from_pairs(len(native_seq), len(gen), pairs)
            new_bb, orig, n2n = edit(bb, ops)
            assert len(orig) == len(gen)
            xyz = np.zeros((len(gen), 37, 3))
            mask = np.zeros((len(gen), 37), bool)
            xyz[:, BB4_SLOT] = new_bb
            mask[:, BB4_SLOT] = True
            write_pdb(os.path.join(dst, f"d{k:02d}.pdb"), [ONE2THREE[c] for c in gen], xyz, mask)
            n_mut = sum(native_seq[i] != gen[j] for i, j in pairs)
            plans.append({"draw": k, "L": len(native_seq), "L_new": len(gen), "ops": [list(o) for o in ops],
                          "orig_idx": orig.tolist(), "native_to_new": n2n.tolist(), "raygun_seq": gen,
                          "noise": seqs[key]["noise"], "n_aligned": len(pairs), "n_mut": int(n_mut),
                          "n_ins": int((orig < 0).sum()), "n_del": int((n2n < 0).sum())})
        json.dump({"key": key, "L": len(native_seq), "native_resnames": gly["native_resnames"], "plans": plans},
                  open(os.path.join(dst, "plans.json"), "w"))
        print(f"{key}: {len(plans)} inputs, mean n_mut {np.mean([x['n_mut'] for x in plans]):.1f} "
              f"ins {np.mean([x['n_ins'] for x in plans]):.1f} del {np.mean([x['n_del'] for x in plans]):.1f}", flush=True)


if __name__ == "__main__":
    main()

"""Raygun-arm inputs THROUGH Protpardelle's real PDB reader: length == L_new, contiguous residue_index,
aatype == the Raygun sequence, survivor CA coordinates == the native's. Env: protpardelle.
Run: python verify_raygun_inputs.py --inputs-dir <inputs_raygun>
"""
import argparse
import json
import os

import numpy as np

from protpardelle.common import residue_constants as rc
from protpardelle.data.pdb_io import load_feats_from_pdb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--ca-tol", type=float, default=2e-3, help="max survivor CA deviation from the native (A)")
    p.add_argument("--backbone-only", action="store_true")
    a = p.parse_args()
    n = n_res = 0
    max_dev = 0.0
    for key in sorted(os.listdir(a.inputs_dir)):
        d = os.path.join(a.inputs_dir, key)
        plans = json.load(open(os.path.join(d, "plans.json")))
        nat, _ = load_feats_from_pdb(os.path.join(d, "native.pdb"), include_pos_feats=True)
        ca_nat = nat["atom_positions"].numpy()[:, 1]
        for rec in plans["plans"]:
            f, _ = load_feats_from_pdb(os.path.join(d, f"d{rec['draw']:02d}.pdb"), include_pos_feats=True)
            Lp = rec["L_new"]
            assert f["aatype"].shape[0] == Lp
            assert np.array_equal(f["residue_index"].numpy(), np.arange(1, Lp + 1))
            assert np.array_equal(f["aatype"].numpy(), [rc.restype_order[c] for c in rec["raygun_seq"]])
            orig = np.array(rec["orig_idx"])
            ca = f["atom_positions"].numpy()[:, 1]
            dev = np.abs(ca[orig >= 0] - ca_nat[orig[orig >= 0]]).max()
            max_dev = max(max_dev, dev)
            assert dev <= a.ca_tol, (key, rec["draw"], dev)
            assert f["atom_mask"].numpy().sum() > 4 * Lp or a.backbone_only, "expected sidechain atoms"
            n += 1
            n_res += Lp
    print(f"verified {n} Raygun-arm inputs through the real reader ({n_res} residues); max survivor CA deviation {max_dev:.4f} A")


if __name__ == "__main__":
    main()

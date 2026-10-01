"""Check every edited input THROUGH Protpardelle's real PDB reader against its plans.json sidecar.
Env: protpardelle. Run: python verify_inputs.py --inputs-dir inputs
Asserts per file: reader length == L_new; contiguous residue_index 1..L_new; aatype of inserted residues == GLY
and of survivors == the native residue type; CA coordinates of survivors equal the native's; counts printed."""
import argparse
import json
import os

import numpy as np

from protpardelle.common import residue_constants as rc
from protpardelle.data.pdb_io import load_feats_from_pdb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    a = p.parse_args()
    n_files = n_ins = n_surv = 0
    for key in sorted(os.listdir(a.inputs_dir)):
        d = os.path.join(a.inputs_dir, key)
        plans = json.load(open(os.path.join(d, "plans.json")))
        nat, _ = load_feats_from_pdb(os.path.join(d, "native.pdb"), include_pos_feats=True)
        L = plans["L"]
        assert nat["aatype"].shape[0] == L
        nat_aa = nat["aatype"].numpy()
        assert np.array_equal(nat["residue_index"].numpy(), np.arange(1, L + 1))
        gly = rc.restype_order["G"]
        for k, rec in enumerate(plans["plans"]):
            f, _ = load_feats_from_pdb(os.path.join(d, f"d{k:02d}.pdb"), include_pos_feats=True)
            Lp = rec["L_new"]
            assert f["aatype"].shape[0] == Lp, (key, k, f["aatype"].shape[0], Lp)
            assert np.array_equal(f["residue_index"].numpy(), np.arange(1, Lp + 1))
            orig = np.array(rec["orig_idx"])
            aa = f["aatype"].numpy()
            assert (aa[orig < 0] == gly).all()
            assert np.array_equal(aa[orig >= 0], nat_aa[orig[orig >= 0]])
            ca_new = f["atom_positions"].numpy()[:, 1] if "atom_positions" in f else None
            if ca_new is not None:
                ca_nat = nat["atom_positions"].numpy()[:, 1]
                assert np.allclose(ca_new[orig >= 0], ca_nat[orig[orig >= 0]], atol=2e-3)
            n_files += 1
            n_ins += int((orig < 0).sum())
            n_surv += int((orig >= 0).sum())
    print(f"verified {n_files} edited inputs through the real reader; {n_surv} survivor + {n_ins} inserted residues")


if __name__ == "__main__":
    main()

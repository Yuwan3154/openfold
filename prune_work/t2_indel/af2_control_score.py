"""Score the AF2 known-good control (af2_complete.py --control): CA RMSD (Kabsch, 1:1 residues) and USalign TM of the prediction to the native
(normalised by the native length), per chain, next to the earlier NO-TEMPLATE AF2 native refold of the same chains (refold_ctl_scores_af2.csv,
seq_kind 0, tm_paired) so the control separates 'the template channel works' from 'AF2 refolds these chains from sequence alone'.
Env: protebm. Run: python af2_control_score.py --inputs-dir D/inputs --ctl-dir D/af2fix_ctl --notemplate-csv D/refold_ctl_scores_af2.csv --out f.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from af2fix_analyze import kabsch_rmsd, tm_pair
from diagnose_indel import native_backbone


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--ctl-dir", required=True)
    p.add_argument("--notemplate-csv", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    nt = pd.read_csv(a.notemplate_csv)
    nt = nt[nt.seq_kind == 0].groupby("chain").tm_paired.mean()
    rows = []
    for f in sorted(os.listdir(a.ctl_dir)):
        if not f.endswith("_native.npz"):
            continue
        c = f[:-len("_native.npz")]
        z = np.load(os.path.join(a.ctl_dir, f))
        pred = z["pred_atom37"][:, 1].astype(np.float64)
        nat = native_backbone(os.path.join(a.inputs_dir, c, "native.pdb"))[:, 1].astype(np.float64)
        assert len(pred) == len(nat), (c, len(pred), len(nat))
        _, tm_nat = tm_pair((pred, nat))
        rows.append(dict(chain=c, L=len(nat), ca_rmsd=kabsch_rmsd(pred, nat), tm_native_template_ctl=tm_nat,
                         plddt=float(z["plddt"].mean()), tm_native_notemplate_af2=float(nt.get(c, np.nan))))
    df = pd.DataFrame(rows)
    assert len(df) > 0
    df.to_csv(a.out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"median CA RMSD {df.ca_rmsd.median():.2f} A | TM with template {df.tm_native_template_ctl.median():.3f} | no-template AF2 {df.tm_native_notemplate_af2.median():.3f}")


if __name__ == "__main__":
    main()

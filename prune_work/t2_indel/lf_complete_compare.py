"""localfold vs ColabDesign AF2 completion on the same partial templates (af2_complete.py arms). Per draw: survivor-CA RMSD to the input structure (Kabsch over survivors),
mean pLDDT (all / inserted / survivors for localfold; all / inserted for ColabDesign), pTM, and for arms with a ColabDesign result the CA RMSD between the two predictions (all residues) and per-residue pLDDT r.
Run: python lf_complete_compare.py CMP_DIR OUT.csv [LF_OUT_SUBDIR=lf_out]   (CMP_DIR holds inputs_<arm>/, af2fix_<arm>/, exp_<tag>/manifest.json, lf_out/)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from lf_complete_export import read_bb
from lf_parity import read_ca, rmsd

cmp_dir, out = sys.argv[1:3]
lf_dir = sys.argv[3] if len(sys.argv) > 3 else "lf_out"
SRC = {"rgn": ("sub_inputs_rg3_full", "af2fix_raygun"), "compn": ("sub_inputs_compF", "af2fix_comp"), "esmc07": ("inputs_esmc_T0.7_p1.0", None)}
rows = []
for tag, (inp, cd) in SRC.items():
    for m in json.load(open(os.path.join(cmp_dir, f"exp_{tag}", "manifest.json"))):
        lf_ca, lf_pl = read_ca(os.path.join(cmp_dir, lf_dir, m["name"] + ".pdb"))
        seq, bb = read_bb(os.path.join(cmp_dir, inp, m["chain"], f"d{m['draw']:02d}.pdb"))
        keep = np.array(m["orig_idx"]) >= 0
        assert len(lf_ca) == len(seq)
        ptm = json.load(open(os.path.join(cmp_dir, lf_dir, m["name"] + "_summary_confidences.json")))["ptm"]
        r = dict(arm=tag, chain=m["chain"], draw=m["draw"], L=len(seq), n_ins=int((~keep).sum()), surv_rmsd_lf=rmsd(lf_ca[keep], bb[keep, 1]),
                 plddt_lf=lf_pl.mean(), plddt_ins_lf=lf_pl[~keep].mean() if (~keep).any() else np.nan, plddt_surv_lf=lf_pl[keep].mean(), ptm_lf=ptm)
        if cd:
            z = np.load(os.path.join(cmp_dir, cd, f"{m['chain']}_d{m['draw']:02d}.npz"))
            assert str(z["seq"]) == seq
            ca = z["pred_atom37"][:, 1].astype(np.float64)
            r.update(surv_rmsd_cd=rmsd(ca[keep], bb[keep, 1]), plddt_cd=100 * z["plddt"].mean(), plddt_ins_cd=100 * z["plddt"][~keep].mean() if (~keep).any() else np.nan,
                     ptm_cd=float(z["ptm"]), rmsd_lf_cd=rmsd(lf_ca, ca), r_plddt=np.corrcoef(lf_pl, 100 * z["plddt"])[0, 1])
        rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
pd.set_option("display.width", 250)
print(df.groupby("arm").agg(n=("L", "size"), ins=("n_ins", "mean"), surv_lf=("surv_rmsd_lf", "median"), surv_cd=("surv_rmsd_cd", "median"), pl_lf=("plddt_lf", "mean"),
                            pl_cd=("plddt_cd", "mean"), plins_lf=("plddt_ins_lf", "mean"), plins_cd=("plddt_ins_cd", "mean"), ptm_lf=("ptm_lf", "mean"), ptm_cd=("ptm_cd", "mean"),
                            lf_vs_cd=("rmsd_lf_cd", "median"), r=("r_plddt", "median")).round(2).to_string())

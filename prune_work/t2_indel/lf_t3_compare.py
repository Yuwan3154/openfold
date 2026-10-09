"""T3 parity: localfold (lf_t3_export.py runs) vs the ColabDesign fp32 af2compat results on the identical template + query. Per fold: pLDDT/pTM of both, CA RMSD of each
prediction to the template (Kabsch, 1:1), CA RMSD between the two predictions, per-residue pLDDT r. Aggregates by (query kind: input seq vs MPNN designs, template-name mode).
No pass/fail threshold. Run: python lf_t3_compare.py T3_DIR LF_OUT_DIR OUT.csv
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from lf_parity import kabsch, read_ca, rmsd

t3, lfout, out = sys.argv[1:4]
rows = []
for f in sorted(glob.glob(os.path.join(lfout, "*.pdb"))):
    name = os.path.basename(f)[:-4]
    head, mode = name.rsplit("_", 1)
    head, k = head.rsplit("_k", 1)
    s, rest = head.split("_", 1)
    chain, ti = rest.rsplit("_t", 1)
    z = np.load(os.path.join(t3, f"af2compat_{s}", f"{chain}_t{ti}.npz"))
    k = int(k)
    ref_ca = z["pred_atom37"][k][:, 1].astype(np.float64)
    ref_pl = z["plddt"][k] * 100
    tpl = z["template_ca"].astype(np.float64)
    lf_ca, lf_pl = read_ca(f)
    assert len(lf_ca) == len(ref_ca) == len(tpl)
    ptm_lf = float(json.load(open(f[:-4] + "_summary_confidences.json"))["ptm"])
    rows.append(dict(set=s, chain=chain, t=int(ti), k=k, kind="input" if k == 0 else "design", mode=mode, L=len(tpl),
                     plddt_lf=lf_pl.mean(), plddt_cd=ref_pl.mean(), ptm_lf=ptm_lf, ptm_cd=float(z["ptm"][k]),
                     rmsd_lf_tpl=rmsd(lf_ca, tpl), rmsd_cd_tpl=rmsd(ref_ca, tpl), rmsd_lf_cd=rmsd(lf_ca, ref_ca), r_plddt=np.corrcoef(lf_pl, ref_pl)[0, 1]))
df = pd.DataFrame(rows)
expected = 2 * sum(len(np.load(f)["seqs"]) for f in glob.glob(os.path.join(t3, "af2compat_*", "*.npz")))
assert len(df) == expected, (len(df), expected, "missing localfold folds")
df.to_csv(out, index=False)
print(len(df), "folds")
pd.set_option("display.width", 250)
g = df.groupby(["kind", "mode"]).agg(n=("L", "size"), plddt_lf=("plddt_lf", "mean"), plddt_cd=("plddt_cd", "mean"), ptm_lf=("ptm_lf", "mean"), ptm_cd=("ptm_cd", "mean"),
                                    rmsd_lf_tpl=("rmsd_lf_tpl", "median"), rmsd_cd_tpl=("rmsd_cd_tpl", "median"), rmsd_lf_cd=("rmsd_lf_cd", "median"), r_plddt=("r_plddt", "median"))
print(g.round(3).to_string())
print(df.groupby(["set", "kind", "mode"])[["plddt_lf", "plddt_cd", "ptm_lf", "ptm_cd", "rmsd_lf_tpl", "rmsd_cd_tpl"]].mean().round(2).to_string())
print("Spearman across folds (plddt_lf vs plddt_cd), by kind/mode:")
print(df.groupby(["kind", "mode"]).apply(lambda d: d.plddt_lf.corr(d.plddt_cd, method="spearman")).round(3).to_string())

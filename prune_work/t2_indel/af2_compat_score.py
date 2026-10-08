"""Score the AF2 compatibility outputs (af2_compat.py): per prediction pTM, mean pLDDT, CA RMSD (Kabsch, 1:1) and USalign TM of the AF2 prediction to the
TEMPLATE (normalised by the template length, i.e. does AF2 stay on the template), then a per-arm table (input sequence vs designs).
Env: protebm (pandas, USalign at ~/.local/bin/USalign). Run: python af2_compat_score.py --pool-root POOL --in-dir DIR --label LABEL --out f.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import indel_pool as ip
from af2fix_analyze import kabsch_rmsd, tm_pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-root", required=True)
    p.add_argument("--in-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for f in sorted(x for x in glob.glob(os.path.join(a.in_dir, "*.npz")) if ".tmp" not in os.path.basename(x)):
        chain, ti = os.path.basename(f)[:-4].rsplit("_t", 1)
        z = np.load(f)
        t = ip.read_template(ip.shard_path(a.pool_root, chain), int(ti))
        tca = z["template_ca"].astype(np.float64)
        for s in range(len(z["seqs"])):
            pca = z["pred_atom37"][s][:, 1].astype(np.float64)
            _, tm_t = tm_pair((pca, tca))  # structure 2 = the template: normalised by the template length
            rows.append(dict(label=a.label, chain=chain, i=int(ti), arm=t["arm"], L=len(tca), seq_kind=int(z["seq_kind"][s]),
                             ptm=float(z["ptm"][s]), plddt=float(z["plddt"][s].mean()), ca_rmsd=kabsch_rmsd(pca, tca), tm_to_template=tm_t))
    df = pd.DataFrame(rows)
    assert len(df) > 0
    df.to_csv(a.out, index=False)
    df["kind"] = np.where(df.seq_kind == 0, "input_seq", "designs")
    print(df.groupby(["label", "kind"]).agg(n=("ptm", "size"), templates=("i", "nunique"), ptm=("ptm", "mean"), plddt=("plddt", "mean"),
                                           rmsd=("ca_rmsd", "median"), tm=("tm_to_template", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()

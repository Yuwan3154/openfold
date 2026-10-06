"""Compare refinement variants against the baseline and against the NATIVE (grounded targets).

Inputs: diag csv dirs (diagnose_indel.py) + score csv dirs (score_indel.py) per variant tag, the panel stats
(native helix/strand fractions), and the chain/draw/rung subset. Targets come from the data itself:
  bonds   background C-N median and broken-link fractions of the variant vs the NATIVE's own bonds (crystal values
          computed from native.pdb) and vs the baseline;
  SSE     template helix / strand fraction vs the chain's native fractions (panel_stats), inserted-residue DSSP
          composition vs the native composition, survivor Q3 near (<=2) and far (>10) from edits.
Run: python eval_variants.py --diag-root <dir with <tag>/> --score-root <dir> --panel panel_stats.csv --tags ...
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def load(root, tag, pat):
    fs = sorted(glob.glob(os.path.join(root, f"{pat.format(tag=tag)}")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diag-root", required=True)
    p.add_argument("--score-root", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--baseline-diag", required=True, help="dir with cc89_<chain>.csv of the baseline sweep")
    p.add_argument("--baseline-score", required=True)
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--draws", type=int, default=8)
    a = p.parse_args()
    nat = pd.read_csv(a.panel).rename(columns={"chain": "chain_id", "helix_frac": "nat_H", "strand_frac": "nat_E"})
    nat["chain"] = nat.pdb + "_" + nat.chain_id
    nat = nat.set_index("chain")[["nat_H", "nat_E"]]
    rows = []
    for tag in ["baseline"] + a.tags:
        if tag == "baseline":
            dg = pd.concat([pd.read_csv(f) for c in a.chains for f in glob.glob(f"{a.baseline_diag}/cc89_{c}.csv")])
            dg = dg[dg.kind == "indel"]
            sc = pd.concat([pd.read_csv(f) for c in a.chains for f in glob.glob(f"{a.baseline_score}/cc89_cc91_{c}.csv")])
            sc = sc[(sc.model == "cc89") & (sc.kind == "indel")]
        else:
            dg = load(a.diag_root, tag, "{tag}_*.csv")  # diag files are per model: <tag>_<chain>.csv
            sc = load(a.score_root, "", "*.csv")  # files are named by the JOINED model list; filter on the model column
            sc = sc[sc.model == tag] if sc is not None else None
            if dg is None:
                continue
        dg = dg[(dg.draw < a.draws) & dg.rewind.isin([250, 300])]
        sc = sc[(sc.draw < a.draws) & sc.rewind.isin([250, 300])] if sc is not None else None
        dg = dg.join(nat, on="chain")
        dg["brk"] = dg.bg_caca_gt45 / dg.n_bg
        dg["anyseam"] = ((dg.seam_caca_gt45 > 0) | (dg.seam_cn_gt2 > 0)).astype(float)
        dg["dH"] = dg.frac_H - dg.nat_H
        dg["dE"] = dg.frac_E - dg.nat_E
        ins = dg.ins_H + dg.ins_E + dg.ins_C
        dg["insH"], dg["insE"], dg["insC"] = dg.ins_H / ins, dg.ins_E / ins, dg.ins_C / ins
        for r, g in dg.groupby("rewind"):
            rec = dict(variant=tag, rewind=r, n=len(g), bg_cn_med=g.bg_cn_med.mean(), seam_cn_med=g.seam_cn_med.mean(),
                       bg_brk=g.brk.mean(), seam_item_brk=g.anyseam.mean(), clash=g.n_clash.mean(),
                       dH=g.dH.mean(), dE=g.dE.mean(), insH=g.insH.mean(), insE=g.insE.mean(), insC=g.insC.mean(),
                       q3_near=g.q3_d0.mean(), q3_far=g.q3_d11.mean(), q3=g.q3_surv.mean())
            if sc is not None:
                rec["tm"] = sc[sc.rewind == r].tm_native.mean()
            rows.append(rec)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

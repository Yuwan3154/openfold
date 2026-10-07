"""Arm comparison tables (Gly / composition / Raygun one-direction), cc89 indel items, per t*: TM, share in band, SSE, bonds, clashes.
Reads the saved per-item csv folders (scores_*/diag_*), exactly as used for RAW §22-25. Run from the folder holding them:
python analysis/analyze_arms.py --panel panel_stats.csv"""
import argparse
import glob

import numpy as np
import pandas as pd


def load(pat):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    a = p.parse_args()
    nat = pd.read_csv(a.panel)
    nat["chain"] = nat.pdb + "_" + nat.chain
    nat = nat.set_index("chain")[["helix_frac", "strand_frac"]]
    arms = {"gly": ("scores_gly/cc89_cc91_*.csv", "diag_gly_v2/cc89_*.csv"), "comp": ("compF_scores/*.csv", "compF_diag/cc89_*.csv"),
            "raygun1dir": ("scores_rg3/*.csv", "diag_rg3/cc89_*.csv")}
    out = []
    for arm, (sp, dp) in arms.items():
        s = load(sp)
        s = s[(s.model == "cc89") & (s.kind == "indel")]
        d = load(dp)
        d = d[d.kind == "indel"] if "kind" in d else d
        d = d.join(nat, on="chain")
        d["helix_gap"], d["strand_gap"] = d.frac_H - d.helix_frac, d.frac_E - d.strand_frac
        d["any_broken_seam"] = ((d.seam_caca_gt45 > 0) | (d.seam_cn_gt2 > 0)).astype(float)
        ins = d.ins_H + d.ins_E + d.ins_C
        d["ins_helix"], d["ins_coil"] = d.ins_H / ins, d.ins_C / ins
        m = d.groupby(["rewind", "chain"])[["bg_cn_med", "any_broken_seam", "n_clash", "helix_gap", "strand_gap", "ins_helix", "ins_coil", "q3_surv"]].mean()
        m = m.reset_index().groupby("rewind").mean(numeric_only=True)
        tm = s.groupby(["rewind", "chain"]).tm_native.mean().reset_index().groupby("rewind").tm_native.mean()
        m["tm_native"] = tm
        m["share_tm_gt_0.5"] = s.assign(i=s.tm_native > 0.5).groupby("rewind").i.mean()
        m.insert(0, "arm", arm)
        out.append(m.reset_index())
    print(pd.concat(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Per-arm x t* summary statistics from the per-item score/diag CSVs (saved before the intermediates are deleted).

Arms (cc89, indel items): gly (scores_gly + diag_gly_v2), comp (compF_*), raygun_1dir (scores_rg3 + diag_rg3),
raygun_full (scores_raygun_full + diag_raygun_full; Raygun at the plan's L_new). Metrics are means over items within
a chain, then over the 24 chains; native helix/strand come from panel_stats.csv. Also writes the refinement-pilot
table (pilot_*), the t*=200 refinement table (p200_*) and the inserted-segment SSE table (coil_gly.csv).
Run from the directory holding the csv folders: python summary_stats.py --out summary_stats.csv
"""
import argparse
import glob

import pandas as pd

CH12 = "6x61_B 6j22_A 5zo3_A 6qla_A 6ve7_W 7aah_A 6rur_V 6z1p_Bf 6kyf_A 6yw5_QQ 6f43_A 6iu3_A".split()


def load(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pattern))], ignore_index=True)


def diag_metrics(d, nat):
    d = d.join(nat, on="chain")
    d["helix_gap"] = d.frac_H - d.helix_frac
    d["strand_gap"] = d.frac_E - d.strand_frac
    d["any_broken_seam"] = ((d.seam_caca_gt45 > 0) | (d.seam_cn_gt2 > 0)).astype(float)
    d["bg_broken_caca"] = d.bg_caca_gt45 / d.n_bg
    ins = d.ins_H + d.ins_E + d.ins_C
    d["ins_helix"], d["ins_strand"], d["ins_coil"] = d.ins_H / ins, d.ins_E / ins, d.ins_C / ins
    cols = ["n_seam", "bg_cn_med", "bg_broken_caca", "any_broken_seam", "n_clash", "helix_gap", "strand_gap",
            "ins_helix", "ins_strand", "ins_coil", "q3_d0", "q3_d11", "q3_surv"]
    return d.groupby(["rewind", "chain"])[cols].mean().reset_index().groupby("rewind")[cols].mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    nat = pd.read_csv("panel_stats.csv")
    nat["chain"] = nat.pdb + "_" + nat.chain
    nat = nat.set_index("chain")[["helix_frac", "strand_frac"]]
    arms = {
        "gly": ("scores_gly/cc89_cc91_*.csv", "diag_gly_v2/cc89_*.csv"),
        "comp": ("compF_scores/*.csv", "compF_diag/cc89_*.csv"),
        "raygun_1dir": ("scores_rg3/*.csv", "diag_rg3/cc89_*.csv"),
        "raygun_full": ("scores_raygun_full/*.csv", "diag_raygun_full/cc89_*.csv"),
    }
    out = []
    for arm, (sp, dp) in arms.items():
        s = load(sp)
        s = s[(s.model == "cc89") & (s.kind == "indel")]
        tm = s.groupby(["rewind", "chain"]).tm_native.mean().reset_index().groupby("rewind").tm_native.mean().rename("tm_native")
        band = s.assign(i=(s.tm_native >= 0.3) & (s.tm_native < 0.9)).groupby("rewind").i.mean().rename("share_tm_0.3_0.9")
        above = s.assign(i=s.tm_native > 0.5).groupby("rewind").i.mean().rename("share_tm_gt_0.5")
        d = load(dp)
        d = d[d.kind == "indel"] if "kind" in d else d
        m = diag_metrics(d, nat).join([tm, band, above])
        m.insert(0, "arm", arm)
        out.append(m.reset_index())
    pd.concat(out).to_csv(a.out, index=False)
    print(pd.concat(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

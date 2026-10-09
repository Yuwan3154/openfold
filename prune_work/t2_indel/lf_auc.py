"""AF2 confidence as a designability filter, ColabDesign vs localfold (user 10-09: rerun the calibration in localfold fast mode, template residue types stripped).
Unit = a template (set, chain, i); score = AF2 pLDDT / pTM of the template's INPUT sequence (seq_kind 0) predicted with the template; label = mean ESMFold2 paired TM over the template's
32 MPNN designs (refold_*_scores.csv, seq_kind >= 1) >= threshold. AUC = P(score of a random positive > score of a random negative), ties 0.5. Also the per-fold parity of localfold vs ColabDesign
(CA RMSD after Kabsch, |dpTM|) on every query. Run: python lf_auc.py T3_DIR AUC_DIR LF_OUT_DIR OUT_PREFIX [THR ...]   (env CD_SAME_DIR: ColabDesign refs run on the SAME template files, lf_cd_run.py npz per query,
instead of the af2compat npz, whose templates were backbone-only)
(T3_DIR holds af2compat_<set>/ npz; AUC_DIR holds refold_*_scores.csv.)"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from lf_parity import read_ca, rmsd

REFOLD = {"control": "refold_cpool_scores", "esmc07": "refold_esmc_scores", "mdc": "refold_mdc_scores", "mdg": "refold_mdg_scores", "mdr": "refold_mdr_scores",
          "milc": "refold_milc_scores", "milg": "refold_milg_scores", "milr": "refold_milr_scores"}


def auc(score, label):
    s, y = np.asarray(score, float), np.asarray(label, bool)
    assert np.isfinite(s).all(), "non-finite score"
    assert y.sum() > 0 and (~y).sum() > 0, "AUC needs both classes"
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def main():
    t3, auc_dir, lfout, out = sys.argv[1:5]
    thrs = [float(x) for x in sys.argv[5:]] or [0.5, 0.7]
    rows = []
    for s, ref in REFOLD.items():
        r = pd.read_csv(os.path.join(auc_dir, ref + ".csv"))
        assert r.method.nunique() == 1 and r.groupby(["chain", "i"]).arm.nunique().max() == 1, ref
        d = r[r.seq_kind >= 1].groupby(["chain", "i"]).tm_paired.agg(["mean", "count"]).reset_index()
        assert glob.glob(os.path.join(t3, f"af2compat_{s}", "*.npz")), s
        for f in sorted(glob.glob(os.path.join(t3, f"af2compat_{s}", "*.npz"))):
            chain, ti = os.path.basename(f)[:-4].rsplit("_t", 1)
            z = np.load(f)
            lab = d[(d.chain == chain) & (d.i == int(ti))]
            assert len(lab) == 1 and lab["count"].iloc[0] == 32, (s, chain, ti, len(lab))
            for k in range(len(z["seqs"])):
                lf = os.path.join(lfout, f"{s}_{chain}_t{ti}_k{k}")
                ca, pl = read_ca(lf + ".pdb")
                if os.environ.get("CD_SAME_DIR"):
                    zs = np.load(os.path.join(os.environ["CD_SAME_DIR"], f"{s}_{chain}_t{ti}_k{k}.npz"))
                    ref_ca, plddt_cd, ptm_cd = zs["ca"].astype(np.float64), float(zs["plddt"].mean()), float(zs["ptm"])
                else:
                    ref_ca, plddt_cd, ptm_cd = z["pred_atom37"][k][:, 1].astype(np.float64), 100 * float(z["plddt"][k].mean()), float(z["ptm"][k])
                rows.append(dict(set=s, chain=chain, i=int(ti), k=k, L=len(ca), tm_design_mean=float(lab["mean"].iloc[0]),
                                 plddt_cd=plddt_cd, ptm_cd=ptm_cd, plddt_lf=pl.mean(),
                                 ptm_lf=float(json.load(open(lf + "_summary_confidences.json"))["ptm"]), rmsd_lf_cd=rmsd(ca, ref_ca)))
    df = pd.DataFrame(rows)
    df.to_csv(out + "_folds.csv", index=False)
    n_expected = 5 * sum(len(glob.glob(os.path.join(t3, f"af2compat_{s}", "*.npz"))) for s in REFOLD)
    assert len(df) == n_expected, (len(df), n_expected)
    df["dptm"] = df.ptm_lf - df.ptm_cd
    df["pass"] = (df.rmsd_lf_cd < 1) & (df.dptm.abs() < 0.01)
    print(f"parity over {len(df)} folds: pass {int(df['pass'].sum())} ({100 * df['pass'].mean():.0f} %); median RMSD {df.rmsd_lf_cd.median():.2f} A, median |dpTM| {df.dptm.abs().median():.3f}")
    print(df.groupby(df.k.eq(0).map({True: "input_seq", False: "designs"})).agg(n=("L", "size"), pass_frac=("pass", "mean"), rmsd_med=("rmsd_lf_cd", "median"), dptm_absmed=("dptm", lambda x: x.abs().median()),
                                                                                  plddt_cd=("plddt_cd", "mean"), plddt_lf=("plddt_lf", "mean")).round(3).to_string())
    t = df[df.k == 0].copy()
    res = []
    for thr in thrs:
        t["pos"] = t.tm_design_mean >= thr
        for name, col in (("pLDDT cd", "plddt_cd"), ("pLDDT lf", "plddt_lf"), ("pTM cd", "ptm_cd"), ("pTM lf", "ptm_lf")):
            res.append(dict(thr=thr, n=len(t), n_pos=int(t.pos.sum()), score=name, auc=round(auc(t[col], t.pos), 3)))
    print(pd.DataFrame(res).to_string(index=False))
    pd.DataFrame(res).to_csv(out + "_auc.csv", index=False)


if __name__ == "__main__":
    main()

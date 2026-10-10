"""Why do some chains end with ZERO surviving templates? (user 10-10: are they rare folds? long? loopy to begin with?) Env pp1c (pandas, scipy, pydssp). Run on a production output dir.
Groups (rows = chain): ZB = zero survivors of 64 and the BREAK gate is the failure (break-gate pass share < 0.3, TM-window share >= 0.3); ZT = zero survivors and TM-window share < 0.3;
C = control chains (random sample of chains with >= 8 survivors). Per chain: native length; extendedness Rg/(2.2 L^0.38); native DSSP loop / helix / strand fractions (pydssp, proline donor mask);
native broken CA-CA steps (> 4.0 A); share of floppy residues (< 4 CA neighbours within 10 A, |i-j| > 2); residues without full backbone; entry metadata of the proteina data frame (number of chains of the entry, experiment type, resolution,
deposition year, name keywords); family size = members of the chain's 25 %-sequence-identity mmseqs cluster; fold frequency = number of chains of the list's data frame with the same CATH C.A.T code (pdb_chain_to_cat.pkl).
Prints: group medians per length bin, partial Spearman correlation of each feature with the chain's break-gate failure (1 - mean break pass) after removing the length rank, and the most frequent name words.
Run: python t2_zero_analysis.py --templates T --natives N --out O.tsv [--n-control 3000]
"""
import argparse
import glob
import os
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from t2_dssp import assign
from t2_native import load_npz, shard_path

PD = "/orcd/pool/006/chenxiou/proteina/data"
DF = f"{PD}/pdb_train/df_pdb_f1_minl50_maxl512_mtprotein_etdiffractionEM_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse.csv"
CL = f"{PD}/pdb_train/cluster_seqid_0.25_df_pdb_f1_minl50_maxl512_mtprotein_etdiffractionEM_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse.tsv"
CAT = f"{PD}/cath_shared/pdb_chain_to_cat.pkl"
BINS = [(0, 100), (100, 150), (150, 200), (200, 300), (300, 10000)]


def native_features(path):
    d = load_npz(path)
    ca = d["pos"][:, 1].astype(np.float64)
    bb = d["bb"]
    L = len(ca)
    rg = np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean())
    dist = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    far = np.abs(np.arange(L)[:, None] - np.arange(L)[None]) > 2
    nb = ((dist < 10.0) & far).sum(1)
    ss = assign(bb, np.array([c == "P" for c in d["names"]]), d["complete"])
    ok = ss >= 0
    return dict(rg_ratio=rg / (2.2 * L ** 0.38), loop=float((ss[ok] == 0).mean()), helix=float((ss[ok] == 1).mean()), strand=float((ss[ok] == 2).mean()),
                native_breaks=int((np.linalg.norm(ca[1:] - ca[:-1], axis=1) > 4.0).sum()), floppy=float((nb < 4).mean()), contacts=float(nb.mean()), n_incomplete=int((~d["complete"]).sum()))


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return spearmanr(ex, ey)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", required=True)
    ap.add_argument("--natives", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-control", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = []
    for f in glob.glob(os.path.join(a.templates, "*", "*.metrics.csv")):
        m = pd.read_csv(f)
        rows.append(dict(chain=m.chain.iloc[0], L=int(m.L_native.iloc[0]), n_pass=int(m.pass_all.sum()), tm_ok=m.pass_tm.mean(), brk_pass=m.pass_break.mean(), n_broken=m.n_broken.mean(), loop_pass=m.pass_loop.mean()))
    t = pd.DataFrame(rows)
    zb = t[(t.n_pass == 0) & (t.tm_ok >= 0.3) & (t.brk_pass < 0.3)].assign(group="ZB")
    zt = t[(t.n_pass == 0) & (t.tm_ok < 0.3)].assign(group="ZT")
    ctrl = t[t.n_pass >= 8].sample(a.n_control, random_state=a.seed).assign(group="C")
    print(f"chains with metrics {len(t)}: ZB (zero survivors, break-gate failure) {len(zb)}, ZT (zero survivors, TM window) {len(zt)}, zero-survivor other {int((t.n_pass == 0).sum()) - len(zb) - len(zt)}; control sample {len(ctrl)} of {int((t.n_pass >= 8).sum())}")
    s = pd.concat([zb, zt, ctrl], ignore_index=True)
    feats = pd.DataFrame([native_features(shard_path(a.natives, c, ".npz")) for c in s.chain])
    s = pd.concat([s, feats], axis=1)
    df = pd.read_csv(DF).set_index("id")
    meta = df.loc[s.chain, ["name", "n_chains", "experiment_type", "resolution", "deposition_date"]].reset_index(drop=True)
    s = pd.concat([s, meta], axis=1)
    s["year"] = pd.to_datetime(s.deposition_date, errors="coerce").dt.year
    member2rep = {}
    for ln in open(CL):
        r, m = ln.rstrip("\n").split("\t")
        member2rep[m] = r
    size = Counter(member2rep.values())
    s["family_size"] = [size.get(member2rep.get(c, c), 1) for c in s.chain]
    cat = pickle.load(open(CAT, "rb"))
    cat_of = {k: ".".join(v[0].split(".")[:3]) if v else None for k, v in cat.items()}
    in_df = {i.lower() for i in df.index}
    freq = Counter(v for k, v in cat_of.items() if v and k in in_df)
    s["cat"] = [cat_of.get(c.lower()) for c in s.chain]
    s["cat_class"] = [c.split(".")[0] if isinstance(c, str) else None for c in s.cat]
    s["fold_freq"] = [freq.get(c, np.nan) if isinstance(c, str) else np.nan for c in s.cat]
    s.to_csv(a.out, sep="\t", index=False)
    num = ["L", "rg_ratio", "loop", "helix", "strand", "native_breaks", "floppy", "contacts", "n_incomplete", "n_chains", "resolution", "year", "family_size", "fold_freq"]
    print("\nMEDIANS per group (rows = chain; ZB = break-gate zero-survivor, ZT = TM-window zero-survivor, C = control with >= 8 survivors)")
    print(s.groupby("group")[num].median().round(3).T.to_string())
    print("\nZB vs C within length bins (median; n chains ZB / C):")
    for lo, hi in BINS:
        b = s[(s.L >= lo) & (s.L < hi)]
        z, c = b[b.group == "ZB"], b[b.group == "C"]
        if len(z) >= 3:
            print(f"L {lo}-{hi}: n {len(z)} / {len(c)} | rg_ratio {z.rg_ratio.median():.2f} vs {c.rg_ratio.median():.2f} | loop {z.loop.median():.2f} vs {c.loop.median():.2f} | helix {z.helix.median():.2f} vs {c.helix.median():.2f} | native breaks {z.native_breaks.mean():.2f} vs {c.native_breaks.mean():.2f} | floppy {z.floppy.median():.2f} vs {c.floppy.median():.2f} | family size {z.family_size.median():.0f} vs {c.family_size.median():.0f} | n_chains {z.n_chains.median():.0f} vs {c.n_chains.median():.0f}")
    print("\nPARTIAL SPEARMAN with the break-gate failure (1 - break pass share) over ALL sampled chains (ZB + ZT + C), length rank removed:")
    s["fail"] = 1 - s.brk_pass
    for f in [x for x in num if x != "L"]:
        v = s[[f, "fail", "L"]].dropna()
        print(f"  {f:14s} r = {partial_spearman(v[f], v.fail, v.L):+.3f} (n {len(v)})")
    print("\nCHAIN CLASS (CATH C: 1 mainly alpha, 2 mainly beta, 3 alpha-beta, 4 few SS) share per group:")
    print(pd.crosstab(s.group, s.cat_class.fillna("none"), normalize="index").round(2).to_string())
    print("\nexperiment type share per group:")
    print(pd.crosstab(s.group, s.experiment_type.fillna("none"), normalize="index").round(2).to_string())
    for g in ("ZB", "ZT", "C"):
        w = Counter(x for n in s[s.group == g].name.dropna() for x in set(re.findall(r"[a-z]{4,}", n.lower())))
        print(f"\nname words {g} (n {int((s.group == g).sum())}):", w.most_common(12))


if __name__ == "__main__":
    main()

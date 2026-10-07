"""Similarity + validity of the AF2-completed templates (af2_complete.py) vs the partial-diffusion templates of the same chains (user 10-07).

Per structure (set = af2fix:<arm> or pool_t250:<arm>): geometry panel (geom_metrics.metrics, Ramachandran histogram from the native panel),
sequence-independent TM to the native normalised by the NATIVE length (tm_native, as every earlier table; user 10-07), mean pLDDT/pTM for AF2 sets, and for AF2 sets the
CA RMSD of the SURVIVORS to the native (Kabsch; does AF2 keep the template). Pairs: within each (set, chain) a seeded random subsample of
--n-div draws, all pairwise TM (mean of the two USalign normalisations, there is no native in a pair). AF2 sets have the pilot's draws only; the
partial-diffusion sets are subsampled (20 per target, user 10-07) from the FULL pool (all draws in the index) into pair CAs only (the structures table's pool_t250 rows are just the AF2-matched pilot draws, a different subset; both pool arms
draw the same 20 draw ids per chain, i.e. paired subsamples). Across sets the TM between
the AF2 and the partial-diffusion template of the SAME draw. Env: protebm (pandas, mdtraj via diagnose_indel) + USalign at ~/.local/bin/USalign.
Run: python af2fix_analyze.py --inputs-dir D/inputs --af2-root D --arms gly comp raygun --pool-root D/pool_t250 --chains .. --out-prefix P
"""
import argparse
import itertools
import os
import tempfile
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

import indel_pool as ip
from diagnose_indel import native_backbone
from geom_metrics import metrics, phipsi
from score_indel import usalign, write_ca_pdb

BB = [0, 1, 2, 4]
POOL_ARM = {"comp": "comp", "raygun": "raygun1dir"}


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    return float(np.sqrt(max((a ** 2).sum() + (b ** 2).sum() - 2 * s.sum(), 0) / len(a)))


def tm_pair(task):
    ca1, ca2 = task
    with tempfile.TemporaryDirectory() as td:
        p1, p2 = os.path.join(td, "a.pdb"), os.path.join(td, "b.pdb")
        write_ca_pdb(p1, ca1)
        write_ca_pdb(p2, ca2)
        tm1, tm2, _, _ = usalign(p1, p2)
    return tm1, tm2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--af2-root", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--pool-root", required=True)
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--n-div", type=int, default=20)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--procs", type=int, default=8)
    a = p.parse_args()
    panel = {c: native_backbone(os.path.join(a.inputs_dir, c, "native.pdb")) for c in sorted(os.listdir(a.inputs_dir))}
    phi, psi = zip(*[phipsi(b) for b in panel.values()])
    H, _, _ = np.histogram2d(np.concatenate(phi), np.concatenate(psi), bins=36, range=[[-180, 180], [-180, 180]])
    P = (H + 1) / (H + 1).sum()

    def rama_nll(ph, ps):
        return -np.log(P[np.clip(((ph + 180) // 10).astype(int), 0, 35), np.clip(((ps + 180) // 10).astype(int), 0, 35)])

    idx = pd.read_csv(os.path.join(a.pool_root, "index.csv"))
    structs = {}  # (set, chain, draw) -> dict(bb, extra)
    for c in a.chains:
        nat = panel[c].astype(np.float64)
        for arm in a.arms:
            fdir = os.path.join(a.af2_root, f"af2fix_{arm}")
            files = sorted(f for f in os.listdir(fdir) if f.startswith(c + "_d") and f.endswith(".npz") and ".tmp" not in f)
            assert files, f"no af2fix outputs for {arm} {c}"
            for f in files:
                z = np.load(os.path.join(fdir, f))
                k = int(f.split("_d")[-1][:2])
                bb = z["pred_atom37"][:, BB].astype(np.float64)
                surv = z["orig_idx"] >= 0
                structs[(f"af2fix:{arm}", c, k)] = dict(
                    bb=bb, plddt=float(z["plddt"].mean()), plddt_ins=float(z["plddt"][~surv].mean()) if (~surv).any() else np.nan,
                    ptm=float(z["ptm"]), surv_rmsd=kabsch_rmsd(bb[surv, 1], nat[z["orig_idx"][surv], 1]), n_ins=int((~surv).sum()))
                if arm in POOL_ARM:
                    r = idx[(idx.chain == c) & (idx.draw == k) & (idx.arm == POOL_ARM[arm])]
                    assert len(r) == 1, (c, k, arm, len(r))
                    t = ip.read_template(os.path.join(a.pool_root, r.iloc[0].file), int(r.iloc[0].i))
                    assert len(t["orig_idx"]) == len(bb) and (np.asarray(t["orig_idx"]) == z["orig_idx"]).all(), (c, k, arm, "pool and AF2 edits differ")
                    structs[(f"pool_t250:{POOL_ARM[arm]}", c, k)] = dict(bb=ip.atom37_coords(t)[:, BB].astype(np.float64), n_ins=int((~surv).sum()))
    rows = [dict(set=s, chain=c, draw=k, **{m: v for m, v in d.items() if m != "bb"}, **metrics(d["bb"], rama_nll)) for (s, c, k), d in structs.items()]
    tasks = [(d["bb"][:, 1], panel[c][:, 1].astype(np.float64)) for (s, c, k), d in structs.items()]
    with Pool(a.procs) as pool:
        for r, (_, tm2) in zip(rows, pool.map(tm_pair, tasks)):
            r["tm_native"] = tm2
        pairs = []
        for (s, c), grp in pd.DataFrame(rows).groupby(["set", "chain"]):
            if s.startswith("af2fix"):  # AF2 sets: the pilot's draws only
                draws = sorted(grp.draw)
                rng = np.random.default_rng([zlib.crc32(c.encode()), 20])
                sub = sorted(rng.choice(draws, size=min(a.n_div, len(draws)), replace=False).tolist())
                for k1, k2 in itertools.combinations(sub, 2):
                    pairs.append(dict(kind="within", set=s, chain=c, d1=k1, d2=k2, ca=(structs[(s, c, k1)]["bb"][:, 1], structs[(s, c, k2)]["bb"][:, 1])))
        for c in a.chains:  # partial-diffusion sets: a seeded random n_div of the FULL pool per target (user 10-07)
            for pa in POOL_ARM.values():
                sel = idx[(idx.chain == c) & (idx.arm == pa)].sort_values("draw")
                assert len(sel) > 0 and sel.draw.is_unique, (c, pa)
                rng = np.random.default_rng([zlib.crc32(c.encode()), 20])
                pick = sel.iloc[sorted(rng.choice(len(sel), size=min(a.n_div, len(sel)), replace=False).tolist())]
                cas = {int(r.draw): ip.atom37_coords(ip.read_template(os.path.join(a.pool_root, r.file), int(r.i)))[:, 1].astype(np.float64)
                       for r in pick.itertuples()}
                for k1, k2 in itertools.combinations(sorted(cas), 2):
                    pairs.append(dict(kind="within", set=f"pool_t250:{pa}", chain=c, d1=k1, d2=k2, ca=(cas[k1], cas[k2])))
        for arm, pa in POOL_ARM.items():
            for (s, c, k) in [key for key in structs if key[0] == f"af2fix:{arm}"]:
                pairs.append(dict(kind="cross", set=f"af2fix:{arm}|pool:{pa}", chain=c, d1=k, d2=k,
                                  ca=(structs[(s, c, k)]["bb"][:, 1], structs[(f"pool_t250:{pa}", c, k)]["bb"][:, 1])))
        tms = pool.map(tm_pair, [pr.pop("ca") for pr in pairs])
    for pr, (tm1, tm2) in zip(pairs, tms):
        pr["tm_sym"] = (tm1 + tm2) / 2
    st, pr = pd.DataFrame(rows), pd.DataFrame(pairs)
    st.to_csv(a.out_prefix + "_structures.csv", index=False)
    pr.to_csv(a.out_prefix + "_pairs.csv", index=False)
    cols = ["tm_native", "plddt", "surv_rmsd", "cn_med", "nca_med", "cac_med", "omega_dev_gt30", "phi_pos_frac", "rama_nll_mean", "clash_per100"]
    print(f"{len(st)} structures, {len(pr)} pairs")
    print(st.groupby("set")[cols].median().round(3).to_string())
    print(pr.groupby(["kind", "set"]).tm_sym.agg(["count", "median", "mean", "min"]).round(3).to_string())


if __name__ == "__main__":
    main()

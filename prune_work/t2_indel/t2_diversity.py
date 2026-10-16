"""Within-chain diversity of the stored templates (user 10-13). Env pp1c (numpy, pandas) + USalign for the validation pairs. Reads the pack (t2_pack_reader.TemplatePack).
Pair score = TM-like score of two templates of the SAME chain over the native residues both still contain (orig_idx >= 0 in both), after an iterative Kabsch superposition, normalised by the NATIVE length with the standard d0 = 1.24 (L - 15)^(1/3) - 1.8 ('proxy TM': fast, needs no alignment because orig_idx is the residue correspondence). Validation: USalign (sequence-independent, both normalisations; mean of the two) on random pairs of the same sample.
Per chain: all pairs among up to --k random stored templates -> max / 90th percentile / median / min pair TM, share of pairs >= 0.9 / 0.8 / 0.7, number of leader clusters at pair TM >= 0.9 / 0.8 / 0.7 (greedy).
Run: python t2_diversity.py --pack P --n-chains 300 --k 24 --n-usalign 6 --out O.tsv
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np

from t2_pack_reader import TemplatePack

USALIGN = os.path.expanduser("~/.local/bin/USalign")


def ca_of(t):
    assert t["atom_mask"][:, 1].all(), "a template residue without a CA atom (a zero row would enter the superposition and USalign)"
    full = np.zeros((len(t["aatype"]), 37, 3), np.float32)
    full[t["atom_mask"]] = t["coords"]
    return full[:, 1].astype(np.float64), t["orig_idx"]


def kabsch(P, Q, w=None):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return R, P.mean(0), Q.mean(0)


def orig_map(o):
    return {int(x): i for i, x in enumerate(o) if x >= 0}


def proxy_tm(a, b, lnat, ma=None, mb=None):
    (ca_a, o_a), (ca_b, o_b) = a, b
    ma = orig_map(o_a) if ma is None else ma
    mb = orig_map(o_b) if mb is None else mb
    common = sorted(set(ma) & set(mb))
    if len(common) < 10:
        return np.nan
    P, Q = ca_a[[ma[c] for c in common]], ca_b[[mb[c] for c in common]]
    d0 = max(1.24 * (max(lnat, 19) - 15) ** (1 / 3) - 1.8, 0.5)
    sel, best = np.arange(len(common)), 0.0
    for _ in range(5):
        if len(sel) < 4:
            break
        R, mp, mq = kabsch(P[sel], Q[sel])
        d = np.linalg.norm((P - mp) @ R - (Q - mq), axis=1)
        best = max(best, float((1.0 / (1.0 + (d / d0) ** 2)).sum() / lnat))
        sel = np.flatnonzero(d < max(d0 + 1.0, 3.5))
    return best


def write_pdb(path, ca):
    with open(path, "w") as f:
        for i, c in enumerate(ca, start=1):
            f.write(f"ATOM  {i:5d}  CA  ALA A{i:4d}    {c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C\n")
        f.write("END\n")


def parse_usalign(out, lnat=None):
    """TM from the USalign text: with lnat the extra line 'normalized by user-specified L=<lnat>' that -u adds (the native length: user 10-13), else the mean of the two own-length normalisations"""
    lines = [l for l in out.splitlines() if l.startswith("TM-score=")]
    if lnat:
        user = [l for l in lines if "user-specified" in l]
        assert len(user) == 1 and f"L={int(lnat)}." in user[0], ("USalign printed no TM-score normalised by the user-specified length", lnat, lines)
        return float(user[0].split()[1])
    t1 = next(float(l.split()[1]) for l in lines if "Structure_1" in l)
    t2 = next(float(l.split()[1]) for l in lines if "Structure_2" in l)
    return (t1 + t2) / 2


def usalign_tm(ca_a, ca_b, lnat=None):
    """USalign TM (sequence-independent alignment) normalised by lnat (-u) when given, else mean of the two own-length normalisations"""
    with tempfile.TemporaryDirectory() as td:
        pa, pb = os.path.join(td, "a.pdb"), os.path.join(td, "b.pdb")
        write_pdb(pa, ca_a); write_pdb(pb, ca_b)
        out = subprocess.run([USALIGN, pa, pb] + (["-u", str(int(lnat))] if lnat else []), capture_output=True, text=True, check=True).stdout
    return parse_usalign(out, lnat)


def leader_clusters(M, tau):
    n, leaders = len(M), []
    for i in range(n):
        if not any(M[i, j] >= tau for j in leaders):
            leaders.append(i)
    return len(leaders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--n-chains", type=int, default=300)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--n-usalign", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    tp = TemplatePack(a.pack)
    ok = [str(c) for c, n in zip(tp.stems, tp.count) if n >= 8]
    chains = [str(c) for c in rng.choice(ok, size=min(a.n_chains, len(ok)), replace=False)]
    rows, val_proxy, val_us = [], [], []
    for cid in chains:
        rr = list(tp.rows(cid))
        pick = sorted(rng.choice(rr, size=min(a.k, len(rr)), replace=False))
        T = [tp._read(r) for r in pick]
        cas = [ca_of(t) for t in T]
        lnat = T[0]["L_native"]
        n = len(T)
        M = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = M[j, i] = proxy_tm(cas[i], cas[j], lnat)
        iu = np.triu_indices(n, 1)
        v = M[iu][~np.isnan(M[iu])]
        if len(v) == 0:
            continue
        Mf = np.where(np.isnan(M), 0, M)
        for _ in range(a.n_usalign):
            i, j = rng.choice(n, size=2, replace=False)
            if not np.isnan(M[i, j]):
                val_proxy.append(M[i, j]); val_us.append(usalign_tm(cas[i][0], cas[j][0]))
        rows.append(dict(chain=cid, L_native=lnat, n=n, tm_native_mean=float(np.mean([t["tm_native"] for t in T])), pmax=float(v.max()), p90=float(np.percentile(v, 90)), pmed=float(np.median(v)), pmin=float(v.min()),
                         f09=float((v >= 0.9).mean()), f08=float((v >= 0.8).mean()), f07=float((v >= 0.7).mean()), c09=leader_clusters(Mf, 0.9), c08=leader_clusters(Mf, 0.8), c07=leader_clusters(Mf, 0.7)))
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(a.out, sep="\t", index=False)
    vp, vu = np.array(val_proxy), np.array(val_us)
    print(f"VALIDATION proxy TM vs USalign TM (mean of both normalisations), {len(vp)} random template pairs of {len(d)} chains: Pearson r {np.corrcoef(vp, vu)[0, 1]:.3f}, mean (proxy - USalign) {np.mean(vp - vu):+.3f}, mean |diff| {np.mean(np.abs(vp - vu)):.3f}; pairs where USalign >= 0.9: {int((vu >= 0.9).sum())}, proxy >= 0.9 among them {int(((vu >= 0.9) & (vp >= 0.9)).sum())}")
    print(f"rows = chain (n = {len(d)} chains, {a.k} random stored templates each = {int(d.n.sum())} templates; {int((d.n * (d.n - 1) / 2).sum())} pairs)")
    for col, name in (("pmax", "max pair TM"), ("p90", "90th percentile pair TM"), ("pmed", "median pair TM"), ("pmin", "min pair TM")):
        q = d[col].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()
        print(f"  {name:25s}: 5/25/50/75/95 % of chains = {q}")
    for thr in (0.7, 0.8, 0.9, 0.95):
        print(f"  chains whose max pair TM < {thr}: {int((d.pmax < thr).sum())} ({100 * (d.pmax < thr).mean():.1f} %)")
    print(f"  share of template pairs with TM >= 0.9 / 0.8 / 0.7 (mean over chains): {d.f09.mean():.3f} / {d.f08.mean():.3f} / {d.f07.mean():.3f}")
    print(f"  leader clusters among {a.k} templates at pair TM >= 0.9 / 0.8 / 0.7 (median over chains): {d.c09.median():.0f} / {d.c08.median():.0f} / {d.c07.median():.0f}; chains with >= 8 clusters at 0.9: {int((d.c09 >= 8).sum())}")
    for lo, hi in [(0, 100), (100, 200), (200, 300), (300, 10000)]:
        s = d[(d.L_native >= lo) & (d.L_native < hi)]
        if len(s):
            print(f"  L {lo}-{hi} (n {len(s)}): median max pair TM {s.pmax.median():.3f}, median pair TM {s.pmed.median():.3f}, median clusters at 0.9: {s.c09.median():.0f}, share of pairs >= 0.9: {s.f09.mean():.3f}")


if __name__ == "__main__":
    main()

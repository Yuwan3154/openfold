"""Decompose how each Protpardelle-1c synthetic template departs from its native.

Answers: when TM-to-native is low, did the secondary-structure elements (SSEs) themselves change,
or are the native SSEs intact but re-packed? Per template:
  - DSSP 3-state (mdtraj compute_dssp simplified: H = H/G/I, E = E/B, C = rest), Q3 vs native,
    composition, and the per-residue native->template transition counts;
  - per native SSE element (maximal run of H or of E): label retention, local CA RMSD after
    superposing the element alone (its internal shape), and CA RMSD of the same residues after
    superposing the WHOLE chain (shape + placement). local << global = intact element, moved;
  - residue contacts, CASP definition (CB, CA for Gly, < 8 A, |i-j| >= 6): fraction of native
    contacts kept, and of the long-range ones (|i-j| >= 24);
  - element-pair contacts (>= 1 residue contact between two elements): native pairs kept, new pairs.
Residues correspond by index: partial diffusion keeps the input length and sequence.

Run: <proteinebm env>/bin/python sse_drift.py --tm-csv pp1c_tm.csv --natives ~/pp1c_work/natives \
     --out-prefix out/bakeoff
"""
import argparse
import os

import mdtraj as md
import numpy as np
import pandas as pd

CONTACT_A = 8.0     # CASP residue-contact definition
MIN_SEP = 6         # CASP: short >= 6, medium >= 12, long >= 24
LONG_SEP = 24


def load(pdb):
    t = md.load_pdb(pdb)
    top = t.topology
    res = [r for r in top.residues if r.is_protein]
    ss = md.compute_dssp(t, simplified=True)[0]
    prot = [i for i, r in enumerate(top.residues) if r.is_protein]
    ss = np.array(ss)[prot]
    xyz = t.xyz[0] * 10.0  # nm -> A
    ca = np.array([xyz[r.atom("CA").index] for r in res])
    cb = np.array([xyz[(r.atom("CB") if r.name != "GLY" else r.atom("CA")).index] for r in res])
    names = [r.name for r in res]
    return ss, ca, cb, names


def kabsch_rmsd(p, q):
    p = p - p.mean(0)
    q = q - q.mean(0)
    u, s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean())), r


def superpose(p, q):
    """p moved onto q (whole-chain fit)."""
    _, r = kabsch_rmsd(p, q)
    return (p - p.mean(0)) @ r + q.mean(0)


def elements(ss):
    out, i = [], 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if ss[i] in ("H", "E"):
            out.append((ss[i], i, j + 1))
        i = j + 1
    return out


def contacts(cb):
    d = np.linalg.norm(cb[:, None] - cb[None], axis=-1)
    n = len(cb)
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None])
    return (d < CONTACT_A) & (sep >= MIN_SEP), sep


def element_pairs(cmap, els):
    pairs = set()
    for a in range(len(els)):
        for b in range(a + 1, len(els)):
            _, i0, i1 = els[a]
            _, j0, j1 = els[b]
            if cmap[i0:i1, j0:j1].any():
                pairs.add((a, b))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tm-csv", required=True)
    p.add_argument("--natives", required=True)
    p.add_argument("--out-prefix", required=True)
    a = p.parse_args()

    df = pd.read_csv(a.tm_csv)
    df["chain_id"] = df["pdb_id"] + "_" + df["chain"]
    nat = {}
    for c in sorted(df["chain_id"].unique()):
        ss, ca, cb, names = load(os.path.join(a.natives, f"{c}.pdb"))
        cm, sep = contacts(cb)
        els = elements(ss)
        nat[c] = dict(ss=ss, ca=ca, cb=cb, names=names, cmap=cm, sep=sep, els=els,
                      epairs=element_pairs(cm, els))
        print(f"native {c}: L={len(ss)} H={np.mean(ss == 'H'):.2f} E={np.mean(ss == 'E'):.2f} "
              f"elements={len(els)} contacts={cm.sum() // 2} element-pairs={len(nat[c]['epairs'])}")

    rows, erows = [], []
    for _, r in df.iterrows():
        n = nat[r["chain_id"]]
        ss, ca, cb, names = load(r["pdb_file"])
        assert names == n["names"], f"{r['pdb_file']}: residue sequence differs from native"
        cm, _ = contacts(cb)
        glob = superpose(ca, n["ca"])
        rec = dict(model=r["model"], chain=r["chain_id"], rewind=int(r["rewind_steps"]),
                   sample=int(r["sample"]), tm=float(r["tm_to_native_byresi"]),
                   rmsd=float(r["rmsd_byresi"]), L=len(ss))
        rec["q3"] = float(np.mean(ss == n["ss"]))
        for s in "HEC":
            rec[f"f{s}"] = float(np.mean(ss == s))
            rec[f"f{s}_native"] = float(np.mean(n["ss"] == s))
            for t in "HEC":
                rec[f"n_{s}to{t}"] = int(((n["ss"] == s) & (ss == t)).sum())
        nc = n["cmap"]
        rec["q_contacts"] = float((cm & nc).sum() / nc.sum())
        lr = nc & (n["sep"] >= LONG_SEP)
        rec["q_contacts_long"] = float((cm & lr).sum() / lr.sum()) if lr.any() else np.nan
        rec["frac_nonnative_contacts"] = float((cm & ~nc).sum() / max(cm.sum(), 1))
        ep = element_pairs(cm, n["els"])
        rec["epairs_kept"] = float(len(ep & n["epairs"]) / max(len(n["epairs"]), 1))
        rec["epairs_new"] = int(len(ep - n["epairs"]))
        w_loc = w_glob = w_n = 0.0
        for k, (lab, i0, i1) in enumerate(n["els"]):
            m = i1 - i0
            ret = float(np.mean(ss[i0:i1] == lab))
            g = float(np.sqrt(((glob[i0:i1] - n["ca"][i0:i1]) ** 2).sum(1).mean()))
            loc = kabsch_rmsd(ca[i0:i1], n["ca"][i0:i1])[0] if m >= 3 else np.nan  # Kabsch needs 3 points
            erows.append(dict(model=rec["model"], chain=rec["chain"], rewind=rec["rewind"],
                              sample=rec["sample"], tm=rec["tm"], element=k, type=lab, start=i0,
                              length=m, retention=ret, local_rmsd=loc, global_rmsd=g))
            if m >= 3:
                w_loc += m * loc ** 2
                w_glob += m * g ** 2
                w_n += m
        rec["elem_local_rmsd"] = float(np.sqrt(w_loc / w_n))
        rec["elem_global_rmsd"] = float(np.sqrt(w_glob / w_n))
        rec["elem_retention"] = float(np.mean([e["retention"] for e in erows[-len(n["els"]):]]))
        rows.append(rec)

    out = pd.DataFrame(rows)
    eout = pd.DataFrame(erows)
    assert len(out) == len(df), (len(out), len(df))
    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
    out.to_csv(f"{a.out_prefix}_sse_drift.csv", index=False)
    eout.to_csv(f"{a.out_prefix}_elements.csv", index=False)
    print(f"wrote {len(out)} templates, {len(eout)} element rows")


if __name__ == "__main__":
    main()

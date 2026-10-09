"""Parity of localfold's CUDA AlphaFold 2 (int5/3-bit weights, TF32/f16 kernels) against the ColabDesign fp32 AF2 used so far (user 10-08).

Per (test, chain): the localfold prediction (<lf-dir>/<test>/<chain>.pdb, B-factor = pLDDT; <chain>_summary_confidences.json = pTM) vs the ColabDesign
reference npz (pred_atom37 (.., L, 37, 3), plddt (0-1), ptm). Reported: CA RMSD between the two predictions (Kabsch, 1:1 residues), a 1:1 TM-like score of
one against the other (Kabsch-superposed, d0 from the length; NOT a TM-align optimum), CA RMSD of each implementation to the native, |mean pLDDT difference|,
|pTM difference|, per-residue pLDDT Pearson r. No pass/fail thresholds are applied.
Run: python lf_parity.py --lf-dir lf_out --ref-t1 ref_t1 --ref-t2 ref_t2 --native native --out parity.csv
"""
import argparse
import csv
import json
import os

import numpy as np

THREE2ONE = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M",
             "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"}


def read_ca(path):
    ca, b = [], []
    for ln in open(path):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            ca.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            b.append(float(ln[60:66]))
    return np.array(ca), np.array(b)


def native_seq(path):
    res = {}
    for ln in open(path):
        if ln.startswith("ATOM"):
            res[int(ln[22:26])] = THREE2ONE[ln[17:20].strip()]
    return "".join(res[i] for i in sorted(res))


def kabsch(a, b):
    """Rotate/translate a onto b; returns aligned a."""
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(ac.T @ bc)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return ac @ r + b.mean(0)


def rmsd(a, b):
    return float(np.sqrt(((kabsch(a, b) - b) ** 2).sum(1).mean()))


def tm_like(a, b):
    L = len(b)
    d0 = 1.24 * max(L - 15, 1) ** (1 / 3) - 1.8
    d = np.linalg.norm(kabsch(a, b) - b, axis=1)
    return float((1 / (1 + (d / d0) ** 2)).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lf-dir", required=True)
    p.add_argument("--ref-t1", required=True)
    p.add_argument("--ref-t2", required=True)
    p.add_argument("--native", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for test, ref_dir, suffix in (("t1", a.ref_t1, "_t000.npz"), ("t2", a.ref_t2, "_native.npz")):
        tdir = os.path.join(a.lf_dir, test)
        assert os.path.isdir(tdir), tdir
        for f in sorted(os.listdir(tdir)):
            if not f.endswith(".pdb"):
                continue
            chain = f[:-4]
            lf_ca, lf_pl = read_ca(os.path.join(tdir, f))
            z = np.load(os.path.join(ref_dir, chain + suffix))
            ref_ca = z["pred_atom37"].reshape(-1, z["pred_atom37"].shape[-3], 37, 3)[0][:, 1].astype(np.float64)
            ref_pl = np.asarray(z["plddt"]).reshape(-1, ref_ca.shape[0])[0] * 100
            nat_ca, _ = read_ca(os.path.join(a.native, chain + ".pdb"))
            assert len(lf_ca) == len(ref_ca) == len(nat_ca), (test, chain, len(lf_ca), len(ref_ca), len(nat_ca))
            js = json.load(open(os.path.join(tdir, chain + "_summary_confidences.json")))
            ptm_lf = float(js["ptm"])
            ptm_ref = float(np.asarray(z["ptm"]).reshape(-1)[0])
            rows.append(dict(test=test, chain=chain, L=len(lf_ca), rmsd_lf_vs_ref=rmsd(lf_ca, ref_ca), tm_like_lf_vs_ref=tm_like(lf_ca, ref_ca),
                             rmsd_lf_to_native=rmsd(lf_ca, nat_ca), rmsd_ref_to_native=rmsd(ref_ca, nat_ca),
                             plddt_lf=float(lf_pl.mean()), plddt_ref=float(ref_pl.mean()), ptm_lf=ptm_lf, ptm_ref=ptm_ref,
                             plddt_resid_r=float(np.corrcoef(lf_pl, ref_pl)[0, 1])))
    assert rows, "no localfold outputs found"
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hdr = list(rows[0])
    print("  ".join(f"{h[:16]:>16s}" for h in hdr))
    for r in rows:
        print("  ".join(f"{r[h]:>16.3f}" if isinstance(r[h], float) else f"{str(r[h]):>16s}" for h in hdr))


if __name__ == "__main__":
    main()

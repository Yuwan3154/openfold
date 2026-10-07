"""Backbone geometry beyond DSSP/TM, for native vs no-indel control vs indel pool backbones (user 10-07: 'is it the geometry?').

Per structure (N,CA,C,O backbone): median C-N / N-CA / CA-C bonds, bond-length scatter (MAD), N-CA-C / CA-C-N / C-N-CA angles
(median, MAD), omega deviation from trans (fraction > 30 deg; cis fraction), fraction phi > 0 (L-chirality check; Gly/Pro are a small
share), Ramachandran NLL = -log p(phi,psi) under a 2-D histogram (36x36 bins, +1 pseudocount) built from the NATIVE panel
residues (mean and 95th percentile per structure), virtual CA-CA-CA angle median, non-local CA clashes (< 3.0 A, |i-j| >= 3) per 100 res.
Sources: native (inputs/<chain>/native.pdb), controls (sweep_gly c<k> at each rung), pool templates (indel_pool).
Run: python geom_metrics.py --inputs-dir inputs --sweep sweep_gly --pool-roots pool_t250 pool_ctl --out geom.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indel_pool as ip  # noqa: E402
from diagnose_indel import native_backbone, unpack_bb  # noqa: E402


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def angle(a, b, c):
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)), -1, 1)))


def phipsi(bb):
    N, CA, C = bb[:, 0], bb[:, 1], bb[:, 2]
    phi = dihedral(C[:-2], N[1:-1], CA[1:-1], C[1:-1])
    psi = dihedral(N[1:-1], CA[1:-1], C[1:-1], N[2:])
    return phi, psi


def mad(x):
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def metrics(bb, rama_nll):
    N, CA, C = bb[:, 0], bb[:, 1], bb[:, 2]
    cn = np.linalg.norm(N[1:] - C[:-1], axis=1)
    nca = np.linalg.norm(CA - N, axis=1)
    cac = np.linalg.norm(C - CA, axis=1)
    a1 = angle(N, CA, C)
    a2 = angle(CA[:-1], C[:-1], N[1:])
    a3 = angle(C[:-1], N[1:], CA[1:])
    om = dihedral(CA[:-1], C[:-1], N[1:], CA[1:])
    phi, psi = phipsi(bb)
    nll = rama_nll(phi, psi)
    d = np.linalg.norm(CA[:, None] - CA[None], axis=-1)
    iu = np.triu_indices(len(CA), k=3)
    vca = angle(CA[:-2], CA[1:-1], CA[2:])
    return dict(L=len(CA), cn_med=np.median(cn), cn_mad=mad(cn), nca_med=np.median(nca), nca_mad=mad(nca), cac_med=np.median(cac),
                cac_mad=mad(cac), ncac_med=np.median(a1), ncac_mad=mad(a1), cacn_med=np.median(a2), cnca_med=np.median(a3),
                cnca_mad=mad(a3), omega_dev_gt30=float((180 - np.abs(om) > 30).mean()), cis_frac=float((np.abs(om) < 30).mean()),
                phi_pos_frac=float((phi > 0).mean()), rama_nll_mean=float(nll.mean()), rama_nll_p95=float(np.percentile(nll, 95)),
                vca_med=float(np.median(vca)), clash_per100=float((d[iu] < 3.0).sum() / len(CA) * 100))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--sweep", required=True)
    p.add_argument("--pool-roots", nargs="+", default=[])
    p.add_argument("--ctl-draws", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    chains = sorted(os.listdir(a.inputs_dir))
    nat = {c: native_backbone(os.path.join(a.inputs_dir, c, "native.pdb")) for c in chains}
    allphi, allpsi = zip(*[phipsi(b) for b in nat.values()])
    H, _, _ = np.histogram2d(np.concatenate(allphi), np.concatenate(allpsi), bins=36, range=[[-180, 180], [-180, 180]])
    P = (H + 1) / (H + 1).sum()

    def rama_nll(phi, psi):
        i = np.clip(((phi + 180) // 10).astype(int), 0, 35)
        j = np.clip(((psi + 180) // 10).astype(int), 0, 35)
        return -np.log(P[i, j])

    rows = []
    for c, bb in nat.items():
        rows.append(dict(set="native", chain=c, draw=-1, rewind=0, tm_native=1.0, **metrics(bb.astype(np.float64), rama_nll)))
    for c in chains:
        for k in range(a.ctl_draws):
            f = os.path.join(a.sweep, "cc89", c, f"c{k:02d}.npz")
            if not os.path.isfile(f):
                continue
            z = np.load(f)
            for r, bb in zip(z["rewind_steps"].tolist(), unpack_bb(z)):
                rows.append(dict(set="control", chain=c, draw=k, rewind=int(r), tm_native=np.nan, **metrics(bb.astype(np.float64), rama_nll)))
    for root in a.pool_roots:
        idx = pd.read_csv(os.path.join(root, "index.csv"))
        for (chain, file), g in idx.groupby(["chain", "file"]):
            for r in g.itertuples():
                t = ip.read_template(os.path.join(root, file), r.i)
                full = ip.atom37_coords(t)
                bb = full[:, [0, 1, 2, 4]].astype(np.float64)
                rows.append(dict(set=f"{os.path.basename(root.rstrip('/'))}:{r.arm}", chain=chain, draw=r.draw, rewind=r.rewind,
                                 tm_native=r.tm_native, **metrics(bb, rama_nll)))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} structures -> {a.out}")


if __name__ == "__main__":
    main()

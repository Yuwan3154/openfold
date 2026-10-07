"""Where does the refold fail: near the edits or everywhere? (H-C)  For each pilot prediction (designed sequences only) of an INDEL template:
superpose the predicted CA onto the template CA using only residues FAR (> --far template positions) from every insertion / deletion seam, then report
the median CA deviation of far residues and of near residues, the fraction of far residues within 3 A, and the far-only superposition RMSD.
Seams = inserted residues (orig_idx -1) and survivor pairs that are not consecutive in the native. Env: protpardelle (numpy), reads refold npz + pool.
Run: python refold_local.py --pool-root pool_t250 --in-dir refold_esm --out local.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import indel_pool as ip


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return R, P.mean(0), Q.mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-root", required=True)
    p.add_argument("--in-dir", required=True)
    p.add_argument("--far", type=int, default=10)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.in_dir, "*.npz"))):
        chain, ti = os.path.basename(f)[:-4].rsplit("_t", 1)
        ti = int(ti)
        z = np.load(f)
        t = ip.read_template(ip.shard_path(a.pool_root, chain), ti)
        orig = t["orig_idx"]
        L = len(orig)
        ev = np.flatnonzero(orig < 0)
        seam = np.flatnonzero((orig[1:] >= 0) & (orig[:-1] >= 0) & ((orig[1:] - orig[:-1]) != 1))
        events = np.concatenate([ev, seam, seam + 1]) if (len(ev) + len(seam)) else np.array([], int)
        dist = np.min(np.abs(np.arange(L)[:, None] - events[None]), axis=1) if len(events) else np.full(L, 10**9)
        far = (dist > a.far) & (orig >= 0)
        near = ~far
        tca = z["template_ca"].astype(np.float64)
        for s in range(1, len(z["seqs"])):                      # 0 = generation sequence
            pca = z["pred_atom37"][s][:, 1].astype(np.float64)
            R, pm, qm = kabsch(pca[far], tca[far]) if far.sum() >= 10 else (np.eye(3), pca.mean(0), tca.mean(0))
            dev = np.linalg.norm((pca - pm) @ R - (tca - qm), axis=1)
            rows.append(dict(chain=chain, i=ti, arm=t["arm"], seq=s, L=L, n_far=int(far.sum()), n_near=int(near.sum()),
                             far_med=float(np.median(dev[far])) if far.any() else np.nan,
                             near_med=float(np.median(dev[near])) if near.any() else np.nan,
                             far_within3=float((dev[far] < 3).mean()) if far.any() else np.nan,
                             near_within3=float((dev[near] < 3).mean()) if near.any() else np.nan,
                             far_rmsd=float(np.sqrt((dev[far] ** 2).mean())) if far.any() else np.nan))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} predictions -> {a.out}")


if __name__ == "__main__":
    main()

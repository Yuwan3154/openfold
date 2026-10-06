"""Why are inserted residues ~80% coil in the Gly arm? Decompose on the EXISTING sweep outputs (CPU).

Per inserted SEGMENT (maximal run of orig<0 in the edited chain): length, DSSP composition after the run at each
t*, and the same for the edited INPUT itself (the t*->0 starting point: interpolated straight lines). Also the
flank context (DSSP of the survivor on each side in the NATIVE), and the chain's native helix/strand fraction.
Env: protebm (mdtraj). Run: python coil_diagnosis.py --inputs-dir <inputs> --out-root <sweep> --model cc89 --out coil.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_indel import dssp, native_backbone, unpack_bb  # noqa: E402


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--model", default="cc89")
    p.add_argument("--out", required=True)
    p.add_argument("--draws", type=int, default=16)
    a = p.parse_args()
    rows = []
    for key in sorted(os.listdir(a.inputs_dir)):
        d = os.path.join(a.inputs_dir, key)
        plans = json.load(open(os.path.join(d, "plans.json")))
        ss_nat = dssp(native_backbone(os.path.join(d, "native.pdb")))
        nat_h, nat_e = float((ss_nat == "H").mean()), float((ss_nat == "E").mean())
        for k in range(a.draws):
            orig = np.array(plans["plans"][k]["orig_idx"])
            segs = runs(orig < 0)
            if not segs:
                continue
            edited = native_backbone(os.path.join(d, f"d{k:02d}.pdb"))
            ss_in = dssp(edited)
            z = np.load(os.path.join(a.out_root, a.model, key, f"d{k:02d}.npz"))
            bbs = unpack_bb(z)
            ss_out = {int(r): dssp(bb) for r, bb in zip(z["rewind_steps"].tolist(), bbs)}
            for (i, j) in segs:
                rec = dict(chain=key, draw=k, seg_len=j - i + 1, nat_H=nat_h, nat_E=nat_e,
                           flank_left=ss_nat[orig[i - 1]] if i > 0 and orig[i - 1] >= 0 else "-",
                           flank_right=ss_nat[orig[j + 1]] if j + 1 < len(orig) and orig[j + 1] >= 0 else "-")
                for tag, ss in [("in", ss_in)] + [(str(r), s) for r, s in ss_out.items()]:
                    seg = ss[i:j + 1]
                    for c in "HEC":
                        rec[f"{tag}_{c}"] = int((seg == c).sum())
                rows.append(rec)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} inserted segments from {a.draws} draws/chain -> {a.out}")


if __name__ == "__main__":
    main()

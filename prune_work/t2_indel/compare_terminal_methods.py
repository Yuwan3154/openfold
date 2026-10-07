"""Terminal-insertion step: current per-atom rescaling (indel_edit.edit) vs a rigid translation of the end residue.

NOTE: 'current' needs the PRE-switch indel_edit.edit (per-atom stepping, commits <= e917060); after the switch to rigid translation both columns coincide.
Both give a CA-CA step of exactly 3.8 A. Old rule: every backbone atom steps by (r1 - r0) * 3.8/|dCA| (survivors may straddle a
deletion, so atoms can drift relative to CA). Rigid (ADOPTED, user 10-07): the end survivor is translated along the CA direction,
so inserted residues copy its internal geometry exactly. Measured on the REAL plans (plans.json ops) of every chain/draw:
deviations of the inserted terminal residues' N-CA/CA-C/C-O lengths and N-CA-C angle from the chain's own native medians, and peptide C-N lengths
(1.33 A) across the terminal block and its junction with the end survivor, split by whether the two end
survivors straddle a deletion. Env: any with numpy. Run: python compare_terminal_methods.py --inputs-dir D --chains ... --out-csv f.csv
"""
import argparse
import csv
import json
import os

import numpy as np

from indel_edit import CA_STEP, edit

BB = ("N", "CA", "C", "O")


def read_native_bb(path):
    res = {}
    for ln in open(path):
        if ln.startswith("ATOM") and ln[12:16].strip() in BB:
            res.setdefault(int(ln[22:26]), {})[ln[12:16].strip()] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
    return np.array([[res[i][a] for a in BB] for i in sorted(res)])


def rigid_terminal(x, orig):
    """Recompute the leading/trailing inserted residues of edit()'s output by rigid translation of the end survivor."""
    x = x.copy()
    a = int(np.argmax(orig >= 0))
    b = len(orig) - 1 - int(np.argmax(orig[::-1] >= 0))
    surv = np.flatnonzero(orig >= 0)
    u = x[surv[1], 1] - x[surv[0], 1]
    u /= np.linalg.norm(u)
    for j in range(1, a + 1):
        x[a - j] = x[a] - j * CA_STEP * u
    u = x[surv[-1], 1] - x[surv[-2], 1]
    u /= np.linalg.norm(u)
    for j in range(1, len(orig) - 1 - b + 1):
        x[b + j] = x[b] + j * CA_STEP * u
    return x


def angle(a, b, c):
    v, w = a - b, c - b
    return np.degrees(np.arccos(np.dot(v, w) / np.linalg.norm(v) / np.linalg.norm(w)))


def native_ref(nat):
    """Chain-own median N-CA, CA-C, C-O lengths and N-CA-C angle: the absolute reference (no invented ideal values)."""
    d = [np.median(np.linalg.norm(nat[:, p] - nat[:, q], axis=1)) for p, q in ((0, 1), (1, 2), (2, 3))]
    return d, np.median([angle(r[0], r[1], r[2]) for r in nat])


def metrics(x, rows, ref):
    """Max deviation (A) of N-CA, CA-C, C-O and (deg) of N-CA-C over the inserted terminal rows from the chain's own medians."""
    bonds, ang0 = ref
    bond = max(abs(np.linalg.norm(x[r, p] - x[r, q]) - d) for r in rows for (p, q), d in zip(((0, 1), (1, 2), (2, 3)), bonds))
    ang = max(abs(angle(x[r, 0], x[r, 1], x[r, 2]) - ang0) for r in rows)
    return bond, ang


def peptide_dev(x, pairs):
    return max(abs(np.linalg.norm(x[i, 2] - x[i + 1, 0]) - 1.33) for i in pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--out-csv", required=True)
    a = p.parse_args()
    out = []
    for c in a.chains:
        nat = read_native_bb(os.path.join(a.inputs_dir, c, "native.pdb"))
        ref = native_ref(nat)
        plans = json.load(open(os.path.join(a.inputs_dir, c, "plans.json")))["plans"]
        for k, pl in enumerate(plans):
            ops = [tuple(o) for o in pl["ops"]]
            x, orig, _ = edit(nat, ops)
            lead = int(np.argmax(orig >= 0))
            trail = len(orig) - 1 - int(np.argmax(orig[::-1] >= 0))
            ntrail = len(orig) - 1 - trail
            if lead == 0 and ntrail == 0:
                continue
            surv = np.flatnonzero(orig >= 0)
            xr = rigid_terminal(x, orig)
            for end, rows, pairs, s0, s1 in (("N", list(range(lead)), list(range(lead)), surv[0], surv[1]),
                                             ("C", list(range(trail + 1, len(orig))), list(range(trail, len(orig) - 1)), surv[-2], surv[-1])):
                if not rows:
                    continue
                straddle = int(orig[s1] - orig[s0] > 1)
                row = dict(chain=c, draw=k, end=end, n_ins=len(rows), straddle=straddle)
                for name, xx in (("current", x), ("rigid", xr)):
                    bond, ang = metrics(xx, rows, ref)
                    row.update({f"{name}_bond_dev": bond, f"{name}_angle_dev": ang, f"{name}_pep_dev": peptide_dev(xx, pairs)})
                out.append(row)
    assert out, "no terminal insertions found"
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for st in (0, 1):
        sub = [r for r in out if r["straddle"] == st]
        print(f"straddle={st}: n={len(sub)}")
        for name in ("current", "rigid"):
            for m in ("bond_dev", "angle_dev", "pep_dev"):
                v = np.array([r[f"{name}_{m}"] for r in sub])
                if len(v):
                    print(f"  {name:8s} {m:10s} median {np.median(v):.3f}  p95 {np.percentile(v, 95):.3f}  max {v.max():.3f}")


if __name__ == "__main__":
    main()

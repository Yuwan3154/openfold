"""Audit a generation tree against its inputs (counts per model/chain/kind, shapes, finiteness, L_new, rungs).
Prints the full per-chain table size and every violation; exits non-zero via assert on any violation.
Run: python audit_outputs.py --inputs-dir <inputs> --out-root <sweep> --models cc89 cc91 --kinds indel control
"""
import argparse
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--kinds", nargs="+", default=["indel", "control"])
    p.add_argument("--rewinds", type=int, nargs="+", default=[375, 300, 250, 200])
    a = p.parse_args()
    bad, n_ok = [], 0
    for m in a.models:
        for key in sorted(os.listdir(a.inputs_dir)):
            plans = json.load(open(os.path.join(a.inputs_dir, key, "plans.json")))
            want = {}
            for k in range(len(plans["plans"])):
                if "indel" in a.kinds:
                    want[f"d{k:02d}.npz"] = plans["plans"][k]["L_new"]
                if "control" in a.kinds:
                    want[f"c{k:02d}.npz"] = plans["L"]
            d = os.path.join(a.out_root, m, key)
            have = set(os.listdir(d)) if os.path.isdir(d) else set()
            if have != set(want):
                bad.append((m, key, "file set differs", len(have), len(want)))
            for fn, L in want.items():
                if fn not in have:
                    continue
                z = np.load(os.path.join(d, fn))
                mask = z["atom_mask"]
                ok = (z["coords"].shape[0] == len(a.rewinds) and z["coords"].shape[2] == 3
                      and z["coords"].shape[1] == int(mask.sum()) and mask.shape[0] == L
                      and int(z["L_new"]) == L and z["rewind_steps"].tolist() == a.rewinds
                      and bool(np.isfinite(z["coords"]).all()) and str(z["model"]) == m)
                if not ok:
                    bad.append((m, key, fn, "shape/finite/L/rungs"))
                n_ok += ok
    print(f"audited ok: {n_ok}; violations: {len(bad)}")
    for b in bad[:30]:
        print("VIOLATION", b)
    assert not bad, "audit failed"


if __name__ == "__main__":
    main()

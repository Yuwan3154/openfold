"""Which Gly-arm draws change under the rigid terminal rule (24f5c50)? Re-edit each chain's native.pdb backbone with the CURRENT
indel_edit and compare with the existing d<k>.pdb backbone atoms. Unaffected draws must agree within PDB rounding (native.pdb and d<k>.pdb
are 3-decimal), so the tolerance is 2e-3 A. Writes a csv (chain, draw, max_abs_diff, changed) and prints the counts.
Control: `changed` must equal `has_term` (the plan inserts before the first / after the last SURVIVING residue); mismatches are
reported, never silently dropped. A shape or orig_idx mismatch is recorded as changed=1 with a note instead of aborting the scan.
Env: numpy only. Run: python verify_regen_scope.py --inputs-dir D --chains ... --out-csv f.csv
"""
import argparse
import csv
import json
import os

import numpy as np

from indel_edit import deleted_mask, edit, insertion_locations

BB = ("N", "CA", "C", "O")
TOL = 2e-3


def read_bb(path):
    res = {}
    for ln in open(path):
        name = {"OT1": "O"}.get(ln[12:16].strip(), ln[12:16].strip())  # cg2all (Raygun-arm) files name the terminal oxygen OT1
        if ln.startswith("ATOM") and name in BB:
            res.setdefault(int(ln[22:26]), {})[name] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
    return np.array([[res[i][a] for a in BB] for i in sorted(res)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--ca-only", action="store_true", help="compare CA only (cg2all-built Raygun-arm files may move N/C/O slightly)")
    a = p.parse_args()
    rows = []
    for c in a.chains:
        nat = read_bb(os.path.join(a.inputs_dir, c, "native.pdb"))
        plans = json.load(open(os.path.join(a.inputs_dir, c, "plans.json")))["plans"]
        for k, pl in enumerate(plans):
            new, orig, _ = edit(nat, [tuple(o) for o in pl["ops"]])
            old = read_bb(os.path.join(a.inputs_dir, c, f"d{k:02d}.pdb"))
            ops = [tuple(o) for o in pl["ops"]]
            locs = insertion_locations(len(nat), ops)
            has_term = int(0 in locs or int((~deleted_mask(len(nat), ops)).sum()) in locs)
            note = ""
            if old.shape != new.shape or orig.tolist() != pl["orig_idx"]:
                d, note = float("inf"), f"shape/orig_idx mismatch {old.shape} vs {new.shape}"
            else:
                d = float(np.abs((old - new)[:, 1] if a.ca_only else old - new).max())
            rows.append(dict(chain=c, draw=k, max_abs_diff=d, changed=int(d > TOL), has_term=has_term, note=note))
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_ch = sum(r["changed"] for r in rows)
    unch = [r["max_abs_diff"] for r in rows if not r["changed"]]
    bad = [(r["chain"], r["draw"]) for r in rows if r["changed"] != r["has_term"]]
    print(f"draws checked {len(rows)} ({len(a.chains)} chains) | changed {n_ch} | has_term {sum(r['has_term'] for r in rows)} | "
          f"unchanged {len(unch)} (max diff {max(unch) if unch else float('nan'):.2e}) | changed!=has_term: {len(bad)} {bad[:10]}")
    assert len(rows) == 64 * len(a.chains), "expected 64 plans per chain"
    assert not bad, "scan disagrees with the plans' terminal inserts"
    for c in a.chains:
        print(c, sum(r["changed"] for r in rows if r["chain"] == c), "changed of", sum(r["chain"] == c for r in rows))


if __name__ == "__main__":
    main()

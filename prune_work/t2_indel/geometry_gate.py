"""Geometry gate (user 10-09: leave the geometry as is; anything within 0.05 A is tolerated). A structure passes when each per-structure MEDIAN backbone bond length (C-N, N-CA, CA-C)
lies inside [native min - tol, native max + tol], the native envelope being the min/max over the native rows of the same table (geom_pool metrics, pool_dssp-style csv from geom_metrics.metrics).
Angles, omega, phi, Ramachandran and clash metrics are reported (share of structures inside the plain native min-max) but NOT gated: no tolerance was given for them.
Run: python geometry_gate.py geom_pools.csv [--tol 0.05] [--out gate.csv]"""
import argparse

import pandas as pd

BONDS = ["cn_med", "nca_med", "cac_med"]
REPORT = ["ncac_med", "cnca_med", "omega_dev_gt30", "cis_frac", "phi_pos_frac", "rama_nll_mean", "clash_per100"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--out")
    a = p.parse_args()
    g = pd.read_csv(a.csv)
    nat = g[g.set == "native"]
    assert len(nat) > 0, "no native rows"
    gate = pd.Series(True, index=g.index)
    for m in BONDS:
        gate &= g[m].between(nat[m].min() - a.tol, nat[m].max() + a.tol)
    g["bond_gate"] = gate
    for m in REPORT:
        g["in_" + m] = g[m].between(nat[m].min(), nat[m].max())
    t = g.groupby("set").agg(n=("bond_gate", "size"), bond_gate=("bond_gate", "mean"), **{m: ("in_" + m, "mean") for m in REPORT})
    print(f"native bond envelopes (+-{a.tol} A): " + ", ".join(f"{m} [{nat[m].min() - a.tol:.3f}, {nat[m].max() + a.tol:.3f}]" for m in BONDS))
    print(t.round(3).to_string())
    if a.out:
        g.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

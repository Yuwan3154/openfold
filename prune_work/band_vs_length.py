"""Does chain length shift the rewind->TM curve, and what rewind range hits the 0.3-0.9 band?

Tests the user's hypothesis (2026-08-14): longer = harder to model, so at a given rewind a longer
chain should land LOWER in TM, meaning the top of the ladder overshoots below 0.3 and should come
down for the next generation pass.

Reads the index sample written by build_template_index.py.
"""

import sys

import numpy as np

LO, HI = 0.3, 0.9

z = np.load(sys.argv[1] if len(sys.argv) > 1 else
            "/home/gridsan/cou/pp1c_work/template_index_sample/index_all.npz", allow_pickle=False)
tm, rw, L = z["tm"], z["rewind"], z["length"]
n = len(L)
print(f"{n} chains, lengths {L.min()}-{L.max()}, {tm.shape[1]} templates each\n")

in_band = (tm > LO) & (tm < HI)
print(f"IN-BAND ({LO} < TM < {HI}) templates per chain: "
      f"median {int(np.median(in_band.sum(1)))}  mean {in_band.sum(1).mean():.1f}  "
      f"min {in_band.sum(1).min()}  max {in_band.sum(1).max()}")
print(f"chains with ZERO in-band: {(in_band.sum(1) == 0).sum()}")
print(f"wasted: TM >= {HI} {100*(tm >= HI).mean():.1f}%   TM <= {LO} {100*(tm <= LO).mean():.1f}%\n")

# length bins, roughly equal counts
edges = np.unique(np.percentile(L, [0, 25, 50, 75, 100])).astype(int)
print(f"{'length bin':>14} {'n':>5} {'in-band/64':>11} {'>=0.9':>7} {'<=0.3':>7} "
      f"{'rewind@0.9':>11} {'rewind@0.3':>11}")
for lo, hi in zip(edges, edges[1:]):
    m = (L >= lo) & (L <= hi if hi == edges[-1] else L < hi)
    if m.sum() < 3:
        continue
    t, r = tm[m], rw[m]
    # rewind at which the chain's TM curve crosses each band edge (interpolated per chain, then
    # averaged) -- rewind is ascending along the ladder and TM falls, so flip for np.interp
    cross = {}
    for edge in (HI, LO):
        vals = []
        for i in range(t.shape[0]):
            o = np.argsort(r[i])
            ti, ri = t[i][o][::-1], r[i][o][::-1]      # TM ascending
            if ti[0] <= edge <= ti[-1]:
                vals.append(np.interp(edge, ti, ri))
        cross[edge] = np.mean(vals) if vals else np.nan
    print(f"{lo:6d}-{hi:<7d} {m.sum():5d} {in_band[m].sum(1).mean():11.1f} "
          f"{100*(t >= HI).mean():6.1f}% {100*(t <= LO).mean():6.1f}% "
          f"{cross[HI]:11.0f} {cross[LO]:11.0f}")

# direct correlation: does a longer chain lose more TM at the SAME rewind?
print("\nmean TM at each rewind rung, by length bin:")
rungs = np.sort(rw[0])[[0, 15, 31, 47, 63]]
hdr = "  ".join(f"r{int(x):>4}" for x in rungs)
print(f"{'length bin':>14}  {hdr}")
for lo, hi in zip(edges, edges[1:]):
    m = (L >= lo) & (L <= hi if hi == edges[-1] else L < hi)
    if m.sum() < 3:
        continue
    row = []
    for rung in rungs:
        sel = rw[m] == rung
        row.append(tm[m][sel].mean() if sel.any() else np.nan)
    print(f"{lo:6d}-{hi:<7d}  " + "  ".join(f"{v:5.3f}" for v in row))

print(f"\ncorr(length, in-band count) = {np.corrcoef(L, in_band.sum(1))[0,1]:+.3f}")
print(f"corr(length, mean TM)       = {np.corrcoef(L, tm.mean(1))[0,1]:+.3f}")

"""Per-chain rewind-ladder START for the round-2 T2 generation pass.

User decision 2026-09-09: round 2 uses the ladder rule that scored best on the full range
(n=81,011, L=5-512) in next_ladder.py -- **each chain's own TM=0.9 crossing + 10, up to 375** --
55.0 in-band templates per 64 rungs at skew 0.99, against 34.1 / 1.37 for round 1's global 90-375.

The rule is reproduced here EXACTLY as next_ladder.py scored it, because that is what the 55.0/0.99
numbers describe:

    crossing_i = rewind at which chain i's own measured TM curve passes 0.9   (linear interp)
    crossing_i = 195.0                                       if it never crosses 0.9
    start_i    = clamp(crossing_i + 10, 90, 370)
    end_i      = 375                                         ⛔ never cut the top: 44% of chains
                                                             (35,439/81,011) never reach TM<0.3
                                                             even at 375, so for them the top of
                                                             the ladder is the only hard end.

⛔ No new measurement is needed and none is invented: the crossing comes from the round-1 index,
which already carries all 64 (rewind, TM) points per chain.

Consumed by generate_templates.py via --ladder-starts.
"""

from __future__ import annotations

import argparse

import numpy as np

NEVER_CROSSES = 195.0        # next_ladder.py's fallback for a chain with no TM=0.9 crossing
OFFSET = 10.0                # the "+10" of the winning rule
CLAMP_LO, CLAMP_HI = 90.0, 370.0
LADDER_TOP = 375


def crossings(tm: np.ndarray, rw: np.ndarray, edge: float) -> np.ndarray:
    """FIRST rewind at which each chain's TM drops below `edge`; NaN if it never does.

    ⛔⛔ "The TM=0.9 crossing" is AMBIGUOUS and next_ladder.py never said so. Measured 2026-09-09:
    **not one of the 82,733 chains has a monotone TM curve**, and **29,984 (36.24%) pop back ABOVE
    0.9 after first dropping below it** -- at high rewind the run-to-run spread exceeds the rung
    spacing, so the curve scatters. 1uue_A reads 0.30 0.88 0.35 0.24 ... 0.92 0.26 across rewind
    307-375. "First crossing" and "last crossing" are therefore different rules with different
    answers for over a third of the set.

    ⛔ The original used `np.interp(edge, ti, ri)` on that scattered curve. np.interp REQUIRES an
    increasing `xp` and silently returns nonsense otherwise. In aggregate it barely mattered
    (median |interp - scan| = 0.0 rewind units, in-band 55.1 vs 55.2), but at the tail it put 16
    chains' ladder START at 370, leaving them a FIVE-rewind-unit window -- 64 templates at one
    noise level -- because a single high-rewind flicker above 0.9 dragged the crossing to the top.

    ✅ USER DECISION 2026-09-09: first-crossing. Scored on next_ladder.py's own two criteria it is
    the only reading with no degenerate windows, at no cost:
        rule    in-band/64   zero-yield   skew   min window   windows <=10
        interp        55.1        0.37%   0.98            5             16
        last          55.2        0.01%   0.89            5             24
        FIRST         55.0        0.01%   1.05           66              0
    No interpolation over a non-monotone axis anywhere: find the first index below the edge, then
    interpolate only across that ONE bracketing pair, where monotonicity holds by construction.
    """
    n = tm.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        o = np.argsort(rw[i])
        r, t = rw[i][o], tm[i][o]                    # rewind ASCENDING
        below = np.flatnonzero(t <= edge)
        if not len(below):
            continue                                 # never leaves the easy regime -> fallback
        k = int(below[0])
        if k == 0:
            out[i] = r[0]                            # already below at the lowest rewind
        else:
            drop = max(t[k - 1] - t[k], 1e-9)
            out[i] = r[k - 1] + (t[k - 1] - edge) * (r[k] - r[k - 1]) / drop
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="index_all.npz (the FULL 64-rung round-1 index)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--edge", type=float, default=0.9, help="the TM the ladder should start at")
    a = ap.parse_args()

    z = np.load(a.index, allow_pickle=False)
    chains, tm, rw, L = z["chains"], z["tm"], z["rewind"].astype(float), z["length"]

    c = crossings(tm, rw, a.edge)
    n_never = int(np.isnan(c).sum())
    c = np.where(np.isnan(c), NEVER_CROSSES, c)
    start = np.clip(c + OFFSET, CLAMP_LO, CLAMP_HI)

    print(f"{len(chains)} chains; {n_never} ({100 * n_never / len(chains):.2f}%) never cross "
          f"TM={a.edge} and take the {NEVER_CROSSES:.0f} fallback")
    q = np.percentile(start, [1, 10, 25, 50, 75, 90, 99])
    print("start rewind   " + "  ".join(f"p{p}={v:.0f}" for p, v in
                                        zip([1, 10, 25, 50, 75, 90, 99], q)))
    print(f"clamped at the low end: {int((start <= CLAMP_LO).sum())};  "
          f"at the high end: {int((start >= CLAMP_HI).sum())}")
    print(f"window width (375 - start): min {LADDER_TOP - start.max():.0f}  "
          f"med {LADDER_TOP - np.median(start):.0f}  max {LADDER_TOP - start.min():.0f}")

    # start vs length, because the crossing moves ~40 rewind units across length quartiles
    edges = np.unique(np.percentile(L, [0, 25, 50, 75, 100])).astype(int)
    print(f"\n{'L bin':>12} {'n':>7} {'start p50':>10}")
    for lo, hi in zip(edges, edges[1:]):
        m = (L >= lo) & (L <= hi if hi == edges[-1] else L < hi)
        if m.sum() >= 3:
            print(f"{f'{lo}-{hi}':>12} {m.sum():>7} {np.median(start[m]):>10.0f}")

    np.savez(a.out, chains=chains, start=start.astype(np.int16),
             top=np.int16(LADDER_TOP), edge=np.float32(a.edge))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

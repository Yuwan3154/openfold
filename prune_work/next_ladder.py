"""What rewind window should the NEXT generation pass use?

`band_vs_length.py` reports the MEAN rewind at which each length bin crosses TM 0.9 and TM 0.3.
A mean is the wrong statistic for choosing a ladder: the ladder is one global setting applied to
every chain, so what matters is the SPREAD of the crossings across chains. A window at the mean
leaves roughly half the chains hanging off each end.

This prints the per-chain crossing distribution so the window can be chosen against an explicit
coverage target, and then scores candidate windows by simulating the resulting ladder.

⛔ Does NOT pick the window -- the coverage target is a user decision.
"""

import argparse

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("index")
p.add_argument("--lo", type=float, default=0.3)
p.add_argument("--hi", type=float, default=0.9)
p.add_argument("--n-rungs", type=int, default=64)
a = p.parse_args()

z = np.load(a.index, allow_pickle=False)
tm, rw, L = z["tm"], z["rewind"], z["length"]
n, k = tm.shape
print(f"{n} chains, lengths {L.min()}-{L.max()}, {k} rungs/chain\n")


def crossings(edge):
    """Rewind at which each chain's TM curve crosses `edge`, NaN if it never does."""
    out = np.full(n, np.nan)
    for i in range(n):
        o = np.argsort(rw[i])
        ti, ri = tm[i][o][::-1], rw[i][o][::-1]        # TM ascending with decreasing rewind
        if ti[0] <= edge <= ti[-1]:
            out[i] = np.interp(edge, ti, ri)
    return out


c_hi, c_lo = crossings(a.hi), crossings(a.lo)          # rewind where TM drops below 0.9 / 0.3
qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"{'crossing':<22} " + " ".join(f"p{q:<3d}" for q in qs) + "   n(defined)")
for name, c in ((f"TM={a.hi} (ladder START)", c_hi), (f"TM={a.lo} (ladder END)", c_lo)):
    v = c[~np.isnan(c)]
    print(f"{name:<22} " + " ".join(f"{x:4.0f}" for x in np.percentile(v, qs)) +
          f"   {len(v)}/{n}")
print(f"\nchains never reaching TM<{a.hi}: {int(np.isnan(c_hi).sum())}   "
      f"never reaching TM<{a.lo}: {int(np.isnan(c_lo).sum())} "
      f"(these stay in band at the ladder's current top)")

# does the window need to move with length?
print("\ncrossing vs length (quartile bins):")
edges = np.unique(np.percentile(L, [0, 25, 50, 75, 100])).astype(int)
print(f"{'length bin':>14} {'n':>5} {'start p10':>10} {'start p50':>10} "
      f"{'end p50':>9} {'end p90':>9}")
for lo, hi in zip(edges, edges[1:]):
    m = (L >= lo) & (L <= hi if hi == edges[-1] else L < hi)
    if m.sum() < 3:
        continue
    s, e = c_hi[m], c_lo[m]
    s, e = s[~np.isnan(s)], e[~np.isnan(e)]
    print(f"{lo:6d}-{hi:<7d} {m.sum():5d} "
          f"{np.percentile(s,10) if len(s) else np.nan:10.0f} "
          f"{np.percentile(s,50) if len(s) else np.nan:10.0f} "
          f"{np.percentile(e,50) if len(e) else np.nan:9.0f} "
          f"{np.percentile(e,90) if len(e) else np.nan:9.0f}")

# ---- score candidate windows -------------------------------------------------------------------
# For a window [r0, r1] the next pass would place n_rungs evenly; a chain's yield is how many of
# those rungs land in band, read off its OWN measured curve by interpolating TM at each rung.
# ⭐ Yield is NOT the only criterion. T2's rationale rests on the synthetic ladder supplying a
# roughly UNIFORM spread of difficulty inside the band (natural templates are easy-skewed, skew
# 3.21; synthetic measured 1.07) -- so a window is also scored on the shape it produces. A narrow
# high-yield window that piles every rung into one sub-band would win on count and lose the thing
# the templates are for.
print(f"\ncandidate windows, {a.n_rungs} evenly spaced rungs:")
print(f"{'window':>14} {'mean':>6} {'med':>4} {'p10':>4} {'min':>4} {'<4':>6} {'=0':>6} "
      f"  " + " ".join(f"{x:.1f}-{x+0.1:.1f}" for x in np.arange(a.lo, a.hi - 1e-9, 0.1)) +
      f" {'skew':>5}")
tm_at = []
for i in range(n):
    o = np.argsort(rw[i])
    tm_at.append((rw[i][o].astype(float), tm[i][o].astype(float)))
sub = np.arange(a.lo, a.hi + 1e-9, 0.1)
for r0, r1 in [(90, 375), (150, 375), (180, 340), (195, 375), (200, 330),
               (210, 350), (220, 320), (240, 340)]:
    rungs = np.linspace(r0, r1, a.n_rungs)
    interp = [np.interp(rungs, x, y) for x, y in tm_at]
    yields = np.array([int(((v > a.lo) & (v < a.hi)).sum()) for v in interp])
    pooled = np.concatenate([v[(v > a.lo) & (v < a.hi)] for v in interp])
    share = np.array([100 * ((pooled >= lo) & (pooled < hi)).mean()
                      for lo, hi in zip(sub, sub[1:])])
    skew = share[-1] / share[0] if share[0] > 0 else np.inf
    print(f"{r0:6d}-{r1:<7d} {yields.mean():6.1f} {np.median(yields):4.0f} "
          f"{np.percentile(yields,10):4.0f} {yields.min():4d} "
          f"{100*(yields < 4).mean():5.1f}% {100*(yields == 0).mean():5.1f}% "
          f"  " + " ".join(f"{s:6.1f}%" for s in share) + f" {skew:5.2f}")
print("⭐ skew = share of the EASIEST 0.1 sub-band / share of the HARDEST; 1.00 is uniform. "
      "Natural templates sit at 3.21, the current ladder's synthetic output at 1.07.")
print("\n⚠️ Interpolating each chain's OWN measured curve, so a candidate window is only as "
      "trustworthy as the ladder's coverage there; windows inside 90-375 are interpolation, "
      "nothing here extrapolates outside it.")

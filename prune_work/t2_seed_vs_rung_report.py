"""Turn the seed-vs-rung sweep into the numbers that describe a ladder's diversity.

⛔⛔ REWORKED 2026-09-14 FOR PER-CHAIN LADDERS (user: "Rework."). Round 1 used one SHARED rewind
ladder, so absolute rewind was a common axis and `rewind[0]` spoke for every chain. Round 2 uses a
per-chain FIRST-CROSSING ladder, so it does not, and the old script said so by dying:

    AssertionError: 81477 chains have a different rewind ROW than chain 0;
                    the separation axis is not shared

⭐ The assertion was RIGHT and the report was obsolete. Measured on the round-2 sweep (all 82,733
chains, job 5621210), every chain descends from a shared rung-0 rewind of 375 to its OWN
first-crossing endpoint, and the spread across chains widens monotonically with depth:

    rung      0    1    2    3    4    5    6    7
    spread    0   29   60   89  119  148  179  208     (max-min absolute rewind, across chains)

⇒ the RUNG INDEX is the only axis every chain shares, so every comparison here is made on it.
Absolute rewind is still reported, as a DISTRIBUTION per rung, so a reader can see what index k
means without it silently becoming the axis again.

⛔⛔ TWO CORRECTIONS FROM THE FIRST VERSION (2026-09-09), both still in force:

1. **A POOLED SEED FLOOR IS MEANINGLESS HERE.** The first run reported one number, "seed floor
   0.7317", over all rungs at once. Its own distribution gave it away: mean 0.7317 but median
   0.8275, p10 0.3077, p90 0.9950. Two samples at a shallow rung are all but identical (there is
   almost no noise to differ in); two at a deep rung are wildly different. Pooling them describes no
   rung that exists. ⇒ the floor is reported PER RUNG and the pooled figure is not reported at all.
   ([[feedback_check_modality_before_quoting_median]])

2. **MEAN PAIRWISE TM REWARDS CLUSTERING, so ranking splits by it is invalid.** It crowned
   "2 rungs x 32 seeds" because two tight clusters far apart give a low MEAN while each holds 32
   near-duplicates. ⇒ every split is reported with its NEAR-DUPLICATE SHARE beside the mean, and no
   single ranking is emitted.

⛔ Reports the measurements. Does NOT choose the split.
"""

from __future__ import annotations

import argparse
import glob
from math import comb

import numpy as np


def load(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no shard files matched {pattern!r}")
    parts = [np.load(f, allow_pickle=False) for f in files]
    parts = [p for p in parts if len(p["chains"])]
    n_pair = parts[0]["tm_pairs"].shape[1]
    for p in parts:
        assert p["tm_pairs"].shape[1] == n_pair, "shards disagree on pair count"
    out = {k: np.concatenate([p[k] for p in parts])
           for k in ("chains", "tm_pairs", "rewind", "length", "n_ca", "skipped")}
    out["pair_i"], out["pair_j"] = parts[0]["pair_i"], parts[0]["pair_j"]
    assert len(np.unique(out["chains"])) == len(out["chains"]), "a chain appears in two shards"
    return out


def rung_axes(z: dict, label: str):
    """Map template SLOTS onto the shared RUNG INDEX, the only axis a per-chain ladder shares."""
    tm, rw = z["tm_pairs"], z["rewind"]
    n_chain, n_slot = rw.shape

    # ⛔ Derive seeds-per-rung from the data. Hardcoding it would silently mis-group the moment a
    #    sweep changes its layout, and the grouping is what every number below rests on.
    blk = 1
    while blk < n_slot and rw[0, blk] == rw[0, 0]:
        blk += 1
    assert n_slot % blk == 0, f"{n_slot} slots is not a whole number of {blk}-seed rungs"
    n_rung = n_slot // blk
    r = rw.reshape(n_chain, n_rung, blk)
    assert (r == r[:, :, :1]).all(), "rewind is not constant within a rung block"
    assert (np.diff(r[:, :, 0], axis=1) < 0).all(), "rungs are not strictly decreasing in rewind"
    per_chain = bool((rw != rw[0]).any())

    print(f"\n{'=' * 78}\n{label}: {len(z['chains'])} chains, "
          f"L {z['length'].min()}-{z['length'].max()} (median {int(np.median(z['length']))}), "
          f"{len(z['skipped'])} skipped")
    print(f"  ladder: {n_rung} rungs x {blk} seeds = {n_slot} templates per chain")
    print(f"  ladder shape: {'PER-CHAIN (first-crossing)' if per_chain else 'SHARED across chains'}")
    print(f"\n  ABSOLUTE rewind per rung index -- a distribution, NOT an axis")
    print(f"  {'rung':>5} {'min':>6} {'p25':>6} {'med':>6} {'p75':>6} {'max':>6} {'spread':>7}")
    depth = r[:, :, 0]
    for k in range(n_rung):
        q = np.percentile(depth[:, k], [0, 25, 50, 75, 100]).astype(int)
        print(f"  {k:>5} {q[0]:>6} {q[1]:>6} {q[2]:>6} {q[3]:>6} {q[4]:>6} "
              f"{int(depth[:, k].max() - depth[:, k].min()):>7}")
    if per_chain:
        print("  ⛔ Chains do NOT share absolute rewind, so every comparison below is on rung INDEX.")

    slot_rung = np.arange(n_slot) // blk
    max_spread = int((depth.max(0) - depth.min(0)).max())
    return tm, slot_rung[z["pair_i"]], slot_rung[z["pair_j"]], n_rung, blk, max_spread


def seed_floor_per_rung(tm, r_i, r_j, n_rung):
    """⭐ THE number that decides seeds-per-rung, and it is strongly rung-dependent."""
    print("\n  ⭐ WHAT A SEED BUYS, PER RUNG (same rung, different noise draw)")
    print(f"  {'rung':>5} {'pairs/chain':>12} {'mean TM':>9} {'med':>7} {'p10':>7} {'p90':>7} "
          f"{'near-dup share':>15}")
    out = {}
    for k in range(n_rung):
        m = (r_i == k) & (r_j == k)
        if not m.any():
            continue
        v = tm[:, m]
        flat = v.reshape(-1)
        out[int(k)] = float(v.mean())
        print(f"  {k:>5} {int(m.sum()):>12} {v.mean():>9.4f} {np.median(flat):>7.4f} "
              f"{np.percentile(flat, 10):>7.4f} {np.percentile(flat, 90):>7.4f} "
              f"{100 * (flat > 0.9).mean():>14.1f}%")
    if out:
        print("  ⛔ These differ by a wide margin across rungs, so there is no single 'seed floor'.")
    return out


def rung_curve(tm, r_i, r_j, label):
    """A(delta) with delta in RUNG INDEX units.

    ⭐ No absolute-rewind window is applied, and that is not an omission: the first-crossing ladder
    already places every chain inside its OWN band, which is the entire point of building it that
    way. Re-imposing one shared rewind window would throw away exactly the per-chain adaptation.
    """
    sep = np.abs(r_i - r_j)
    print(f"\n  {label}")
    print(f"  {'delta':>7} {'pairs/chain':>12} {'mean TM':>9} {'near-dup share':>15}")
    xs, ys = [], []
    for s in np.unique(sep[sep > 0]):
        m = sep == s
        v = tm[:, m]
        xs.append(float(s))
        ys.append(float(v.mean()))
        print(f"  {int(s):>7} {int(m.sum()):>12} {v.mean():>9.4f} "
              f"{100 * (v.reshape(-1) > 0.9).mean():>14.1f}%")
    # ⛔ np.interp is silently wrong on non-monotone xp; np.unique returns sorted, so this holds.
    xs = np.array(xs)
    assert (np.diff(xs) > 0).all(), "rung separations are not strictly increasing"
    return xs, np.array(ys)


def split_table(sep, tm_curve, floors, budget, n_rung, max_spread):
    """Per split: the mean, AND the near-duplicate share the mean hides. No single winner."""
    span = n_rung - 1
    print(f"\n{'=' * 78}\nCANDIDATE SPLITS of a {budget}-template budget over {n_rung} rung indices")
    print("⛔ NOT a ranking. Mean pairwise TM alone crowns the degenerate 2-rung split, because two")
    print("   tight clusters far apart average low while each holds many near-duplicates. Read the")
    print("   `same-rung share` column beside it: that is the fraction of the budget spent on pairs")
    print("   that only differ by a noise draw at ONE difficulty level.")
    print("⛔ Spacing is in RUNG INDEX units. On a per-chain ladder it cannot be quoted in absolute")
    # ⛔ MEASURED, never a literal: a 3-shard sample said 171 while the full 82,733 chains say 208.
    print("   rewind, because one index step is a different number of rewind units on every chain")
    print(f"   (measured spread across chains reaches {max_spread} rewind units).\n")
    floor = float(np.mean(list(floors.values()))) if floors else float(tm_curve[0])
    print(f"  seed floor averaged over all {len(floors)} rungs: {floor:.4f}"
          f"   (per-rung: {', '.join(f'{r}:{floors[r]:.3f}' for r in sorted(floors))})")
    print(f"\n  {'rungs':>6} {'seeds':>6} {'spacing':>8} {'same-rung share':>16} "
          f"{'mean TM':>9} {'clamped?':>9}")
    for R in sorted({r for r in range(2, budget + 1) if budget % r == 0}):
        S = budget // R
        spacing = span / (R - 1)
        n_same = R * comb(S, 2)
        num, den = n_same * floor, n_same
        for d in range(1, R):
            cnt = S * S * (R - d)
            num += cnt * float(np.interp(d * spacing, sep, tm_curve))
            den += cnt
        clamped = "YES" if spacing < sep.min() else ""
        print(f"  {R:>6} {S:>6} {spacing:>8.2f} {100 * n_same / den:>15.1f}% "
              f"{num / den:>9.4f} {clamped:>9}")
    print(f"\n  ⚠️ A(delta) is measured only for delta {sep.min():.0f}-{sep.max():.0f} rung indices; "
          f"a smaller spacing is CLAMPED to the nearest measured point, not extrapolated.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouped", required=True)
    ap.add_argument("--tiered", default=None)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--start", type=int, default=None,
                    help="round-1 only: absolute-rewind window floor. Ignored on a per-chain ladder.")
    ap.add_argument("--top", type=int, default=None,
                    help="round-1 only: absolute-rewind window top. Ignored on a per-chain ladder.")
    a = ap.parse_args()

    if a.start is not None or a.top is not None:
        print(f"⛔ --start/--top ({a.start}/{a.top}) describe an absolute-rewind window and are "
              f"IGNORED: on a per-chain first-crossing ladder there is no shared rewind axis to "
              f"window. Comparisons are on rung INDEX. Drop them from the launcher.")

    g = load(a.grouped)
    tm_g, gi, gj, n_rung, blk, max_spread = rung_axes(g, f"GROUPED  ({g['rewind'].shape[1]} templates/chain)")
    floors = seed_floor_per_rung(tm_g, gi, gj, n_rung)
    sep, curve = rung_curve(tm_g, gi, gj, "rung curve, grouped (delta in rung indices)")

    if a.tiered is not None:
        t = load(a.tiered)
        tm_t, ti, tj, n_rung_t, _, max_spread = rung_axes(t, "TIERED")
        sep, curve = rung_curve(tm_t, ti, tj, "rung curve, tiered (delta in rung indices)")
        n_rung = n_rung_t
        print("\n  ⛔ The tiered population may have no same-rung pairs, so its floor cannot be "
              "measured directly; the per-rung floors above come from the grouped population, which "
              "is a DIFFERENT length band and TM rises with L. Treat the split table as indicative "
              "of ORDERING, not as absolute TM.")

    split_table(sep, curve, floors, a.budget, n_rung, max_spread)


if __name__ == "__main__":
    main()

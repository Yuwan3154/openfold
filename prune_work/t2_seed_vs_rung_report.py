"""Turn the seed-vs-rung sweep into the numbers that choose a round-2 ladder split.

⛔⛔ TWO CORRECTIONS TO THE FIRST VERSION OF THIS SCRIPT (2026-09-09). Both were mine, both would
have produced a wrong recommendation, and both are instances of traps this project has already
recorded.

1. **A POOLED SEED FLOOR IS MEANINGLESS HERE.** The first run reported one number, "seed floor
   0.7317", over all 8 rungs at once. Its own distribution gave it away: mean 0.7317 but median
   0.8275, p10 0.3077, p90 0.9950. Two samples at rewind 90 are all but identical (there is almost
   no noise to differ in); two at rewind 375 are wildly different. Pooling them describes no rewind
   that exists, and round 2's window (~216-375) uses only the noisy end where seeds ARE diverse.
   ⇒ the floor is reported PER RUNG, and the pooled figure is not reported at all.
   ([[feedback_check_modality_before_quoting_median]])

2. **MEAN PAIRWISE TM REWARDS CLUSTERING, so ranking splits by it is invalid.** It crowned
   "2 rungs x 32 seeds" because two tight clusters 159 rewind units apart give a low MEAN while
   containing 32 near-duplicates each. That is precisely the failure `next_ladder.py` was written to
   avoid -- there, chasing in-band YIELD picked a window that destroyed the uniform difficulty
   spread the templates exist to supply. A pool of two clusters is not a diverse pool.
   ⇒ every split is reported with its NEAR-DUPLICATE SHARE beside the mean, so the artifact is
   visible instead of hidden inside an average, and no single ranking is emitted.

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


def pair_axes(z: dict, label: str):
    tm, rw = z["tm_pairs"], z["rewind"]
    ii, jj = z["pair_i"], z["pair_j"]
    bad = np.flatnonzero((rw != rw[0]).any(axis=1))
    assert len(bad) == 0, (
        f"{label}: {len(bad)} chains have a different rewind ROW than chain 0 "
        f"(e.g. {z['chains'][bad[0]]}); the separation axis is not shared"
    )
    r_i, r_j = rw[0][ii].astype(int), rw[0][jj].astype(int)
    print(f"\n{'=' * 78}\n{label}: {len(z['chains'])} chains, "
          f"L {z['length'].min()}-{z['length'].max()} (median {int(np.median(z['length']))}), "
          f"{len(z['skipped'])} skipped")
    return tm, r_i, r_j, np.unique(rw[0])[::-1]


def seed_floor_per_rung(tm, r_i, r_j, ladder):
    """⭐ THE number that decides seeds-per-rung, and it is strongly rewind-dependent."""
    print("\n  ⭐ WHAT A SEED BUYS, PER RUNG (same rewind, different noise draw)")
    print(f"  {'rewind':>7} {'pairs/chain':>12} {'mean TM':>9} {'med':>7} {'p10':>7} {'p90':>7} "
          f"{'near-dup share':>15}")
    out = {}
    for r in ladder:
        m = (r_i == r) & (r_j == r)
        if not m.any():
            continue
        v = tm[:, m]
        flat = v.reshape(-1)
        out[int(r)] = float(v.mean())
        print(f"  {int(r):>7} {int(m.sum()):>12} {v.mean():>9.4f} {np.median(flat):>7.4f} "
              f"{np.percentile(flat, 10):>7.4f} {np.percentile(flat, 90):>7.4f} "
              f"{100 * (flat > 0.9).mean():>14.1f}%")
    if out:
        print("  ⛔ These differ by a wide margin across rungs, so there is no single 'seed floor'. "
              "Read the rows inside the window round 2 will actually use.")
    return out


def rung_curve(tm, r_i, r_j, min_rewind, label):
    """A(delta), restricted to pairs whose BOTH members lie in the round-2 window."""
    keep = (r_i >= min_rewind) & (r_j >= min_rewind)
    sep = np.abs(r_i - r_j)
    print(f"\n  {label}: pairs with both members >= rewind {min_rewind} "
          f"({int(keep.sum())} of {len(sep)} per chain)")
    print(f"  {'delta':>7} {'pairs/chain':>12} {'mean TM':>9} {'near-dup share':>15}")
    pts = []
    for s in np.unique(sep[keep & (sep > 0)]):
        m = keep & (sep == s)
        v = tm[:, m]
        pts.append((float(s), float(v.mean())))
        if s in (np.unique(sep[keep & (sep > 0)])[:12]) or s % 40 < 5:
            print(f"  {int(s):>7} {int(m.sum()):>12} {v.mean():>9.4f} "
                  f"{100 * (v.reshape(-1) > 0.9).mean():>14.1f}%")
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def split_table(sep, tm_curve, floors, budget, start, top):
    """Per split: the mean, AND the near-duplicate share the mean hides. No single winner."""
    print(f"\n{'=' * 78}\nCANDIDATE SPLITS of a {budget}-template budget over {start}-{top}")
    print("⛔ NOT a ranking. Mean pairwise TM alone crowns the degenerate 2-rung split, because two")
    print("   tight clusters far apart average low while each holds 32 near-duplicates. Read the")
    print("   `same-rung share` column beside it: that is the fraction of the budget spent on pairs")
    print("   that only differ by a noise draw at ONE difficulty level.\n")
    # floor for the window: average the per-rung floors that fall inside it, so the number
    # describes the rewinds this split would actually sample
    inw = [v for r, v in floors.items() if r >= start]
    floor = float(np.mean(inw)) if inw else float(tm_curve[0])
    print(f"  seed floor averaged over the rungs inside {start}-{top}: {floor:.4f}"
          f"   (per-rung values: {', '.join(f'{r}:{floors[r]:.3f}' for r in sorted(floors, reverse=True) if r >= start)})")
    print(f"\n  {'rungs':>6} {'seeds':>6} {'spacing':>8} {'same-rung share':>16} "
          f"{'mean TM':>9} {'clamped?':>9}")
    for R in sorted({r for r in range(2, budget + 1) if budget % r == 0}):
        S = budget // R
        spacing = (top - start) / (R - 1)
        n_same = R * comb(S, 2)
        num, den = n_same * floor, n_same
        for d in range(1, R):
            cnt = S * S * (R - d)
            num += cnt * float(np.interp(d * spacing, sep, tm_curve))
            den += cnt
        clamped = "YES" if spacing < sep.min() else ""
        print(f"  {R:>6} {S:>6} {spacing:>8.1f} {100 * n_same / den:>15.1f}% "
              f"{num / den:>9.4f} {clamped:>9}")
    print(f"\n  ⚠️ A(delta) is measured only for delta {sep.min():.0f}-{sep.max():.0f}; a smaller "
          f"spacing is CLAMPED to the nearest measured point, not extrapolated.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouped", required=True)
    ap.add_argument("--tiered", default=None)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--start", type=int, required=True,
                    help="median per-chain ladder start; also the window floor for A(delta)")
    ap.add_argument("--top", type=int, default=375)
    a = ap.parse_args()

    g = load(a.grouped)
    tm_g, gi, gj, lad_g = pair_axes(g, "GROUPED  (L>300, 8 rungs x 8 seeds)")
    floors = seed_floor_per_rung(tm_g, gi, gj, lad_g)
    rung_curve(tm_g, gi, gj, a.start, "rung curve, grouped (resolves multiples of ~41 only)")

    if a.tiered is None:
        print("\n⚠️ --tiered not supplied: no fine-resolution rung curve, so no split table.")
        return
    t = load(a.tiered)
    tm_t, ti, tj, _ = pair_axes(t, "TIERED   (L<=300, 64 rungs x 1 seed)")
    sep, curve = rung_curve(tm_t, ti, tj, a.start,
                            "rung curve, tiered (~4.5-unit resolution)")
    print(f"\n  ⛔ The tiered population has NO same-rewind pairs, so its floor cannot be measured "
          f"directly; the per-rung floors above come from the grouped population, which is a "
          f"DIFFERENT length band (L>300 vs L<=300) and TM rises with L. Treat the split table as "
          f"indicative of ORDERING, not as absolute TM.")
    split_table(sep, curve, floors, a.budget, a.start, a.top)


if __name__ == "__main__":
    main()

"""Draw an indel plan for a chain of native length L (user spec 2026-10-01).

Per operation (insertion, deletion) independently:
  fraction ~ U(0.10, 0.30) of L; total T = round(fraction * L);
  k ~ U{1..5} segments; T is split into k positive integers uniformly (stars and bars);
  if T < k the plan degenerates to T segments of length 1.
Deletion segments are placed uniformly at random, non-overlapping, separated by >= 1 retained residue
(terminal deletions allowed). Insertion spots are drawn uniformly without replacement from the S+1 effective locations
(before the first survivor .. after the last survivor), S = retained residues, so no two coincide. Segment lengths are assigned to the sorted
placements in order, so the joint law over (lengths, positions) is uniform.
Seeding: default_rng([global_seed, crc32(chain), draw]) -- stable across processes (no builtin hash()).
"""
import zlib

import numpy as np

FRAC_LO, FRAC_HI = 0.10, 0.30
K_MAX = 5


def split_total(T, k, rng):
    if T < k:
        return [1] * T
    cuts = np.sort(rng.choice(np.arange(1, T), size=k - 1, replace=False)) if k > 1 else np.array([], int)
    return np.diff(np.concatenate([[0], cuts, [T]])).astype(int).tolist()


def place_deletions(L, lengths, rng):
    """Uniform over placements. Retained residues R = L - D fill k+1 gaps g_0..g_k; interior gaps carry
    one mandatory retained residue each, so the free residues (R - (k-1)) are split among the k+1 gaps
    by stars and bars (k bars among free+k slots)."""
    k, D = len(lengths), sum(lengths)
    free = (L - D) - (k - 1)
    assert free >= 0, "not enough retained residues to separate the segments"
    bars = np.sort(rng.choice(np.arange(free + k), size=k, replace=False)).tolist()
    edges = [-1] + bars + [free + k]
    g = [edges[i + 1] - edges[i] - 1 for i in range(k + 1)]
    assert sum(g) == free
    ops, pos = [], g[0]
    for i, ln in enumerate(lengths):
        ops.append(("del", pos, pos + ln - 1))
        pos += ln + 1 + (g[i + 1] if i + 1 < k else 0)
    return ops


def draw_plan(L, chain, draw, global_seed=0):
    rng = np.random.default_rng([global_seed, zlib.crc32(chain.encode()), draw])
    rec = {}
    for name in ("ins", "del"):
        frac = float(rng.uniform(FRAC_LO, FRAC_HI))
        T = int(round(frac * L))
        k = int(rng.integers(1, K_MAX + 1))
        segs = split_total(T, k, rng)
        rec[name] = {"frac": frac, "T": T, "k_drawn": k, "segments": segs}
    dels = place_deletions(L, rec["del"]["segments"], rng)
    dele = np.zeros(L, bool)
    for _, s, e in dels:
        dele[s:e + 1] = True
    surv = np.flatnonzero(~dele)
    n_ins = len(rec["ins"]["segments"])
    locs = sorted(rng.choice(len(surv) + 1, size=n_ins, replace=False).tolist())   # 0..S, uniform, distinct
    ins = [("ins", -1 if loc == 0 else int(surv[loc - 1]), int(k)) for loc, k in zip(locs, rec["ins"]["segments"])]
    rec["ops"] = [list(o) for o in dels + ins]
    rec["chain"], rec["draw"], rec["L"], rec["global_seed"] = chain, draw, L, global_seed
    return rec

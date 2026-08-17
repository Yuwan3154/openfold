"""Prove the pruned template tree is REDUNDANCY-ONLY: identical features, fewer bytes.

The static argument is that `SyntheticTemplatePool.sample_features` is the only training-path
consumer of the npz coords and it only ever picks from `eligible` (the TM band), so rungs outside
the band are unreachable. This is the empirical half: for real chains, build features from the
ORIGINAL 64-rung tree and from the PRUNED tree with the SAME rng, and require every array to be
bit-identical.

⭐ Same rng is the whole trick: `eligible` is identical in both indexes (the pruned index keeps the
full rectangular `tm`), so `rng.choice` draws the same rungs, and any difference in the output can
only come from the slot translation being wrong.
"""

import argparse
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--index", required=True, help="original index_all.npz")
p.add_argument("--pruned-index", required=True)
p.add_argument("--src-root", required=True)
p.add_argument("--dst-root", required=True)
p.add_argument("--n-chains", type=int, default=200)
p.add_argument("--n-sample", type=int, default=4)
p.add_argument("--min-tm", type=float, default=0.3)
p.add_argument("--max-tm", type=float, default=0.9)
a = p.parse_args()

from openfold.data.synthetic_templates import SyntheticTemplatePool

orig = SyntheticTemplatePool(a.index, a.src_root, min_tm=a.min_tm, max_tm=a.max_tm)
pruned = SyntheticTemplatePool(a.pruned_index, a.dst_root, min_tm=a.min_tm, max_tm=a.max_tm)
assert pruned.slot is not None, "pruned index has no `slot` array -- wrong file?"
assert orig.slot is None, "original index should not carry `slot`"

# the eligible sets must be identical, or the two are not comparable at all
assert len(orig.eligible) == len(pruned.eligible)
mism = [i for i in range(len(orig.eligible))
        if not np.array_equal(orig.eligible[i], pruned.eligible[i])]
assert not mism, f"eligible sets differ for {len(mism)} chains, e.g. row {mism[0]}"
print(f"✅ eligible sets identical across all {len(orig.eligible)} chains")

# only chains whose pruned npz actually exists, so this works on a PILOT prune of a subset as well
# as on the full tree -- the gate has to be runnable before committing to the full 103 GB pass
eligible_chains = [c for c in orig.row_of if c in orig and pruned.npz_path(c).is_file()]
assert eligible_chains, f"no pruned npz found under {a.dst_root}"
print(f"{len(eligible_chains)} chains present in the pruned tree and in band")
rng = np.random.default_rng(0)
pick_chains = [eligible_chains[i] for i in
               rng.choice(len(eligible_chains), size=min(a.n_chains, len(eligible_chains)),
                          replace=False)]

n_bad = 0
n_tmpl = 0
for j, chain in enumerate(pick_chains):
    seed = int(np.frombuffer(chain.encode().ljust(4, b"\0")[:4], dtype=np.uint32)[0])
    fo = orig.sample_features(chain, a.n_sample, np.random.default_rng(seed))
    fp = pruned.sample_features(chain, a.n_sample, np.random.default_rng(seed))
    assert (fo is None) == (fp is None), chain
    if fo is None:
        continue
    n_tmpl += fo["template_all_atom_positions"].shape[0]
    assert set(fo) == set(fp), (chain, set(fo) ^ set(fp))
    for k in fo:
        x, y = np.asarray(fo[k]), np.asarray(fp[k])
        if x.shape != y.shape or not np.array_equal(x, y):
            n_bad += 1
            print(f"  ⛔ {chain}: key {k} differs (shapes {x.shape} vs {y.shape})")
            break

print(f"\ncompared {len(pick_chains)} chains / {n_tmpl} sampled templates: "
      f"{n_bad} mismatches")

# the band the tree was actually pruned to must be recorded, so the widening guard has something to
# check (the guard firing is pinned by test_pruned_index_refuses_a_wider_band)
z = np.load(a.pruned_index, allow_pickle=False)
print(f"pruned index records its band as TM {float(z['min_tm'])}-{float(z['max_tm'])}"
      f" (requested {a.min_tm}-{a.max_tm})")
# float32 storage means exact float64 equality does not hold -- compare in the stored precision
assert np.float32(z["min_tm"]) == np.float32(a.min_tm)
assert np.float32(z["max_tm"]) == np.float32(a.max_tm)

# every retained row must be reachable, and nothing beyond
slot = z["slot"]
band = (z["tm"] > a.min_tm) & (z["tm"] < a.max_tm)
assert np.array_equal(slot >= 0, band), "slot>=0 disagrees with the band it claims to encode"
per_chain_rows = (slot >= 0).sum(1)
# each chain's retained slots must be exactly 0..k-1 in ascending rung order, since the npz rows
# were written in that order
for i in range(slot.shape[0]):
    kept = slot[i][slot[i] >= 0]
    assert np.array_equal(kept, np.arange(len(kept))), f"row {i}: slots not 0..k-1 ({kept[:8]})"
print(f"✅ slot>=0 matches the band exactly; retained rows/chain median "
      f"{int(np.median(per_chain_rows))}, total {int(per_chain_rows.sum()):,} of {slot.size:,} "
      f"({100*per_chain_rows.sum()/slot.size:.1f}%)")

print("\nRESULT:", "FAIL" if n_bad else "PASS")
sys.exit(1 if n_bad else 0)

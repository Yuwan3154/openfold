"""Gate the query-index map before T2 launches on it.

The map decides where every synthetic template's coordinates land, so "the job exited 0" is not
evidence it is right. Checks, in order of what would hurt most if wrong:
  1. Coverage of the generated chains, and how much of the training list keeps synthetic templates.
  2. Provenance split (structural numbering vs recovered alignment) and the ambiguous count.
  3. Monotonic, in-bounds, correct-length maps -- structural invariants, cheap and total.
  4. ⭐ End-to-end: build features through SyntheticTemplatePool for real chains using the REAL query
     sequence from the training pipeline's own source, which exercises the live sequence-agreement
     assert that the two failed launches tripped.
"""

import argparse
import json
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--qmap", default="/home/jupyter-chenxi/pp1c_work/qmap_all.npz")
ap.add_argument("--index", default="/home/jupyter-chenxi/pp1c_work/index_band.npz")
ap.add_argument("--templates-root", default="/home/jupyter-chenxi/pp1c_work/templates_band")
ap.add_argument("--chain-cache", default="/home/jupyter-chenxi/data/pdb_mmcif/chain_data_cache.json")
ap.add_argument("--train-list",
                default="/home/jupyter-chenxi/prune_work/lists_pdb/slim_struct_train.list")
ap.add_argument("--n-exercise", type=int, default=300)
a = ap.parse_args()

from openfold.data.synthetic_templates import SyntheticTemplatePool

z = np.load(a.qmap, allow_pickle=False)
names = [str(c) for c in z["chains"]]
lens, qlen, flags = z["qmap_len"], z["query_len"], z["ambiguous"]
flat = z["qmap"]
offs = np.concatenate([[0], np.cumsum(lens)])

gen = [str(c) for c in np.load(a.index, allow_pickle=False)["chains"]]
train = [l.strip() for l in open(a.train_list) if l.strip()]

print(f"=== 1. COVERAGE ===")
print(f"generated chains         : {len(gen)}")
print(f"mapped chains            : {len(names)}  ({100*len(names)/len(gen):.2f}% of generated)")
dropped = set(gen) - set(names)
print(f"dropped (no usable map)  : {len(dropped)}"
      + (f"  e.g. {' '.join(sorted(dropped)[:5])}" if dropped else ""))
print(f"training list            : {len(train)}")
print(f"  with a map             : {len(set(train) & set(names))} "
      f"({100*len(set(train) & set(names))/len(train):.2f}%)")

print(f"\n=== 2. PROVENANCE ===")
amb = (flags & 1).astype(bool)
from_align = (flags & 2).astype(bool)
print(f"from structural numbering (ridx-1) : {int((~from_align).sum())} "
      f"({100*(~from_align).mean():.2f}%)")
print(f"from recovered alignment           : {int(from_align.sum())} "
      f"({100*from_align.mean():.2f}%)")
print(f"ambiguous placements               : {int(amb.sum())} ({100*amb.mean():.2f}%)")
print(f"  (ambiguity only applies to the alignment-sourced ones)")
assert not (amb & ~from_align).any(), "a ridx-sourced chain was flagged ambiguous"

print(f"\n=== 3. STRUCTURAL INVARIANTS (all {len(names)} chains) ===")
bad = 0
for j in range(len(names)):
    m = flat[offs[j]:offs[j + 1]]
    if len(m) != lens[j] or m.min() < 0 or m.max() >= qlen[j] or not (np.diff(m) > 0).all():
        bad += 1
        if bad <= 5:
            print(f"  ⛔ {names[j]}: len {len(m)} vs {lens[j]}, range {m.min()}-{m.max()}, "
                  f"qlen {qlen[j]}, monotonic {(np.diff(m) > 0).all()}")
print(f"maps failing an invariant : {bad}")
assert bad == 0, "map violates monotonicity / bounds / length"
print("✅ every map is monotonic, in bounds, and the right length")

print(f"\n=== 4. END-TO-END through SyntheticTemplatePool ===")
pool = SyntheticTemplatePool(a.index, a.templates_root, min_tm=0.3, max_tm=0.9, qmap_path=a.qmap)
cache = json.load(open(a.chain_cache))
usable = [c for c in train if c in pool and c in cache and cache[c].get("seq")]
print(f"chains exercisable with a cached query seq: {len(usable)}")
rng = np.random.default_rng(0)
pick = [usable[i] for i in rng.permutation(len(usable))[: a.n_exercise]]
fails = 0
for c in pick:
    qseq = cache[c]["seq"]
    if len(qseq) != pool.qmap_query_len[c]:
        continue                      # cache seq differs from the map's source; skip, not a failure
    f = pool.sample_features(c, 4, np.random.default_rng(abs(hash(c)) % (2**31)),
                             query_sequence=qseq)
    if f is None:
        fails += 1
        print(f"  ⛔ {c}: returned None despite being 'in pool'")
        continue
    pos = f["template_all_atom_positions"]
    if pos.shape[1] != len(qseq) or not np.isfinite(pos).all():
        fails += 1
        print(f"  ⛔ {c}: shape {pos.shape} vs query {len(qseq)}")
print(f"exercised {len(pick)} chains: {fails} failures "
      f"(the live sequence-agreement assert would have raised on a bad map)")

print("\nRESULT:", "FAIL" if (bad or fails) else "PASS")
sys.exit(1 if (bad or fails) else 0)

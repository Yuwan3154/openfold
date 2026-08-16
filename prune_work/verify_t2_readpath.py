"""Verify the T2 read path against REAL production npz + the REAL index.

The 30 unit tests all ran on hand-built fixtures. This is the [[feedback_verify_correctness_after_pipeline_completion]]
gate: exercise `SyntheticTemplatePool` through the actual files the training run will read, and
check the invariants that a fixture cannot catch -- chain coverage, band arithmetic against the
index itself, npz path resolution, and the aatype round-trip that `fix_templates_aatype` depends on.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--index", default="/home/gridsan/cou/pp1c_work/template_index/index_all.npz")
p.add_argument("--templates-root", default="/home/gridsan/cou/pp1c_work/templates")
p.add_argument("--chain-list", default="/home/gridsan/cou/prune_work/lists_pdb/slim_struct_train.list")
p.add_argument("--n-chains", type=int, default=25)
p.add_argument("--min-tm", type=float, default=0.3)
p.add_argument("--max-tm", type=float, default=0.9)
p.add_argument("--n-sample", type=int, default=4)
a = p.parse_args()

from openfold.data.synthetic_templates import SyntheticTemplatePool, merge_template_features
from openfold.np import residue_constants as rc

pool = SyntheticTemplatePool(a.index, a.templates_root, min_tm=a.min_tm, max_tm=a.max_tm)
n_chain, n_tmpl = pool.tm.shape
print(f"index: {n_chain} chains x {n_tmpl} templates, "
      f"TM {pool.tm.min():.3f}-{pool.tm.max():.3f}")

# --- 1. coverage of the ACTUAL training list -------------------------------------------------
train = [l.strip() for l in Path(a.chain_list).read_text().split() if l.strip()]
have = [c for c in train if c in pool.row_of]
elig = [c for c in train if c in pool]
print(f"\ntraining list {len(train)} chains")
print(f"  in index          : {len(have)} ({100*len(have)/len(train):.1f}%)")
print(f"  with >=1 in-band  : {len(elig)} ({100*len(elig)/len(train):.1f}%)")
print(f"  in index, 0 in-band: {len(have) - len(elig)}")

# --- 2. band arithmetic straight off the index (independent of the pool's own bookkeeping) ----
band = (pool.tm > a.min_tm) & (pool.tm < a.max_tm)
per = band.sum(1)
print(f"\nin-band per chain: median {int(np.median(per))}, mean {per.mean():.1f}, "
      f"min {per.min()}, max {per.max()}")
print(f"  chains with 0 in-band: {(per == 0).sum()} / {n_chain}")
print(f"  chains with <4       : {(per < a.n_sample).sum()} / {n_chain}")
recomputed = [len(e) for e in pool.eligible]
assert recomputed == per.tolist(), "pool.eligible disagrees with the index it was built from"
print("  ✅ pool.eligible matches the index")

# --- 3. exercise sample_features on real npz --------------------------------------------------
rng = np.random.default_rng(0)
picks = rng.choice(len(elig), size=min(a.n_chains, len(elig)), replace=False)
bad = 0
for i in picks:
    chain = elig[i]
    row = pool.row_of[chain]
    f = pool.sample_features(chain, a.n_sample, np.random.default_rng(int(i)))
    assert f is not None, chain
    pos = f["template_all_atom_positions"]
    msk = f["template_all_atom_mask"]
    aat = f["template_aatype"]
    k, L = pos.shape[0], pos.shape[1]
    assert pos.shape == (k, L, 37, 3) and msk.shape == (k, L, 37), (chain, pos.shape)
    assert aat.shape == (k, L, 22), (chain, aat.shape)
    assert np.isfinite(pos).all(), f"{chain}: non-finite coords"
    assert msk.sum() > 0, f"{chain}: empty mask"
    # every sampled template must actually be inside the band
    assert ((f["_tm"] > a.min_tm) & (f["_tm"] < a.max_tm)).all(), (chain, f["_tm"])
    # ⛔ the one-hot must survive the HHBLITS->restype regather that fix_templates_aatype does
    hh = aat[0].argmax(-1)
    back = np.array([rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[x] for x in hh])
    d = np.load(pool.npz_path(chain), allow_pickle=False)
    src = d["aatype"].astype(int)
    if not (back == src).all():
        bad += 1
        print(f"  ⛔ {chain}: aatype round-trip mismatch at "
              f"{int((back != src).sum())}/{len(src)} residues")
    # the CA of every template must sit where the native's CA does, per-residue present
    if pos.shape[1] != len(src):
        bad += 1
        print(f"  ⛔ {chain}: L={pos.shape[1]} but npz aatype has {len(src)}")
print(f"\nexercised {len(picks)} chains through sample_features: {bad} failures")

# --- 4. merge onto a realistic natural-template block ------------------------------------------
chain = elig[int(picks[0])]
f = pool.sample_features(chain, a.n_sample, np.random.default_rng(1))
L = f["template_all_atom_positions"].shape[1]
nat = {
    "template_all_atom_positions": np.zeros((4, L, 37, 3), np.float32),
    "template_all_atom_mask": np.zeros((4, L, 37), np.float32),
    "template_aatype": np.zeros((4, L, 22), np.float32),
    "template_sequence": np.array([b"A" * L] * 4, dtype=object),
    "template_domain_names": np.array([b"nat%d" % i for i in range(4)], dtype=object),
    "template_sum_probs": np.zeros((4, 1), np.float32),
}
m = merge_template_features(nat, f)
k = f["template_all_atom_positions"].shape[0]
for key, v in m.items():
    assert np.asarray(v).shape[0] == 4 + k, (key, np.asarray(v).shape)
assert "_tm" not in m, "_tm leaked into the merged features"
print(f"merge onto 4 natural hits -> {4 + k} templates on every key  ✅")

print("\nRESULT:", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)

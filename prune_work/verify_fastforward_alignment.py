"""Prove the sampler fast-forward lands on the parent run's actual dataset position.

The parent's t4_pool index records, per rank, {chain, epoch, step, sample} for every datapoint it
trained on. That is ground truth. This builds a datamodule with the SAME seed and chain list,
replays N epochs of sampler draws, and checks the resulting chain set against the pool's record for
epoch N -- which pins the off-by-one instead of deriving it.

⛔ Runs CPU-only and touches no GPU. It does construct the real dataset, so it needs the chain data
cache and the mmcif dir, and takes a couple of minutes.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfold.config import model_config
from openfold.data.data_modules import OpenFoldDataModule

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True, help="parent run's t4_pool dir")
ap.add_argument("--epochs", type=int, nargs="+", required=True, help="epochs to check")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--train_epoch_len", type=int, default=3000)
ap.add_argument("--train_data_dir", required=True)
ap.add_argument("--train_alignment_dir", required=True)
ap.add_argument("--train_chain_list_path", required=True)
ap.add_argument("--template_mmcif_dir", required=True)
ap.add_argument("--max_template_date", default="2018-04-30")
ap.add_argument("--train_chain_data_cache_path", default=None)
ap.add_argument("--template_release_dates_cache_path", default=None)
args = ap.parse_args()

# --- ground truth: what the parent actually trained on, per epoch ----------------------
truth = defaultdict(set)
for f in sorted(glob.glob(os.path.join(args.pool, "rank*", "index.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        truth[r["epoch"]].add(r["chain"])
print(f"pool ground truth: epochs {min(truth)}..{max(truth)}; "
      f"chains/epoch = {sorted({len(v) for v in truth.values()})[:5]}")

config = model_config("finetuning_ptm", train=True, low_prec=True)
config.data.common.max_extra_msa = 1
config.data.common.max_msa_clusters = 1
config.data.train.max_extra_msa = 1
config.data.train.max_msa_clusters = 1


def chains_after_replaying(n):
    """Chain ids the sampler yields on the reroll immediately after replaying `n` epochs."""
    dm = OpenFoldDataModule(
        config=config.data,
        template_mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        train_data_dir=args.train_data_dir,
        train_alignment_dir=args.train_alignment_dir,
        train_chain_list_path=args.train_chain_list_path,
        train_chain_data_cache_path=args.train_chain_data_cache_path,
        template_release_dates_cache_path=args.template_release_dates_cache_path,
        train_epoch_len=args.train_epoch_len,
        batch_seed=args.seed,
        fastforward_epochs=n,
    )
    dm.setup()
    ds = dm.train_dataset
    ds.reroll()                                  # the reroll the first training epoch would do
    inner = ds.datasets[0]
    return {inner.idx_to_chain_id(int(i)) for _, i in ds.datapoints}


ok = True
for e in args.epochs:
    assert e in truth, f"pool has no record for epoch {e}"
    got = chains_after_replaying(e)
    want = truth[e]
    inter = got & want
    frac = len(inter) / max(1, len(want))
    verdict = "MATCH" if frac > 0.99 else "MISMATCH"
    ok &= frac > 0.99
    print(f"  replay {e:>3d} epochs -> drew {len(got)} chains; pool epoch {e} has {len(want)}; "
          f"overlap {len(inter)} ({100*frac:.2f}%)  [{verdict}]")
    if frac <= 0.99:                             # locate the true offset instead of just failing
        for cand in (e - 1, e + 1):
            if cand in truth:
                f2 = len(got & truth[cand]) / max(1, len(truth[cand]))
                print(f"      vs pool epoch {cand}: {100*f2:.2f}%")

print("\nFASTFORWARD ALIGNMENT " + ("VERIFIED" if ok else "FAILED"))
sys.exit(0 if ok else 1)

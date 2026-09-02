"""Preflight: prove the training dataset is the intended one BEFORE a run starts.

Run this on any new host before launching. It checks the things that fail SILENTLY:

  1. the chain list is actually applied (not discarded for os.listdir(alignment_dir))
  2. the resolved chain count equals the list's
  3. force_query_only_msa is on, so no a3m is ever opened (hhr-only deployments are then exact)
  4. every dependency the run dereferences per chain is present for a sample of chains
  5. the guard itself fires when the enhanced path is unavailable (negative control)

⛔ CPU-only; touches no GPU.
"""
import argparse
import os
import random
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
# The live launcher supplies this as PYTHONPATH; without it chain_list_path is silently ignored.
sys.path.insert(0, os.path.join(_REPO, "openfold"))

import openfold.data.data_modules as dm_mod
from openfold.config import model_config
from openfold.data.data_modules import OpenFoldDataModule

ap = argparse.ArgumentParser()
ap.add_argument("--train_data_dir", required=True)
ap.add_argument("--train_alignment_dir", required=True)
ap.add_argument("--train_chain_list_path", required=True)
ap.add_argument("--template_mmcif_dir", required=True)
ap.add_argument("--template_release_dates_cache_path", required=True)
ap.add_argument("--max_template_date", default="2018-04-30")
ap.add_argument("--train_epoch_len", type=int, default=3000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sample", type=int, default=200, help="chains to dereference on disk")
args = ap.parse_args()

config = model_config("finetuning_ptm", train=True, low_prec=True)
config.data.common.max_extra_msa = 1
config.data.common.max_msa_clusters = 1
config.data.train.max_extra_msa = 1
config.data.train.max_msa_clusters = 1

print(f"[0] ENHANCED_UTILS_AVAILABLE = {dm_mod.ENHANCED_UTILS_AVAILABLE}")
assert dm_mod.ENHANCED_UTILS_AVAILABLE, (
    f"FALSE -- chain_list_path would be ignored. Set PYTHONPATH={os.path.join(_REPO, 'openfold')}")


def build(force_query_only_msa=True):
    dm = OpenFoldDataModule(
        config=config.data,
        template_mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        train_data_dir=args.train_data_dir,
        train_alignment_dir=args.train_alignment_dir,
        train_chain_list_path=args.train_chain_list_path,
        template_release_dates_cache_path=args.template_release_dates_cache_path,
        train_epoch_len=args.train_epoch_len,
        batch_seed=args.seed,
        force_query_only_msa=force_query_only_msa,
    )
    dm.setup()
    return dm


with open(args.train_chain_list_path) as fh:
    listed = [l.strip() for l in fh if l.strip()]
print(f"[1] chain list: {len(listed)} entries")

dm = build()
inner = dm.train_dataset.datasets[0]
print(f"[2] dataset resolved {len(inner)} chains")
assert len(inner) == len(listed), f"MISMATCH: {len(inner)} vs {len(listed)}"

got = {inner.idx_to_chain_id(i) for i in range(len(inner))}
missing = set(listed) - got
extra = got - set(listed)
print(f"    identity vs the list: missing={len(missing)} extra={len(extra)}")
assert not missing and not extra, (sorted(missing)[:5], sorted(extra)[:5])

fqo = inner.data_pipeline.force_query_only_msa
print(f"[3] force_query_only_msa = {fqo}  ({'a3m never opened' if fqo else 'a3m IS read'})")
assert fqo, "a3m files WOULD be read, so an hhr-only deployment is NOT equivalent here"

# 4. dereference what the pipeline actually opens per chain, on a random sample
random.seed(args.seed)
sample = random.sample(listed, min(args.sample, len(listed)))
bad_struct, bad_hhr, bad_a3m_present = [], [], 0
for chain in sample:
    pdb_id = chain.split("_")[0]
    if not any(os.path.exists(os.path.join(args.train_data_dir, pdb_id + ext))
               for ext in (".cif", ".cif.gz", ".pdb")):
        bad_struct.append(chain)
    adir = os.path.join(args.train_alignment_dir, chain)
    if not os.path.exists(os.path.join(adir, "pdb70_hits.hhr")):
        bad_hhr.append(chain)
    if os.path.exists(os.path.join(adir, "uniref90_hits.a3m")):
        bad_a3m_present += 1
print(f"[4] sampled {len(sample)} chains: missing structure={len(bad_struct)} "
      f"missing pdb70_hits.hhr={len(bad_hhr)}  (a3m present for {bad_a3m_present}, irrelevant)")
if bad_struct:
    print(f"    e.g. missing structure: {bad_struct[:5]}")
if bad_hhr:
    print(f"    e.g. missing hhr: {bad_hhr[:5]}")
assert not bad_struct and not bad_hhr, "dependencies missing -- the run would crash per chain"

# 5. negative control: the guard must fire when the enhanced path is gone
dm_mod.ENHANCED_UTILS_AVAILABLE = False
try:
    build()
except RuntimeError as e:
    ok = "SILENTLY IGNORED" in str(e)
    print(f"[5] guard fires without the enhanced path: {ok}")
    assert ok, f"wrong error: {e}"
else:
    raise AssertionError("guard did NOT fire -- the silent-fallback bug is still reachable")
finally:
    dm_mod.ENHANCED_UTILS_AVAILABLE = True

print("\nPREFLIGHT PASSED -- the training dataset is the intended one")

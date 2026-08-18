"""What does the ExtraMSA track actually hold in the no-MSA recipe -- the query, or a real homolog?

⚠️ USER HYPOTHESIS BEING TESTED (2026-08-18): "the extra MSA track in our case simply uses the query
sequence itself." This prints the direct evidence rather than a derived statistic, because the previous
script's headline check turned out to be vacuous (it compared the row to the query and got False in both
conditions, since the a3m-absent row is constant PADDING which also is not the query).

Three things decide it, and all three are printed per chain:
  1. the row DECODED TO LETTERS beside the query -- HHBLITS ids, not restype ids, so it needs
     ID_TO_HHBLITS_AA; decoding with `restypes` would silently print the wrong amino acids
  2. its percent identity to the query over the crop
  3. `extra_msa_mask` -- ⭐ if the row is masked OUT the track is inert whatever it contains, which
     would make the hypothesis right in the only way that matters
"""

import argparse
import random

import numpy as np
import torch

from openfold.config import model_config
from openfold.data.data_modules import OpenFoldSingleDataset
from openfold.np import residue_constants as rc

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--aln-dir", required=True)
ap.add_argument("--chain-list", required=True)
ap.add_argument("--obsolete", required=True)
ap.add_argument("--kalign", required=True)
ap.add_argument("--template-cache", required=True)
ap.add_argument("--n-chains", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = model_config("finetuning_ptm", train=True, low_prec=True)
cfg.data.common.max_extra_msa = 1
cfg.data.common.max_msa_clusters = 1
cfg.data.train.max_extra_msa = 1
cfg.data.train.max_msa_clusters = 1
cfg.loss.masked_msa.weight = 0.0
cfg.data.train.crop_size = min(cfg.data.train.crop_size, 256)
print(f"max_msa_clusters={cfg.data.train.max_msa_clusters} "
      f"max_extra_msa={cfg.data.train.max_extra_msa} "
      f"extra_msa.enabled={cfg.model.extra_msa.enabled} "
      f"masked_msa.weight={cfg.loss.masked_msa.weight}")

ds = OpenFoldSingleDataset(
    data_dir=a.data_dir, alignment_dir=a.aln_dir, template_mmcif_dir=a.data_dir,
    max_template_date="2018-04-30", config=cfg.data, chain_data_cache_path=None,
    kalign_binary_path=a.kalign, max_template_hits=cfg.data.train.max_template_hits,
    shuffle_top_k_prefiltered=cfg.data.train.shuffle_top_k_prefiltered,
    template_release_dates_cache_path=a.template_cache,
    obsolete_pdbs_file_path=a.obsolete, mode="train", chain_list_path=a.chain_list,
)
chains = [l.strip() for l in open(a.chain_list) if l.strip()]
random.seed(a.seed)
pick = random.sample(chains, a.n_chains)
idx_of = {ds.idx_to_chain_id(i): i for i in range(len(ds))}

# ⛔⛔ THE MSA IS NOT HHBLITS-CODED BY THE TIME WE SEE IT. `input_pipeline.py:27` runs
# `correct_msa_restypes` as one of the FIRST transforms, gathering through
# MAP_HHBLITS_AATYPE_TO_OUR_AATYPE, so `true_msa`/`extra_msa` are in OUR restype order with X=20 and
# GAP=21 -- i.e. `restypes_with_x_and_gap`. Decoding them with ID_TO_HHBLITS_AA (as this script first
# did) prints a plausible-looking but WRONG sequence, and the tell is that the decoded `true_msa[0]`
# does not match `aatype` even though the cluster row IS the query. Asserted below rather than trusted.
MSA_ALPHA = np.array(rc.restypes_with_x_and_gap)          # 22 symbols: 20 aa + X + '-'
RT = np.array(rc.restypes + ["X"])


def de_msa(v):
    v = np.asarray(v, int)
    return "".join(MSA_ALPHA[np.clip(v, 0, len(MSA_ALPHA) - 1)])


def de_rt(v):
    v = np.asarray(v, int)
    return "".join(RT[np.clip(v, 0, len(RT) - 1)])


n_query_like, n_homolog, n_masked_out = 0, 0, 0
gaps = []
for c in pick:
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    f = ds[idx_of[c]]
    # ⛔ `msa` does NOT survive the feature pipeline -- it is consumed into `msa_feat` and `true_msa`.
    # `true_msa` is the pre-masking cluster MSA, still HHBLITS-coded, so it is the right comparison row.
    strip = lambda t: t[..., 0]                      # every feature carries a trailing recycling dim
    msa = strip(f["true_msa"])
    ex = strip(f["extra_msa"])
    exm = strip(f["extra_msa_mask"])
    exrm = strip(f["extra_msa_row_mask"])
    aat = strip(f["aatype"])

    q_rt = de_rt(aat.numpy())
    m0 = de_msa(msa[0].numpy())
    e0 = de_msa(ex[0].numpy())
    live = float(exm[0].float().mean())
    # identity of the extra row to the CLUSTER row (both HHBLITS-coded, so directly comparable)
    same = int((ex[0] == msa[0]).sum())
    ident = 100.0 * same / len(m0)
    # ⭐ the load-bearing sanity check: the cluster row must BE the query, decoded in the same
    # alphabet. If this fails the decode is wrong and nothing below can be trusted.
    gap = 100.0 * e0.count("-") / len(e0)
    print(f"\n=== {c} ===  true_msa {tuple(msa.shape)}  extra_msa {tuple(ex.shape)}")
    print(f"  cluster row == query?  {'YES' if m0 == q_rt else 'NO <-- DECODE IS WRONG'}")
    print(f"  extra_msa[0] gap fraction: {gap:.1f}%")
    print(f"  aatype (restype ids) : {q_rt[:80]}")
    print(f"  true_msa[0] (cluster): {m0[:80]}")
    print(f"  extra_msa[0]         : {e0[:80]}")
    print(f"  extra_msa[0] identity to true_msa[0]: {ident:.1f}%  ({same}/{len(m0)})")
    print(f"  extra_msa_mask[0] mean = {live:.3f}   extra_msa_row_mask[0] = {float(exrm[0]):.3f}"
          f"  -> row is {'LIVE (attended)' if live > 0 and float(exrm[0]) > 0 else 'MASKED OUT (inert)'}")
    print(f"  distinct symbols in extra_msa[0]: {sorted(set(e0))[:25]}")
    if live == 0 or float(exrm[0]) == 0:
        n_masked_out += 1
    elif ident > 95:
        n_query_like += 1
    else:
        n_homolog += 1
    gaps.append(gap)

print(f"\n================ VERDICT ================")
print(f"  rows masked out entirely (track inert) : {n_masked_out}/{len(pick)}")
print(f"  rows >95% identical to the query row   : {n_query_like}/{len(pick)}")
print(f"  rows that are a DIFFERENT sequence     : {n_homolog}/{len(pick)}")
print(f"  gap fraction of the extra row: median {np.median(gaps):.1f}%  "
      f"min {min(gaps):.1f}%  max {max(gaps):.1f}%")
print("  => the hypothesis 'extra MSA is just the query' holds only if the first two account for all.")
print("  => a HIGH gap fraction means the leaked row is mostly empty even when it is a real homolog.")

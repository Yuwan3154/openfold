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

# ⛔ extra_msa/msa hold HHBLITS-ordered ids; aatype holds restype-ordered ids. Two different alphabets.
HH = np.array([rc.ID_TO_HHBLITS_AA[i] for i in range(len(rc.ID_TO_HHBLITS_AA))])
RT = np.array(rc.restypes + ["X"])


def de_hh(v):
    v = np.asarray(v, int)
    return "".join(HH[np.clip(v, 0, len(HH) - 1)])


def de_rt(v):
    v = np.asarray(v, int)
    return "".join(RT[np.clip(v, 0, len(RT) - 1)])


n_query_like, n_homolog, n_masked_out = 0, 0, 0
for c in pick:
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    f = ds[idx_of[c]]
    take = lambda t: t[..., 0] if t.dim() > (2 if t.dim() > 2 else 1) else t
    msa = f["msa"][..., 0] if f["msa"].dim() > 2 else f["msa"]
    ex = f["extra_msa"][..., 0] if f["extra_msa"].dim() > 2 else f["extra_msa"]
    exm = f["extra_msa_mask"][..., 0] if f["extra_msa_mask"].dim() > 2 else f["extra_msa_mask"]
    aat = f["aatype"][..., 0] if f["aatype"].dim() > 1 else f["aatype"]

    q_rt = de_rt(aat.numpy())
    m0 = de_hh(msa[0].numpy())
    e0 = de_hh(ex[0].numpy())
    live = float(exm[0].float().mean())
    # identity of the extra row to the CLUSTER row (both HHBLITS-coded, so directly comparable)
    same = int((ex[0] == msa[0]).sum())
    ident = 100.0 * same / len(m0)
    print(f"\n=== {c} ===  msa {tuple(msa.shape)}  extra_msa {tuple(ex.shape)}")
    print(f"  aatype (restype ids) : {q_rt[:80]}")
    print(f"  msa[0]   (=cluster)  : {m0[:80]}")
    print(f"  extra_msa[0]         : {e0[:80]}")
    print(f"  extra_msa[0] identity to msa[0]: {ident:.1f}%  ({same}/{len(m0)})")
    print(f"  extra_msa_mask[0] mean = {live:.3f}  -> row is "
          f"{'LIVE (attended)' if live > 0 else 'MASKED OUT (inert)'}")
    print(f"  distinct symbols in extra_msa[0]: {sorted(set(e0))[:25]}")
    if live == 0:
        n_masked_out += 1
    elif ident > 95:
        n_query_like += 1
    else:
        n_homolog += 1

print(f"\n================ VERDICT ================")
print(f"  rows masked out entirely (track inert) : {n_masked_out}/{len(pick)}")
print(f"  rows >95% identical to the query row   : {n_query_like}/{len(pick)}")
print(f"  rows that are a DIFFERENT sequence     : {n_homolog}/{len(pick)}")
print("  => the hypothesis 'extra MSA is just the query' holds only if the first two account for all.")

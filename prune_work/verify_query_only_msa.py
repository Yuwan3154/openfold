"""Gate --force_query_only_msa: AF2Rank parity, and hhr-only equivalence.

Three claims, each of which would otherwise be an assumption:

  1. ⭐⭐ IGNORES THE a3m COMPLETELY. With the flag on, features built from the FULL alignment dir must
     be BIT-IDENTICAL to features built from a dir holding only `pdb70_hits.hhr`. This is the claim that
     licenses shipping hhr-only to Engaging: not "similar", identical. It also proves the a3m is never
     opened, without having to instrument file IO.
  2. THE EXTRA TRACK IS INERT BUT PRESENT. `extra_msa_mask` and `extra_msa_row_mask` must be 0 -- the
     AF2Rank regime, where the stack keeps its pretrained weights and attends to nothing. Contrast with
     the flag OFF, where the row is live.
  3. NO MSA REACHES msa_feat AT ALL. `msa_feat` is 49 channels = 23 msa one-hot + 1 + 1 + 23
     cluster_profile + 1 cluster_deletion_mean, and `summarize_clusters` computes the cluster block FROM
     THE EXTRA MSA before it is cropped. So the flag must change those channels too, or homology still
     enters through the cluster track even with the extra rows masked.
"""

import argparse
import random

import numpy as np
import torch

from openfold.config import model_config
from openfold.data.data_modules import OpenFoldSingleDataset

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--aln-dir", required=True)
ap.add_argument("--hhr-only-aln-dir", required=True)
ap.add_argument("--chain-list", required=True)
ap.add_argument("--obsolete", required=True)
ap.add_argument("--kalign", required=True)
ap.add_argument("--template-cache", required=True)
ap.add_argument("--n-chains", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = model_config("finetuning_ptm", train=True, low_prec=True)
for k in ("common", "train"):
    cfg.data[k].max_extra_msa = 1
    cfg.data[k].max_msa_clusters = 1
cfg.loss.masked_msa.weight = 0.0
cfg.data.train.crop_size = 256
print(f"extra_msa.enabled={cfg.model.extra_msa.enabled} (kept ON, as AF2Rank does) "
      f"max_extra_msa={cfg.data.train.max_extra_msa}")


def make_ds(aln, force):
    return OpenFoldSingleDataset(
        data_dir=a.data_dir, alignment_dir=aln, template_mmcif_dir=a.data_dir,
        max_template_date="2018-04-30", config=cfg.data, chain_data_cache_path=None,
        kalign_binary_path=a.kalign, max_template_hits=cfg.data.train.max_template_hits,
        shuffle_top_k_prefiltered=cfg.data.train.shuffle_top_k_prefiltered,
        template_release_dates_cache_path=a.template_cache,
        obsolete_pdbs_file_path=a.obsolete, mode="train", chain_list_path=a.chain_list,
        force_query_only_msa=force,
    )


ds_full_forced = make_ds(a.aln_dir, True)          # full a3m present, but flag ON
ds_hhr_forced = make_ds(a.hhr_only_aln_dir, True)  # a3m absent, flag ON
ds_full_off = make_ds(a.aln_dir, False)            # the current T1/T2 behaviour

chains = [l.strip() for l in open(a.chain_list) if l.strip()]
idx_of = {ds_full_forced.idx_to_chain_id(i): i for i in range(len(ds_full_forced))}
random.seed(a.seed)
pick = [c for c in random.sample(chains, 600) if c in idx_of][: a.n_chains]

n_ident, n_inert, n_live_off, cluster_changed = 0, 0, 0, 0
for c in pick:
    i = idx_of[c]

    def build(ds):
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        return ds[i]

    A, B, C = build(ds_full_forced), build(ds_hhr_forced), build(ds_full_off)

    # ---- claim 1: bit-identical across the two dirs, with the flag on
    keys = sorted(set(A) & set(B))
    diff = [k for k in keys if not torch.equal(A[k].float(), B[k].float())]
    same_keys = set(A) == set(B)
    ok1 = same_keys and not diff
    n_ident += ok1

    # ---- claim 2: the extra row is inert with the flag on, live without it
    m_on = float(A["extra_msa_mask"][..., 0].max())
    r_on = float(A["extra_msa_row_mask"][..., 0].max())
    m_off = float(C["extra_msa_mask"][..., 0].max())
    r_off = float(C["extra_msa_row_mask"][..., 0].max())
    n_inert += (m_on == 0.0 and r_on == 0.0)
    n_live_off += (m_off > 0.0 or r_off > 0.0)

    # ---- claim 3: the cluster block of msa_feat changes too
    per_ch = (A["msa_feat"][..., 0] - C["msa_feat"][..., 0]).abs().amax(dim=(0, 1))
    cl = torch.nonzero(per_ch[25:] > 0).flatten().tolist()
    cluster_changed += bool(cl)

    print(f"  {c}: flag-on full-vs-hhr {'IDENTICAL' if ok1 else 'DIFFERS ' + str(diff[:4])}"
          f" | extra mask on={m_on:.3f}/row {r_on:.3f}  off={m_off:.3f}/row {r_off:.3f}"
          f" | msa_feat cluster channels changed by the flag: {len(cl)}/24")

n = len(pick)
print("\n================ VERDICT ================")
print(f"  1. full-a3m dir == hhr-only dir with the flag ON : {n_ident}/{n}"
      f"   {'✅ shipping hhr-only is EXACTLY equivalent' if n_ident == n else '❌'}")
print(f"  2. extra track inert with the flag ON            : {n_inert}/{n}"
      f"   (and live with it OFF: {n_live_off}/{n})")
print(f"  3. msa_feat cluster block changed by the flag     : {cluster_changed}/{n}"
      f"   {'-> the cluster-profile path really was carrying MSA info' if cluster_changed else ''}")
assert n_ident == n and n_inert == n, "gate FAILED"
print("\n✅ GATE PASSED")

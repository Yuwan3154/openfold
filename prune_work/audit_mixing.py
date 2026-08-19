"""Does the count-matched mixing actually reproduce T1's delivered-template distribution?

⭐ This is the ONE claim the whole count-matched design rests on, and it has only ever been checked
analytically and in unit tests with synthetic pools. Here it is measured on the REAL dataset, through
the REAL feature pipeline: draw many examples and histogram how many templates the model is handed.

Reference (analytic, T1): with a pool of 4 and `templates_crop_start ~ U{0..pool}` INCLUSIVE,
delivered = min(pool - start, 4) => P(0)=1/5, and P(k)=1/5 for k=1..4 as well => mean 2.00.
Run B must match that, because the pool size is held at the natural count.
⛔ The exception, by design: chains with <4 PREFILTERED natural hits get topped up, which RAISES their
count. That is 1.04% of the list, so it should perturb the histogram only slightly.
"""

import argparse
import collections
import random

import numpy as np
import torch

from openfold.config import model_config
from openfold.data.data_modules import OpenFoldSingleDataset
from openfold.data.synthetic_templates import SyntheticTemplatePool

ap = argparse.ArgumentParser()
for f in ("data-dir", "aln-dir", "train-list", "obsolete", "kalign", "template-cache",
          "t2-index", "t2-root", "t2-qmap", "pref-counts"):
    ap.add_argument("--" + f, required=True)
ap.add_argument("--n", type=int, default=150)
a = ap.parse_args()

cfg = model_config("finetuning_ptm", train=True, low_prec=True)
for k in ("common", "train"):
    cfg.data[k].max_extra_msa = 1
    cfg.data[k].max_msa_clusters = 1
cfg.data.train.crop_size = 256

common = dict(data_dir=a.data_dir, template_mmcif_dir=a.data_dir, max_template_date="2018-04-30",
              config=cfg.data, chain_data_cache_path=None, kalign_binary_path=a.kalign,
              max_template_hits=cfg.data.train.max_template_hits,
              shuffle_top_k_prefiltered=cfg.data.train.shuffle_top_k_prefiltered,
              template_release_dates_cache_path=a.template_cache,
              obsolete_pdbs_file_path=a.obsolete, mode="train",
              chain_list_path=a.train_list, alignment_dir=a.aln_dir, force_query_only_msa=True)

pool = SyntheticTemplatePool(a.t2_index, a.t2_root, min_tm=0.3, max_tm=0.9, qmap_path=a.t2_qmap)
ds_match = OpenFoldSingleDataset(synthetic_template_pool=pool, n_synthetic_templates=0,
                                 t2_replace_prob=0.5, t2_topup_to=20,
                                 prefiltered_counts_path=a.pref_counts, **common)
ds_t1 = OpenFoldSingleDataset(**common)          # no pool at all == T1's template regime

random.seed(0)
idxs = random.sample(range(len(ds_match)), a.n)


def histogram(ds, label):
    h, syn = collections.Counter(), []
    for i in idxs:
        torch.manual_seed(i); np.random.seed(i)
        f = ds[i]
        tm_ = f["template_mask"][..., 0]
        k = int((tm_ > 0).sum())
        h[k] += 1
        names = f.get("template_domain_names")
        if names is not None and k:
            dn = names[..., 0] if hasattr(names, "dim") and names.dim() > 1 else names
            try:
                syn.append(sum(1 for x in dn[:k] if bytes(x).startswith(b"pp1c_")) / k)
            except Exception:
                pass
    n = sum(h.values())
    print(f"\n{label}  (n={n})")
    print("   delivered:  " + "  ".join(f"{k}:{100*h[k]/n:5.1f}%" for k in range(5)))
    mean = sum(k * h[k] for k in h) / n
    print(f"   mean = {mean:.3f}   P(0) = {100*h[0]/n:.1f}%")
    if syn:
        print(f"   synthetic share of delivered = {100*sum(syn)/len(syn):.1f}%")
    return h, mean


h1, m1 = histogram(ds_t1, "T1 regime (natural only, no pool)")
h2, m2 = histogram(ds_match, "Run B (count-matched: replace_prob 0.5 + topup 20)")
print("\nANALYTIC T1 reference: each of 0..4 at 20.0%, mean 2.000")
print(f"\nΔ mean (RunB - T1) = {m2 - m1:+.3f}")
print("⭐ they should agree to sampling noise; the topup tier (1.04% of chains) shifts it slightly up")

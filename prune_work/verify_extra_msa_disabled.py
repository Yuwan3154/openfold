"""Gate the extra-MSA-track removal. Four things, because three of them can fail silently.

⛔ Why a dedicated gate: "disabled" can mean four different degrees of disabled, and only one of them
is what was asked for.
  1. MODULES GONE, not just skipped. `model.py:117` gates construction on `extra_msa.enabled`, so the
     embedder and stack must be ABSENT from the module tree -- a skipped forward would leave
     parameters that receive no gradient, which under DDP raises "did not receive grad".
  2. THE JAX WARM-START STILL WORKS. `import_weights.py` reached `model.extra_msa_stack.blocks`
     unconditionally, so before the guard this raised AttributeError before loading a single weight.
     A warm-start that dies is obvious; one that silently loads nothing is not, so the loaded weights
     are compared against the npz.
  3. A FORWARD PASS RUNS. `model.py:386/791/826` branch on the same flag; if any site were missed the
     forward would KeyError on extra_msa features.
  4. ⛔⛔ THE RESIDUAL LEAK. `msa_feat` is 49 channels = 23 msa one-hot + 1 + 1 + 23 CLUSTER_PROFILE +
     1 cluster_deletion_mean, and the cluster profile is computed FROM THE EXTRA MSA by
     `summarize_clusters` (input_pipeline.py:115-117), BEFORE the extra MSA is cropped/deleted. So
     disabling the model's extra track does NOT by itself stop homology reaching the model. This
     measures how much still gets through.
"""

import argparse
import random

import numpy as np
import torch

from openfold.config import model_config
from openfold.data.data_modules import OpenFoldSingleDataset
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--aln-dir", required=True)
ap.add_argument("--hhr-only-aln-dir", required=True,
                help="a mirror of --aln-dir holding ONLY pdb70_hits.hhr, for the leak measurement")
ap.add_argument("--chain-list", required=True)
ap.add_argument("--obsolete", required=True)
ap.add_argument("--kalign", required=True)
ap.add_argument("--template-cache", required=True)
ap.add_argument("--jax-params", required=True)
ap.add_argument("--n-chains", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()


def single_seq_config(disable_extra):
    """Exactly what train_openfold.py's --enable_single_seq_mode branch produces."""
    c = model_config("finetuning_ptm", train=True, low_prec=True)
    c.data.common.max_extra_msa = 1
    c.data.common.max_msa_clusters = 1
    c.data.train.max_extra_msa = 1
    c.data.train.max_msa_clusters = 1
    if disable_extra:
        c.model.extra_msa.enabled = False
    c.loss.masked_msa.weight = 0.0
    c.data.train.crop_size = 256
    return c


print("=" * 88)
print("CHECK 1: the modules are ABSENT, not merely skipped")
cfg_off = single_seq_config(True)
cfg_on = single_seq_config(False)
m_off, m_on = AlphaFold(cfg_off.model), AlphaFold(cfg_on.model)
for name, m, want in (("extra_msa DISABLED", m_off, False), ("extra_msa ENABLED", m_on, True)):
    has_stack = hasattr(m, "extra_msa_stack")
    has_emb = hasattr(m, "extra_msa_embedder")
    n = sum(p.numel() for nm, p in m.named_parameters() if "extra_msa" in nm)
    print(f"  {name:20s} extra_msa_stack={has_stack} extra_msa_embedder={has_emb} "
          f"params_named_extra_msa={n:,}")
    assert has_stack is want and has_emb is want
assert sum(p.numel() for nm, p in m_off.named_parameters() if "extra_msa" in nm) == 0
_off_total = sum(p.numel() for p in m_off.parameters())
_on_total = sum(p.numel() for p in m_on.parameters())
print(f"  ✅ removed {_on_total - _off_total:,} parameters ({_on_total:,} -> {_off_total:,})")

print("=" * 88)
print("CHECK 2: the JAX warm-start still works with the track removed")
import_jax_weights_(m_off, a.jax_params, version="model_1_ptm")
npz = np.load(a.jax_params)
ref = npz["alphafold/alphafold_iteration/evoformer/preprocess_msa//weights"]
got = m_off.input_embedder.linear_msa_m.weight.detach().numpy()
assert got.shape == ref.T.shape, (got.shape, ref.T.shape)
assert np.allclose(got, ref.T, atol=1e-6), np.abs(got - ref.T).max()
print(f"  ✅ import_jax_weights_ returned, and preprocess_msa matches the npz "
      f"(max|d| = {np.abs(got - ref.T).max():.2e})")
print(f"  ✅ msa_feat input dim unchanged: linear_msa_m expects "
      f"{m_off.input_embedder.linear_msa_m.weight.shape[1]} channels")

print("=" * 88)
print("CHECK 3 + 4: a real batch -- forward runs, and how much MSA still reaches msa_feat")


def make_ds(cfg, aln):
    return OpenFoldSingleDataset(
        data_dir=a.data_dir, alignment_dir=aln, template_mmcif_dir=a.data_dir,
        max_template_date="2018-04-30", config=cfg.data, chain_data_cache_path=None,
        kalign_binary_path=a.kalign, max_template_hits=cfg.data.train.max_template_hits,
        shuffle_top_k_prefiltered=cfg.data.train.shuffle_top_k_prefiltered,
        template_release_dates_cache_path=a.template_cache,
        obsolete_pdbs_file_path=a.obsolete, mode="train", chain_list_path=a.chain_list,
    )


ds_full = make_ds(cfg_off, a.aln_dir)
ds_hhr = make_ds(cfg_off, a.hhr_only_aln_dir)
chains = [l.strip() for l in open(a.chain_list) if l.strip()]
random.seed(a.seed)
idx_of = {ds_full.idx_to_chain_id(i): i for i in range(len(ds_full))}
avail = [c for c in random.sample(chains, 400) if c in idx_of]

m_off.eval()
n_fwd = 0
leak_rows = []
for c in avail:
    if n_fwd >= a.n_chains:
        break
    i = idx_of[c]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    f_full = ds_full[i]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    f_hhr = ds_hhr[i]

    have_extra = [k for k in f_full if k.startswith("extra_msa")]
    mf_full, mf_hhr = f_full["msa_feat"][..., 0], f_hhr["msa_feat"][..., 0]
    # channel layout: [0:23] msa one-hot | 23 has_del | 24 del_value | [25:48] cluster_profile | 48 cdm
    per_ch = (mf_full - mf_hhr).abs().amax(dim=(0, 1))
    diff_ch = torch.nonzero(per_ch > 0).flatten().tolist()
    onehot_diff = [x for x in diff_ch if x < 25]
    cluster_diff = [x for x in diff_ch if x >= 25]
    leak_rows.append((c, len(onehot_diff), len(cluster_diff), float(per_ch[25:].max())))
    print(f"  {c}: msa_feat channels differing a3m-vs-hhr -> "
          f"query/deletion block(0-24): {len(onehot_diff)}   "
          f"CLUSTER block(25-48): {len(cluster_diff)}   max|d| in cluster block "
          f"{float(per_ch[25:].max()):.4f}")
    print(f"     extra_msa* keys still produced by the data pipeline: {have_extra}")

    if n_fwd == 0:
        batch = {k: v.unsqueeze(0) for k, v in f_full.items() if torch.is_tensor(v)}
        with torch.no_grad():
            out = m_off(batch)
        print(f"     ✅ forward ran with the track removed: "
              f"final_atom_positions {tuple(out['final_atom_positions'].shape)}")
    n_fwd += 1

n_leak = sum(1 for r in leak_rows if r[2] > 0)
print("=" * 88)
print(f"VERDICT: cluster-profile channels still differ with vs without the a3m in "
      f"{n_leak}/{len(leak_rows)} chains.")
print("  -> if n_leak > 0, disabling the model's extra-MSA track does NOT make the recipe MSA-free:")
print("     summarize_clusters computes cluster_profile from the EXTRA MSA before it is dropped, and")
print("     those 24 channels are concatenated into the 49-channel msa_feat the model consumes.")

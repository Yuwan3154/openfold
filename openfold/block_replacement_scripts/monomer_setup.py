"""Monomer (regular AF2, single-chain) confidence-hallucination setup for the gradient estimator.
Load vanilla AF2 model_1_ptm (import_jax_weights from params_model_1_ptm.npz), no templates.
Φ = ColabDesign hallucination loss (de novo, no target structure): plddt + pae + con (intra), reusing the
verified loss math from binder_setup. Single-chain features via H.make_feature_batch(is_multimer=False)."""
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
import jvp_sanity_check as J
import binder_setup as B

PARAMS = str(Path.home() / "params/params_model_1_ptm.npz")
MONO_WEIGHTS = dict(pae=0.4, con=1.0)  # plddt EXCLUDED (user): no structure module needed for Φ


def load_monomer_model(device, preset="model_1_ptm"):
    cfg = model_config(preset)
    cfg.model.template.enabled = False
    cfg.globals.chunk_size = None
    cfg.model.num_recycle = 1
    model = AlphaFold(cfg)
    import_jax_weights_(model, PARAMS, version=preset)
    model = model.to(device).float().eval()
    J.set_jvp_safe(model)
    model.evoformer.blocks_per_ckpt = 1  # ckpt for memory (single backward is reentrant-ckpt safe)
    if hasattr(model, "extra_msa_stack"):
        model.extra_msa_stack.ckpt = True
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def monomer_loss(out, L, residue_index, model, weights=None):
    weights = weights or MONO_WEIGHTS
    dgram = out["distogram_logits"] if "distogram_logits" in out else model.aux_heads.distogram(out["pair"])
    tm = out["tm_logits"] if "tm_logits" in out else model.aux_heads.tm(out["pair"])
    pae = B._pae(tm) / 31.0
    pae = (pae + pae.transpose(-1, -2)) / 2
    pae_loss = pae.mean()
    ones = torch.ones(L, device=dgram.device)
    con = B._con_loss(dgram, residue_index, B.CON_INTRA, mask_1d=ones, mask_1b=ones)
    terms = {"pae": pae_loss, "con": con}
    if weights.get("plddt", 0.0) != 0:  # only run the lddt head (needs structure module) if actually weighted
        lddt = out["lddt_logits"] if "lddt_logits" in out else model.aux_heads.plddt(out["single"])
        terms["plddt"] = (1.0 - B._plddt(lddt)).mean()
    total = sum(weights.get(k, 0.0) * v for k, v in terms.items())
    return total, {k: float(v) for k, v in terms.items() if weights.get(k, 0.0) != 0}

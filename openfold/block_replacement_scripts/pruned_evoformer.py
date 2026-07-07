"""Pruned-AF2 Evoformer as a gradient estimator (user's ceiling test). Wrap the real 48-block EvoformerStack
with a surrogate-compatible forward(m,z,msa_mask,pair_mask)->(m,z). Pruning = drop column attention
(no_column_attention flag) + triangle attention (replace pair_stack.tri_att_start/end with a no-op), keeping
row-attention-with-pair-bias, OuterProductMean, triangle MULTIPLICATION, and both transitions."""
import torch
import torch.nn as nn


class _NoOpZ(nn.Module):
    """Residual no-op: returns zeros shaped like the input pair tensor, so add(z, .) is identity."""
    def forward(self, z, *args, **kwargs):
        return torch.zeros_like(z)


def prune_blocks(evoformer):
    for b in evoformer.blocks:
        b.no_column_attention = True
        if hasattr(b, "msa_att_col"):
            del b.msa_att_col  # remove dead column-attn params (DDP find_unused_parameters, optimizer, ckpt)
        b.pair_stack.tri_att_start = _NoOpZ()
        b.pair_stack.tri_att_end = _NoOpZ()
    return evoformer


def freeze_all_except_evoformer(model):
    # Evoformer-only fine-tune: freeze structure module + heads, train only the kept Evoformer stack.
    for prm in model.parameters():
        prm.requires_grad_(False)
    for prm in model.evoformer.parameters():
        prm.requires_grad_(True)
    # The ESMFold2-inspired contractive pair update (openfold/model/contractive_recycling.py)
    # lives in model.recycling_embedder, NOT model.evoformer -- without this, its learnable
    # Delta/A/B parameters would be silently frozen (dead weights) whenever use_contractive=True
    # is combined with an evoformer-only freeze scheme.
    contractive = getattr(model.recycling_embedder, "contractive_pair_update", None)
    if contractive is not None:
        for prm in contractive.parameters():
            prm.requires_grad_(True)
    return model


def freeze_all_except_heads(model, train_distogram=False):
    # Confidence-head-only fine-tune (WS2): train only the AuxiliaryHeads confidence heads
    # (plddt, experimentally_resolved, tm=pTM/pAE); freeze Evoformer + structure module + embedders.
    keep = ("aux_heads.plddt", "aux_heads.experimentally_resolved", "aux_heads.tm")
    if train_distogram:
        keep = keep + ("aux_heads.distogram",)
    n_tr = n_all = 0
    for name, prm in model.named_parameters():
        n_all += prm.numel()
        if name.startswith(keep):
            prm.requires_grad_(True)
            n_tr += prm.numel()
        else:
            prm.requires_grad_(False)
    return n_tr, n_all


class RealEvoEstimator(nn.Module):
    def __init__(self, evoformer, prune=False):
        super().__init__()
        if prune:
            prune_blocks(evoformer)
        self.evo = evoformer

    def forward(self, m, z, msa_mask=None, pair_mask=None):
        out = self.evo(
            m, z, msa_mask=msa_mask, pair_mask=pair_mask, outputs={}, cycle_no=0,
            chunk_size=None, use_deepspeed_evo_attention=False, use_lma=False, use_flash=False,
            use_cuequivariance_attention=False, use_cuequivariance_multiplicative_update=False,
            inplace_safe=False, _mask_trans=False, use_torch_sdpa=False, use_torch_vanilla=False,
            use_torch_cueq=False,
        )
        return out[0], out[1]

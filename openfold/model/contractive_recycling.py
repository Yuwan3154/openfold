"""Prototype building blocks from ESMFold2's Recurrent Folding Layers (Appendix A.2.5 of
"Language Modeling Materializes a World Model of Protein Biology", Candido/Hayes/.../Rives 2026),
for inference-time recycle scaling on our own AF2-family models -- NOT wired into any training
pipeline yet, standalone and unit-testable first.

ContractivePairUpdate replaces OpenFold's plain-additive RecyclingEmbedder combination step
(z_new = z_fresh + Linear(distogram_bins(prev_x)), a standard residual update the source paper
explicitly identifies as unstable for large recycle counts) with a channel-wise linear-SSM-style
contractive recurrence (Prairie et al. 2026, "Parcae: Scaling Laws For Stable Looped Language
Models", arXiv:2604.12946 -- adapted by ESMFold2 from stabilizing looped LMs to stabilizing the
pair-representation recycling loop). This module only replaces the COMBINATION step; the existing
pair-folding stack itself (triangle multiplication + transition) is unchanged/reused as-is.

sample_gaussian_pair_init provides the independent, seed-varying initial recurrent pair state
ESMFold2 uses (z_0 ~ trunc_norm(0, 2/(5*d_pair)), +/-3 sigma) -- the mechanism for inference-time
sampling diversity that doesn't depend on MSA masking.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContractivePairUpdate(nn.Module):
    """z_{t+1}_input = Abar * z_t + Bbar * LayerNorm(u_t), eq. (2) of ESMFold2 Appendix A.2.5.

    Abar = exp(-delta * a), Bbar = delta * b -- delta, a, b are learnable, per-channel (shape
    [c_z]) scalars. delta = softplus(log_delta), a = exp(log_a) -- this asymmetric split (not
    softplus for both, not exp for both) matches the official Mamba/S6 reference implementation
    (state-spaces/mamba, mamba_simple.py: `dt = F.softplus(...)`, `A = -torch.exp(self.A_log)`),
    the exact lineage Parcae (arXiv:2604.12946, Sec 4.1) cites for this discretization scheme
    (refs [17,29] = Dao & Gu 2024, Gu & Dao 2023). ESMFold2/Parcae's own papers write both as
    plain "exp" in their compact notation, but neither paper's code is public; Mamba's IS public
    and is the literature both papers explicitly build on, so its code is the disambiguating
    source here. -delta*a < 0 always => Abar in (0,1) elementwise by construction (contractive),
    regardless of training dynamics -- this is the property that keeps the recurrent state's
    magnitude bounded across arbitrarily many loop iterations.
    """

    def __init__(self, c_z: int):
        super().__init__()
        self.c_z = c_z
        # log-parametrized so the raw learnable params are unconstrained.
        self.log_delta = nn.Parameter(torch.zeros(c_z))
        self.log_a = nn.Parameter(torch.zeros(c_z))
        self.b = nn.Parameter(torch.ones(c_z))
        self.layer_norm_u = nn.LayerNorm(c_z)

    def discretized_params(self):
        delta = F.softplus(self.log_delta)
        a = torch.exp(self.log_a)
        a_bar = torch.exp(-delta * a)
        b_bar = delta * self.b
        return a_bar, b_bar

    def forward(self, z_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """z_t, u_t: [..., c_z] (any leading batch/residue-pair dims). Returns the combined
        signal to feed into the pair-folding stack (triangle mult + transition), NOT the final
        post-pair-folding-layers representation."""
        a_bar, b_bar = self.discretized_params()
        return a_bar * z_t + b_bar * self.layer_norm_u(u_t)


def sample_gaussian_pair_init(shape, d_pair: int, device=None, dtype=None, generator=None):
    """z_0 ~ trunc_norm(0, 2/(5*d_pair)), +/-3 sigma truncation (ESMFold2 Appendix A.2.5).
    Independent of input features -- the seed-varying source of structural diversity."""
    std = math.sqrt(2.0 / (5.0 * d_pair))
    z0 = torch.empty(shape, device=device, dtype=dtype)
    with torch.no_grad():
        nn.init.trunc_normal_(z0, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)
    return z0

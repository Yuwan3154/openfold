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


def inv_softplus(y):
    """The inverse of F.softplus: log(exp(y) - 1). Used to solve for the raw `log_delta` that
    yields a WANTED delta, which is how a floor is added without moving delta."""
    return torch.log(torch.expm1(y))


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

    def __init__(self, c_z: int, per_position_delta: bool = False,
                 delta_floor: float = None):
        super().__init__()
        self.c_z = c_z
        self.per_position_delta = per_position_delta
        # log-parametrized so the raw learnable params are unconstrained.
        self.log_delta = nn.Parameter(torch.zeros(c_z))
        self.log_a = nn.Parameter(torch.zeros(c_z))
        # ⛔⛔ B is a FULL [c_z, c_z] matrix, matching the reference implementation
        # (esm/models/esmfold2/model.py: `parcae_b_cont = nn.Parameter(torch.eye(d_pair))`,
        # applied as `F.linear(LN(u), delta[:, None] * B)`). Identity init makes
        # `delta[:, None] * I == diag(delta)`, so step 0 is BIT-IDENTICAL to the previous
        # per-channel `b = ones(c_z)`; the two differ only as training moves B off-diagonal.
        # The paper's A.2.5 text calls B "channel-wise", which is what the vector version followed.
        self.b = nn.Parameter(torch.eye(c_z))
        self.layer_norm_u = nn.LayerNorm(c_z)

        if per_position_delta:
            assert delta_floor is not None, "per_position_delta needs an explicit delta_floor"
            # ⛔ The floor is a BUFFER, not a constant: it must travel inside the checkpoint so
            # _load_from_state_dict can tell which parameterization the stored log_delta was
            # trained under. Registered ONLY in this mode, so a flag-off state_dict keeps its exact
            # historical key set and still loads strict=True.
            self.register_buffer("delta_floor", torch.tensor(float(delta_floor)))
            self.delta_head = nn.Linear(c_z, 1)
            # Zero weight AND bias => s == 0 at init => delta reduces exactly to the per-channel
            # value, so flag-on-at-step-0 matches the flag-off path.
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
            # ⛔⛔ Re-solve log_delta for the floor. delta = floor + softplus(log_delta), so leaving
            # log_delta at 0 would start the run at floor + softplus(0) instead of softplus(0) --
            # a silent shift away from every previous run's initial delta.
            with torch.no_grad():
                self.log_delta.fill_(float(inv_softplus(
                    F.softplus(torch.zeros((), dtype=torch.float64))
                    - torch.tensor(delta_floor, dtype=torch.float64))))

    def discretized_params(self):
        """Per-channel (a_bar, b_bar). ⛔ Meaningless in per-position mode, where delta depends on
        z_t and cannot be reduced to a [c_z] vector -- reading it there would silently report a
        delta the model never uses."""
        assert not self.per_position_delta, (
            "discretized_params() is per-channel only; in per_position_delta mode delta is a "
            "function of z_t. Call per_position_delta_from_state(z_t) instead.")
        delta = F.softplus(self.log_delta)
        a = torch.exp(self.log_a)
        a_bar = torch.exp(-delta * a)
        b_bar = delta[:, None] * self.b
        return a_bar, b_bar

    def per_position_delta_from_state(self, z_t: torch.Tensor) -> torch.Tensor:
        """delta[i,j,c] = floor + softplus(log_delta[c] + symmetrize(Linear(z_t))[i,j]).

        Returns [..., L, L, c_z]. The head reads the RECYCLED state z_t, so "how much do I
        overwrite here" becomes a per-residue-pair decision instead of a constant.

        `s` is symmetrized across the two pair axes because the pair track carries approximate
        (i,j) <-> (j,i) symmetry -- the distogram head relies on it, doing z + z^T explicitly --
        and an asymmetric gate would break a symmetry the trunk maintains.

        The floor bounds the freeze failure mode: a per-position gate can otherwise drive delta -> 0
        somewhere (high confidence -> retain -> state unchanged -> still high confidence) and pin
        that region for the whole run. Since softplus > 0, delta > floor strictly, so
        a_bar = exp(-delta*a) < exp(-floor*a) < 1 everywhere.
        """
        s = self.delta_head(z_t)                            # [..., L, L, 1]
        s = 0.5 * (s + s.transpose(-3, -2))                 # the two pair axes, not the channel axis
        return self.delta_floor + F.softplus(self.log_delta + s)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Expand a pre-matrix checkpoint's per-channel `b` into `diag(b)`, and re-solve
        `log_delta` whenever the checkpoint's delta floor differs from this module's.

        Every checkpoint written before this change stores `b` with shape [c_z]. `diag(b)` is the
        exact matrix that reproduces the old elementwise behaviour, so the migration is lossless
        rather than an approximation. ⛔ It does NOT migrate optimizer state: resuming a run across
        this change needs a fresh optimizer state for this parameter, since its shape changed.
        """
        # ⛔⛔ FLOOR MIGRATION. Every pre-floor checkpoint stores log_delta under
        # delta = softplus(log_delta); loading it verbatim into a floored module would give
        # delta + floor -- a silent jump (at floor=0.05 on our measured delta=0.714, +7%) that no
        # loss curve would reveal. Re-solve so the EFFECTIVE delta is preserved exactly. Runs in
        # both directions, treating a checkpoint with no delta_floor buffer as floor 0.
        old_floor = float(state_dict[prefix + "delta_floor"]) \
            if prefix + "delta_floor" in state_dict else 0.0
        new_floor = float(self.delta_floor) if self.per_position_delta else 0.0
        lk = prefix + "log_delta"
        if old_floor != new_floor and lk in state_dict:
            old_delta = F.softplus(state_dict[lk].double()) + old_floor
            assert bool((old_delta > new_floor).all()), (
                f"delta floor {new_floor} is >= the checkpoint's smallest per-channel delta "
                f"{float(old_delta.min()):.5f}; log_delta has no solution. Pick a floor below it.")
            print(f"ContractivePairUpdate: re-solving {lk} for delta floor "
                  f"{old_floor} -> {new_floor} (effective delta preserved)")
            state_dict[lk] = inv_softplus(old_delta - new_floor).to(state_dict[lk].dtype)

        k = prefix + "b"
        if k in state_dict and state_dict[k].dim() == 1:
            print(f"ContractivePairUpdate: migrating {k} from per-channel vector to diag(b) "
                  f"[{state_dict[k].shape[0]} -> {tuple(self.b.shape)}]")
            state_dict[k] = torch.diag(state_dict[k].to(self.b.dtype))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, z_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """z_t, u_t: [..., c_z] (any leading batch/residue-pair dims). Returns the combined
        signal to feed into the pair-folding stack (triangle mult + transition), NOT the final
        post-pair-folding-layers representation."""
        if not self.per_position_delta:
            a_bar, b_bar = self.discretized_params()
            return a_bar * z_t + F.linear(self.layer_norm_u(u_t), b_bar)

        delta = self.per_position_delta_from_state(z_t)
        a_bar = torch.exp(-delta * torch.exp(self.log_a))
        # ⛔⛔ NEVER form b_bar = delta[..., :, None] * b per position: that is [..., L, L, c_z, c_z],
        # ~1e12 elements at L=256/c_z=128. Use the row-scaling identity diag(d) @ B @ x == d * (B @ x)
        # -- apply the STATIC matrix first, then scale elementwise. One extra tensor the shape of z,
        # no new matmul. (The per-channel branch above keeps its historical delta[:, None] * b form
        # so that path stays bit-identical; the two orders differ only in float rounding.)
        return a_bar * z_t + delta * F.linear(self.layer_norm_u(u_t), self.b)


def sample_gaussian_pair_init(shape, d_pair: int, device=None, dtype=None, generator=None,
                             scale: float = 1.0):
    """z_0 ~ trunc_norm(0, scale^2 * 2/(5*d_pair)), +/-3 sigma truncation (ESMFold2 App. A.2.5).
    Independent of input features -- the seed-varying source of structural diversity.

    `scale` multiplies sigma (and the truncation with it, so the shape of the distribution is
    preserved and only its width changes). scale=1.0 is the paper value and is BIT-IDENTICAL to the
    unscaled call, since 1.0*x == x exactly.

    ⛔⛔ A scale is only meaningful on the CONTRACTIVE path. With use_contractive=False,
    RecyclingEmbedder.forward feeds z_prev through `layer_norm_z`, and LayerNorm is scale-invariant:
    LN(scale*z) == LN(z) apart from the eps in its denominator. Measured on this code (c_z=128,
    2026-08-19): scale=4 and scale=100 both differ from scale=1 by the SAME 7.8e-3 -- the deviation
    saturates, which is the signature of an eps artifact rather than a real change. With
    use_contractive=True the update is `a_bar * z_prev + b_bar * LN(u_t)` and z_prev is used RAW, so
    the scale passes through linearly and does change the sample.
    """
    std = scale * math.sqrt(2.0 / (5.0 * d_pair))
    # ⛔⛔ scale=0 is a REAL, WANTED rung -- the deterministic "T=0" replica of a temperature ladder,
    # identical to stock AF2's all-zero pair init. It must short-circuit: torch's trunc_normal_
    # computes norm_cdf((a-mean)/std) and raises ZeroDivisionError on std=0. Caught by the sweep smoke
    # test; the training path would have crashed the same way at the first step of any ladder
    # containing 0.
    if std == 0.0:
        return torch.zeros(shape, device=device, dtype=dtype)
    z0 = torch.empty(shape, device=device, dtype=dtype)
    with torch.no_grad():
        nn.init.trunc_normal_(z0, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)
    return z0

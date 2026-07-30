"""Iterative single-surrogate for the whole 48-block Evoformer (user's diagnosis: the prior surrogate had only ONE
round of m<->z communication, ~single direction; the Evoformer iterates 48x). N_rep mini-Evoformer repetitions, each:
  z->m (pair-biased row attention) + m FFN  ->  m->z (OuterProductMean on the UPDATED m)  ->  z->z (dilated conv).
Parameter budget spread over the reps. Drop-in for StraightThroughStack (same forward signature as StackSurrogate).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from openfold.model.primitives import Linear, LayerNorm, Attention


class _ConvZ(nn.Module):
    """z->z via dilated conv (local, additive)."""
    def __init__(self, c_z, c_hidden, kernel, dilation):
        super().__init__()
        self.z_ln = LayerNorm(c_z)
        self.z_down = Linear(c_z, c_hidden)
        self.conv = nn.Conv2d(c_hidden, c_hidden, kernel, padding=dilation * (kernel // 2), dilation=dilation)
        self.z_up = Linear(c_hidden, c_z, init="final")

    def forward(self, z):
        x = self.z_down(F.elu(self.z_ln(z))).permute(0, 3, 1, 2)
        x = self.conv(x).permute(0, 2, 3, 1)
        return z + self.z_up(F.elu(x))


class _TriMulZ(nn.Module):
    """z->z via triangle multiplication (outgoing + incoming) — the real Evoformer pair op (global, multiplicative)."""
    def __init__(self, c_z, c_hidden):
        super().__init__()
        self.ln = LayerNorm(c_z)
        self.a = Linear(c_z, c_hidden); self.ag = Linear(c_z, c_hidden)
        self.b = Linear(c_z, c_hidden); self.bg = Linear(c_z, c_hidden)
        self.out_ln = LayerNorm(c_hidden)
        self.out = Linear(c_hidden, c_z, init="final")
        self.g = Linear(c_z, c_z, init="final")

    def forward(self, z):
        zl = self.ln(z)
        a = self.a(zl) * torch.sigmoid(self.ag(zl))
        b = self.b(zl) * torch.sigmoid(self.bg(zl))
        out = torch.einsum("bikc,bjkc->bijc", a, b) + torch.einsum("bkic,bkjc->bijc", a, b)  # outgoing + incoming
        out = self.out(self.out_ln(out)) * torch.sigmoid(self.g(zl))
        return z + out


class _ViTZ(nn.Module):
    """z->z via explicit long-range reasoning: 2D windowed self-attention over the LxL pair grid (local/mid-range)
    + coarse-grid global attention over window summaries (explicit long-range), broadcast back."""
    def __init__(self, c_z, c_hidden, window=16, no_heads=4):
        super().__init__()
        self.window = window
        self.ln = LayerNorm(c_z)
        self.down = Linear(c_z, c_hidden)
        self.win_ln = LayerNorm(c_hidden)
        self.win_attn = Attention(c_hidden, c_hidden, c_hidden, c_hidden // no_heads, no_heads, gating=True)
        self.coarse_ln = LayerNorm(c_hidden)
        self.coarse_attn = Attention(c_hidden, c_hidden, c_hidden, c_hidden // no_heads, no_heads, gating=True)
        self.up = Linear(c_hidden, c_z, init="final")

    def forward(self, z):
        B, L = z.shape[0], z.shape[1]
        x = self.down(F.elu(self.ln(z)))
        ch = x.shape[-1]
        w = self.window
        pad = (w - L % w) % w
        if pad:
            x = F.pad(x, (0, 0, 0, pad, 0, pad))
        Lp = L + pad
        G = Lp // w
        # windowed self-attention: group each wxw block of the LxL grid into w*w tokens
        xw = x.reshape(B, G, w, G, w, ch).permute(0, 1, 3, 2, 4, 5).reshape(B * G * G, w * w, ch)
        h = self.win_ln(xw)
        xw = xw + self.win_attn(h, h)
        # coarse-grid global attention over the G*G window summaries
        grid = xw.reshape(B, G * G, w * w, ch).mean(2)
        hc = self.coarse_ln(grid)
        grid = grid + self.coarse_attn(hc, hc)
        xw = xw.reshape(B, G * G, w * w, ch) + grid[:, :, None, :]
        x = xw.reshape(B, G, G, w, w, ch).permute(0, 1, 3, 2, 4, 5).reshape(B, Lp, Lp, ch)
        if pad:
            x = x[:, :L, :L]
        return z + self.up(F.elu(x))


class _Rep(nn.Module):
    def __init__(self, c_m, c_z, d_opm, no_heads, c_hidden, kernel, dilation, z_op="conv"):
        super().__init__()
        self.no_heads = no_heads
        # z->m : pair-biased row attention over L
        self.m_ln = LayerNorm(c_m)
        self.row_attn = Attention(c_m, c_m, c_m, c_m // no_heads, no_heads, gating=True)
        self.zbias_ln = LayerNorm(c_z)
        self.zbias = Linear(c_z, no_heads, bias=False, init="final")
        self.m_ffn_ln = LayerNorm(c_m)
        self.m_ffn1 = Linear(c_m, 2 * c_m, init="relu")
        self.m_ffn2 = Linear(2 * c_m, c_m, init="final")
        # m->z : OuterProductMean on the UPDATED m
        self.opm_ln = LayerNorm(c_m)
        self.opm_a = Linear(c_m, d_opm)
        self.opm_b = Linear(c_m, d_opm)
        self.opm_out = Linear(d_opm * d_opm, c_z, init="final")
        # z->z : conv (local) / triangle multiplication (global) / windowed+coarse attention (explicit long-range)
        if z_op == "trimul":
            self.z_update = _TriMulZ(c_z, c_hidden)
        elif z_op == "vit":
            self.z_update = _ViTZ(c_z, c_hidden)
        else:
            self.z_update = _ConvZ(c_z, c_hidden, kernel, dilation)
        # z-track FFN (the real Evoformer's PairTransition; was missing) — LN -> Linear -> ReLU -> Linear, residual
        self.z_ffn_ln = LayerNorm(c_z)
        self.z_ffn1 = Linear(c_z, 2 * c_z, init="relu")
        self.z_ffn2 = Linear(2 * c_z, c_z, init="final")

    def forward(self, m, z, msa_mask):
        B, S, L = m.shape[0], m.shape[1], m.shape[2]
        # z -> m
        zb = self.zbias(self.zbias_ln(z)).permute(0, 3, 1, 2).unsqueeze(1)          # [B,1,H,L,L]
        bias = zb.expand(B, S, self.no_heads, L, L).reshape(B * S, self.no_heads, L, L)
        h = self.m_ln(m).reshape(B * S, L, -1)
        m = m + self.row_attn(h, h, biases=[bias]).reshape(B, S, L, -1)
        m = m + self.m_ffn2(F.gelu(self.m_ffn1(self.m_ffn_ln(m))))
        # m -> z (OPM on updated m)
        mln = self.opm_ln(m)
        a, b = self.opm_a(mln), self.opm_b(mln)
        if msa_mask is not None:
            a = a * msa_mask[..., None]
            b = b * msa_mask[..., None]
        z = z + self.opm_out((torch.einsum("bsid,bsje->bijde", a, b) / float(S)).reshape(B, L, L, -1))
        # z -> z, then z-track FFN (PairTransition)
        z = self.z_update(z)
        z = z + self.z_ffn2(F.relu(self.z_ffn1(self.z_ffn_ln(z))))
        return m, z


class IterativeStackSurrogate(nn.Module):
    def __init__(self, c_m=256, c_z=128, c_m_proj=64, c_z_proj=48, d_opm=8, no_heads=4,
                 c_hidden=32, kernel=5, dilations=(1, 2, 4, 8), n_rep=8, z_op="conv"):
        super().__init__()
        self.m_in_ln = LayerNorm(c_m); self.m_down = Linear(c_m, c_m_proj)
        self.z_in_ln = LayerNorm(c_z); self.z_down = Linear(c_z, c_z_proj)
        self.reps = nn.ModuleList([
            _Rep(c_m_proj, c_z_proj, d_opm, no_heads, c_hidden, kernel, dilations[i % len(dilations)], z_op=z_op)
            for i in range(n_rep)])
        self.m_up_ln = LayerNorm(c_m_proj); self.m_up = Linear(c_m_proj, c_m, init="final")
        self.z_up_ln = LayerNorm(c_z_proj); self.z_up = Linear(c_z_proj, c_z, init="final")

    def forward(self, m, z, msa_mask=None, pair_mask=None):
        sq_m, sq_z = m.dim() == 3, z.dim() == 3
        if sq_m:
            m = m.unsqueeze(0)
            if msa_mask is not None and msa_mask.dim() == 2:
                msa_mask = msa_mask.unsqueeze(0)
        if sq_z:
            z = z.unsqueeze(0)
        m_in, z_in = m, z
        mx = self.m_down(F.elu(self.m_in_ln(m_in)))
        zx = self.z_down(F.elu(self.z_in_ln(z_in)))
        for rep in self.reps:
            mx, zx = rep(mx, zx, msa_mask)
        m_out = m_in + self.m_up(F.elu(self.m_up_ln(mx)))
        z_out = z_in + self.z_up(F.elu(self.z_up_ln(zx)))
        if msa_mask is not None:
            m_out = m_out * msa_mask[..., None]
        if pair_mask is not None:
            z_out = z_out * pair_mask[..., None]
        if sq_m:
            m_out = m_out.squeeze(0)
        if sq_z:
            z_out = z_out.squeeze(0)
        return m_out, z_out

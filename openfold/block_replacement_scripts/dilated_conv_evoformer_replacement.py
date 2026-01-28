#!/usr/bin/env python3

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MResNetBlock(nn.Module):
    def __init__(self, c_m: int, kernel_size: int, dilation: int):
        super().__init__()
        if c_m % 2 != 0:
            raise ValueError(f"c_m must be even for down-projection, got {c_m}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd to preserve length, got {kernel_size}")
        c_hidden = c_m // 2

        padding = dilation * (kernel_size // 2)
        self.layer_gate = nn.Linear(c_m, c_m)

        self.ln_in = nn.LayerNorm(c_m)
        self.down = nn.Linear(c_m, c_hidden)

        self.ln_hidden = nn.LayerNorm(c_hidden)
        self.conv = nn.Conv1d(
            in_channels=c_hidden,
            out_channels=c_hidden,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.ln_out = nn.LayerNorm(c_hidden)
        self.up = nn.Linear(c_hidden, c_m)

    def forward(self, m: torch.Tensor, msa_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # m: [B, S, N, C_m]
        residual = m
        layer_gate = torch.sigmoid(self.layer_gate(residual))

        x = self.ln_in(m)
        x = F.gelu(x)
        x = self.down(x)  # [B, S, N, C_hidden]

        x = self.ln_hidden(x)
        x = F.gelu(x)
        b, s, n, c_h = x.shape
        x = x.reshape(b * s, n, c_h).transpose(1, 2)  # [B*S, C_hidden, N]
        x = self.conv(x)
        x = x.transpose(1, 2).reshape(b, s, n, c_h)  # [B, S, N, C_hidden]

        x = self.ln_out(x)
        x = F.gelu(x)
        x = self.up(x)  # [B, S, N, C_m]

        x = x * layer_gate

        if msa_mask is not None:
            x = x * msa_mask[..., None]

        return residual + x


class _ZResNetBlock(nn.Module):
    def __init__(self, c_z: int, kernel_size: int, dilation: int):
        super().__init__()
        if c_z % 2 != 0:
            raise ValueError(f"c_z must be even for down-projection, got {c_z}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd to preserve spatial dims, got {kernel_size}")
        c_hidden = c_z // 2

        padding = dilation * (kernel_size // 2)
        self.layer_gate = nn.Linear(c_z, c_z)

        self.ln_in = nn.LayerNorm(c_z)
        self.down = nn.Linear(c_z, c_hidden)

        self.ln_hidden = nn.LayerNorm(c_hidden)
        self.conv = nn.Conv2d(
            in_channels=c_hidden,
            out_channels=c_hidden,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.ln_out = nn.LayerNorm(c_hidden)
        self.up = nn.Linear(c_hidden, c_z)

    def forward(self, z: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # z: [B, N, N, C_z]
        residual = z
        layer_gate = torch.sigmoid(self.layer_gate(residual))

        x = self.ln_in(z)
        x = F.gelu(x)
        x = self.down(x)  # [B, N, N, C_hidden]

        x = self.ln_hidden(x)
        x = F.gelu(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C_hidden, N, N]
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, N, N, C_hidden]

        x = self.ln_out(x)
        x = F.gelu(x)
        x = self.up(x)  # [B, N, N, C_z]

        x = x * layer_gate

        if pair_mask is not None:
            x = x * pair_mask[..., None]

        return residual + x


class _MSharedProjConvStack(nn.Module):
    def __init__(self, c_m: int, kernel_size: int, dilations: Sequence[int]):
        super().__init__()
        if c_m % 2 != 0:
            raise ValueError(f"c_m must be even for down-projection, got {c_m}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd to preserve length, got {kernel_size}")
        c_hidden = c_m // 2

        pad_factor = kernel_size // 2
        self.layer_gate = nn.Linear(c_m, c_m)

        self.ln_in = nn.LayerNorm(c_m)
        self.down = nn.Linear(c_m, c_hidden)

        self.conv_lns = nn.ModuleList([nn.LayerNorm(c_hidden) for _ in dilations])
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=c_hidden,
                    out_channels=c_hidden,
                    kernel_size=kernel_size,
                    padding=int(d) * pad_factor,
                    dilation=int(d),
                )
                for d in dilations
            ]
        )

        self.ln_out = nn.LayerNorm(c_hidden)
        self.up = nn.Linear(c_hidden, c_m)

    def forward(self, m: torch.Tensor, msa_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # m: [B, S, N, C_m]
        residual = m
        layer_gate = torch.sigmoid(self.layer_gate(residual))

        x = self.ln_in(m)
        x = F.gelu(x)
        x = self.down(x)  # [B, S, N, C_hidden]
        if msa_mask is not None:
            x = x * msa_mask[..., None]

        b, s, n, c_h = x.shape
        for ln, conv in zip(self.conv_lns, self.convs):
            x = ln(x)
            x = F.gelu(x)
            if msa_mask is not None:
                x = x * msa_mask[..., None]
            y = x.reshape(b * s, n, c_h).transpose(1, 2)  # [B*S, C_hidden, N]
            y = conv(y)
            x = y.transpose(1, 2).reshape(b, s, n, c_h)
            if msa_mask is not None:
                x = x * msa_mask[..., None]

        x = self.ln_out(x)
        x = F.gelu(x)
        x = self.up(x)  # [B, S, N, C_m]

        x = x * layer_gate
        if msa_mask is not None:
            x = x * msa_mask[..., None]

        return residual + x


class _ZSharedProjConvStack(nn.Module):
    def __init__(self, c_z: int, kernel_size: int, dilations: Sequence[int]):
        super().__init__()
        if c_z % 2 != 0:
            raise ValueError(f"c_z must be even for down-projection, got {c_z}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd to preserve spatial dims, got {kernel_size}")
        c_hidden = c_z // 2

        pad_factor = kernel_size // 2
        self.layer_gate = nn.Linear(c_z, c_z)

        self.ln_in = nn.LayerNorm(c_z)
        self.down = nn.Linear(c_z, c_hidden)

        self.conv_lns = nn.ModuleList([nn.LayerNorm(c_hidden) for _ in dilations])
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=c_hidden,
                    out_channels=c_hidden,
                    kernel_size=kernel_size,
                    padding=int(d) * pad_factor,
                    dilation=int(d),
                )
                for d in dilations
            ]
        )

        self.ln_out = nn.LayerNorm(c_hidden)
        self.up = nn.Linear(c_hidden, c_z)

    def forward(self, z: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # z: [B, N, N, C_z]
        residual = z
        layer_gate = torch.sigmoid(self.layer_gate(residual))

        x = self.ln_in(z)
        x = F.gelu(x)
        x = self.down(x)  # [B, N, N, C_hidden]
        if pair_mask is not None:
            x = x * pair_mask[..., None]

        for ln, conv in zip(self.conv_lns, self.convs):
            x = ln(x)
            x = F.gelu(x)
            if pair_mask is not None:
                x = x * pair_mask[..., None]
            y = x.permute(0, 3, 1, 2).contiguous()  # [B, C_hidden, N, N]
            y = conv(y)
            x = y.permute(0, 2, 3, 1).contiguous()  # [B, N, N, C_hidden]
            if pair_mask is not None:
                x = x * pair_mask[..., None]

        x = self.ln_out(x)
        x = F.gelu(x)
        x = self.up(x)  # [B, N, N, C_z]

        x = x * layer_gate
        if pair_mask is not None:
            x = x * pair_mask[..., None]

        return residual + x


class DilatedConvEvoformerReplacement(nn.Module):
    """
    Replacement for EvoformerBlock using 3 ResNet-style blocks (dilations 1/2/4)
    for both MSA representation (m) and pair representation (z).

    - LayerNorm, GeLU
    - Down projection to C/2
    - Dilated conv (k=3 for m, 3x3 for z)
    - Up projection back to C
    - Residual connection
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4),
        mode: str = "per_block",
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.dilations = tuple(int(d) for d in dilations)
        self.mode = str(mode)

        if self.mode == "per_block":
            self.m_blocks = nn.ModuleList([_MResNetBlock(c_m=c_m, kernel_size=kernel_size, dilation=d) for d in self.dilations])
            self.z_blocks = nn.ModuleList([_ZResNetBlock(c_z=c_z, kernel_size=kernel_size, dilation=d) for d in self.dilations])
            self.m_stack = None
            self.z_stack = None
        elif self.mode == "shared_proj":
            self.m_blocks = None
            self.z_blocks = None
            self.m_stack = _MSharedProjConvStack(c_m=c_m, kernel_size=kernel_size, dilations=self.dilations)
            self.z_stack = _ZSharedProjConvStack(c_z=c_z, kernel_size=kernel_size, dilations=self.dilations)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'per_block' or 'shared_proj'.")

    def forward(
        self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del (
            chunk_size,
            use_deepspeed_evo_attention,
            use_lma,
            use_flash,
            inplace_safe,
            _attn_chunk_size,
            _offload_inference,
            _offloadable_inputs,
        )
        msa_trans_mask = msa_mask if _mask_trans else None
        pair_trans_mask = pair_mask if _mask_trans else None

        if m is not None:
            if self.mode == "per_block":
                for blk in self.m_blocks:
                    m = blk(m, msa_trans_mask)
            else:
                m = self.m_stack(m, msa_trans_mask)
        if z is not None:
            if self.mode == "per_block":
                for blk in self.z_blocks:
                    z = blk(z, pair_trans_mask)
            else:
                z = self.z_stack(z, pair_trans_mask)
        return m, z

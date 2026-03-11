#!/usr/bin/env python3
"""
Test memory usage when replacing/removing triangular attention in Evoformer blocks.

Configurations tested:
1. Baseline: Full original Evoformer with triangular attention
2. No tri_att: Skip triangular attention entirely
3. Conv replace (1,2,4,8): Replace tri_att with conv stack
4. Conv replace (1,2,4,8,1,2,4,8): Replace with deeper conv stack
"""

import argparse
import gc
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import pytree compatibility shim before other OpenFold imports
from openfold.block_replacement_scripts import _torch_pytree_compat  # noqa: F401

# OpenFold imports
from openfold.model.evoformer import EvoformerStack, EvoformerBlock
from openfold.model.triangular_attention import TriangleAttention
from openfold.config import model_config


class TriAttConvReplacement(nn.Module):
    """
    Conv-based replacement for TriangleAttention.
    Input/output: [*, I, J, C_z]
    """
    def __init__(self, c_z: int, kernel_size: int = 5, dilations: Sequence[int] = (1, 2, 4, 8)):
        super().__init__()
        if c_z % 2 != 0:
            raise ValueError(f"c_z must be even, got {c_z}")
        c_hidden = c_z // 2
        pad_factor = kernel_size // 2
        
        self.layer_gate = nn.Linear(c_z, c_z)
        self.ln_in = nn.LayerNorm(c_z)
        self.down = nn.Linear(c_z, c_hidden)
        
        self.conv_lns = nn.ModuleList([nn.LayerNorm(c_hidden) for _ in dilations])
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=c_hidden,
                out_channels=c_hidden,
                kernel_size=kernel_size,
                padding=int(d) * pad_factor,
                dilation=int(d),
            )
            for d in dilations
        ])
        
        self.ln_out = nn.LayerNorm(c_hidden)
        self.up = nn.Linear(c_hidden, c_z)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [*, I, J, C_z] pair representation
        Returns:
            [*, I, J, C_z] output
        """
        del chunk_size, use_memory_efficient_kernel, use_deepspeed_evo_attention, use_lma, inplace_safe
        
        residual = x
        layer_gate = torch.sigmoid(self.layer_gate(residual))
        
        z = self.ln_in(x)
        z = F.gelu(z)
        z = self.down(z)  # [*, I, J, C_hidden]
        
        # Apply mask if provided
        if mask is not None:
            z = z * mask[..., None]
        
        # Handle batch dimensions
        orig_shape = z.shape[:-3]  # [*]
        i, j, c_h = z.shape[-3:]
        z = z.reshape(-1, i, j, c_h)  # [B, I, J, C_hidden]
        
        for ln, conv in zip(self.conv_lns, self.convs):
            z = ln(z)
            z = F.gelu(z)
            y = z.permute(0, 3, 1, 2).contiguous()  # [B, C_hidden, I, J]
            y = conv(y)
            z = y.permute(0, 2, 3, 1).contiguous()  # [B, I, J, C_hidden]
        
        z = self.ln_out(z)
        z = F.gelu(z)
        z = self.up(z)  # [B, I, J, C_z]
        
        # Restore original shape
        z = z.reshape(*orig_shape, i, j, -1)
        
        z = z * layer_gate
        if mask is not None:
            z = z * mask[..., None]
        
        return z


class SkipTriangleAttention(nn.Module):
    """Dummy module that returns zero (used with residual connection to skip tri_att)."""
    def __init__(self, c_z: int):
        super().__init__()
        self.c_z = c_z
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        return torch.zeros_like(x)


class StraightThroughTriAtt(nn.Module):
    """
    Wraps original TriangleAttention with straight-through gradient estimation.
    Forward: uses original tri_att output
    Backward: gradient flows through conv replacement
    """
    def __init__(self, original_tri_att: TriangleAttention, replacement: nn.Module):
        super().__init__()
        self.original = original_tri_att
        self.replacement = replacement
        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        # Original forward (no grad)
        with torch.no_grad():
            orig_out = self.original(
                x, mask=mask, chunk_size=chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma, inplace_safe=inplace_safe,
            )
        
        # Replacement forward (with grad)
        repl_out = self.replacement(
            x, mask=mask, chunk_size=chunk_size,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma, inplace_safe=inplace_safe,
        )
        
        # Straight-through: forward uses orig, backward uses repl
        return repl_out + (orig_out - repl_out).detach()


def replace_triangle_attention(
    evoformer_stack: EvoformerStack,
    mode: str,
    dilations: Tuple[int, ...] = (1, 2, 4, 8),
    kernel_size: int = 5,
) -> None:
    """
    Replace triangle attention in all Evoformer blocks.
    
    Args:
        evoformer_stack: The EvoformerStack to modify
        mode: One of "baseline", "skip", "conv_st"
        dilations: Dilation factors for conv replacement
        kernel_size: Kernel size for conv replacement
    """
    if mode == "baseline":
        return  # No changes
    
    c_z = evoformer_stack.blocks[0].pair_stack.tri_att_start.c_in
    
    for block in evoformer_stack.blocks:
        pair_stack = block.pair_stack
        
        if mode == "skip":
            # Replace with skip (returns zeros)
            pair_stack.tri_att_start = SkipTriangleAttention(c_z)
            pair_stack.tri_att_end = SkipTriangleAttention(c_z)
        elif mode == "conv_st":
            # Wrap with straight-through conv replacement
            repl_start = TriAttConvReplacement(c_z, kernel_size, dilations)
            repl_end = TriAttConvReplacement(c_z, kernel_size, dilations)
            
            # Move to same device/dtype as original
            device = next(pair_stack.parameters()).device
            dtype = next(pair_stack.parameters()).dtype
            repl_start = repl_start.to(device=device, dtype=dtype)
            repl_end = repl_end.to(device=device, dtype=dtype)
            
            pair_stack.tri_att_start = StraightThroughTriAtt(
                pair_stack.tri_att_start, repl_start
            )
            pair_stack.tri_att_end = StraightThroughTriAtt(
                pair_stack.tri_att_end, repl_end
            )


def create_evoformer_stack(
    num_blocks: int = 48,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
    enable_checkpointing: bool = True,
) -> Tuple[EvoformerStack, int, int]:
    """Create an EvoformerStack with OpenFold defaults."""
    cfg = model_config("model_1_ptm")
    evo_cfg = cfg.model.evoformer_stack
    c_m = evo_cfg.c_m
    c_z = evo_cfg.c_z
    blocks_per_ckpt = 1 if enable_checkpointing else None
    
    stack = EvoformerStack(
        c_m=c_m,
        c_z=c_z,
        c_hidden_msa_att=evo_cfg.c_hidden_msa_att,
        c_hidden_opm=evo_cfg.c_hidden_opm,
        c_hidden_mul=evo_cfg.c_hidden_mul,
        c_hidden_pair_att=evo_cfg.c_hidden_pair_att,
        c_s=evo_cfg.c_s,
        no_heads_msa=evo_cfg.no_heads_msa,
        no_heads_pair=evo_cfg.no_heads_pair,
        no_blocks=num_blocks,
        transition_n=evo_cfg.transition_n,
        msa_dropout=0.0,  # Disable dropout for determinism
        pair_dropout=0.0,
        no_column_attention=evo_cfg.no_column_attention,
        opm_first=evo_cfg.opm_first,
        fuse_projection_weights=evo_cfg.fuse_projection_weights,
        blocks_per_ckpt=blocks_per_ckpt,
        inf=evo_cfg.inf,
        eps=evo_cfg.eps,
        clear_cache_between_blocks=False,
        tune_chunk_size=False,
    ).to(device=device, dtype=dtype)
    
    return stack, c_m, c_z


def measure_memory(
    seq_len: int,
    mode: str,
    dilations: Tuple[int, ...],
    kernel_size: int,
    num_blocks: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[float, float, float]:
    """
    Measure GPU memory for forward + backward pass.
    
    Returns:
        (mem_after_fwd, mem_after_bwd, peak_mem) in GiB
    """
    gc.collect()
    torch.cuda.empty_cache()
    # Initialize CUDA if not already done
    torch.cuda.set_device(device)
    _ = torch.zeros(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    
    # Create model
    stack, c_m, c_z = create_evoformer_stack(num_blocks, device, dtype, enable_checkpointing=True)
    replace_triangle_attention(stack, mode, dilations, kernel_size)
    stack.train()
    
    # Create inputs (batch size = 1)
    batch_size = 1
    msa_depth = 1  # Single sequence for simplicity
    m = torch.randn(batch_size, msa_depth, seq_len, c_m, device=device, dtype=dtype, requires_grad=True)
    z = torch.randn(batch_size, seq_len, seq_len, c_z, device=device, dtype=dtype, requires_grad=True)
    msa_mask = torch.ones(batch_size, msa_depth, seq_len, device=device, dtype=dtype)
    pair_mask = torch.ones(batch_size, seq_len, seq_len, device=device, dtype=dtype)
    
    # Forward pass
    torch.cuda.synchronize()
    m_out, z_out, _ = stack(
        m, z, msa_mask, pair_mask,
        outputs=None, cycle_no=0,
        chunk_size=None, use_deepspeed_evo_attention=False,
        use_lma=False, use_flash=False,
        inplace_safe=False, _mask_trans=True,
    )
    torch.cuda.synchronize()
    mem_after_fwd = torch.cuda.memory_allocated(device) / (1024**3)
    
    # Backward pass
    loss = (m_out.sum() + z_out.sum())
    loss.backward()
    torch.cuda.synchronize()
    mem_after_bwd = torch.cuda.memory_allocated(device) / (1024**3)
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
    
    # Check gradient flows to input
    if m.grad is None or z.grad is None:
        print(f"  WARNING: No gradient on input tensors!")
    else:
        m_grad_norm = m.grad.norm().item()
        z_grad_norm = z.grad.norm().item()
        print(f"  Input gradient norms: m={m_grad_norm:.4f}, z={z_grad_norm:.4f}")
    
    # Cleanup
    del stack, m, z, m_out, z_out, loss, msa_mask, pair_mask
    gc.collect()
    torch.cuda.empty_cache()
    
    return mem_after_fwd, mem_after_bwd, peak_mem


def main():
    parser = argparse.ArgumentParser(description="Test triangle attention replacement memory")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--num_blocks", type=int, default=48, help="Number of Evoformer blocks")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    dtype = torch.bfloat16
    
    print("=" * 70)
    print("Triangle Attention Replacement Memory Test")
    print("=" * 70)
    print(f"Sequence length: {args.seq_len}")
    print(f"Num blocks: {args.num_blocks}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print()
    
    configs = [
        ("baseline", (1,), 5),  # dilations ignored for baseline
        ("skip", (1,), 5),      # dilations ignored for skip
        ("conv_st (1,2,4,8)", (1, 2, 4, 8), 5),
        ("conv_st (1,2,4,8,1,2,4,8)", (1, 2, 4, 8, 1, 2, 4, 8), 5),
    ]
    
    results = []
    for name, dilations, kernel_size in configs:
        mode = name.split()[0] if name.startswith("conv_st") else name
        if name.startswith("conv_st"):
            mode = "conv_st"
        
        print(f"\n--- {name} ---")
        mem_fwd, mem_bwd, peak_mem = measure_memory(
            args.seq_len, mode, dilations, kernel_size,
            args.num_blocks, device, dtype,
        )
        results.append((name, mem_fwd, mem_bwd, peak_mem))
        print(f"  Memory after forward:  {mem_fwd:.3f} GiB")
        print(f"  Memory after backward: {mem_bwd:.3f} GiB")
        print(f"  Peak memory:           {peak_mem:.3f} GiB")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Peak Memory (GiB)':>18} {'Savings':>10}")
    print("-" * 70)
    
    baseline_peak = results[0][3]
    for name, mem_fwd, mem_bwd, peak_mem in results:
        savings = baseline_peak - peak_mem
        savings_str = f"{savings:+.3f}" if name != "baseline" else "---"
        print(f"{name:<35} {peak_mem:>18.3f} {savings_str:>10}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

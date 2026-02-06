#!/usr/bin/env python3
"""
Diagnostic script to pinpoint memory discrepancy between isolated EvoformerStack test
and full AlphaFold model test.

Goal: Identify where memory is consumed and why the ~5 GiB savings seen in
isolated EvoformerStack test don't appear in full model.
"""

import argparse
import contextlib
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.block_replacement_scripts import _torch_pytree_compat  # noqa: F401
from openfold.block_replacement_scripts.custom_evoformer_replacement import (
    SimpleEvoformerReplacement,
)
from openfold.block_replacement_scripts.hallucination_straight_through import (
    GradientStraightThroughBlock,
)
from openfold.config import model_config
from openfold.model.evoformer import EvoformerStack


def _bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def _log_memory(label: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"[{label}] alloc={_bytes_to_gib(alloc):.3f} GiB, reserved={_bytes_to_gib(reserved):.3f} GiB")


def _reset_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _build_evoformer_stack(
    preset: str,
    num_blocks: int,
    blocks_per_ckpt: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> EvoformerStack:
    cfg = model_config(preset)
    evo_cfg = cfg.model.evoformer_stack
    stack = EvoformerStack(
        c_m=evo_cfg.c_m,
        c_z=evo_cfg.c_z,
        c_hidden_msa_att=evo_cfg.c_hidden_msa_att,
        c_hidden_opm=evo_cfg.c_hidden_opm,
        c_hidden_mul=evo_cfg.c_hidden_mul,
        c_hidden_pair_att=evo_cfg.c_hidden_pair_att,
        c_s=evo_cfg.c_s,
        no_heads_msa=evo_cfg.no_heads_msa,
        no_heads_pair=evo_cfg.no_heads_pair,
        no_blocks=num_blocks,
        transition_n=evo_cfg.transition_n,
        msa_dropout=evo_cfg.msa_dropout,
        pair_dropout=evo_cfg.pair_dropout,
        no_column_attention=evo_cfg.no_column_attention,
        opm_first=evo_cfg.opm_first,
        fuse_projection_weights=evo_cfg.fuse_projection_weights,
        blocks_per_ckpt=blocks_per_ckpt,
        inf=evo_cfg.inf,
        eps=evo_cfg.eps,
        clear_cache_between_blocks=False,
        tune_chunk_size=False,
    ).to(device=device, dtype=dtype)
    return stack


def _wrap_all_blocks_with_replacement(
    stack: EvoformerStack,
    c_m: int,
    c_z: int,
    ckpt_replacement_only: bool,
) -> None:
    for idx, block in enumerate(stack.blocks):
        replacement = SimpleEvoformerReplacement(c_m=c_m, c_z=c_z, linear_type="full")
        replacement = replacement.to(
            device=next(stack.parameters()).device,
            dtype=next(stack.parameters()).dtype,
        )
        wrapped = GradientStraightThroughBlock(block, replacement, block_idx=idx)
        wrapped.use_straight_through = True
        wrapped.ckpt_replacement_only = ckpt_replacement_only
        stack.blocks[idx] = wrapped


def _freeze_non_replacement_params(stack: EvoformerStack) -> int:
    frozen = 0
    for name, param in stack.named_parameters():
        if ".replacement_block." in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def run_test_case(
    name: str,
    seq_len: int,
    msa_depth: int,
    num_blocks: int,
    preset: str,
    device: torch.device,
    dtype: torch.dtype,
    use_replacement: bool,
    freeze_base: bool,
    blocks_per_ckpt: Optional[int],
    ckpt_replacement_only: bool,
    input_requires_grad: bool,
) -> Dict[str, float]:
    """Run a single test case and return memory stats."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  seq_len={seq_len}, msa_depth={msa_depth}, num_blocks={num_blocks}")
    print(f"  use_replacement={use_replacement}, freeze_base={freeze_base}")
    print(f"  blocks_per_ckpt={blocks_per_ckpt}, ckpt_replacement_only={ckpt_replacement_only}")
    print(f"  input_requires_grad={input_requires_grad}")
    print("=" * 60)

    _reset_memory()

    cfg = model_config(preset)
    evo_cfg = cfg.model.evoformer_stack
    c_m = evo_cfg.c_m
    c_z = evo_cfg.c_z

    stack = _build_evoformer_stack(
        preset=preset,
        num_blocks=num_blocks,
        blocks_per_ckpt=blocks_per_ckpt,
        device=device,
        dtype=dtype,
    )

    if use_replacement:
        _wrap_all_blocks_with_replacement(stack, c_m, c_z, ckpt_replacement_only)
        if freeze_base:
            frozen = _freeze_non_replacement_params(stack)
            print(f"  Frozen {frozen:,} parameters")

    _log_memory("after model init")

    # Create input tensors
    m = torch.randn(1, msa_depth, seq_len, c_m, device=device, dtype=dtype)
    z = torch.randn(1, seq_len, seq_len, c_z, device=device, dtype=dtype)
    msa_mask = torch.ones(1, msa_depth, seq_len, device=device, dtype=dtype)
    pair_mask = torch.ones(1, seq_len, seq_len, device=device, dtype=dtype)

    if input_requires_grad:
        m = m.requires_grad_(True)
        z = z.requires_grad_(True)

    _log_memory("after input creation")

    m_bytes = m.numel() * m.element_size()
    z_bytes = z.numel() * z.element_size()
    print(f"  Input m: {m.shape}, {_bytes_to_gib(m_bytes):.4f} GiB")
    print(f"  Input z: {z.shape}, {_bytes_to_gib(z_bytes):.4f} GiB")

    stack.train()
    stack.zero_grad(set_to_none=True)

    # Forward pass
    m_out, z_out, _ = stack(
        m,
        z,
        msa_mask,
        pair_mask,
        outputs=None,
        cycle_no=0,
        chunk_size=None,
        use_deepspeed_evo_attention=False,
        use_lma=False,
        use_flash=False,
        inplace_safe=False,
        _mask_trans=True,
    )
    loss = m_out.square().mean() + z_out.square().mean()

    torch.cuda.synchronize()
    fwd_alloc = torch.cuda.memory_allocated()
    fwd_peak = torch.cuda.max_memory_allocated()
    _log_memory("after forward")

    # Backward pass
    loss.backward()

    torch.cuda.synchronize()
    bwd_alloc = torch.cuda.memory_allocated()
    bwd_peak = torch.cuda.max_memory_allocated()
    _log_memory("after backward")

    # Check gradient flow
    if input_requires_grad:
        m_grad_ok = m.grad is not None and torch.any(m.grad != 0)
        z_grad_ok = z.grad is not None and torch.any(z.grad != 0)
        print(f"  Gradient reaches input m: {m_grad_ok}")
        print(f"  Gradient reaches input z: {z_grad_ok}")

    print(f"\n  Peak memory (forward): {_bytes_to_gib(fwd_peak):.3f} GiB")
    print(f"  Peak memory (total):   {_bytes_to_gib(bwd_peak):.3f} GiB")

    # Cleanup
    del stack, m, z, msa_mask, pair_mask, m_out, z_out, loss
    torch.cuda.empty_cache()

    return {
        "fwd_alloc_gib": _bytes_to_gib(fwd_alloc),
        "fwd_peak_gib": _bytes_to_gib(fwd_peak),
        "bwd_alloc_gib": _bytes_to_gib(bwd_alloc),
        "bwd_peak_gib": _bytes_to_gib(bwd_peak),
    }


def main():
    parser = argparse.ArgumentParser(description="Debug memory discrepancy")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--msa_depth", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=48)
    parser.add_argument("--preset", type=str, default="model_1_ptm")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Memory Discrepancy Analysis")
    print("=" * 70)
    print(f"seq_len={args.seq_len}, msa_depth={args.msa_depth}, num_blocks={args.num_blocks}")
    print(f"device={device}, dtype={dtype}")
    print("=" * 70)

    results = {}

    # Test 1: Baseline (no replacement, with checkpointing)
    results["baseline_ckpt"] = run_test_case(
        name="Baseline (no replacement, ckpt, input_grad=False)",
        seq_len=args.seq_len,
        msa_depth=args.msa_depth,
        num_blocks=args.num_blocks,
        preset=args.preset,
        device=device,
        dtype=dtype,
        use_replacement=False,
        freeze_base=False,
        blocks_per_ckpt=1,
        ckpt_replacement_only=False,
        input_requires_grad=False,
    )

    # Test 2: Replacement with freeze (as in isolated test)
    results["replacement_freeze_ckpt"] = run_test_case(
        name="Replacement + freeze (ckpt, input_grad=False)",
        seq_len=args.seq_len,
        msa_depth=args.msa_depth,
        num_blocks=args.num_blocks,
        preset=args.preset,
        device=device,
        dtype=dtype,
        use_replacement=True,
        freeze_base=True,
        blocks_per_ckpt=1,
        ckpt_replacement_only=False,
        input_requires_grad=False,
    )

    # Test 3: Replacement with freeze + ckpt_replacement_only
    results["replacement_freeze_ckpt_repl_only"] = run_test_case(
        name="Replacement + freeze + ckpt_replacement_only (input_grad=False)",
        seq_len=args.seq_len,
        msa_depth=args.msa_depth,
        num_blocks=args.num_blocks,
        preset=args.preset,
        device=device,
        dtype=dtype,
        use_replacement=True,
        freeze_base=True,
        blocks_per_ckpt=1,
        ckpt_replacement_only=True,
        input_requires_grad=False,
    )

    # Test 4: Baseline with input requiring grad (simulates full model)
    results["baseline_ckpt_input_grad"] = run_test_case(
        name="Baseline (no replacement, ckpt, input_grad=True)",
        seq_len=args.seq_len,
        msa_depth=args.msa_depth,
        num_blocks=args.num_blocks,
        preset=args.preset,
        device=device,
        dtype=dtype,
        use_replacement=False,
        freeze_base=False,
        blocks_per_ckpt=1,
        ckpt_replacement_only=False,
        input_requires_grad=True,
    )

    # Test 5: Replacement with freeze + input requiring grad
    results["replacement_freeze_ckpt_input_grad"] = run_test_case(
        name="Replacement + freeze (ckpt, input_grad=True)",
        seq_len=args.seq_len,
        msa_depth=args.msa_depth,
        num_blocks=args.num_blocks,
        preset=args.preset,
        device=device,
        dtype=dtype,
        use_replacement=True,
        freeze_base=True,
        blocks_per_ckpt=1,
        ckpt_replacement_only=False,
        input_requires_grad=True,
    )

    # Test 6: Replacement with freeze + ckpt_replacement_only + input requiring grad
    results["replacement_freeze_ckpt_repl_only_input_grad"] = run_test_case(
        name="Replacement + freeze + ckpt_replacement_only (input_grad=True)",
        seq_len=args.seq_len,
        msa_depth=args.msa_depth,
        num_blocks=args.num_blocks,
        preset=args.preset,
        device=device,
        dtype=dtype,
        use_replacement=True,
        freeze_base=True,
        blocks_per_ckpt=1,
        ckpt_replacement_only=True,
        input_requires_grad=True,
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test Case':<50} {'Peak Total (GiB)':>18}")
    print("-" * 70)
    for name, stats in results.items():
        print(f"{name:<50} {stats['bwd_peak_gib']:>18.3f}")

    # Calculate savings
    print("\n" + "-" * 70)
    print("Memory savings analysis:")
    baseline = results["baseline_ckpt"]["bwd_peak_gib"]
    repl_freeze = results["replacement_freeze_ckpt"]["bwd_peak_gib"]
    repl_freeze_only = results["replacement_freeze_ckpt_repl_only"]["bwd_peak_gib"]
    baseline_grad = results["baseline_ckpt_input_grad"]["bwd_peak_gib"]
    repl_freeze_grad = results["replacement_freeze_ckpt_input_grad"]["bwd_peak_gib"]
    repl_freeze_only_grad = results["replacement_freeze_ckpt_repl_only_input_grad"]["bwd_peak_gib"]

    print(f"\nWithout input grad (isolated-like):")
    print(f"  Baseline:            {baseline:.3f} GiB")
    print(f"  Replacement+freeze:  {repl_freeze:.3f} GiB (saves {baseline - repl_freeze:.3f} GiB)")
    print(f"  + ckpt_repl_only:    {repl_freeze_only:.3f} GiB (saves {baseline - repl_freeze_only:.3f} GiB)")

    print(f"\nWith input grad (full-model-like):")
    print(f"  Baseline:            {baseline_grad:.3f} GiB")
    print(f"  Replacement+freeze:  {repl_freeze_grad:.3f} GiB (saves {baseline_grad - repl_freeze_grad:.3f} GiB)")
    print(f"  + ckpt_repl_only:    {repl_freeze_only_grad:.3f} GiB (saves {baseline_grad - repl_freeze_only_grad:.3f} GiB)")


if __name__ == "__main__":
    main()

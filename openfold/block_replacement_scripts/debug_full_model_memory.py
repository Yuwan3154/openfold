#!/usr/bin/env python3
"""
Debug memory usage in full AlphaFold model context.
Compare with isolated EvoformerStack test to find the discrepancy.
"""

import argparse
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from openfold.block_replacement_scripts import _torch_pytree_compat  # noqa: F401
from openfold.block_replacement_scripts.hallucination_straight_through import (
    GradientStraightThroughBlock,
    load_model,
    make_feature_batch,
    _set_straight_through_blocks,
    _set_replacement_checkpoint_only,
)
from openfold.utils.loss import distogram_loss


def _bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def _log_memory(label: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        print(f"[{label}] alloc={_bytes_to_gib(alloc):.3f} GiB, reserved={_bytes_to_gib(reserved):.3f} GiB, peak={_bytes_to_gib(peak):.3f} GiB")


def _reset_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _freeze_all_except_replacement(model) -> int:
    """Freeze all parameters except those in replacement blocks."""
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if ".replacement_block." in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"  Frozen {frozen:,} parameters, trainable {trainable:,}")
    return frozen


def _count_requires_grad(model) -> Dict[str, int]:
    """Count parameters that require grad by module."""
    counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Get top-level module name
            parts = name.split(".")
            top_module = parts[0] if parts else "unknown"
            counts[top_module] = counts.get(top_module, 0) + param.numel()
    return counts


def _analyze_model_structure(model):
    """Print model structure and parameter counts."""
    print("\nModel structure analysis:")
    total = 0
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        requires_grad = sum(p.numel() for p in child.parameters() if p.requires_grad)
        print(f"  {name}: {params:,} params, {requires_grad:,} requires_grad")
        total += params
    print(f"  TOTAL: {total:,} params")


def run_full_model_test(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    seq_len: int,
    use_replacement: bool,
    freeze_non_replacement: bool,
    disable_chunking: bool,
    disable_extra_msa: bool,
    ckpt_replacement_only: bool,
    straight_through_all: bool,
) -> Dict[str, float]:
    """Run test with full AlphaFold model."""
    print(f"\n{'='*70}")
    print(f"Full Model Test")
    print(f"  use_replacement={use_replacement}")
    print(f"  freeze_non_replacement={freeze_non_replacement}")
    print(f"  disable_chunking={disable_chunking}")
    print(f"  disable_extra_msa={disable_extra_msa}")
    print(f"  ckpt_replacement_only={ckpt_replacement_only}")
    print(f"  straight_through_all={straight_through_all}")
    print("=" * 70)

    _reset_memory()

    # Load model
    model = load_model(
        config_path,
        checkpoint_path,
        device,
        straight_through=use_replacement,
        allow_replacements=use_replacement,
        disable_attention_opts=True,  # Use vanilla attention for comparison
        disable_chunking=disable_chunking,
    )

    if disable_extra_msa:
        model.config.extra_msa.enabled = False
        print("  Extra MSA disabled")

    # Set up straight-through for all blocks if requested
    num_blocks = len(model.evoformer.blocks)
    if use_replacement and straight_through_all:
        st_blocks = tuple(range(num_blocks))
        _set_straight_through_blocks(model, st_blocks)
        print(f"  Straight-through enabled for all {num_blocks} blocks")

    if ckpt_replacement_only:
        _set_replacement_checkpoint_only(model, enabled=True)
        print("  ckpt_replacement_only enabled")

    # Freeze parameters if requested
    if freeze_non_replacement:
        _freeze_all_except_replacement(model)

    _analyze_model_structure(model)
    _log_memory("after model init")

    # Create input
    seq_logits = nn.Parameter(torch.zeros(seq_len, 20, device=device))
    residue_index = torch.arange(seq_len, device=device, dtype=torch.long)
    batch = make_feature_batch(seq_logits, residue_index, msa_depth=1)
    batch["return_representations"] = True

    _log_memory("after input creation")

    model.train()

    # Forward pass
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
        outputs = model(batch)
        dist_logits = model.aux_heads.distogram(outputs["pair"])
        pseudo_beta = torch.randn(seq_len, 3, device=device, dtype=torch.float32)
        pseudo_mask = torch.ones(seq_len, device=device, dtype=torch.float32)
        loss = distogram_loss(logits=dist_logits, pseudo_beta=pseudo_beta, pseudo_beta_mask=pseudo_mask)

    torch.cuda.synchronize()
    fwd_peak = torch.cuda.max_memory_allocated()
    _log_memory("after forward")

    # Backward pass
    loss.backward()

    torch.cuda.synchronize()
    bwd_peak = torch.cuda.max_memory_allocated()
    _log_memory("after backward")

    # Check gradient flow
    if seq_logits.grad is not None:
        grad_norm = seq_logits.grad.norm().item()
        print(f"  seq_logits gradient norm: {grad_norm:.6f}")
    else:
        print("  WARNING: seq_logits has no gradient!")

    # Check which replacement blocks got gradients
    if use_replacement:
        blocks_with_grad = 0
        blocks_without_grad = 0
        for blk in model.evoformer.blocks:
            if isinstance(blk, GradientStraightThroughBlock):
                for p in blk.replacement_block.parameters():
                    if p.requires_grad and p.grad is not None and torch.any(p.grad != 0):
                        blocks_with_grad += 1
                        break
                else:
                    blocks_without_grad += 1
        print(f"  Replacement blocks with nonzero grad: {blocks_with_grad}")
        print(f"  Replacement blocks without grad: {blocks_without_grad}")

    # Check original blocks
    if use_replacement:
        orig_with_grad = 0
        for blk in model.evoformer.blocks:
            if isinstance(blk, GradientStraightThroughBlock):
                for p in blk.original_block.parameters():
                    if p.grad is not None:
                        orig_with_grad += 1
                        break
        print(f"  Original blocks with grad: {orig_with_grad} (should be 0 if frozen)")

    print(f"\n  Peak memory (forward): {_bytes_to_gib(fwd_peak):.3f} GiB")
    print(f"  Peak memory (total):   {_bytes_to_gib(bwd_peak):.3f} GiB")

    return {
        "fwd_peak_gib": _bytes_to_gib(fwd_peak),
        "bwd_peak_gib": _bytes_to_gib(bwd_peak),
    }


def main():
    parser = argparse.ArgumentParser(description="Debug full model memory")
    parser.add_argument("--config_path", type=Path, default=Path("~/AFdistill/configs/repr_distill_per-block.yaml"))
    parser.add_argument("--checkpoint_path", type=Path, default=Path("~/AFdistill/results/repr_distill_per-block/checkpoints/last.ckpt"))
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    config_path = args.config_path.expanduser()
    checkpoint_path = args.checkpoint_path.expanduser()

    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)

    print("\n" + "=" * 70)
    print("FULL MODEL MEMORY DEBUG")
    print("=" * 70)
    print(f"config_path={config_path}")
    print(f"checkpoint_path={checkpoint_path}")
    print(f"seq_len={args.seq_len}")
    print("=" * 70)

    results = {}

    # Test 1: Baseline (no replacement)
    results["baseline"] = run_full_model_test(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        seq_len=args.seq_len,
        use_replacement=False,
        freeze_non_replacement=False,
        disable_chunking=True,
        disable_extra_msa=True,
        ckpt_replacement_only=False,
        straight_through_all=False,
    )

    # Test 2: With replacement but NO freezing (current behavior)
    results["replacement_no_freeze"] = run_full_model_test(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        seq_len=args.seq_len,
        use_replacement=True,
        freeze_non_replacement=False,
        disable_chunking=True,
        disable_extra_msa=True,
        ckpt_replacement_only=True,
        straight_through_all=True,
    )

    # Test 3: With replacement AND freezing (should match isolated test)
    results["replacement_with_freeze"] = run_full_model_test(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        seq_len=args.seq_len,
        use_replacement=True,
        freeze_non_replacement=True,
        disable_chunking=True,
        disable_extra_msa=True,
        ckpt_replacement_only=True,
        straight_through_all=True,
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test Case':<40} {'Peak Total (GiB)':>18}")
    print("-" * 60)
    for name, stats in results.items():
        print(f"{name:<40} {stats['bwd_peak_gib']:>18.3f}")

    print("\n" + "-" * 60)
    baseline = results["baseline"]["bwd_peak_gib"]
    no_freeze = results["replacement_no_freeze"]["bwd_peak_gib"]
    with_freeze = results["replacement_with_freeze"]["bwd_peak_gib"]

    print(f"Baseline:                    {baseline:.3f} GiB")
    print(f"Replacement (no freeze):     {no_freeze:.3f} GiB (saves {baseline - no_freeze:.3f} GiB)")
    print(f"Replacement (with freeze):   {with_freeze:.3f} GiB (saves {baseline - with_freeze:.3f} GiB)")

    print(f"\nExpected savings from isolated test: ~5.8 GiB")
    print(f"Actual savings with freeze: {baseline - with_freeze:.3f} GiB")

    if baseline - with_freeze < 4.0:
        print("\n*** WARNING: Memory savings much less than expected!")
        print("    Possible causes:")
        print("    1. InputEmbedder or other non-Evoformer components still have requires_grad=True")
        print("    2. Checkpointing not working as expected")
        print("    3. Extra components (structure module, etc.) consuming memory")


if __name__ == "__main__":
    main()

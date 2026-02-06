#!/usr/bin/env python3
"""
Debug: Test memory when ALL Evoformer blocks are wrapped, not just those with checkpoint weights.
This should match the isolated EvoformerStack test conditions.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from openfold.block_replacement_scripts import _torch_pytree_compat  # noqa: F401
from openfold.block_replacement_scripts.custom_evoformer_replacement import (
    SimpleEvoformerReplacement,
)
from openfold.block_replacement_scripts.hallucination_straight_through import (
    GradientStraightThroughBlock,
    load_model,
    make_feature_batch,
    _set_straight_through_blocks,
    _set_replacement_checkpoint_only,
)
from openfold.config import model_config
from openfold.utils.loss import distogram_loss


def _bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def _wrap_all_blocks(model, c_m: int, c_z: int, linear_type: str = "full") -> int:
    """Wrap ALL Evoformer blocks with GradientStraightThroughBlock."""
    wrapped = 0
    for idx, block in enumerate(model.evoformer.blocks):
        if isinstance(block, GradientStraightThroughBlock):
            # Already wrapped
            wrapped += 1
            continue
        # Create a new replacement
        replacement = SimpleEvoformerReplacement(c_m=c_m, c_z=c_z, linear_type=linear_type)
        replacement = replacement.to(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        wrapped_block = GradientStraightThroughBlock(block, replacement, idx)
        wrapped_block.use_straight_through = True
        wrapped_block.ckpt_replacement_only = True
        model.evoformer.blocks[idx] = wrapped_block
        wrapped += 1
    return wrapped


def _freeze_all_except_replacement(model) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        if ".replacement_block." in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def run_test(
    name: str,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    seq_len: int,
    wrap_all: bool,
    freeze_non_replacement: bool,
) -> dict:
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"  wrap_all={wrap_all}, freeze_non_replacement={freeze_non_replacement}")
    print("=" * 70)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load model
    model = load_model(
        config_path,
        checkpoint_path,
        device,
        straight_through=True,
        allow_replacements=True,
        disable_attention_opts=True,
        disable_chunking=True,
    )
    model.config.extra_msa.enabled = False

    # Get Evoformer config
    cfg = model_config("model_1_ptm")
    evo_cfg = cfg.model.evoformer_stack
    c_m = evo_cfg.c_m
    c_z = evo_cfg.c_z

    # Wrap all blocks if requested
    if wrap_all:
        wrapped = _wrap_all_blocks(model, c_m, c_z)
        print(f"  Wrapped ALL {wrapped} blocks")

    # Set straight-through for all blocks
    num_blocks = len(model.evoformer.blocks)
    st_blocks = tuple(range(num_blocks))
    _set_straight_through_blocks(model, st_blocks)
    _set_replacement_checkpoint_only(model, enabled=True)

    # Count wrapped blocks
    wrapped_count = sum(1 for blk in model.evoformer.blocks if isinstance(blk, GradientStraightThroughBlock))
    print(f"  Wrapped blocks: {wrapped_count}/{num_blocks}")
    print(f"  blocks_per_ckpt: {model.evoformer.blocks_per_ckpt}")

    if freeze_non_replacement:
        frozen = _freeze_all_except_replacement(model)
        print(f"  Frozen {frozen:,} parameters")

    # Create input
    seq_logits = nn.Parameter(torch.zeros(seq_len, 20, device=device))
    residue_index = torch.arange(seq_len, device=device, dtype=torch.long)
    batch = make_feature_batch(seq_logits, residue_index, msa_depth=1)
    batch["return_representations"] = True

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
    print(f"  Peak memory (forward): {_bytes_to_gib(fwd_peak):.3f} GiB")

    # Backward pass
    loss.backward()

    torch.cuda.synchronize()
    bwd_peak = torch.cuda.max_memory_allocated()
    print(f"  Peak memory (total): {_bytes_to_gib(bwd_peak):.3f} GiB")

    # Check gradients
    has_seq_grad = seq_logits.grad is not None and torch.any(seq_logits.grad != 0)
    print(f"  seq_logits has gradient: {has_seq_grad}")

    return {
        "fwd_peak_gib": _bytes_to_gib(fwd_peak),
        "bwd_peak_gib": _bytes_to_gib(bwd_peak),
        "has_seq_grad": has_seq_grad,
    }


def main():
    parser = argparse.ArgumentParser(description="Debug wrap all blocks")
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

    print("=" * 70)
    print("DEBUG: Wrap All Blocks Memory Test")
    print("=" * 70)

    results = {}

    # Test 1: Baseline (no replacement, no freeze)
    results["baseline"] = run_test(
        name="Baseline",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        seq_len=args.seq_len,
        wrap_all=False,
        freeze_non_replacement=False,
    )

    # Test 2: Only checkpoint-wrapped blocks (46/48) + freeze
    results["partial_wrap_freeze"] = run_test(
        name="Partial wrap (46/48) + freeze",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        seq_len=args.seq_len,
        wrap_all=False,
        freeze_non_replacement=True,
    )

    # Test 3: All blocks wrapped (48/48) + freeze
    results["full_wrap_freeze"] = run_test(
        name="Full wrap (48/48) + freeze",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        seq_len=args.seq_len,
        wrap_all=True,
        freeze_non_replacement=True,
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    baseline = results["baseline"]["bwd_peak_gib"]
    partial = results["partial_wrap_freeze"]["bwd_peak_gib"]
    full = results["full_wrap_freeze"]["bwd_peak_gib"]
    
    print(f"\nBaseline:                    {baseline:.3f} GiB")
    print(f"Partial wrap (46/48) + freeze: {partial:.3f} GiB (saves {baseline - partial:.3f} GiB)")
    print(f"Full wrap (48/48) + freeze:    {full:.3f} GiB (saves {baseline - full:.3f} GiB)")
    
    print(f"\nExpected savings (isolated test): ~5.8 GiB")
    print(f"\nDifference between partial and full wrap: {partial - full:.3f} GiB")
    
    if baseline - full > 4.0:
        print("\n*** SUCCESS: Full wrapping achieves expected memory savings!")
    else:
        print("\n*** Still not achieving expected savings. Other factors may be involved.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Debug: Verify that the correct checkpointing path is being taken.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

# IMPORTANT: Patch checkpoint_blocks BEFORE importing openfold modules
def _patch_checkpointing():
    """Patch checkpoint_blocks to add debug logging."""
    import openfold.utils.checkpointing as ckpt_module
    
    original_checkpoint_blocks = ckpt_module.checkpoint_blocks
    
    def logged_checkpoint_blocks(blocks, args, blocks_per_ckpt, outputs=None, cycle_no=None):
        print(f"\n>>> [checkpoint_blocks CALLED]", flush=True)
        # Count block types
        total = len(blocks)
        replacement_only = sum(1 for b in blocks if ckpt_module._is_replacement_only_block(b))
        st_blocks = 0
        non_st_blocks = 0
        
        for b in blocks:
            func, _, _ = ckpt_module._unwrap_block(b)
            if hasattr(func, "use_straight_through"):
                if func.use_straight_through:
                    st_blocks += 1
                else:
                    non_st_blocks += 1
        
        print(f"[checkpoint_blocks] total_blocks={total}, replacement_only_detected={replacement_only}, "
              f"st_enabled={st_blocks}, st_disabled={non_st_blocks}, blocks_per_ckpt={blocks_per_ckpt}", flush=True)
        
        # Check first block in detail
        if blocks:
            func, _, _ = ckpt_module._unwrap_block(blocks[0])
            print(f"[checkpoint_blocks] First block type: {type(func).__name__}", flush=True)
            if hasattr(func, "ckpt_replacement_only"):
                print(f"[checkpoint_blocks] First block ckpt_replacement_only: {func.ckpt_replacement_only}", flush=True)
            if hasattr(func, "use_straight_through"):
                print(f"[checkpoint_blocks] First block use_straight_through: {func.use_straight_through}", flush=True)
        
        print(f"[checkpoint_blocks] grad_enabled={torch.is_grad_enabled()}", flush=True)
        
        result = original_checkpoint_blocks(blocks, args, blocks_per_ckpt, outputs, cycle_no)
        print(f">>> [checkpoint_blocks DONE]\n", flush=True)
        return result
    
    ckpt_module.checkpoint_blocks = logged_checkpoint_blocks
    print(">>> Patched openfold.utils.checkpointing.checkpoint_blocks", flush=True)
    return original_checkpoint_blocks

# Patch BEFORE importing other openfold modules
_original_ckpt_fn = _patch_checkpointing()

# Now import the rest
from openfold.block_replacement_scripts import _torch_pytree_compat  # noqa: F401
from openfold.block_replacement_scripts.hallucination_straight_through import (
    GradientStraightThroughBlock,
    load_model,
    make_feature_batch,
    _set_straight_through_blocks,
    _set_replacement_checkpoint_only,
)
from openfold.utils.loss import distogram_loss

# Patch evoformer module's copy too
import openfold.model.evoformer as evo_module
import openfold.utils.checkpointing as ckpt_module
evo_module.checkpoint_blocks = ckpt_module.checkpoint_blocks
print(">>> Patched openfold.model.evoformer.checkpoint_blocks", flush=True)


def _bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def main():
    parser = argparse.ArgumentParser(description="Debug checkpointing path")
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
    print("DEBUG: Checkpointing Path Verification")
    print("=" * 70)

    # Load model with replacements
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

    # Enable straight-through for all blocks
    num_blocks = len(model.evoformer.blocks)
    st_blocks = tuple(range(num_blocks))
    _set_straight_through_blocks(model, st_blocks)
    _set_replacement_checkpoint_only(model, enabled=True)

    print(f"\nModel has {num_blocks} Evoformer blocks")
    print(f"Evoformer blocks_per_ckpt: {model.evoformer.blocks_per_ckpt}")
    
    # Count wrapped blocks
    wrapped_count = sum(1 for blk in model.evoformer.blocks if isinstance(blk, GradientStraightThroughBlock))
    print(f"Wrapped blocks: {wrapped_count}/{num_blocks}")

    # Freeze non-replacement params
    frozen = 0
    for name, param in model.named_parameters():
        if ".replacement_block." in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Frozen {frozen:,} parameters")

    # Create input
    seq_logits = nn.Parameter(torch.zeros(args.seq_len, 20, device=device))
    residue_index = torch.arange(args.seq_len, device=device, dtype=torch.long)
    batch = make_feature_batch(seq_logits, residue_index, msa_depth=1)
    batch["return_representations"] = True

    model.train()
    print(f"\nModel training mode: {model.training}")
    print(f"Grad enabled: {torch.is_grad_enabled()}")

    print("\n" + "=" * 70)
    print("Running forward pass (watch for checkpoint_blocks logging)...")
    print("=" * 70)

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
        outputs = model(batch)
        dist_logits = model.aux_heads.distogram(outputs["pair"])
        pseudo_beta = torch.randn(args.seq_len, 3, device=device, dtype=torch.float32)
        pseudo_mask = torch.ones(args.seq_len, device=device, dtype=torch.float32)
        loss = distogram_loss(logits=dist_logits, pseudo_beta=pseudo_beta, pseudo_beta_mask=pseudo_mask)

    torch.cuda.synchronize()
    fwd_peak = torch.cuda.max_memory_allocated()
    print(f"\nPeak memory (forward): {_bytes_to_gib(fwd_peak):.3f} GiB")

    print("\n" + "=" * 70)
    print("Running backward pass...")
    print("=" * 70)

    loss.backward()

    torch.cuda.synchronize()
    bwd_peak = torch.cuda.max_memory_allocated()
    print(f"\nPeak memory (total): {_bytes_to_gib(bwd_peak):.3f} GiB")

    # Check gradients
    has_seq_grad = seq_logits.grad is not None and torch.any(seq_logits.grad != 0)
    print(f"\nseq_logits has gradient: {has_seq_grad}")


if __name__ == "__main__":
    main()

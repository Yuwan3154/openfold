#!/usr/bin/env python3
"""
Debug: Test if upstream activations (InputEmbedder) cause the memory discrepancy.

Hypothesis: In the full model, even with Evoformer params frozen, InputEmbedder
activations must be kept for backprop to seq_logits. The isolated test doesn't
have this overhead.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.model.evoformer import EvoformerStack
from openfold.utils.loss import distogram_loss
from openfold.utils.import_weights import import_jax_weights_


def _bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def _log_memory(label: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        print(f"[{label}] alloc={_bytes_to_gib(alloc):.3f} GiB, peak={_bytes_to_gib(peak):.3f} GiB")


def _reset_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _freeze_all_except_replacement(model) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        if ".replacement_block." in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen += param.numel()
    return frozen


class DetachedInputEmbedder(nn.Module):
    """Wrapper that detaches the output of InputEmbedder."""
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
    
    def forward(self, *args, **kwargs):
        result = self.embedder(*args, **kwargs)
        # Detach the outputs to break gradient flow
        if isinstance(result, tuple):
            return tuple(r.detach() if isinstance(r, torch.Tensor) else r for r in result)
        elif isinstance(result, torch.Tensor):
            return result.detach()
        return result


def run_test(
    name: str,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    seq_len: int,
    use_replacement: bool,
    freeze_non_replacement: bool,
    detach_input_embedder: bool,
    ckpt_replacement_only: bool,
) -> Dict[str, float]:
    """Run a single test case."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"  use_replacement={use_replacement}")
    print(f"  freeze_non_replacement={freeze_non_replacement}")
    print(f"  detach_input_embedder={detach_input_embedder}")
    print(f"  ckpt_replacement_only={ckpt_replacement_only}")
    print("=" * 70)

    _reset_memory()

    # Load model
    model = load_model(
        config_path,
        checkpoint_path,
        device,
        straight_through=use_replacement,
        allow_replacements=use_replacement,
        disable_attention_opts=True,
        disable_chunking=True,
    )
    model.config.extra_msa.enabled = False

    # Set up straight-through for all blocks
    num_blocks = len(model.evoformer.blocks)
    if use_replacement:
        st_blocks = tuple(range(num_blocks))
        _set_straight_through_blocks(model, st_blocks)

    if ckpt_replacement_only:
        _set_replacement_checkpoint_only(model, enabled=True)

    # Detach input embedder output if requested
    if detach_input_embedder:
        original_forward = model.input_embedder.forward
        def detached_forward(*args, **kwargs):
            m, z = original_forward(*args, **kwargs)
            return m.detach(), z.detach()
        model.input_embedder.forward = detached_forward
        print("  InputEmbedder outputs will be detached")

    # Freeze parameters if requested
    if freeze_non_replacement:
        frozen = _freeze_all_except_replacement(model)
        print(f"  Frozen {frozen:,} parameters")

    _log_memory("after model init")

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
    _log_memory("after forward")

    # Backward pass
    loss.backward()

    torch.cuda.synchronize()
    bwd_peak = torch.cuda.max_memory_allocated()
    _log_memory("after backward")

    # Check gradient flow
    has_seq_grad = seq_logits.grad is not None and torch.any(seq_logits.grad != 0)
    print(f"  seq_logits has gradient: {has_seq_grad}")

    # Count replacement blocks with grad
    if use_replacement:
        blocks_with_grad = sum(
            1 for blk in model.evoformer.blocks
            if isinstance(blk, GradientStraightThroughBlock) and
            any(p.grad is not None and torch.any(p.grad != 0) 
                for p in blk.replacement_block.parameters() if p.requires_grad)
        )
        print(f"  Replacement blocks with grad: {blocks_with_grad}/{num_blocks}")

    print(f"\n  Peak memory (forward): {_bytes_to_gib(fwd_peak):.3f} GiB")
    print(f"  Peak memory (total):   {_bytes_to_gib(bwd_peak):.3f} GiB")

    return {
        "fwd_peak_gib": _bytes_to_gib(fwd_peak),
        "bwd_peak_gib": _bytes_to_gib(bwd_peak),
        "has_seq_grad": has_seq_grad,
    }


def main():
    parser = argparse.ArgumentParser(description="Debug upstream activations")
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
    print("DEBUG: Upstream Activations Impact on Memory")
    print("=" * 70)

    results = {}

    # Test 1: Baseline
    results["baseline"] = run_test(
        name="Baseline (no replacement)",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        seq_len=args.seq_len,
        use_replacement=False,
        freeze_non_replacement=False,
        detach_input_embedder=False,
        ckpt_replacement_only=False,
    )

    # Test 2: Replacement + freeze (current approach)
    results["replacement_freeze"] = run_test(
        name="Replacement + freeze (current)",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        seq_len=args.seq_len,
        use_replacement=True,
        freeze_non_replacement=True,
        detach_input_embedder=False,
        ckpt_replacement_only=True,
    )

    # Test 3: Replacement + freeze + detach InputEmbedder
    # This simulates the isolated test: no gradient flow through InputEmbedder
    results["replacement_freeze_detach"] = run_test(
        name="Replacement + freeze + detach InputEmbedder",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        seq_len=args.seq_len,
        use_replacement=True,
        freeze_non_replacement=True,
        detach_input_embedder=True,
        ckpt_replacement_only=True,
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Impact of InputEmbedder gradient flow")
    print("=" * 70)
    
    baseline = results["baseline"]["bwd_peak_gib"]
    repl_freeze = results["replacement_freeze"]["bwd_peak_gib"]
    repl_freeze_detach = results["replacement_freeze_detach"]["bwd_peak_gib"]
    
    print(f"\nBaseline:                       {baseline:.3f} GiB")
    print(f"Replacement + freeze:           {repl_freeze:.3f} GiB (saves {baseline - repl_freeze:.3f} GiB)")
    print(f"Replacement + freeze + detach:  {repl_freeze_detach:.3f} GiB (saves {baseline - repl_freeze_detach:.3f} GiB)")
    
    print(f"\nExpected savings (isolated test): ~5.8 GiB")
    print(f"\nInputEmbedder gradient overhead: {repl_freeze - repl_freeze_detach:.3f} GiB")
    
    if results["replacement_freeze"]["has_seq_grad"] and not results["replacement_freeze_detach"]["has_seq_grad"]:
        print("\nNote: Detaching InputEmbedder breaks gradient flow to seq_logits")
        print("      This is expected - you can't have seq_logits gradients without")
        print("      storing InputEmbedder activations.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
One-block toy test for memory/runtime behavior.
Runs a single Evoformer block with optional replacement wrapping.
"""

import argparse
import contextlib
import time

import torch
import torch.nn as nn

from openfold.block_replacement_scripts import _torch_pytree_compat  # noqa: F401
from openfold.block_replacement_scripts.custom_evoformer_replacement import (
    SimpleEvoformerReplacement,
)
from openfold.block_replacement_scripts.hallucination_straight_through import (
    GradientStraightThroughBlock,
    make_feature_batch,
)
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.loss import distogram_loss


def _bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def _build_model(
    device: torch.device,
    enable_checkpointing: bool,
    use_replacement: bool,
    ckpt_replacement_only: bool,
    preset: str,
) -> AlphaFold:
    cfg = model_config(preset)
    cfg.model.template.enabled = False
    cfg.model.num_recycle = 1
    cfg.globals.chunk_size = None
    cfg.model.evoformer_stack.no_blocks = 1
    cfg.model.evoformer_stack.tune_chunk_size = False
    cfg.model.template.template_pair_stack.tune_chunk_size = False
    cfg.model.extra_msa.extra_msa_stack.tune_chunk_size = False

    cfg.globals.use_flash = False
    cfg.globals.use_lma = False
    cfg.globals.use_deepspeed_evo_attention = False

    if enable_checkpointing:
        cfg.globals.blocks_per_ckpt = 1
        cfg.model.evoformer_stack.blocks_per_ckpt = 1
        cfg.model.template.template_pair_stack.blocks_per_ckpt = 1
        cfg.model.extra_msa.extra_msa_stack.ckpt = True
    else:
        cfg.globals.blocks_per_ckpt = None
        cfg.model.evoformer_stack.blocks_per_ckpt = None
        cfg.model.template.template_pair_stack.blocks_per_ckpt = None
        cfg.model.extra_msa.extra_msa_stack.ckpt = False

    model = AlphaFold(cfg).to(device)
    if device.type == "cuda":
        model = model.bfloat16()

    if use_replacement:
        evo_cfg = getattr(cfg.model, "evoformer", None) or getattr(cfg.model, "evoformer_stack")
        c_m = evo_cfg.c_m
        c_z = evo_cfg.c_z
        replacement = SimpleEvoformerReplacement(c_m=c_m, c_z=c_z, linear_type="full")
        dtype = next(model.parameters()).dtype
        replacement = replacement.to(device=device, dtype=dtype)
        evoformer = getattr(model, "evoformer", None) or model.evoformer_stack
        original_block = evoformer.blocks[0]
        wrapped = GradientStraightThroughBlock(original_block, replacement, block_idx=0)
        wrapped.use_straight_through = True
        wrapped.ckpt_replacement_only = ckpt_replacement_only
        evoformer.blocks[0] = wrapped

    model.eval()
    return model


def run_one(
    model: AlphaFold,
    device: torch.device,
    seq_len: int,
    msa_depth: int,
    no_grad: bool,
    dist_scale: float,
):
    seq_logits = nn.Parameter(torch.zeros(seq_len, 20, device=device))
    residue_index = torch.arange(seq_len, device=device, dtype=torch.long)
    batch = make_feature_batch(seq_logits, residue_index, msa_depth=msa_depth)
    batch["return_representations"] = True

    pseudo_beta = torch.randn(seq_len, 3, device=device, dtype=torch.float32)
    pseudo_mask = torch.ones(seq_len, device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    if not no_grad:
        model.zero_grad(set_to_none=True)

    t_start = time.perf_counter()
    no_grad_ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with no_grad_ctx, torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        outputs = model(batch)
        dist_logits = outputs.get("distogram_logits", model.aux_heads.distogram(outputs["pair"]))
        m_out = outputs["msa"]
        z_out = outputs["pair"]
        dist_loss = distogram_loss(
            logits=dist_logits,
            pseudo_beta=pseudo_beta,
            pseudo_beta_mask=pseudo_mask,
        )
        loss = dist_scale * dist_loss

    if device.type == "cuda":
        torch.cuda.synchronize()
        fwd_alloc = int(torch.cuda.memory_allocated())
        fwd_reserved = int(torch.cuda.memory_reserved())
        peak_alloc = int(torch.cuda.max_memory_allocated())
    else:
        fwd_alloc = 0
        fwd_reserved = 0
        peak_alloc = 0

    bwd_alloc = None
    bwd_reserved = None
    if not no_grad:
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
            bwd_alloc = int(torch.cuda.memory_allocated())
            bwd_reserved = int(torch.cuda.memory_reserved())

    elapsed = time.perf_counter() - t_start

    print(
        f"m_shape={tuple(m_out.shape)} z_shape={tuple(z_out.shape)} "
        f"m_bytes={m_out.numel() * m_out.element_size()} "
        f"z_bytes={z_out.numel() * z_out.element_size()}",
        flush=True,
    )
    print(
        f"mem_fwd_alloc_gib={_bytes_to_gib(fwd_alloc):.3f} "
        f"mem_fwd_reserved_gib={_bytes_to_gib(fwd_reserved):.3f} "
        f"mem_bwd_alloc_gib={_bytes_to_gib(bwd_alloc) if bwd_alloc is not None else 'NA'} "
        f"mem_bwd_reserved_gib={_bytes_to_gib(bwd_reserved) if bwd_reserved is not None else 'NA'} "
        f"peak_alloc_gib={_bytes_to_gib(peak_alloc):.3f} "
        f"loss={float(loss):.4f} time_s={elapsed:.3f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="One-block toy memory test")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--msa_depth", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--preset", type=str, default="model_1_ptm")
    parser.add_argument("--use_replacement", action="store_true", default=False)
    parser.add_argument("--disable_checkpointing", action="store_true", default=False)
    parser.add_argument("--ckpt_replacement_only", action="store_true", default=True)
    parser.add_argument("--no_grad", action="store_true", default=False)
    parser.add_argument("--dist_scale", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this toy test.")

    model = _build_model(
        device=device,
        enable_checkpointing=not args.disable_checkpointing,
        use_replacement=args.use_replacement,
        ckpt_replacement_only=args.ckpt_replacement_only,
        preset=args.preset,
    )

    run_one(
        model=model,
        device=device,
        seq_len=args.seq_len,
        msa_depth=args.msa_depth,
        no_grad=args.no_grad,
        dist_scale=args.dist_scale,
    )


if __name__ == "__main__":
    main()

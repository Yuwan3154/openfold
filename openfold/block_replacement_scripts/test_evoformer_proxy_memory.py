#!/usr/bin/env python3
"""
Proxy-style memory test using OpenFold Evoformer blocks and linear replacements.
Mirrors the proxy vs checkpoint comparison from Untitled834.ipynb.
"""

import argparse
import contextlib
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
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


def _format_gib(value: Optional[int]) -> str:
    if value is None:
        return "NA"
    return f"{_bytes_to_gib(value):.3f}"


@dataclass
class RunStats:
    fwd_alloc: int
    fwd_reserved: int
    peak_fwd: int
    bwd_alloc: Optional[int]
    bwd_reserved: Optional[int]
    peak_total: Optional[int]
    elapsed_s: float
    m_shape: torch.Size
    z_shape: torch.Size
    m_bytes: int
    z_bytes: int


def _build_evoformer_stack(
    preset: str,
    num_blocks: int,
    enable_checkpointing: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> EvoformerStack:
    cfg = model_config(preset)
    evo_cfg = cfg.model.evoformer_stack
    blocks_per_ckpt = 1 if enable_checkpointing else None
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
    linear_type: str,
    ckpt_replacement_only: bool,
) -> None:
    for idx, block in enumerate(stack.blocks):
        replacement = SimpleEvoformerReplacement(c_m=c_m, c_z=c_z, linear_type=linear_type)
        replacement = replacement.to(device=next(stack.parameters()).device, dtype=next(stack.parameters()).dtype)
        wrapped = GradientStraightThroughBlock(block, replacement, block_idx=idx)
        wrapped.use_straight_through = True
        wrapped.ckpt_replacement_only = ckpt_replacement_only
        stack.blocks[idx] = wrapped


def _freeze_non_replacement_params(stack: EvoformerStack) -> None:
    for name, param in stack.named_parameters():
        if ".replacement_block." in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def _make_inputs(
    batch_size: int,
    msa_depth: int,
    seq_len: int,
    c_m: int,
    c_z: int,
    device: torch.device,
    dtype: torch.dtype,
    use_target: bool,
) -> Dict[str, Optional[torch.Tensor]]:
    m = torch.randn(batch_size, msa_depth, seq_len, c_m, device=device, dtype=dtype)
    z = torch.randn(batch_size, seq_len, seq_len, c_z, device=device, dtype=dtype)
    msa_mask = torch.ones(batch_size, msa_depth, seq_len, device=device, dtype=dtype)
    pair_mask = torch.ones(batch_size, seq_len, seq_len, device=device, dtype=dtype)
    target_m = torch.randn_like(m) if use_target else None
    target_z = torch.randn_like(z) if use_target else None
    return {
        "m": m,
        "z": z,
        "msa_mask": msa_mask,
        "pair_mask": pair_mask,
        "target_m": target_m,
        "target_z": target_z,
    }


def _run_pass(
    stack: EvoformerStack,
    inputs: Dict[str, Optional[torch.Tensor]],
    use_target: bool,
    no_grad: bool,
) -> RunStats:
    m = inputs["m"]
    z = inputs["z"]
    msa_mask = inputs["msa_mask"]
    pair_mask = inputs["pair_mask"]
    target_m = inputs["target_m"]
    target_z = inputs["target_z"]
    device = m.device

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    if no_grad:
        stack.eval()
    else:
        stack.train()
        stack.zero_grad(set_to_none=True)

    t_start = time.perf_counter()
    grad_ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with grad_ctx:
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
        if use_target:
            loss = F.mse_loss(m_out, target_m) + F.mse_loss(z_out, target_z)
        else:
            loss = m_out.square().mean() + z_out.square().mean()

    if device.type == "cuda":
        torch.cuda.synchronize()
        fwd_alloc = int(torch.cuda.memory_allocated())
        fwd_reserved = int(torch.cuda.memory_reserved())
        peak_fwd = int(torch.cuda.max_memory_allocated())
    else:
        fwd_alloc = 0
        fwd_reserved = 0
        peak_fwd = 0

    bwd_alloc = None
    bwd_reserved = None
    peak_total = None
    if not no_grad:
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
            bwd_alloc = int(torch.cuda.memory_allocated())
            bwd_reserved = int(torch.cuda.memory_reserved())
            peak_total = int(torch.cuda.max_memory_allocated())

    elapsed = time.perf_counter() - t_start
    return RunStats(
        fwd_alloc=fwd_alloc,
        fwd_reserved=fwd_reserved,
        peak_fwd=peak_fwd,
        bwd_alloc=bwd_alloc,
        bwd_reserved=bwd_reserved,
        peak_total=peak_total,
        elapsed_s=elapsed,
        m_shape=m_out.shape,
        z_shape=z_out.shape,
        m_bytes=m_out.numel() * m_out.element_size(),
        z_bytes=z_out.numel() * z_out.element_size(),
    )


def _print_mode_stats(name: str, stats: RunStats) -> None:
    print(f"\n[{name}]")
    print(
        f"m_shape={tuple(stats.m_shape)} z_shape={tuple(stats.z_shape)} "
        f"m_bytes={stats.m_bytes} z_bytes={stats.z_bytes}"
    )
    print(
        f"fwd_alloc_gib={_format_gib(stats.fwd_alloc)} "
        f"fwd_reserved_gib={_format_gib(stats.fwd_reserved)} "
        f"peak_fwd_gib={_format_gib(stats.peak_fwd)} "
        f"bwd_alloc_gib={_format_gib(stats.bwd_alloc)} "
        f"bwd_reserved_gib={_format_gib(stats.bwd_reserved)} "
        f"peak_total_gib={_format_gib(stats.peak_total)} "
        f"time_s={stats.elapsed_s:.3f}"
    )


def _print_summary(results: Dict[str, RunStats]) -> None:
    if "standard" not in results:
        return
    baseline = results["standard"].peak_total or results["standard"].peak_fwd
    print("\nSummary (peak_total vs standard):")
    print(f"{'strategy':<16} {'peak_gib':>10} {'vs_std':>10}")
    for name, stats in results.items():
        peak = stats.peak_total or stats.peak_fwd
        if baseline and peak:
            pct = (1 - peak / baseline) * 100
            vs_std = f"{pct:+.1f}%"
        else:
            vs_std = "NA"
        print(f"{name:<16} { _format_gib(peak):>10} {vs_std:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evoformer proxy memory test")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--msa_depth", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--preset", type=str, default="model_1_ptm")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--linear_type", type=str, default="full")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--use_target_mse", action="store_true", default=False)
    parser.add_argument("--skip_inference", action="store_true", default=False)
    parser.add_argument("--skip_standard", action="store_true", default=False)
    parser.add_argument("--skip_checkpoint", action="store_true", default=False)
    parser.add_argument("--skip_proxy", action="store_true", default=False)
    parser.add_argument("--ckpt_replacement_only", action="store_true", default=False)
    parser.add_argument("--proxy_checkpointing", action="store_true", default=False)
    parser.add_argument("--no_freeze_base", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this test.")

    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    cfg = model_config(args.preset)
    evo_cfg = cfg.model.evoformer_stack

    print("\nConfig:")
    print(
        f"seq_len={args.seq_len} msa_depth={args.msa_depth} "
        f"batch_size={args.batch_size} num_blocks={args.num_blocks} "
        f"dtype={args.dtype} linear_type={args.linear_type}"
    )

    results: Dict[str, RunStats] = {}

    if not args.skip_inference:
        stack = _build_evoformer_stack(
            preset=args.preset,
            num_blocks=args.num_blocks,
            enable_checkpointing=False,
            device=device,
            dtype=dtype,
        )
        inputs = _make_inputs(
            batch_size=args.batch_size,
            msa_depth=args.msa_depth,
            seq_len=args.seq_len,
            c_m=evo_cfg.c_m,
            c_z=evo_cfg.c_z,
            device=device,
            dtype=dtype,
            use_target=args.use_target_mse,
        )
        stats = _run_pass(stack, inputs, use_target=args.use_target_mse, no_grad=True)
        results["inference"] = stats
        _print_mode_stats("inference", stats)
        del stack, inputs
        torch.cuda.empty_cache()

    if not args.skip_standard:
        stack = _build_evoformer_stack(
            preset=args.preset,
            num_blocks=args.num_blocks,
            enable_checkpointing=False,
            device=device,
            dtype=dtype,
        )
        inputs = _make_inputs(
            batch_size=args.batch_size,
            msa_depth=args.msa_depth,
            seq_len=args.seq_len,
            c_m=evo_cfg.c_m,
            c_z=evo_cfg.c_z,
            device=device,
            dtype=dtype,
            use_target=args.use_target_mse,
        )
        stats = _run_pass(stack, inputs, use_target=args.use_target_mse, no_grad=False)
        results["standard"] = stats
        _print_mode_stats("standard", stats)
        del stack, inputs
        torch.cuda.empty_cache()

    if not args.skip_checkpoint:
        stack = _build_evoformer_stack(
            preset=args.preset,
            num_blocks=args.num_blocks,
            enable_checkpointing=True,
            device=device,
            dtype=dtype,
        )
        inputs = _make_inputs(
            batch_size=args.batch_size,
            msa_depth=args.msa_depth,
            seq_len=args.seq_len,
            c_m=evo_cfg.c_m,
            c_z=evo_cfg.c_z,
            device=device,
            dtype=dtype,
            use_target=args.use_target_mse,
        )
        stats = _run_pass(stack, inputs, use_target=args.use_target_mse, no_grad=False)
        results["checkpoint"] = stats
        _print_mode_stats("checkpoint", stats)
        del stack, inputs
        torch.cuda.empty_cache()

    if not args.skip_proxy:
        stack = _build_evoformer_stack(
            preset=args.preset,
            num_blocks=args.num_blocks,
            enable_checkpointing=args.proxy_checkpointing,
            device=device,
            dtype=dtype,
        )
        _wrap_all_blocks_with_replacement(
            stack=stack,
            c_m=evo_cfg.c_m,
            c_z=evo_cfg.c_z,
            linear_type=args.linear_type,
            ckpt_replacement_only=args.ckpt_replacement_only,
        )
        if not args.no_freeze_base:
            _freeze_non_replacement_params(stack)
        inputs = _make_inputs(
            batch_size=args.batch_size,
            msa_depth=args.msa_depth,
            seq_len=args.seq_len,
            c_m=evo_cfg.c_m,
            c_z=evo_cfg.c_z,
            device=device,
            dtype=dtype,
            use_target=args.use_target_mse,
        )
        stats = _run_pass(stack, inputs, use_target=args.use_target_mse, no_grad=False)
        results["proxy"] = stats
        _print_mode_stats("proxy", stats)
        del stack, inputs
        torch.cuda.empty_cache()

    _print_summary(results)


if __name__ == "__main__":
    main()

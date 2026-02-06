#!/usr/bin/env python3
"""
Measure runtime and GPU memory scaling vs. sequence length.

Protocol mirrors hallucination_straight_through, but runs a single
optimization step per length with a random distogram target.
"""

import argparse
import contextlib
import csv
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from openfold.block_replacement_scripts import _torch_pytree_compat  # noqa: F401

from openfold.block_replacement_scripts.hallucination_straight_through import (
    GradientStraightThroughBlock,
    _set_straight_through,
    _set_straight_through_blocks,
    _set_replacement_checkpoint_only,
    load_model,
    make_feature_batch,
)
from openfold.utils.loss import distogram_loss


def _freeze_replaced_evoformer_params(model, st_blocks: Tuple[int, ...]) -> None:
    st_set = set(int(i) for i in st_blocks)
    total_params = 0
    frozen_params = 0
    frozen_blocks = []
    for param in model.parameters():
        total_params += param.numel()
        param.requires_grad = True
    for blk in model.evoformer.blocks:
        if isinstance(blk, GradientStraightThroughBlock) and blk.block_idx in st_set:
            for param in blk.original_block.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            frozen_blocks.append(blk.block_idx)
    print(
        f"Froze replaced Evoformer params: frozen={frozen_params} total={total_params} "
        f"blocks={sorted(frozen_blocks)}",
        flush=True,
    )


def _freeze_all_non_replacement_params(model) -> None:
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if ".replacement_block." in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    print(
        f"Froze all non-replacement params: trainable={trainable_params} total={total_params}",
        flush=True,
    )


def _freeze_all_params(model) -> None:
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        param.requires_grad = False
    print(f"Froze all params: total={total_params}", flush=True)


def _parse_int_list(value: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected a non-empty list for --lengths")
    lengths: List[int] = []
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"Invalid length entry: {part}")
        lengths.append(int(part))
    return tuple(lengths)


def _parse_block_indices(value: str) -> Tuple[int, ...]:
    if value.strip() == "":
        return ()
    out: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if part == "":
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = a.strip()
            b = b.strip()
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Invalid block range: {part}")
            start = int(a)
            end = int(b)
            if end < start:
                raise ValueError(f"Invalid block range (end<start): {part}")
            out.extend(list(range(start, end + 1)))
        else:
            if not part.isdigit():
                raise ValueError(f"Invalid block index: {part}")
            out.append(int(part))
    return tuple(sorted(set(out)))


def _get_optimizer(seq_logits: nn.Parameter, name: str, lr: float):
    if name == "SGD":
        return torch.optim.SGD([seq_logits], lr=lr)
    if name == "Adam":
        return torch.optim.Adam([seq_logits], lr=lr)
    raise ValueError(f"Invalid optimizer: {name}")


def run_one_pass(
    model,
    seq_len: int,
    device: torch.device,
    dist_scale: float,
    lr: float,
    init_seq: str,
    optimizer_name: str,
    norm_grad: bool,
    straight_through_selection: str,
    straight_through_blocks: Tuple[int, ...],
    orig_steps_per_cycle: int,
    repl_steps_per_cycle: int,
    msa_depth: int = 1,
    log_fwd_bwd_mem: bool = False,
    log_mz_sizes: bool = False,
    no_grad: bool = False,
    assert_grad_nonzero: bool = False,
    assert_seq_grad_nonzero: bool = False,
    assert_orig_grad_none: bool = False,
):
    if init_seq == "0":
        seq_logits = nn.Parameter(torch.zeros(seq_len, 20, device=device))
    elif init_seq == "gaussian":
        seq_logits = nn.Parameter(torch.randn(seq_len, 20, device=device))
    else:
        raise ValueError(f"Invalid initial sequence: {init_seq}")

    optimizer = None if no_grad else _get_optimizer(seq_logits, optimizer_name, lr)
    residue_index = torch.arange(seq_len, device=device, dtype=torch.long)

    if straight_through_selection == "static_blocks":
        _set_straight_through_blocks(model, straight_through_blocks)
        use_straight_through = len(straight_through_blocks) > 0
    else:
        cycle = orig_steps_per_cycle + repl_steps_per_cycle
        if cycle > 0:
            use_straight_through = 0 >= orig_steps_per_cycle
            _set_straight_through(model, enabled=use_straight_through)
        else:
            use_straight_through = True
            _set_straight_through(model, enabled=True)

    batch = make_feature_batch(seq_logits, residue_index, msa_depth=msa_depth)
    batch["return_representations"] = True

    pseudo_beta = torch.randn(seq_len, 3, device=device, dtype=torch.float32)
    pseudo_mask = torch.ones(seq_len, device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    if optimizer is not None:
        optimizer.zero_grad()
    no_grad_ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with no_grad_ctx, torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        outputs = model(batch)
        if "distogram_logits" in outputs:
            dist_logits = outputs["distogram_logits"]
        else:
            dist_logits = model.aux_heads.distogram(outputs["pair"])
        if log_mz_sizes:
            m_out = outputs["msa"]
            z_out = outputs["pair"]
            m_bytes = int(m_out.numel() * m_out.element_size())
            z_bytes = int(z_out.numel() * z_out.element_size())
            print(
                f"m_shape={tuple(m_out.shape)} m_dtype={m_out.dtype} m_bytes={m_bytes} "
                f"z_shape={tuple(z_out.shape)} z_dtype={z_out.dtype} z_bytes={z_bytes}",
                flush=True,
            )
        dist_loss = distogram_loss(
            logits=dist_logits,
            pseudo_beta=pseudo_beta,
            pseudo_beta_mask=pseudo_mask,
        )
        loss = dist_scale * dist_loss
    if log_fwd_bwd_mem and device.type == "cuda":
        torch.cuda.synchronize()
        fwd_alloc = int(torch.cuda.memory_allocated())
        fwd_reserved = int(torch.cuda.memory_reserved())
    if not no_grad:
        loss.backward()
        if assert_grad_nonzero and use_straight_through:
            total_params = 0
            nonzero_params = 0
            none_params = 0
            for blk in model.evoformer.blocks:
                if hasattr(blk, "replacement_block"):
                    for param in blk.replacement_block.parameters():
                        if not param.requires_grad:
                            continue
                        total_params += 1
                        if param.grad is None:
                            none_params += 1
                        elif torch.any(param.grad != 0):
                            nonzero_params += 1
            print(
                f"repl_grad_nonzero={nonzero_params} repl_grad_total={total_params} "
                f"repl_grad_none={none_params}",
                flush=True,
            )
            if nonzero_params == 0:
                raise RuntimeError("No nonzero replacement gradients detected.")
        if assert_seq_grad_nonzero:
            if seq_logits.grad is None:
                raise RuntimeError("No gradient found for seq_logits.")
            if not torch.any(seq_logits.grad != 0):
                raise RuntimeError("seq_logits gradient is all zeros.")
            print(f"seq_grad_norm={seq_logits.grad.norm().item():.6f}", flush=True)
        if assert_orig_grad_none and use_straight_through:
            grad_params = 0
            for blk in model.evoformer.blocks:
                if isinstance(blk, GradientStraightThroughBlock) and blk.use_straight_through:
                    for param in blk.original_block.parameters():
                        if param.grad is not None:
                            grad_params += 1
            if grad_params != 0:
                raise RuntimeError("Original block gradients detected for straight-through blocks.")
            print("orig_block_grad_none=True", flush=True)
        if log_fwd_bwd_mem and device.type == "cuda":
            torch.cuda.synchronize()
            bwd_alloc = int(torch.cuda.memory_allocated())
            bwd_reserved = int(torch.cuda.memory_reserved())
            print(
                f"mem_after_fwd_alloc={fwd_alloc} mem_after_fwd_reserved={fwd_reserved} "
                f"mem_after_bwd_alloc={bwd_alloc} mem_after_bwd_reserved={bwd_reserved}",
                flush=True,
            )
        if norm_grad:
            seq_logits.grad = seq_logits.grad * seq_logits.shape[0] ** 0.5 / seq_logits.grad.norm()
        optimizer.step()
    elif log_fwd_bwd_mem and device.type == "cuda":
        print(
            f"mem_after_fwd_alloc={fwd_alloc} mem_after_fwd_reserved={fwd_reserved} "
            f"mem_after_bwd_alloc=NA mem_after_bwd_reserved=NA",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t_start

    peak_mem_allocated = None
    peak_mem_reserved = None
    if device.type == "cuda":
        peak_mem_allocated = int(torch.cuda.max_memory_allocated())
        peak_mem_reserved = int(torch.cuda.max_memory_reserved())

    return {
        "seq_len": int(seq_len),
        "loss": float(loss.item()),
        "dist_loss": float(dist_loss.item()),
        "elapsed_s": float(elapsed_s),
        "peak_mem_allocated_bytes": peak_mem_allocated,
        "peak_mem_reserved_bytes": peak_mem_reserved,
        "use_straight_through": bool(use_straight_through),
    }


def main():
    parser = argparse.ArgumentParser(description="Scaling test: length vs GPU memory/runtime")
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="Training config used for replacement blocks",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Checkpoint containing replacement weights",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        required=True,
        help="Comma-separated sequence lengths, e.g. 64,128,256",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("~/AFdistill/outputs"),
        help="Directory to save outputs",
    )
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--dist_scale", type=float, default=1.0, help="Weight for distogram loss")
    parser.add_argument("--init_seq", type=str, default="0", help="Initial sequence: 0 or gaussian")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Optimizer to use: SGD or Adam",
    )
    parser.add_argument(
        "--norm_grad",
        action="store_true",
        default=False,
        help="Normalize the gradient of the initial sequence (SGD only recommended)",
    )
    parser.add_argument(
        "--straight_through_selection",
        type=str,
        default="cycle",
        choices=["cycle", "static_blocks"],
        help="How to choose straight-through usage: cycle or static_blocks.",
    )
    parser.add_argument(
        "--straight_through_blocks",
        type=str,
        default="",
        help="Block indices (0-based) to use straight-through when selection=static_blocks.",
    )
    parser.add_argument(
        "--triangle_st_only",
        action="store_true",
        default=False,
        help="Use straight-through replacements only for triangle ops (tri_mul/tri_att).",
    )
    parser.add_argument(
        "--triangle_kernel_size",
        type=int,
        default=3,
        help="Kernel size for triangle-op replacement convs.",
    )
    parser.add_argument(
        "--triangle_dilation_pattern",
        type=str,
        default="1,2,4,8",
        help="Comma-separated dilation pattern for triangle-op replacements.",
    )
    parser.add_argument(
        "--triangle_dilation_repeats",
        type=int,
        default=1,
        help="Number of times to repeat the triangle dilation pattern.",
    )
    parser.add_argument(
        "--use_base_model",
        action="store_true",
        default=False,
        help="Use unmodified OpenFold model (no replacement wrapping).",
    )
    parser.add_argument(
        "--disable_attention_opts",
        action="store_true",
        default=False,
        help="Disable flash/LMA/DeepSpeed attention optimizations.",
    )
    parser.add_argument(
        "--enable_flash_attention",
        action="store_true",
        default=False,
        help="Enable FlashAttention (disables LMA/DeepSpeed attention).",
    )
    parser.add_argument(
        "--enable_deepspeed_attention",
        action="store_true",
        default=False,
        help="Enable DeepSpeed Evoformer attention (disables Flash/LMA).",
    )
    parser.add_argument(
        "--disable_chunking",
        action="store_true",
        default=False,
        help="Disable chunked attention and chunk-size tuning.",
    )
    parser.add_argument(
        "--disable_extra_msa",
        action="store_true",
        default=False,
        help="Disable extra MSA stack for memory comparisons.",
    )
    parser.add_argument(
        "--msa_depth",
        type=int,
        default=1,
        help="Number of MSA rows to synthesize in the toy batch.",
    )
    parser.add_argument(
        "--straight_through_all",
        action="store_true",
        default=False,
        help="Use straight-through for all Evoformer blocks (static_blocks only).",
    )
    parser.add_argument(
        "--wrap_all_blocks",
        action="store_true",
        default=False,
        help="Wrap ALL Evoformer blocks (including those without pretrained weights) for max memory savings.",
    )
    parser.add_argument(
        "--orig_steps_per_cycle",
        type=int,
        default=0,
        help="Original-backprop steps per cycle (cycle mode).",
    )
    parser.add_argument(
        "--repl_steps_per_cycle",
        type=int,
        default=1,
        help="Replacement-straight-through steps per cycle (cycle mode).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cpu or cuda). Default: auto-detect.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=16,
        help="Number of repeats per length (median is reported).",
    )
    parser.add_argument(
        "--disable_checkpointing",
        action="store_true",
        default=False,
        help="Disable activation checkpointing (use with caution).",
    )
    parser.add_argument(
        "--ckpt_replacement_only",
        action="store_true",
        default=False,
        help="Checkpoint replacement blocks only when straight-through is enabled.",
    )
    parser.add_argument(
        "--freeze_non_replacement_params",
        action="store_true",
        default=False,
        help="Freeze only original Evoformer params for replaced blocks.",
    )
    parser.add_argument(
        "--freeze_all_non_replacement_params",
        action="store_true",
        default=False,
        help="Freeze all params except replacement blocks (proxy-style).",
    )
    parser.add_argument(
        "--freeze_all_params",
        action="store_true",
        default=False,
        help="Freeze all model parameters.",
    )
    parser.add_argument(
        "--assert_seq_grad_nonzero",
        action="store_true",
        default=False,
        help="Assert seq_logits receives nonzero gradient.",
    )
    parser.add_argument(
        "--assert_orig_grad_none",
        action="store_true",
        default=False,
        help="Assert original block grads are None when straight-through is enabled.",
    )
    parser.add_argument(
        "--log_fwd_bwd_mem",
        action="store_true",
        default=False,
        help="Log memory after forward and backward (single pass).",
    )
    parser.add_argument(
        "--log_mz_sizes",
        action="store_true",
        default=False,
        help="Log msa/pair tensor shapes and sizes (single pass).",
    )
    parser.add_argument(
        "--no_grad",
        action="store_true",
        default=False,
        help="Run forward-only (no backward, no optimizer step).",
    )
    parser.add_argument(
        "--assert_grad_nonzero",
        action="store_true",
        default=False,
        help="Assert replacement gradients are nonzero when straight-through is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    lengths = _parse_int_list(args.lengths)
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    st_blocks = ()
    if args.straight_through_selection == "static_blocks":
        st_blocks = _parse_block_indices(args.straight_through_blocks)

    config_path = args.config_path.expanduser()
    output_base_dir = args.output_dir.expanduser()
    config_name = config_path.stem
    replacement_description = (
        f"_orig-{args.orig_steps_per_cycle}_repl-{args.repl_steps_per_cycle}"
        if args.straight_through_selection == "cycle"
        else f"_st-blocks-{args.straight_through_blocks}"
    )
    out_dir = output_base_dir / config_name / f"length_scaling{replacement_description}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This scaling test requires CUDA (attention core is CUDA-only).")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.enable_flash_attention and args.disable_attention_opts:
        raise ValueError("--enable_flash_attention and --disable_attention_opts are mutually exclusive")
    if args.enable_deepspeed_attention and args.disable_attention_opts:
        raise ValueError("--enable_deepspeed_attention and --disable_attention_opts are mutually exclusive")
    if args.enable_deepspeed_attention and args.enable_flash_attention:
        raise ValueError("--enable_deepspeed_attention and --enable_flash_attention are mutually exclusive")
    if (
        args.freeze_non_replacement_params
        and args.freeze_all_non_replacement_params
        or args.freeze_all_params
        and (args.freeze_non_replacement_params or args.freeze_all_non_replacement_params)
    ):
        raise ValueError("Freeze flags are mutually exclusive")
    model = load_model(
        config_path,
        args.checkpoint_path,
        device,
        straight_through=not args.use_base_model,
        allow_replacements=not args.use_base_model,
        disable_attention_opts=args.disable_attention_opts,
        enable_flash_attention=args.enable_flash_attention,
        enable_deepspeed_attention=args.enable_deepspeed_attention,
        disable_chunking=args.disable_chunking,
        wrap_all_blocks=args.wrap_all_blocks,
        triangle_st_only=args.triangle_st_only,
        triangle_kernel_size=args.triangle_kernel_size,
        triangle_dilation_pattern=tuple(
            int(d) for d in args.triangle_dilation_pattern.split(",") if d.strip() != ""
        ),
        triangle_dilation_repeats=args.triangle_dilation_repeats,
    )
    if args.disable_extra_msa:
        model.config.extra_msa.enabled = False
        print("Extra MSA stack disabled", flush=True)
    if args.straight_through_all:
        if args.straight_through_selection != "static_blocks":
            raise ValueError("--straight_through_all requires straight_through_selection=static_blocks")
        st_blocks = tuple(range(len(model.evoformer.blocks)))
    if args.disable_checkpointing:
        if hasattr(model, "evoformer"):
            model.evoformer.blocks_per_ckpt = None
        if hasattr(model, "template_embedder") and hasattr(model.template_embedder, "template_pair_stack"):
            model.template_embedder.template_pair_stack.blocks_per_ckpt = None
        if hasattr(model, "extra_msa_stack"):
            if hasattr(model.extra_msa_stack, "ckpt"):
                model.extra_msa_stack.ckpt = False
            if hasattr(model.extra_msa_stack, "blocks"):
                for block in model.extra_msa_stack.blocks:
                    if hasattr(block, "ckpt"):
                        block.ckpt = False
        print("Activation checkpointing disabled", flush=True)
    if args.ckpt_replacement_only:
        _set_replacement_checkpoint_only(model, enabled=True)
        print("Checkpointing replacement blocks only", flush=True)
    if args.freeze_non_replacement_params:
        _freeze_replaced_evoformer_params(model, st_blocks)
    if args.freeze_all_non_replacement_params:
        _freeze_all_non_replacement_params(model)
    if args.freeze_all_params:
        _freeze_all_params(model)

    ckpt_tag = "no-ckpt" if args.disable_checkpointing else "ckpt"
    results_path = out_dir / f"length_scaling{replacement_description}_{ckpt_tag}.csv"
    existing_lengths = set()
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                seq_val = row.get("seq_len", "")
                if not str(seq_val).isdigit():
                    continue
                existing_lengths.add(int(seq_val))
    write_header = True
    if results_path.exists() and results_path.stat().st_size > 0:
        write_header = False
    open_mode = "a" if results_path.exists() else "w"
    with open(results_path, open_mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seq_len",
                "loss",
                "dist_loss",
                "elapsed_s",
                "peak_mem_allocated_bytes",
                "peak_mem_reserved_bytes",
                "use_straight_through",
                "repeats",
            ],
        )
        if write_header:
            writer.writeheader()
        for seq_len in lengths:
            if int(seq_len) in existing_lengths:
                print(f"Skipping existing length {seq_len}", flush=True)
                continue
            elapsed_s_list: List[float] = []
            mem_alloc_list: List[float] = []
            mem_reserved_list: List[float] = []
            loss_list: List[float] = []
            dist_loss_list: List[float] = []
            use_straight_through = None
            for _ in range(args.repeats):
                row = run_one_pass(
                    model=model,
                    seq_len=seq_len,
                    device=device,
                    dist_scale=args.dist_scale,
                    lr=args.lr,
                    init_seq=args.init_seq,
                    optimizer_name=args.optimizer,
                    norm_grad=args.norm_grad,
                    straight_through_selection=args.straight_through_selection,
                    straight_through_blocks=st_blocks,
                    orig_steps_per_cycle=args.orig_steps_per_cycle,
                    repl_steps_per_cycle=args.repl_steps_per_cycle,
                    msa_depth=args.msa_depth,
                    log_fwd_bwd_mem=args.log_fwd_bwd_mem,
                    log_mz_sizes=args.log_mz_sizes,
                    no_grad=args.no_grad,
                    assert_grad_nonzero=args.assert_grad_nonzero,
                    assert_seq_grad_nonzero=args.assert_seq_grad_nonzero,
                    assert_orig_grad_none=args.assert_orig_grad_none,
                )
                elapsed_s_list.append(row["elapsed_s"])
                mem_alloc_list.append(float(row["peak_mem_allocated_bytes"]))
                mem_reserved_list.append(float(row["peak_mem_reserved_bytes"]))
                loss_list.append(row["loss"])
                dist_loss_list.append(row["dist_loss"])
                if use_straight_through is None:
                    use_straight_through = row["use_straight_through"]

            med_row = {
                "seq_len": int(seq_len),
                "loss": float(np.median(np.array(loss_list, dtype=np.float64))),
                "dist_loss": float(np.median(np.array(dist_loss_list, dtype=np.float64))),
                "elapsed_s": float(np.median(np.array(elapsed_s_list, dtype=np.float64))),
                "peak_mem_allocated_bytes": int(np.median(np.array(mem_alloc_list, dtype=np.float64))),
                "peak_mem_reserved_bytes": int(np.median(np.array(mem_reserved_list, dtype=np.float64))),
                "use_straight_through": bool(use_straight_through),
                "repeats": int(args.repeats),
            }
            writer.writerow(med_row)
            handle.flush()
            print(
                f"len={med_row['seq_len']} loss={med_row['loss']:.4f} "
                f"time={med_row['elapsed_s']:.3f}s "
                f"mem_alloc={med_row['peak_mem_allocated_bytes']} "
                f"repeats={med_row['repeats']}",
                flush=True,
            )

    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()

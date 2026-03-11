#!/usr/bin/env python3

import argparse
import json
from typing import Tuple

import torch
import torch._dynamo as dynamo
from torch._dynamo.utils import counters

from openfold.config import model_config
from openfold.block_replacement_scripts.dilated_conv_evoformer_replacement import (
    DilatedConvEvoformerReplacement,
)


def _run_train_steps(
    block: torch.nn.Module,
    m: torch.Tensor,
    z: torch.Tensor,
    msa_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    m_tgt: torch.Tensor,
    z_tgt: torch.Tensor,
    iters: int,
    autocast: bool,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
        for _ in range(iters):
            block.zero_grad(set_to_none=True)
            m_out, z_out = block(m, z, msa_mask=msa_mask, pair_mask=pair_mask, _mask_trans=False)
            loss = (m_out.float() - m_tgt.float()).square().mean() + (z_out.float() - z_tgt.float()).square().mean()
            loss.backward()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark replacement block torch.compile vs eager (forward+backward).")
    parser.add_argument("--preset", type=str, default="model_1_ptm")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--dilations", type=str, default="1,2,4,8,16")
    parser.add_argument("--replacement_mode", type=str, default="shared_proj", choices=["per_block", "shared_proj"])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--autocast", action="store_true", default=True)
    parser.add_argument("--no_autocast", action="store_true", default=False)
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")
    parser.add_argument("--dynamic", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise ValueError("This benchmark requires CUDA.")

    if args.no_autocast:
        autocast = False
    else:
        autocast = bool(args.autocast)

    dils: Tuple[int, ...] = tuple(int(d) for d in str(args.dilations).split(",") if str(d).strip() != "")

    cfg = model_config(args.preset)
    evo = getattr(cfg.model, "evoformer", None) or getattr(cfg.model, "evoformer_stack")
    c_m = int(evo.c_m)
    c_z = int(evo.c_z)
    n = int(args.seq_len)

    block = DilatedConvEvoformerReplacement(
        c_m=c_m,
        c_z=c_z,
        kernel_size=int(args.kernel_size),
        dilations=dils,
        mode=args.replacement_mode,
    ).cuda()
    block.train()

    m = torch.randn(1, 1, n, c_m, device=device, dtype=torch.float32)
    z = torch.randn(1, n, n, c_z, device=device, dtype=torch.float32)
    msa_mask = torch.ones(1, 1, n, device=device, dtype=torch.float32)
    pair_mask = torch.ones(1, n, n, device=device, dtype=torch.float32)
    m_tgt = torch.randn_like(m)
    z_tgt = torch.randn_like(z)

    # Eager baseline (exclude any compilation)
    eager_avg_ms = _run_train_steps(
        block=block,
        m=m,
        z=z,
        msa_mask=msa_mask,
        pair_mask=pair_mask,
        m_tgt=m_tgt,
        z_tgt=z_tgt,
        iters=args.iters,
        autocast=autocast,
    )

    dynamo.config.suppress_errors = True
    counters.clear()

    compile_mode = None if args.compile_mode.lower() in {"none", "null"} else args.compile_mode
    block_compiled = torch.compile(block, mode=compile_mode, dynamic=bool(args.dynamic))

    # Compilation batch (excluded from timing)
    _ = _run_train_steps(
        block=block_compiled,
        m=m,
        z=z,
        msa_mask=msa_mask,
        pair_mask=pair_mask,
        m_tgt=m_tgt,
        z_tgt=z_tgt,
        iters=1,
        autocast=autocast,
    )

    compiled_avg_ms = _run_train_steps(
        block=block_compiled,
        m=m,
        z=z,
        msa_mask=msa_mask,
        pair_mask=pair_mask,
        m_tgt=m_tgt,
        z_tgt=z_tgt,
        iters=args.iters,
        autocast=autocast,
    )

    result = {
        "preset": args.preset,
        "seq_len": n,
        "kernel_size": int(args.kernel_size),
        "dilations": list(dils),
        "replacement_mode": args.replacement_mode,
        "iters": int(args.iters),
        "autocast_bf16": bool(autocast),
        "compile_mode": compile_mode,
        "dynamic": bool(args.dynamic),
        "eager_avg_ms": float(eager_avg_ms),
        "compiled_avg_ms": float(compiled_avg_ms),
        "speedup": float(eager_avg_ms / compiled_avg_ms) if compiled_avg_ms > 0 else None,
        "dynamo": {
            "frames_ok": int(counters["frames"]["ok"]) if "frames" in counters else 0,
            "unique_graphs": int(counters["stats"]["unique_graphs"]) if "stats" in counters else 0,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


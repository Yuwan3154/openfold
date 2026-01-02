#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import statistics
from typing import Dict, List, Optional, Tuple

import torch

from openfold.config import model_config
from openfold.model.evoformer import EvoformerBlock
from openfold.block_replacement_scripts.dilated_conv_evoformer_replacement import (
    DilatedConvEvoformerReplacement,
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def _get_evo_cfg(cfg):
    evo = getattr(cfg.model, "evoformer", None) or getattr(cfg.model, "evoformer_stack")
    return evo


def _make_block(preset: str, device: torch.device, dtype: torch.dtype) -> EvoformerBlock:
    cfg = model_config(preset)
    evo = _get_evo_cfg(cfg)
    block = EvoformerBlock(
        c_m=evo.c_m,
        c_z=evo.c_z,
        c_hidden_msa_att=evo.c_hidden_msa_att,
        c_hidden_opm=evo.c_hidden_opm,
        c_hidden_mul=evo.c_hidden_mul,
        c_hidden_pair_att=evo.c_hidden_pair_att,
        no_heads_msa=evo.no_heads_msa,
        no_heads_pair=evo.no_heads_pair,
        transition_n=evo.transition_n,
        msa_dropout=evo.msa_dropout,
        pair_dropout=evo.pair_dropout,
        no_column_attention=evo.no_column_attention,
        opm_first=evo.opm_first,
        fuse_projection_weights=getattr(evo, "fuse_projection_weights", False),
        inf=evo.inf,
        eps=evo.eps,
    ).to(device=device, dtype=dtype)
    block.eval()
    for p in block.parameters():
        p.requires_grad_(False)
    return block


def _make_replacement(
    preset: str,
    device: torch.device,
    dtype: torch.dtype,
    kernel_size: int,
    dilations: Tuple[int, ...],
    replacement_mode: str,
) -> DilatedConvEvoformerReplacement:
    cfg = model_config(preset)
    evo = _get_evo_cfg(cfg)
    rep = DilatedConvEvoformerReplacement(
        c_m=evo.c_m,
        c_z=evo.c_z,
        kernel_size=kernel_size,
        dilations=dilations,
        mode=replacement_mode,
    ).to(device=device, dtype=dtype)
    rep.eval()
    for p in rep.parameters():
        p.requires_grad_(False)
    return rep


def _make_inputs(
    preset: str,
    batch: int,
    n_seq: int,
    n_res: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = model_config(preset)
    evo = _get_evo_cfg(cfg)
    m = torch.randn(batch, n_seq, n_res, evo.c_m, device=device, dtype=dtype, requires_grad=True)
    z = torch.randn(batch, n_res, n_res, evo.c_z, device=device, dtype=dtype, requires_grad=True)
    msa_mask = torch.ones(batch, n_seq, n_res, device=device, dtype=dtype)
    pair_mask = torch.ones(batch, n_res, n_res, device=device, dtype=dtype)
    return m, z, msa_mask, pair_mask


def _loss_from_outputs(m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # Simple scalar loss that touches both outputs.
    return (m.float().square().mean() + z.float().square().mean())


@torch.no_grad()
def _warmup_autocast_once(device: torch.device, dtype: torch.dtype, enabled: bool) -> None:
    # Force autocast context initialization to reduce first-iter variance.
    if device.type != "cuda" or not enabled:
        return
    x = torch.ones((1,), device=device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        _ = x + 1
    torch.cuda.synchronize()


def _bench_mode(
    mode: str,
    orig: EvoformerBlock,
    rep: DilatedConvEvoformerReplacement,
    preset: str,
    batch: int,
    n_seq: int,
    n_res: int,
    device: torch.device,
    dtype: torch.dtype,
    autocast_enabled: bool,
    chunk_size: Optional[int],
    warmup: int,
    iters: int,
    dilations: str,
) -> Dict:
    if device.type != "cuda":
        raise ValueError("This benchmark is intended to run on CUDA GPUs.")

    _warmup_autocast_once(device=device, dtype=dtype, enabled=autocast_enabled)

    def forward_and_backward() -> Tuple[float, int]:
        m, z, msa_mask, pair_mask = _make_inputs(
            preset=preset,
            batch=batch,
            n_seq=n_seq,
            n_res=n_res,
            device=device,
            dtype=dtype,
        )

        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
            if mode == "original":
                m_out, z_out = orig(
                    m,
                    z,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                    chunk_size=chunk_size,
                    inplace_safe=False,
                )
                loss = _loss_from_outputs(m_out, z_out)
            elif mode == "straight_through":
                with torch.no_grad():
                    m_o, z_o = orig(
                        m,
                        z,
                        msa_mask=msa_mask,
                        pair_mask=pair_mask,
                        chunk_size=chunk_size,
                        inplace_safe=True,
                    )
                m_r, z_r = rep(
                    m,
                    z,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                )
                m_out = m_r + m_o.detach() - m_r.detach()
                z_out = z_r + z_o.detach() - z_r.detach()
                loss = _loss_from_outputs(m_out, z_out)
            elif mode == "replacement_only":
                m_r, z_r = rep(
                    m,
                    z,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                )
                loss = _loss_from_outputs(m_r, z_r)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        backward_ms = float(start.elapsed_time(end))
        peak_bytes = int(torch.cuda.max_memory_allocated())
        return backward_ms, peak_bytes

    # Warmup
    for _ in range(warmup):
        _ = forward_and_backward()

    times: List[float] = []
    peaks: List[int] = []
    for _ in range(iters):
        t, p = forward_and_backward()
        times.append(t)
        peaks.append(p)

    return {
        "mode": mode,
        "iters": iters,
        "warmup": warmup,
        "backward_ms_median": float(statistics.median(times)) if len(times) > 0 else None,
        "backward_ms_mean": float(statistics.mean(times)) if len(times) > 0 else None,
        "peak_memory_bytes_median": int(statistics.median(peaks)) if len(peaks) > 0 else None,
        "peak_memory_bytes_max": int(max(peaks)) if len(peaks) > 0 else None,
    }


def _ratio(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    if numer is None or denom is None:
        return None
    if denom == 0:
        return None
    return float(numer) / float(denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dilated-conv Evoformer replacement vs EvoformerBlock (one block).")
    parser.add_argument("--preset", type=str, default="model_1_ptm", help="OpenFold model preset for dimensions.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--n_seq", type=int, default=1, help="Number of MSA sequences")
    parser.add_argument("--n_res", type=int, default=128, help="Number of residues")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"], help="Computation dtype")
    parser.add_argument("--autocast", action="store_true", default=False, help="Enable CUDA autocast")
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size for EvoformerBlock (baseline/orig forward)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="Measured iterations")
    parser.add_argument("--out_json", type=Path, default=None, help="Optional output JSON path")
    parser.add_argument("--max_time_ratio", type=float, default=None, help="Pass if (straight_through/original) <= this")
    parser.add_argument("--max_mem_ratio", type=float, default=None, help="Pass if (straight_through/original) <= this")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for the dilated convolution")
    parser.add_argument("--dilations", type=str, default="1,2,4", help="Dilations for the dilated convolution; determines the number of convolution layers in each block")
    parser.add_argument(
        "--replacement_mode",
        type=str,
        default="per_block",
        choices=["per_block", "shared_proj"],
        help="Replacement architecture mode: per_block (current) or shared_proj (project down once, all convs in hidden, project up once)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    orig = _make_block(preset=args.preset, device=device, dtype=dtype)
    dilations = tuple(int(d) for d in args.dilations.split(","))
    rep = _make_replacement(
        preset=args.preset,
        device=device,
        dtype=dtype,
        kernel_size=args.kernel_size,
        dilations=dilations,
        replacement_mode=args.replacement_mode,
    )

    results = {
        "preset": args.preset,
        "batch": args.batch,
        "n_seq": args.n_seq,
        "n_res": args.n_res,
        "dtype": args.dtype,
        "autocast": bool(args.autocast),
        "chunk_size": args.chunk_size,
        "dilations": list(dilations),
        "replacement_mode": args.replacement_mode,
        "modes": [],
    }

    for mode in ["original", "straight_through", "replacement_only"]:
        res = _bench_mode(
            mode=mode,
            orig=orig,
            rep=rep,
            preset=args.preset,
            batch=args.batch,
            n_seq=args.n_seq,
            n_res=args.n_res,
            device=device,
            dtype=dtype,
            autocast_enabled=bool(args.autocast),
            chunk_size=args.chunk_size,
            warmup=args.warmup,
            iters=args.iters,
            dilations=args.dilations,
        )
        results["modes"].append(res)

    by_mode = {m["mode"]: m for m in results["modes"]}
    t_ratio = _ratio(by_mode["straight_through"]["backward_ms_median"], by_mode["original"]["backward_ms_median"])
    m_ratio = _ratio(by_mode["straight_through"]["peak_memory_bytes_median"], by_mode["original"]["peak_memory_bytes_median"])
    results["ratios_vs_original"] = {
        "straight_through_backward_time_median": t_ratio,
        "straight_through_peak_mem_median": m_ratio,
    }

    passed_time = True
    passed_mem = True
    if args.max_time_ratio is not None:
        passed_time = (t_ratio is not None) and (t_ratio <= args.max_time_ratio)
    if args.max_mem_ratio is not None:
        passed_mem = (m_ratio is not None) and (m_ratio <= args.max_mem_ratio)
    results["pass"] = bool(passed_time and passed_mem)

    print(json.dumps(results, indent=2))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    if args.max_time_ratio is not None or args.max_mem_ratio is not None:
        if not results["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()




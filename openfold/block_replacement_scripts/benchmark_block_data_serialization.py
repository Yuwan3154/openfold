#!/usr/bin/env python3
"""
Benchmark block-data serialization strategies on a single example protein/block.

Example:
  python benchmark_block_data_serialization.py \
    --input_pt /home/jupyter-chenxi/data/af2rank_single/af2_block_data_single/block_01/1aaj_A.pt \
    --out_dir /home/jupyter-chenxi/tmp/block_data_ser_bench
"""

import argparse
import gzip
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from safetensors.torch import safe_open as safetensors_safe_open

from openfold.block_replacement_scripts.block_data_io import (
    _df11_decode_tensor_bf16,
    _unflatten_block_sample,
    _zipnn_safetensors_decompress_all,
    load_block_sample,
    save_block_sample,
)


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())


def _collect_errors(ref: Dict, loaded: Dict) -> Tuple[float, float, float, float]:
    return (
        _max_abs_err(ref["input"]["m"], loaded["input"]["m"]),
        _max_abs_err(ref["input"]["z"], loaded["input"]["z"]),
        _max_abs_err(ref["output"]["m"], loaded["output"]["m"]),
        _max_abs_err(ref["output"]["z"], loaded["output"]["z"]),
    )


def _rss_mb() -> float:
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return kb / 1024.0
    return float("nan")


def _bitwise_equal_bf16(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape:
        return False
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        return False
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_pt",
        type=Path,
        default=Path("/home/jupyter-chenxi/data/af2rank_single/af2_block_data_single/block_01/1aaj_A.pt"),
        help="Path to an existing .pt block-data sample to benchmark against",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/jupyter-chenxi/tmp/block_data_serialization_bench"),
        help="Directory to write benchmark artifacts",
    )
    parser.add_argument(
        "--df11_gpu_device",
        type=str,
        default="",
        help="If set (e.g. 'cuda:0'), run an additional DF11 GPU decode correctness + leak test loop",
    )
    parser.add_argument(
        "--df11_gpu_iters",
        type=int,
        default=200,
        help="Number of iterations for DF11 GPU decode leak test (only when --df11_gpu_device is set)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref = torch.load(args.input_pt, map_location="cpu")
    # Normalize: ensure we have the expected dict structure
    sample = {
        "input": {"m": ref["input"]["m"], "z": ref["input"]["z"]},
        "output": {"m": ref["output"]["m"], "z": ref["output"]["z"]},
        "chain_id": ref.get("chain_id", "unknown"),
        "block_idx": int(ref.get("block_idx", -1)),
    }
    # Reference for error checks: compare against BF16-casted tensors to isolate *compression* loss.
    sample_ref = {
        "input": {
            "m": sample["input"]["m"].to(torch.bfloat16),
            "z": sample["input"]["z"].to(torch.bfloat16),
        },
        "output": {
            "m": sample["output"]["m"].to(torch.bfloat16),
            "z": sample["output"]["z"].to(torch.bfloat16),
        },
        "chain_id": sample["chain_id"],
        "block_idx": sample["block_idx"],
    }

    configs = [
        # format, save_dtype, quantization, extension
        ("pt", "bf16", "none", "pt"),
        ("pt.gz", "bf16", "none", "pt.gz"),
        ("safetensors", "bf16", "none", "safetensors"),
        ("safetensors.gz", "bf16", "none", "safetensors.gz"),
        ("safetensors.znn", "bf16", "none", "safetensors.znn"),
        ("df11.safetensors", "bf16", "none", "df11.safetensors"),
    ]

    print(
        "\t".join(
            [
                "format",
                "save_dtype",
                "quant",
                "bytes",
                "save_s",
                "load_s",
                "decompress_s",
                "err_in_m",
                "err_in_z",
                "err_out_m",
                "err_out_z",
                "path",
            ]
        )
    )

    for fmt, save_dtype, quant, ext in configs:
        out_path = args.out_dir / f"sample_{fmt.replace('.', '_')}_{save_dtype}_{quant}.{ext}"

        t0 = time.perf_counter()
        save_block_sample(
            sample,
            out_path,
            save_dtype=save_dtype,  # type: ignore[arg-type]
            quantization=quant,  # type: ignore[arg-type]
        )
        t1 = time.perf_counter()

        decompress_s = 0.0
        t_load0 = time.perf_counter()

        if out_path.name.endswith(".pt.gz"):
            raw = out_path.read_bytes()
            t_dec0 = time.perf_counter()
            raw2 = gzip.decompress(raw)
            t_dec1 = time.perf_counter()
            loaded = torch.load(io.BytesIO(raw2), map_location="cpu")
            decompress_s = t_dec1 - t_dec0

        elif out_path.name.endswith(".safetensors.gz"):
            raw = out_path.read_bytes()
            t_dec0 = time.perf_counter()
            raw2 = gzip.decompress(raw)
            t_dec1 = time.perf_counter()
            # Use safetensors load for gz bytes
            from safetensors.torch import load as safetensors_load

            st_tensors = safetensors_load(raw2)
            loaded = {
                "input": {"m": st_tensors["input.m"], "z": st_tensors["input.z"]},
                "output": {"m": st_tensors["output.m"], "z": st_tensors["output.z"]},
                "chain_id": None,
                "block_idx": None,
            }
            decompress_s = t_dec1 - t_dec0

        elif out_path.name.endswith(".safetensors.znn"):
            t_dec0 = time.perf_counter()
            tensors, metadata = _zipnn_safetensors_decompress_all(out_path, device="cpu")
            t_dec1 = time.perf_counter()
            loaded = _unflatten_block_sample(tensors, metadata=metadata)
            decompress_s = t_dec1 - t_dec0

        elif out_path.name.endswith(".df11.safetensors"):
            with safetensors_safe_open(str(out_path), framework="pt", device="cpu") as f:
                metadata = f.metadata()
                df11_meta = json.loads(metadata.get("df11", "{}"))

                tensors_out: Dict[str, torch.Tensor] = {}
                dec_total = 0.0
                for k, info in df11_meta.items():
                    enc = f.get_tensor(f"{k}.encoded_exponent")
                    sm = f.get_tensor(f"{k}.sign_mantissa")
                    counter_raw = info["counter"]
                    counter = {int(kk): int(vv) for kk, vv in counter_raw.items()}
                    shape = tuple(int(x) for x in info["shape"])

                    t_dec0 = time.perf_counter()
                    tensors_out[k] = _df11_decode_tensor_bf16(enc, sm, counter, shape)
                    t_dec1 = time.perf_counter()
                    dec_total += t_dec1 - t_dec0

            loaded = _unflatten_block_sample(tensors_out, metadata=metadata)
            decompress_s = dec_total

        else:
            loaded = load_block_sample(out_path, map_location="cpu")

        t2 = time.perf_counter()

        size = out_path.stat().st_size
        err_in_m, err_in_z, err_out_m, err_out_z = _collect_errors(sample_ref, loaded)

        print(
            "\t".join(
                [
                    fmt,
                    save_dtype,
                    quant,
                    str(size),
                    f"{(t1 - t0):.6f}",
                    f"{(t2 - t_load0):.6f}",
                    f"{decompress_s:.6f}",
                    f"{err_in_m:.6g}",
                    f"{err_in_z:.6g}",
                    f"{err_out_m:.6g}",
                    f"{err_out_z:.6g}",
                    str(out_path),
                ]
            )
        )

        # Optional: DF11 GPU decode correctness + leak test.
        if (
            fmt == "df11.safetensors"
            and args.df11_gpu_device
            and torch.cuda.is_available()
            and args.df11_gpu_device.startswith("cuda")
        ):
            device = torch.device(args.df11_gpu_device)
            torch.cuda.set_device(device)
            _ = torch.empty(1, device=device)  # initialize CUDA context
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            # One-shot correctness check (bitwise BF16).
            loaded_gpu = load_block_sample(out_path, map_location=device)
            ok = (
                _bitwise_equal_bf16(sample_ref["input"]["m"], loaded_gpu["input"]["m"].cpu())
                and _bitwise_equal_bf16(sample_ref["input"]["z"], loaded_gpu["input"]["z"].cpu())
                and _bitwise_equal_bf16(sample_ref["output"]["m"], loaded_gpu["output"]["m"].cpu())
                and _bitwise_equal_bf16(sample_ref["output"]["z"], loaded_gpu["output"]["z"].cpu())
            )
            del loaded_gpu
            torch.cuda.synchronize(device)

            print(
                f"[df11_gpu_check] device={args.df11_gpu_device} bitwise_ok={ok} "
                f"rss_mb={_rss_mb():.1f} cuda_alloc_mb={torch.cuda.memory_allocated(device)/1024**2:.1f} "
                f"cuda_peak_mb={torch.cuda.max_memory_allocated(device)/1024**2:.1f}",
                flush=True,
            )

            # Leak test loop: repeated DF11 GPU loads/decodes.
            rss0 = _rss_mb()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            alloc0 = torch.cuda.memory_allocated(device) / 1024**2

            for i in range(1, int(args.df11_gpu_iters) + 1):
                x = load_block_sample(out_path, map_location=device)
                # Touch one tensor to ensure decode ran.
                _ = float(x["input"]["z"].reshape(-1)[:1024].float().sum().item())
                del x
                torch.cuda.synchronize(device)
                if i % 20 == 0 or i == int(args.df11_gpu_iters):
                    print(
                        f"[df11_gpu_leak] iter={i} rss_mb={_rss_mb():.1f} "
                        f"cuda_alloc_mb={torch.cuda.memory_allocated(device)/1024**2:.1f} "
                        f"cuda_peak_mb={torch.cuda.max_memory_allocated(device)/1024**2:.1f}",
                        flush=True,
                    )

            rss1 = _rss_mb()
            alloc1 = torch.cuda.memory_allocated(device) / 1024**2
            print(
                f"[df11_gpu_leak_done] iters={args.df11_gpu_iters} rss_mb_delta={rss1 - rss0:.1f} "
                f"cuda_alloc_mb_delta={alloc1 - alloc0:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()



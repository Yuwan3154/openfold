#!/usr/bin/env python3
"""
Convert per-block DF11 cache files into merged per-sample safetensors files.

Input layout:  block_data_cache/block_{XX}/{sample_id}.df11.safetensors  (48 files per sample)
Output layout: block_data_cache_merged/{sample_id}.safetensors           (1 file per sample)

Each merged file contains 48*4=192 bf16 tensors keyed as:
  block_{XX}.input.m, block_{XX}.input.z, block_{XX}.output.m, block_{XX}.output.z

Usage:
  # Single GPU:
  python convert_to_merged_cache.py --src_dir data/.../block_data_cache --dst_dir data/.../block_data_cache_merged

  # Multi-GPU (4 GPUs):
  torchrun --nproc_per_node=4 convert_to_merged_cache.py --src_dir ... --dst_dir ...
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file as safetensors_save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openfold.block_replacement_scripts.block_data_io import load_block_sample


def get_all_sample_ids(src_dir: Path, block_idx: int = 1) -> list:
    """Discover all sample IDs from a reference block directory."""
    block_dir = src_dir / f"block_{block_idx:02d}"
    ids = []
    for f in sorted(block_dir.iterdir()):
        if f.name.startswith(".") or f.name.startswith("tmp"):
            continue
        name = f.name
        dot = name.index(".")
        sid = name[:dot]
        ext = name[dot:]
        ids.append((sid, ext))
    return ids


def convert_one_sample(src_dir: Path, dst_dir: Path, sid: str, ext: str, device: torch.device, n_blocks: int = 48):
    """Convert 48 per-block files into 1 merged bf16 safetensors file."""
    out_path = dst_dir / f"{sid}.safetensors"
    if out_path.exists():
        return "skip"

    all_tensors = {}
    blocks_found = 0
    for bi in range(n_blocks):
        path = src_dir / f"block_{bi:02d}" / f"{sid}{ext}"
        if not path.exists():
            continue
        data = load_block_sample(path, map_location=device)
        for role in ("input", "output"):
            for key in ("m", "z"):
                tensor_name = f"block_{bi:02d}.{role}.{key}"
                all_tensors[tensor_name] = data[role][key].cpu().to(torch.bfloat16).contiguous()
        blocks_found += 1

    if blocks_found == 0:
        return "empty"

    tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.safetensors")
    safetensors_save_file(all_tensors, str(tmp_path))
    os.replace(tmp_path, out_path)
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Convert per-block cache to merged per-sample cache")
    parser.add_argument("--src_dir", type=str, required=True, help="Source block_data_cache directory")
    parser.add_argument("--dst_dir", type=str, required=True, help="Destination merged cache directory")
    parser.add_argument("--device", type=str, default=None, help="GPU device (default: auto from LOCAL_RANK)")
    args = parser.parse_args()

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Discover samples
    all_ids = get_all_sample_ids(src_dir)
    print(f"[rank {local_rank}/{world_size}] Found {len(all_ids)} samples, using device {device}", flush=True)

    # Shard across ranks
    my_ids = [x for i, x in enumerate(all_ids) if i % world_size == local_rank]
    print(f"[rank {local_rank}] Processing {len(my_ids)} samples", flush=True)

    t0 = time.perf_counter()
    ok = 0
    skip = 0
    empty = 0
    for i, (sid, ext) in enumerate(my_ids):
        result = convert_one_sample(src_dir, dst_dir, sid, ext, device)
        if result == "ok":
            ok += 1
        elif result == "skip":
            skip += 1
        else:
            empty += 1
        if (i + 1) % 100 == 0 or i == len(my_ids) - 1:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(my_ids) - i - 1) / rate if rate > 0 else 0
            print(
                f"[rank {local_rank}] {i+1}/{len(my_ids)} "
                f"({ok} converted, {skip} skipped, {empty} empty) "
                f"{rate:.1f} samples/s, ETA {eta:.0f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(f"[rank {local_rank}] Done in {elapsed:.1f}s: {ok} converted, {skip} skipped, {empty} empty", flush=True)


if __name__ == "__main__":
    main()

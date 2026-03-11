"""
Focused: measure GPU decode time for actual training pattern (48 blocks per sample).
Only tests GPU decode approaches since CPU decode (~1.2s/file) is 23x slower.
"""
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openfold.block_replacement_scripts.block_data_io import load_block_sample


def get_sample_ids(block_data_dir, block_idx=1, limit=5):
    block_dir = Path(block_data_dir) / f"block_{block_idx:02d}"
    files = sorted(block_dir.glob("*.df11.safetensors"))[:limit]
    if not files:
        files = sorted(block_dir.glob("*.safetensors"))[:limit]
    ids_and_ext = []
    for f in files:
        name = f.name
        dot = name.index('.')
        ids_and_ext.append((name[:dot], name[dot:]))
    if ids_and_ext:
        print(f"Found {len(ids_and_ext)} samples, ext={ids_and_ext[0][1]}")
        sizes = [f.stat().st_size for f in files]
        print(f"Avg file size: {sum(sizes)/len(sizes)/1024:.1f} KB")
    return ids_and_ext


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_data_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    ids_ext = get_sample_ids(args.block_data_dir, limit=args.limit)
    if not ids_ext:
        print("No files found!")
        return
    bd = Path(args.block_data_dir)
    n_blocks = 48

    # Warmup GPU
    sid, ext = ids_ext[0]
    path = bd / "block_01" / f"{sid}{ext}"
    _ = load_block_sample(path, map_location=device)
    torch.cuda.synchronize(device)

    # 1. Single file GPU decode timing
    print("\n--- Single file GPU decode ---")
    times = []
    for _ in range(20):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = load_block_sample(path, map_location=device)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  {avg*1000:.1f}ms/file (20 trials, median={sorted(times)[10]*1000:.1f}ms)")

    # 2. Sequential 48-block load per sample (current training pattern)
    print(f"\n--- Sequential GPU: {len(ids_ext)} samples x {n_blocks} blocks ---")
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    total_loaded = 0
    per_sample = []
    for sid, ext in ids_ext:
        ts = time.perf_counter()
        for block_idx in range(n_blocks):
            path = bd / f"block_{block_idx:02d}" / f"{sid}{ext}"
            if path.exists():
                data = load_block_sample(path, map_location=device)
                total_loaded += 1
        torch.cuda.synchronize(device)
        per_sample.append(time.perf_counter() - ts)
    total = time.perf_counter() - t0
    print(f"  Total: {total:.3f}s, {total_loaded} files loaded")
    print(f"  Per sample: {[f'{t:.3f}' for t in per_sample]}")
    print(f"  Avg: {total/len(ids_ext):.3f}s/sample, {total/total_loaded*1000:.1f}ms/file")

    # 3. Measure I/O vs decode split: read file to bytes, then decode
    print(f"\n--- I/O vs decode split (single file) ---")
    import json
    from safetensors import safe_open as safetensors_safe_open
    sid, ext = ids_ext[0]
    path = bd / "block_01" / f"{sid}{ext}"
    
    # Time just the file read
    io_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with open(path, 'rb') as f:
            raw = f.read()
        io_times.append(time.perf_counter() - t0)
    avg_io = sum(io_times) / len(io_times)
    print(f"  Raw file read: {avg_io*1000:.2f}ms")
    
    # Time the full GPU load
    full_times = []
    for _ in range(10):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = load_block_sample(path, map_location=device)
        torch.cuda.synchronize(device)
        full_times.append(time.perf_counter() - t0)
    avg_full = sum(full_times) / len(full_times)
    print(f"  Full GPU load: {avg_full*1000:.2f}ms")
    print(f"  Decode overhead: {(avg_full - avg_io)*1000:.2f}ms ({(avg_full-avg_io)/avg_full*100:.0f}%)")

    print("\n--- Summary ---")
    print(f"  Per-sample loading (48 blocks): {total/len(ids_ext):.3f}s")
    print(f"  I/O is {avg_io/avg_full*100:.0f}% of load time, decode is {(avg_full-avg_io)/avg_full*100:.0f}%")
    print(f"  Optimization target: overlap {total/len(ids_ext):.3f}s data loading with training compute")


if __name__ == "__main__":
    main()

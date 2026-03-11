"""
Focused profiling of training data loading: GPU decode vs threaded CPU+transfer.
Simulates the actual training loop pattern (48 blocks per sample).
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
    print(f"Found {len(ids_and_ext)} samples, ext={ids_and_ext[0][1] if ids_and_ext else '?'}")
    print(f"Avg file size: {sum(f.stat().st_size for f in files)/len(files)/1024:.1f} KB")
    return ids_and_ext


def test_sequential_gpu(block_data_dir, sample_ids_ext, device, n_blocks=48):
    """Current approach: load each block file sequentially with GPU decode."""
    bd = Path(block_data_dir)
    n = len(sample_ids_ext)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    loaded = 0
    for sid, ext in sample_ids_ext:
        for block_idx in range(n_blocks):
            path = bd / f"block_{block_idx:02d}" / f"{sid}{ext}"
            if path.exists():
                data = load_block_sample(path, map_location=device)
                loaded += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    print(f"[Sequential GPU]     {elapsed:.3f}s total, {elapsed/n:.3f}s/sample, {loaded} files, {elapsed/loaded*1000:.1f}ms/file")
    return elapsed


def test_threaded_cpu_then_gpu(block_data_dir, sample_ids_ext, device, n_blocks=48, n_workers=8):
    """Load all 48 blocks on CPU in parallel threads, then transfer to GPU."""
    bd = Path(block_data_dir)
    n = len(sample_ids_ext)

    def load_cpu(path):
        return load_block_sample(path, map_location="cpu")

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    loaded = 0
    for sid, ext in sample_ids_ext:
        paths = []
        for block_idx in range(n_blocks):
            path = bd / f"block_{block_idx:02d}" / f"{sid}{ext}"
            if path.exists():
                paths.append(path)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(load_cpu, paths))
        loaded += len(results)
        for data in results:
            for role in ("input", "output"):
                for k, v in data[role].items():
                    data[role][k] = v.to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    print(f"[Threaded CPU→GPU {n_workers}w] {elapsed:.3f}s total, {elapsed/n:.3f}s/sample, {loaded} files, {elapsed/loaded*1000:.1f}ms/file")
    return elapsed


def test_threaded_cpu_pin_gpu(block_data_dir, sample_ids_ext, device, n_blocks=48, n_workers=8):
    """Load on CPU threads, pin memory, then async transfer to GPU."""
    bd = Path(block_data_dir)
    n = len(sample_ids_ext)

    def load_cpu_pin(path):
        data = load_block_sample(path, map_location="cpu")
        for role in ("input", "output"):
            for k, v in data[role].items():
                data[role][k] = v.pin_memory()
        return data

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    loaded = 0
    for sid, ext in sample_ids_ext:
        paths = []
        for block_idx in range(n_blocks):
            path = bd / f"block_{block_idx:02d}" / f"{sid}{ext}"
            if path.exists():
                paths.append(path)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(load_cpu_pin, paths))
        loaded += len(results)
        for data in results:
            for role in ("input", "output"):
                for k, v in data[role].items():
                    data[role][k] = v.to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    print(f"[Threaded CPU+pin {n_workers}w] {elapsed:.3f}s total, {elapsed/n:.3f}s/sample, {loaded} files, {elapsed/loaded*1000:.1f}ms/file")
    return elapsed


def test_single_file_gpu(block_data_dir, sample_ids_ext, device):
    """Measure single file GPU decode time."""
    bd = Path(block_data_dir)
    sid, ext = sample_ids_ext[0]
    path = bd / "block_01" / f"{sid}{ext}"
    # Warmup
    _ = load_block_sample(path, map_location=device)
    torch.cuda.synchronize(device)
    
    times = []
    for _ in range(10):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = load_block_sample(path, map_location=device)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"[Single file GPU]    {avg*1000:.1f}ms/file (10 trials)")


def test_single_file_cpu(block_data_dir, sample_ids_ext):
    """Measure single file CPU decode time."""
    bd = Path(block_data_dir)
    sid, ext = sample_ids_ext[0]
    path = bd / "block_01" / f"{sid}{ext}"
    # Warmup
    _ = load_block_sample(path, map_location="cpu")
    
    t0 = time.perf_counter()
    _ = load_block_sample(path, map_location="cpu")
    elapsed = time.perf_counter() - t0
    print(f"[Single file CPU]    {elapsed*1000:.1f}ms/file")


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
        return

    print("\n--- Per-file decode time ---")
    test_single_file_gpu(args.block_data_dir, ids_ext, device)
    test_single_file_cpu(args.block_data_dir, ids_ext)

    print(f"\n--- Full training loop pattern ({len(ids_ext)} samples x 48 blocks) ---")
    t_seq = test_sequential_gpu(args.block_data_dir, ids_ext, device)
    t_thr = test_threaded_cpu_then_gpu(args.block_data_dir, ids_ext, device, n_workers=4)
    test_threaded_cpu_then_gpu(args.block_data_dir, ids_ext, device, n_workers=8)
    t_pin = test_threaded_cpu_pin_gpu(args.block_data_dir, ids_ext, device, n_workers=4)
    test_threaded_cpu_pin_gpu(args.block_data_dir, ids_ext, device, n_workers=8)

    print(f"\nSummary: seq_gpu={t_seq:.3f}s, best_threaded={min(t_thr,t_pin):.3f}s")


if __name__ == "__main__":
    main()

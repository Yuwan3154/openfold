"""
Profile data loading performance for block data cache files.
Tests: sequential loading, threaded prefetching, and multiprocess prefetching.
"""
import os
import sys
import time
import queue
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch

# Ensure openfold is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openfold.block_replacement_scripts.block_data_io import load_block_sample


def find_sample_files(block_data_dir: str, block_idx: int = 0, limit: int = 50):
    """Find sample files for a given block."""
    block_dir = Path(block_data_dir) / f"block_{block_idx:02d}"
    if not block_dir.exists():
        raise FileNotFoundError(f"Block dir not found: {block_dir}")
    files = sorted(block_dir.glob("*.df11.safetensors"))[:limit]
    if not files:
        files = sorted(block_dir.glob("*.safetensors"))[:limit]
    if not files:
        files = sorted(block_dir.glob("*.pt"))[:limit]
    print(f"Found {len(files)} files in {block_dir}")
    return files


def benchmark_sequential_cpu(files, n_repeats=1):
    """Load files sequentially on CPU."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        for f in files:
            data = load_block_sample(f, map_location="cpu")
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = sum(times) / len(times)
    print(f"Sequential CPU:  {avg:.3f}s for {len(files)} files ({avg/len(files)*1000:.1f}ms/file)")
    return avg


def benchmark_sequential_gpu(files, device, n_repeats=1):
    """Load files sequentially with GPU decode."""
    times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for f in files:
            data = load_block_sample(f, map_location=device)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = sum(times) / len(times)
    print(f"Sequential GPU:  {avg:.3f}s for {len(files)} files ({avg/len(files)*1000:.1f}ms/file)")
    return avg


def benchmark_threaded_cpu(files, n_workers=4, n_repeats=1):
    """Load files on CPU using ThreadPoolExecutor."""
    def load_one(f):
        return load_block_sample(f, map_location="cpu")

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(load_one, files))
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = sum(times) / len(times)
    print(f"Threaded CPU ({n_workers}w): {avg:.3f}s for {len(files)} files ({avg/len(files)*1000:.1f}ms/file)")
    return avg


def benchmark_threaded_cpu_then_gpu(files, device, n_workers=4, n_repeats=1):
    """Load files on CPU using threads, then transfer to GPU."""
    def load_one(f):
        return load_block_sample(f, map_location="cpu")

    times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(load_one, files))
        # Transfer all to GPU
        for data in results:
            for role in ("input", "output"):
                for k, v in data[role].items():
                    data[role][k] = v.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = sum(times) / len(times)
    print(f"Threaded CPU→GPU ({n_workers}w): {avg:.3f}s for {len(files)} files ({avg/len(files)*1000:.1f}ms/file)")
    return avg


def benchmark_prefetch_pipeline(files, device, n_ahead=2, n_repeats=1):
    """
    Simulate training loop with prefetch:
    - Background thread loads next batch(es) on CPU
    - Main thread processes current batch on GPU
    """
    def loader_thread(file_queue, result_queue, stop_event):
        while not stop_event.is_set():
            try:
                f = file_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            data = load_block_sample(f, map_location="cpu")
            # Pin memory for faster transfer
            for role in ("input", "output"):
                for k, v in data[role].items():
                    data[role][k] = v.pin_memory()
            result_queue.put(data)

    times = []
    for _ in range(n_repeats):
        file_q = queue.Queue()
        result_q = queue.Queue(maxsize=n_ahead)
        stop = threading.Event()

        # Start loader threads
        threads = []
        for _ in range(n_ahead):
            t = threading.Thread(target=loader_thread, args=(file_q, result_q, stop), daemon=True)
            t.start()
            threads.append(t)

        # Enqueue all files
        for f in files:
            file_q.put(f)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(len(files)):
            data = result_q.get()
            # Transfer to GPU (fast from pinned memory)
            for role in ("input", "output"):
                for k, v in data[role].items():
                    data[role][k] = v.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        stop.set()
        for t in threads:
            t.join(timeout=2)
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    print(f"Prefetch pipeline ({n_ahead} ahead): {avg:.3f}s for {len(files)} files ({avg/len(files)*1000:.1f}ms/file)")
    return avg


def benchmark_training_loop_simulation(files, device, n_blocks=48, n_repeats=1):
    """
    Simulate actual training loop pattern:
    - For each sample, load 48 blocks sequentially (current behavior)
    vs
    - Prefetch all 48 blocks for next sample while processing current
    """
    block_data_dir = files[0].parent.parent
    # Extract sample ID and full extension correctly
    # e.g. "1abc_A.df11.safetensors" -> id="1abc_A", ext=".df11.safetensors"
    example_name = files[0].name
    first_dot = example_name.index('.')
    sample_ids = [f.name[:f.name.index('.')] for f in files]
    ext = example_name[first_dot:]

    # Current approach: sequential load per block
    n_samples = min(5, len(sample_ids))
    print(f"\n--- Training loop simulation ({n_samples} samples x {n_blocks} blocks) ---")

    # Sequential (current)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for sid in sample_ids[:n_samples]:
        for block_idx in range(n_blocks):
            path = block_data_dir / f"block_{block_idx:02d}" / f"{sid}{ext}"
            if path.exists():
                data = load_block_sample(path, map_location=device)
    torch.cuda.synchronize(device)
    t_seq = time.perf_counter() - t0
    print(f"Sequential GPU per sample: {t_seq:.3f}s total, {t_seq/n_samples:.3f}s/sample")

    # Threaded CPU load + GPU transfer
    def load_cpu(path):
        return load_block_sample(path, map_location="cpu")

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for sid in sample_ids[:n_samples]:
        paths = []
        for block_idx in range(n_blocks):
            path = block_data_dir / f"block_{block_idx:02d}" / f"{sid}{ext}"
            if path.exists():
                paths.append(path)
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(load_cpu, paths))
        # Transfer to GPU
        for data in results:
            for role in ("input", "output"):
                for k, v in data[role].items():
                    data[role][k] = v.to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    t_thread = time.perf_counter() - t0
    print(f"Threaded CPU→GPU per sample: {t_thread:.3f}s total, {t_thread/n_samples:.3f}s/sample")

    # Threaded CPU load + pin_memory + GPU transfer
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for sid in sample_ids[:n_samples]:
        paths = []
        for block_idx in range(n_blocks):
            path = block_data_dir / f"block_{block_idx:02d}" / f"{sid}{ext}"
            if path.exists():
                paths.append(path)
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(load_cpu, paths))
        # Pin then transfer
        for data in results:
            for role in ("input", "output"):
                for k, v in data[role].items():
                    data[role][k] = v.pin_memory().to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    t_pin = time.perf_counter() - t0
    print(f"Threaded CPU+pin→GPU per sample: {t_pin:.3f}s total, {t_pin/n_samples:.3f}s/sample")

    speedup_thread = t_seq / t_thread if t_thread > 0 else float('inf')
    speedup_pin = t_seq / t_pin if t_pin > 0 else float('inf')
    print(f"\nSpeedup: threaded={speedup_thread:.2f}x, threaded+pin={speedup_pin:.2f}x")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_data_dir", type=str, required=True)
    parser.add_argument("--block_idx", type=int, default=1)
    parser.add_argument("--limit", type=int, default=20, help="Number of files to test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_blocks", type=int, default=48)
    args = parser.parse_args()

    device = torch.device(args.device)
    files = find_sample_files(args.block_data_dir, block_idx=args.block_idx, limit=args.limit)
    if not files:
        print("No files found!")
        return

    # Show file size
    sizes = [f.stat().st_size for f in files]
    avg_size = sum(sizes) / len(sizes)
    print(f"Avg file size: {avg_size/1024:.1f} KB")

    print("\n--- Single block loading benchmarks ---")
    benchmark_sequential_cpu(files)
    benchmark_sequential_gpu(files, device)
    benchmark_threaded_cpu(files, n_workers=4)
    benchmark_threaded_cpu(files, n_workers=8)
    benchmark_threaded_cpu_then_gpu(files, device, n_workers=4)
    benchmark_threaded_cpu_then_gpu(files, device, n_workers=8)
    benchmark_prefetch_pipeline(files, device, n_ahead=2)
    benchmark_prefetch_pipeline(files, device, n_ahead=4)

    # Full training loop simulation
    benchmark_training_loop_simulation(files, device, n_blocks=args.n_blocks)


if __name__ == "__main__":
    main()

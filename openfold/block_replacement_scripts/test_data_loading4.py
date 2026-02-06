"""
Test threaded I/O for GPU decode path.
Since I/O is 98% of load time and releases GIL, threading should help.
"""
import os
import sys
import time
import threading
import queue
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
    if ids_ext := ids_and_ext:
        print(f"Found {len(ids_ext)} samples, ext={ids_ext[0][1]}")
    return ids_and_ext


def test_sequential_gpu(bd, sid, ext, device, n_blocks=48):
    """Load 48 blocks sequentially with GPU decode."""
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    loaded = 0
    results = {}
    for block_idx in range(n_blocks):
        path = bd / f"block_{block_idx:02d}" / f"{sid}{ext}"
        if path.exists():
            results[block_idx] = load_block_sample(path, map_location=device)
            loaded += 1
    torch.cuda.synchronize(device)
    return time.perf_counter() - t0, loaded


def test_threaded_gpu(bd, sid, ext, device, n_blocks=48, n_workers=4):
    """Load 48 blocks using thread pool with GPU decode."""
    paths = []
    for block_idx in range(n_blocks):
        path = bd / f"block_{block_idx:02d}" / f"{sid}{ext}"
        if path.exists():
            paths.append((block_idx, path))

    def load_one(args):
        idx, path = args
        return idx, load_block_sample(path, map_location=device)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = dict(ex.map(load_one, paths))
    torch.cuda.synchronize(device)
    return time.perf_counter() - t0, len(paths)


def test_threaded_read_seq_decode(bd, sid, ext, device, n_blocks=48, n_workers=4):
    """Read raw bytes in threads, then decode on GPU sequentially."""
    paths = []
    for block_idx in range(n_blocks):
        path = bd / f"block_{block_idx:02d}" / f"{sid}{ext}"
        if path.exists():
            paths.append((block_idx, path))

    def read_bytes(args):
        idx, path = args
        with open(path, 'rb') as f:
            raw = f.read()
        return idx, path, raw

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    # Phase 1: read all files in parallel (I/O, releases GIL)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        raw_data = list(ex.map(read_bytes, paths))
    t_io = time.perf_counter() - t0

    # Phase 2: decode sequentially on GPU
    t_dec_start = time.perf_counter()
    results = {}
    for idx, path, raw in raw_data:
        # We still need to use load_block_sample since decode needs safetensors parsing
        # But the OS will serve from page cache since we just read it
        results[idx] = load_block_sample(path, map_location=device)
    torch.cuda.synchronize(device)
    t_total = time.perf_counter() - t0
    t_dec = time.perf_counter() - t_dec_start
    return t_total, t_io, t_dec, len(paths)


def test_prefetch_next_sample(bd, sample_ids_ext, device, n_blocks=48, n_workers=4):
    """
    Simulate overlap: while 'processing' current sample (sleep),
    prefetch next sample's data in background.
    """
    processing_time = 0.5  # simulated training time per sample

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    prefetch_q = queue.Queue(maxsize=1)
    stop = threading.Event()

    def prefetch_worker():
        for sid, ext in sample_ids_ext:
            if stop.is_set():
                break
            paths = []
            for block_idx in range(n_blocks):
                path = bd / f"block_{block_idx:02d}" / f"{sid}{ext}"
                if path.exists():
                    paths.append(path)

            def load_one(p):
                return load_block_sample(p, map_location=device)

            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(load_one, paths))
            torch.cuda.synchronize(device)
            prefetch_q.put(results)

    t = threading.Thread(target=prefetch_worker, daemon=True)
    t.start()

    for i in range(len(sample_ids_ext)):
        data = prefetch_q.get()
        # Simulate processing
        time.sleep(processing_time)

    stop.set()
    t.join(timeout=2)
    total = time.perf_counter() - t0
    ideal = len(sample_ids_ext) * processing_time
    print(f"  Prefetch pipeline: {total:.3f}s (ideal {ideal:.3f}s if fully overlapped)")
    return total


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
    bd = Path(args.block_data_dir)

    # Warmup
    sid, ext = ids_ext[0]
    path = bd / "block_01" / f"{sid}{ext}"
    _ = load_block_sample(path, map_location=device)
    torch.cuda.synchronize(device)

    print(f"\n--- Per-sample (48 blocks) loading comparison ---")
    for sid, ext in ids_ext[:3]:
        print(f"\nSample: {sid}")
        t_seq, n = test_sequential_gpu(bd, sid, ext, device)
        print(f"  Sequential GPU:           {t_seq:.3f}s ({n} files, {t_seq/n*1000:.1f}ms/file)")

        t_thr2, n = test_threaded_gpu(bd, sid, ext, device, n_workers=2)
        print(f"  Threaded GPU (2w):        {t_thr2:.3f}s ({t_thr2/n*1000:.1f}ms/file, {t_seq/t_thr2:.2f}x)")

        t_thr4, n = test_threaded_gpu(bd, sid, ext, device, n_workers=4)
        print(f"  Threaded GPU (4w):        {t_thr4:.3f}s ({t_thr4/n*1000:.1f}ms/file, {t_seq/t_thr4:.2f}x)")

        t_thr8, n = test_threaded_gpu(bd, sid, ext, device, n_workers=8)
        print(f"  Threaded GPU (8w):        {t_thr8:.3f}s ({t_thr8/n*1000:.1f}ms/file, {t_seq/t_thr8:.2f}x)")

        t_rsd, t_io, t_dec, n = test_threaded_read_seq_decode(bd, sid, ext, device, n_workers=8)
        print(f"  Read//+SeqDecode (8w):    {t_rsd:.3f}s (io={t_io:.3f}s, dec={t_dec:.3f}s, {t_seq/t_rsd:.2f}x)")

    print(f"\n--- Prefetch pipeline simulation (5 samples, 0.5s simulated training) ---")
    test_prefetch_next_sample(bd, ids_ext, device, n_workers=4)


if __name__ == "__main__":
    main()

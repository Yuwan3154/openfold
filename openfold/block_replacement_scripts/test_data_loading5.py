"""
Test optimal thread count for batch_size=4, 48 blocks pattern.
Also test whether DataLoader num_workers helps.
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openfold.block_replacement_scripts.block_data_io import load_block_sample


def get_samples(block_data_dir, block_idx=1, limit=20):
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
        print(f"File sizes: min={min(sizes)/1024:.0f}KB, max={max(sizes)/1024:.0f}KB, avg={sum(sizes)/len(sizes)/1024:.0f}KB")
    return ids_and_ext


def load_batch_sequential(bd, batch_ids_ext, device, n_blocks=48):
    """Current pattern: load all blocks for all samples in batch sequentially."""
    loaded = 0
    for sid, ext in batch_ids_ext:
        for block_idx in range(n_blocks):
            path = bd / f"block_{block_idx:02d}" / f"{sid}{ext}"
            if path.exists():
                _ = load_block_sample(path, map_location=device)
                loaded += 1
    return loaded


def load_batch_threaded(bd, batch_ids_ext, device, n_blocks=48, n_workers=2):
    """Preload all block files for a batch using thread pool."""
    tasks = []
    for sid, ext in batch_ids_ext:
        for block_idx in range(n_blocks):
            path = bd / f"block_{block_idx:02d}" / f"{sid}{ext}"
            if path.exists():
                tasks.append(path)

    def _load(p):
        return load_block_sample(p, map_location=device)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_load, tasks))
    return len(results)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_batches", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    bd = Path(args.block_data_dir)
    all_ids = get_samples(args.block_data_dir, limit=args.batch_size * args.n_batches)
    if len(all_ids) < args.batch_size:
        print("Not enough samples")
        return

    # Make batches
    batches = []
    for i in range(0, len(all_ids) - args.batch_size + 1, args.batch_size):
        batches.append(all_ids[i:i + args.batch_size])
    batches = batches[:args.n_batches]
    print(f"\nTesting {len(batches)} batches of {args.batch_size} samples each (48 blocks/sample)")
    print(f"Files per batch: {args.batch_size * 48}")

    # Warmup
    sid, ext = all_ids[0]
    path = bd / "block_01" / f"{sid}{ext}"
    _ = load_block_sample(path, map_location=device)
    torch.cuda.synchronize(device)

    # Drop page cache if possible (unlikely without root, but try)
    os.system("sync")

    # Test sequential
    print("\n--- Sequential ---")
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    total_files = 0
    for batch in batches:
        total_files += load_batch_sequential(bd, batch, device)
    torch.cuda.synchronize(device)
    t_seq = time.perf_counter() - t0
    print(f"  {t_seq:.3f}s total, {t_seq/len(batches):.3f}s/batch, {t_seq/total_files*1000:.1f}ms/file")

    # Test threaded with different worker counts
    for nw in [2, 3, 4, 6, 8]:
        print(f"\n--- Threaded ({nw} workers) ---")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        total_files = 0
        for batch in batches:
            total_files += load_batch_threaded(bd, batch, device, n_workers=nw)
        torch.cuda.synchronize(device)
        t_thr = time.perf_counter() - t0
        speedup = t_seq / t_thr
        print(f"  {t_thr:.3f}s total, {t_thr/len(batches):.3f}s/batch, {t_thr/total_files*1000:.1f}ms/file, {speedup:.2f}x")

    # Test: does num_workers in DataLoader help?
    print("\n--- DataLoader num_workers test ---")
    print("  In cache-only mode, DataLoader only loads seq metadata + creates masks.")
    print("  Heavy I/O (block files) happens in training_step, not DataLoader.")
    print("  So num_workers for DataLoader is irrelevant to the I/O bottleneck.")
    print("  However, num_workers>0 is safe in cache-only mode (no CUDA in collate_fn).")


if __name__ == "__main__":
    main()

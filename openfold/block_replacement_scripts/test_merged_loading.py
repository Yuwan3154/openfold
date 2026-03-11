"""
Test: does merging 48 per-block files into 1 file per sample speed up loading?

Current: 48 separate .df11.safetensors files, each opened/parsed independently.
Proposed: 1 merged safetensors file with all 48 blocks' tensors keyed by block index.
"""
import os
import sys
import time
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import safe_open as safetensors_safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openfold.block_replacement_scripts.block_data_io import load_block_sample


def get_sample_ids(block_data_dir, block_idx=1, limit=10):
    block_dir = Path(block_data_dir) / f"block_{block_idx:02d}"
    files = sorted(block_dir.glob("*.df11.safetensors"))[:limit]
    ids_and_ext = []
    for f in files:
        name = f.name
        dot = name.index('.')
        ids_and_ext.append((name[:dot], name[dot:]))
    return ids_and_ext


def load_48_separate(bd, sid, ext, device):
    """Current approach: load 48 separate files."""
    results = {}
    for bi in range(48):
        path = bd / f"block_{bi:02d}" / f"{sid}{ext}"
        if path.exists():
            results[bi] = load_block_sample(path, map_location=device)
    return results


def create_merged_bf16(bd, sid, ext, device, out_path):
    """
    Create a single merged safetensors file containing all 48 blocks' tensors
    decoded to bf16 (no DF11 compression, just raw bf16 safetensors).
    """
    all_tensors = {}
    for bi in range(48):
        path = bd / f"block_{bi:02d}" / f"{sid}{ext}"
        if not path.exists():
            continue
        data = load_block_sample(path, map_location=device)
        for role in ("input", "output"):
            for key in ("m", "z"):
                tensor_name = f"block_{bi:02d}.{role}.{key}"
                all_tensors[tensor_name] = data[role][key].cpu().to(torch.bfloat16).contiguous()
    safetensors_save_file(all_tensors, str(out_path))
    return len(all_tensors)


def load_merged_bf16(path, device):
    """Load all 48 blocks from a single merged safetensors file."""
    results = {}
    with safetensors_safe_open(str(path), framework="pt", device=str(device)) as f:
        keys = list(f.keys())
        for key in keys:
            # Parse "block_XX.role.tensor" format
            parts = key.split(".")
            bi = int(parts[0].split("_")[1])
            role = parts[1]
            tensor_key = parts[2]
            if bi not in results:
                results[bi] = {"input": {}, "output": {}}
            results[bi][role][tensor_key] = f.get_tensor(key)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    bd = Path(args.block_data_dir)
    ids_ext = get_sample_ids(args.block_data_dir, limit=args.n_samples)
    if not ids_ext:
        print("No samples found")
        return

    print(f"Testing {len(ids_ext)} samples")

    # Warmup
    sid, ext = ids_ext[0]
    path = bd / "block_01" / f"{sid}{ext}"
    _ = load_block_sample(path, map_location=device)
    torch.cuda.synchronize(device)

    # Create merged files in a temp dir
    tmpdir = Path(tempfile.mkdtemp())
    print(f"\nCreating merged files in {tmpdir}...")
    for sid, ext in ids_ext:
        out_path = tmpdir / f"{sid}.safetensors"
        n_tensors = create_merged_bf16(bd, sid, ext, device, out_path)
    torch.cuda.synchronize(device)

    # Report file sizes
    for sid, ext in ids_ext[:3]:
        merged_size = (tmpdir / f"{sid}.safetensors").stat().st_size
        sep_total = 0
        for bi in range(48):
            p = bd / f"block_{bi:02d}" / f"{sid}{ext}"
            if p.exists():
                sep_total += p.stat().st_size
        print(f"  {sid}: separate={sep_total/1024:.0f}KB, merged={merged_size/1024:.0f}KB, ratio={merged_size/sep_total:.2f}x")

    # Benchmark: 48 separate DF11 files (current)
    print(f"\n--- 48 separate DF11 files (current approach) ---")
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for sid, ext in ids_ext:
        _ = load_48_separate(bd, sid, ext, device)
    torch.cuda.synchronize(device)
    t_sep = time.perf_counter() - t0
    print(f"  {t_sep:.3f}s total, {t_sep/len(ids_ext)*1000:.0f}ms/sample")

    # Benchmark: 1 merged bf16 safetensors file
    print(f"\n--- 1 merged bf16 safetensors file ---")
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for sid, ext in ids_ext:
        _ = load_merged_bf16(tmpdir / f"{sid}.safetensors", device)
    torch.cuda.synchronize(device)
    t_merged = time.perf_counter() - t0
    print(f"  {t_merged:.3f}s total, {t_merged/len(ids_ext)*1000:.0f}ms/sample")

    print(f"\n--- Summary ---")
    print(f"  Separate (48 DF11 files): {t_sep/len(ids_ext)*1000:.0f}ms/sample")
    print(f"  Merged (1 bf16 file):     {t_merged/len(ids_ext)*1000:.0f}ms/sample")
    print(f"  Speedup: {t_sep/t_merged:.2f}x")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()

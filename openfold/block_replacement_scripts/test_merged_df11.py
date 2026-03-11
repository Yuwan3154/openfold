"""
Test option 2: merge all 48 blocks' DF11 components into a single safetensors file.
Keeps compression (no disk size increase) while eliminating 47 file-open overheads.
"""
import os
import sys
import time
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import safe_open as safetensors_safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openfold.block_replacement_scripts.block_data_io import (
    load_block_sample,
    DF11_THREADS_PER_BLOCK,
    DF11_BYTES_PER_THREAD,
    _df11_decode_tensor_bf16_v2_gpu,
    _df11_decode_tensor_bf16,
    DFLOAT11_CUPY_AVAILABLE,
)


def get_sample_ids(block_data_dir, block_idx=1, limit=10):
    block_dir = Path(block_data_dir) / f"block_{block_idx:02d}"
    files = sorted(block_dir.glob("*.df11.safetensors"))[:limit]
    ids_and_ext = []
    for f in files:
        name = f.name
        dot = name.index('.')
        ids_and_ext.append((name[:dot], name[dot:]))
    return ids_and_ext


def create_merged_df11(bd, sid, ext, out_path, n_blocks=48):
    """
    Merge 48 separate DF11 safetensors files into one.
    Tensor keys: block_XX.{role}.{key}.{component}  (e.g. block_00.input.m.encoded_exponent)
    Metadata: per-block df11 decode info stored in a single JSON blob.
    """
    all_tensors = {}
    merged_df11_meta = {}  # block_XX.role.key -> {shape, counter, threads_per_block, bytes_per_thread}

    for bi in range(n_blocks):
        path = bd / f"block_{bi:02d}" / f"{sid}{ext}"
        if not path.exists():
            continue
        prefix = f"block_{bi:02d}"

        with safetensors_safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata()
            df11_meta = json.loads(metadata.get("df11", "{}"))
            tpb = int(metadata.get("df11_threads_per_block", str(DF11_THREADS_PER_BLOCK)))
            bpt = int(metadata.get("df11_bytes_per_thread", str(DF11_BYTES_PER_THREAD)))

            for k, info in df11_meta.items():
                # k is like "input.m", "output.z"
                full_key = f"{prefix}.{k}"
                for component in ("encoded_exponent", "sign_mantissa", "luts", "output_positions", "gaps"):
                    tensor_name = f"{full_key}.{component}"
                    all_tensors[tensor_name] = f.get_tensor(f"{k}.{component}")
                merged_df11_meta[full_key] = {
                    "shape": info["shape"],
                    "counter": info["counter"],
                    "threads_per_block": tpb,
                    "bytes_per_thread": bpt,
                }

    metadata = {
        "schema": "openfold_block_data_merged_df11_v1",
        "df11": json.dumps(merged_df11_meta),
    }
    safetensors_save_file(all_tensors, str(out_path), metadata=metadata)
    return len(all_tensors)


def load_merged_df11(path, device):
    """Load all 48 blocks from a single merged DF11 safetensors file."""
    results = {}
    dev = device if isinstance(device, torch.device) else torch.device(str(device))
    allow_gpu = dev.type == "cuda" and DFLOAT11_CUPY_AVAILABLE

    with safetensors_safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata()
        df11_meta = json.loads(metadata.get("df11", "{}"))

        for full_key, info in df11_meta.items():
            # full_key: "block_XX.role.key"
            parts = full_key.split(".")
            bi = int(parts[0].split("_")[1])
            role = parts[1]
            tensor_key = parts[2]

            shape = tuple(int(x) for x in info["shape"])
            tpb = info["threads_per_block"]
            bpt = info["bytes_per_thread"]

            enc = f.get_tensor(f"{full_key}.encoded_exponent")
            sm = f.get_tensor(f"{full_key}.sign_mantissa")

            if allow_gpu:
                luts = f.get_tensor(f"{full_key}.luts")
                outpos = f.get_tensor(f"{full_key}.output_positions")
                gaps = f.get_tensor(f"{full_key}.gaps")
                decoded = _df11_decode_tensor_bf16_v2_gpu(
                    luts=luts,
                    encoded_exponent=enc,
                    sign_mantissa=sm,
                    output_positions=outpos,
                    gaps=gaps,
                    shape=shape,
                    device=dev,
                    threads_per_block=tpb,
                    bytes_per_thread=bpt,
                )
            else:
                counter_raw = info["counter"]
                counter = {int(kk): int(vv) for kk, vv in counter_raw.items()}
                decoded = _df11_decode_tensor_bf16(enc, sm, counter, shape)

            if bi not in results:
                results[bi] = {"input": {}, "output": {}}
            results[bi][role][tensor_key] = decoded

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

    # Create merged DF11 files
    tmpdir = Path(tempfile.mkdtemp())
    print(f"\nCreating merged DF11 files in {tmpdir}...")
    for sid, ext in ids_ext:
        out_path = tmpdir / f"{sid}.merged.df11.safetensors"
        n = create_merged_df11(bd, sid, ext, out_path)
    torch.cuda.synchronize(device)

    # Report file sizes
    print("\nFile sizes:")
    for sid, ext in ids_ext[:3]:
        merged_size = (tmpdir / f"{sid}.merged.df11.safetensors").stat().st_size
        sep_total = 0
        for bi in range(48):
            p = bd / f"block_{bi:02d}" / f"{sid}{ext}"
            if p.exists():
                sep_total += p.stat().st_size
        print(f"  {sid}: separate={sep_total/1024:.0f}KB, merged_df11={merged_size/1024:.0f}KB, ratio={merged_size/sep_total:.2f}x")

    # Verify correctness on first sample
    print("\nVerifying correctness...")
    sid, ext = ids_ext[0]
    sep_data = {}
    for bi in range(48):
        p = bd / f"block_{bi:02d}" / f"{sid}{ext}"
        if p.exists():
            sep_data[bi] = load_block_sample(p, map_location=device)
    merged_data = load_merged_df11(tmpdir / f"{sid}.merged.df11.safetensors", device)
    torch.cuda.synchronize(device)

    all_match = True
    for bi in sorted(sep_data.keys()):
        for role in ("input", "output"):
            for key in ("m", "z"):
                t_sep = sep_data[bi][role][key]
                t_merged = merged_data[bi][role][key]
                if not torch.equal(t_sep, t_merged):
                    print(f"  MISMATCH block {bi} {role}.{key}: max_diff={torch.max(torch.abs(t_sep.float()-t_merged.float())).item()}")
                    all_match = False
    print(f"  {'All tensors match!' if all_match else 'MISMATCHES found'}")

    # Benchmark: 48 separate DF11 files
    print(f"\n--- 48 separate DF11 files ---")
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for sid, ext in ids_ext:
        for bi in range(48):
            p = bd / f"block_{bi:02d}" / f"{sid}{ext}"
            if p.exists():
                _ = load_block_sample(p, map_location=device)
    torch.cuda.synchronize(device)
    t_sep = time.perf_counter() - t0
    print(f"  {t_sep:.3f}s total, {t_sep/len(ids_ext)*1000:.0f}ms/sample")

    # Benchmark: 1 merged DF11 file
    print(f"\n--- 1 merged DF11 safetensors file ---")
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for sid, ext in ids_ext:
        _ = load_merged_df11(tmpdir / f"{sid}.merged.df11.safetensors", device)
    torch.cuda.synchronize(device)
    t_merged_df11 = time.perf_counter() - t0
    print(f"  {t_merged_df11:.3f}s total, {t_merged_df11/len(ids_ext)*1000:.0f}ms/sample")

    print(f"\n--- Summary ---")
    print(f"  Separate (48 DF11 files):   {t_sep/len(ids_ext)*1000:.0f}ms/sample")
    print(f"  Merged DF11 (1 file):       {t_merged_df11/len(ids_ext)*1000:.0f}ms/sample")
    print(f"  Speedup: {t_sep/t_merged_df11:.2f}x")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()

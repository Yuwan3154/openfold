"""
Compare: loading merged files in training_step vs in DataLoader workers.

Approach A (current): DataLoader returns {seq_id, masks}. training_step loads merged file.
Approach B (proposed): DataLoader loads merged file in collate_fn, returns all block data.
                       training_step just does .to(device) per block.
"""
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import safe_open as safetensors_safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openfold.block_replacement_scripts.block_data_io import load_merged_block_samples, sanitize_id


class SequenceDatasetWithMergedCache(Dataset):
    """Dataset that returns seq metadata + optionally loads merged cache in __getitem__."""
    def __init__(self, seq_ids, seq_lengths, cache_dir, load_in_dataset=False):
        self.seq_ids = seq_ids
        self.seq_lengths = seq_lengths
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.load_in_dataset = load_in_dataset

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        sid = self.seq_ids[idx]
        slen = self.seq_lengths[idx]
        item = {"id": sid, "length": slen}
        if self.load_in_dataset and self.cache_dir is not None:
            safe_sid = sanitize_id(str(sid))
            path = self.cache_dir / f"{safe_sid}.safetensors"
            # Load as CPU tensors (mmap) - safe for worker processes
            item["block_data"] = load_merged_block_samples(path, map_location="cpu")
        return item


def collate_metadata_only(batch):
    """Approach A: just return IDs + masks (current behavior)."""
    seq_ids = [s["id"] for s in batch]
    seq_lens = [s["length"] for s in batch]
    bsz = len(seq_lens)
    n_max = max(seq_lens)
    seq_mask = torch.zeros((bsz, n_max), dtype=torch.float32)
    for i, n in enumerate(seq_lens):
        seq_mask[i, :n] = 1.0
    return {
        "seq_id": seq_ids,
        "seq_length": torch.tensor(seq_lens, dtype=torch.long),
        "seq_mask": seq_mask,
    }


def collate_with_block_data(batch):
    """Approach B: return IDs + masks + pre-loaded block data."""
    seq_ids = [s["id"] for s in batch]
    seq_lens = [s["length"] for s in batch]
    bsz = len(seq_lens)
    n_max = max(seq_lens)
    seq_mask = torch.zeros((bsz, n_max), dtype=torch.float32)
    for i, n in enumerate(seq_lens):
        seq_mask[i, :n] = 1.0
    block_data = [s["block_data"] for s in batch]
    return {
        "seq_id": seq_ids,
        "seq_length": torch.tensor(seq_lens, dtype=torch.long),
        "seq_mask": seq_mask,
        "block_data": block_data,  # list of dict[block_idx] -> {input: ..., output: ...}
    }


def simulate_training_step_A(batch, cache_dir, device, n_blocks=48):
    """Approach A: load merged files in training_step."""
    sids = [sanitize_id(str(s)) for s in batch["seq_id"]]
    # Load merged files
    merged_all = {}
    for sid in sids:
        path = cache_dir / f"{sid}.safetensors"
        merged_all[sid] = load_merged_block_samples(path, map_location="cpu")
    # Simulate block loop: transfer to GPU per block
    for bi in range(n_blocks):
        for sid in sids:
            bdata = merged_all[sid].get(bi)
            if bdata is None:
                continue
            for role in ("input", "output"):
                for k, v in bdata[role].items():
                    _ = v.to(device, non_blocking=True)
    torch.cuda.synchronize(device)


def simulate_training_step_B(batch, device, n_blocks=48):
    """Approach B: block data already loaded by DataLoader, just transfer to GPU."""
    block_data_list = batch["block_data"]  # list[dict[block_idx] -> ...]
    for bi in range(n_blocks):
        for bdata_per_sample in block_data_list:
            bdata = bdata_per_sample.get(bi)
            if bdata is None:
                continue
            for role in ("input", "output"):
                for k, v in bdata[role].items():
                    _ = v.to(device, non_blocking=True)
    torch.cuda.synchronize(device)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True, help="Merged cache directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)

    # Discover samples
    files = sorted(cache_dir.glob("*.safetensors"))
    n_needed = args.batch_size * args.n_steps
    files = files[:n_needed]
    print(f"Using {len(files)} samples, batch_size={args.batch_size}, {args.n_steps} steps")

    seq_ids = [f.stem for f in files]
    # Get approximate lengths from the first block's m tensor
    seq_lengths = []
    for f in files:
        with safetensors_safe_open(str(f), framework="pt", device="cpu") as sf:
            keys = [k for k in sf.keys() if k.endswith(".input.m")]
            if keys:
                t = sf.get_tensor(keys[0])
                seq_lengths.append(t.shape[-2] if t.dim() >= 2 else t.shape[0])
            else:
                seq_lengths.append(64)

    # Warmup
    _ = load_merged_block_samples(files[0], map_location="cpu")

    # --- Approach A: DataLoader returns metadata, training_step loads files ---
    print(f"\n--- Approach A: load in training_step (num_workers=0) ---")
    ds_a = SequenceDatasetWithMergedCache(seq_ids, seq_lengths, cache_dir, load_in_dataset=False)
    dl_a = DataLoader(ds_a, batch_size=args.batch_size, shuffle=False, num_workers=0,
                      collate_fn=collate_metadata_only)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for i, batch in enumerate(dl_a):
        if i >= args.n_steps:
            break
        simulate_training_step_A(batch, cache_dir, device)
    torch.cuda.synchronize(device)
    t_a = time.perf_counter() - t0
    print(f"  {t_a:.3f}s total, {t_a/args.n_steps*1000:.0f}ms/step")

    # --- Approach B: DataLoader loads files, various num_workers ---
    for nw in [0, 2, 4]:
        pf = 2 if nw > 0 else None
        label = f"num_workers={nw}" + (f", prefetch={pf}" if pf else "")
        print(f"\n--- Approach B: load in DataLoader ({label}) ---")
        ds_b = SequenceDatasetWithMergedCache(seq_ids, seq_lengths, cache_dir, load_in_dataset=True)
        dl_b = DataLoader(ds_b, batch_size=args.batch_size, shuffle=False,
                          num_workers=nw, collate_fn=collate_with_block_data,
                          prefetch_factor=pf, persistent_workers=(nw > 0))
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for i, batch in enumerate(dl_b):
            if i >= args.n_steps:
                break
            simulate_training_step_B(batch, device)
        torch.cuda.synchronize(device)
        t_b = time.perf_counter() - t0
        print(f"  {t_b:.3f}s total, {t_b/args.n_steps*1000:.0f}ms/step, {t_a/t_b:.2f}x vs A")
        del dl_b, ds_b


if __name__ == "__main__":
    main()

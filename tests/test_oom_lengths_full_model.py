#!/usr/bin/env python
"""
Whole-model OOM diagnostic: find max residue length that fits in GPU memory.

Tests the FULL AlphaFold model (embedders + extra MSA stack + evoformer +
structure module + aux heads) at various n_res values.

Tests both fp32 and fp16 (autocast) modes.
Uses synthetic batches (same pattern as tests/test_model.py:test_dry_run).
"""
import gc
import time

import numpy as np
import torch
import torch.nn as nn

from openfold.config import model_config
from openfold.data import data_transforms
from openfold.model.model import AlphaFold
from openfold.utils.tensor_utils import tensor_tree_map

# ── Parameters ────────────────────────────────────────────────────────────────
LENGTHS = [128, 192, 256, 320, 384, 448, 512]
N_SEQ = 128           # typical MSA depth
N_TEMPL = 4           # typical for AF2Rank
N_EXTRA = 1024        # realistic for inference
NUM_RECYCLES = 1      # minimize time; memory dominated by single iteration
CHUNK_SIZE = 4


# ── Synthetic feature generation (inlined from tests/data_utils.py) ──────────

def _random_template_feats(n_templ, n):
    """Generate random template features (monomer only)."""
    batch = {
        "template_mask": np.random.randint(0, 2, (n_templ,)).astype(np.float32),
        "template_pseudo_beta_mask": np.random.randint(0, 2, (n_templ, n)).astype(np.float32),
        "template_pseudo_beta": np.random.rand(n_templ, n, 3).astype(np.float32),
        "template_aatype": np.random.randint(0, 22, (n_templ, n)).astype(np.int64),
        "template_all_atom_mask": np.random.randint(0, 2, (n_templ, n, 37)).astype(np.float32),
        "template_all_atom_positions": (np.random.rand(n_templ, n, 37, 3) * 10).astype(np.float32),
        "template_torsion_angles_sin_cos": np.random.rand(n_templ, n, 7, 2).astype(np.float32),
        "template_alt_torsion_angles_sin_cos": np.random.rand(n_templ, n, 7, 2).astype(np.float32),
        "template_torsion_angles_mask": np.random.rand(n_templ, n, 7).astype(np.float32),
    }
    return batch


def _random_extra_msa_feats(n_extra, n):
    """Generate random extra MSA features."""
    batch = {
        "extra_msa": np.random.randint(0, 22, (n_extra, n)).astype(np.int64),
        "extra_has_deletion": np.random.randint(0, 2, (n_extra, n)).astype(np.float32),
        "extra_deletion_value": np.random.rand(n_extra, n).astype(np.float32),
        "extra_msa_mask": np.random.randint(0, 2, (n_extra, n)).astype(np.float32),
    }
    return batch


# ── Batch and run logic ──────────────────────────────────────────────────────

def make_batch(cfg, n_res):
    """Create a synthetic batch matching model expectations."""
    batch = {}

    tf = torch.randint(cfg.model.input_embedder.tf_dim - 1, size=(n_res,))
    batch["target_feat"] = nn.functional.one_hot(
        tf, cfg.model.input_embedder.tf_dim
    ).float()
    batch["aatype"] = torch.argmax(batch["target_feat"], dim=-1)
    batch["residue_index"] = torch.arange(n_res)

    batch["msa_feat"] = torch.rand(
        (N_SEQ, n_res, cfg.model.input_embedder.msa_dim)
    )

    # Template features
    t_feats = _random_template_feats(N_TEMPL, n_res)
    batch.update({k: torch.tensor(v) for k, v in t_feats.items()})

    # Extra MSA features
    extra_feats = _random_extra_msa_feats(N_EXTRA, n_res)
    batch.update({k: torch.tensor(v) for k, v in extra_feats.items()})

    batch["msa_mask"] = torch.ones((N_SEQ, n_res)).float()
    batch["seq_mask"] = torch.ones((n_res,)).float()
    batch.update(data_transforms.make_atom14_masks(batch))
    batch["no_recycling_iters"] = torch.tensor(float(NUM_RECYCLES))

    # Add recycling dimension
    add_recycling_dims = lambda t: (
        t.unsqueeze(-1).expand(*t.shape, cfg.data.common.max_recycling_iters)
    )
    batch = tensor_tree_map(add_recycling_dims, batch)

    # Move to GPU
    batch = tensor_tree_map(lambda t: t.cuda(), batch)

    return batch


def run_length(model, cfg, n_res, use_fp16=False):
    """Run full model at given n_res, return (status, elapsed, peak_gb)."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        batch = make_batch(cfg, n_res)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return "OOM_BATCH", 0.0, 0.0
        raise

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            if use_fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model(batch)
            else:
                out = model(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        # Check for NaN in key outputs (fp16 stability check)
        has_nan = False
        if use_fp16:
            for key in ["pair", "plddt", "ptm_score"]:
                if key in out and torch.is_tensor(out[key]):
                    if torch.isnan(out[key]).any():
                        has_nan = True
                        break
            if has_nan:
                return "NaN", elapsed, peak_gb

        return "PASS", elapsed, peak_gb
    except RuntimeError as e:
        elapsed = time.perf_counter() - t0
        msg = str(e).lower()
        if "out of memory" in msg:
            torch.cuda.empty_cache()
            return "OOM", elapsed, 0.0
        if "unsupported datatype" in msg:
            return "DTYPE_ERR", elapsed, 0.0
        raise
    except IndexError:
        # compute_tm argmax fails on NaN input
        elapsed = time.perf_counter() - t0
        return "NaN", elapsed, 0.0
    finally:
        del batch


def run_suite(model, cfg, weight_mem, label, use_fp16=False):
    """Run all lengths for one mode (fp32 or fp16)."""
    print(f"\n=== {label} ===")
    print(f"{'n_res':>6}  {'status':>10}  {'time(s)':>8}  {'peak(GB)':>10}  {'act(GB)':>10}")
    print("-" * 56)

    skip_remaining = False
    for n_res in LENGTHS:
        if skip_remaining:
            print(f"{n_res:>6}  {'SKIP':>10}")
            continue

        status, elapsed, peak_gb = run_length(model, cfg, n_res, use_fp16=use_fp16)
        if status in ("OOM", "OOM_BATCH"):
            marker = " <-- OOM"
            if status == "OOM_BATCH":
                marker += " (batch creation)"
            print(f"{n_res:>6}  {status:>10}  {'---':>8}  {'---':>10}{marker}")
            skip_remaining = True
        elif status == "NaN":
            print(f"{n_res:>6}  {status:>10}  {elapsed:>8.1f}  {peak_gb:>10.2f}  "
                  f"{peak_gb - weight_mem:>10.2f}  <-- NaN in outputs")
        elif status == "DTYPE_ERR":
            print(f"{n_res:>6}  {status:>10}  {'---':>8}  {'---':>10}  <-- unsupported dtype")
            skip_remaining = True
        else:
            act_gb = peak_gb - weight_mem
            print(f"{n_res:>6}  {status:>10}  {elapsed:>8.1f}  {peak_gb:>10.2f}  {act_gb:>10.2f}")

        gc.collect()
        torch.cuda.empty_cache()


def main():
    cfg = model_config("model_1_ptm")
    # Use full 48-block evoformer, no checkpointing, chunk_size=4
    cfg.model.evoformer_stack.blocks_per_ckpt = None
    cfg.globals.chunk_size = CHUNK_SIZE
    cfg.globals.use_deepspeed_evo_attention = False
    cfg.globals.use_cuequivariance_attention = False
    cfg.globals.use_cuequivariance_multiplicative_update = False
    cfg.globals.use_lma = False
    cfg.globals.use_flash = False
    cfg.globals.offload_inference = False

    print(f"Config: model_1_ptm, full 48 blocks, chunk_size={CHUNK_SIZE}")
    print(f"n_seq={N_SEQ}, n_templ={N_TEMPL}, n_extra={N_EXTRA}, recycles={NUM_RECYCLES}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")

    # Measure model weight memory
    torch.cuda.reset_peak_memory_stats()
    model = AlphaFold(cfg).cuda().eval()
    torch.cuda.synchronize()
    weight_mem = torch.cuda.memory_allocated() / 1e9

    # Detailed parameter accounting
    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M ({param_bytes / 1e9:.3f} GB)")
    print(f"GPU memory after model load: {weight_mem:.3f} GB")

    # Run fp32 suite
    run_suite(model, cfg, weight_mem, "FP32", use_fp16=False)

    # Run fp16 autocast suite (weights stay fp32, autocast handles matmul/conv in fp16)
    run_suite(model, cfg, weight_mem, "FP16 (autocast)", use_fp16=True)

    print(f"\nNote: act(GB) = peak(GB) - GPU_after_load({weight_mem:.3f} GB)")


if __name__ == "__main__":
    main()

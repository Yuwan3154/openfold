#!/usr/bin/env python
"""
Diagnostic: find max residue length that fits in GPU memory at chunk_size=4.
Tests EvoformerStack (all 48 blocks) for n_res in {256, 320, 384, 448, 512}.
chunk_size tuning is DISABLED — always uses chunk_size=4.
Tests both fp32 and fp16 (model.half()) modes.
"""
import gc
import time
import torch
from openfold.config import model_config
from openfold.model.evoformer import EvoformerStack

N_SEQ = 128       # typical MSA depth after trimming in model_1_ptm
LENGTHS = [256, 320, 384, 448, 512]
CHUNK_SIZE = 4


def make_stack(fp16: bool = False) -> EvoformerStack:
    cfg = model_config("model_1_ptm")
    gc_cfg = cfg.globals
    ec = cfg.model.evoformer_stack
    stack = EvoformerStack(
        c_m=gc_cfg.c_m,
        c_z=gc_cfg.c_z,
        c_hidden_msa_att=ec.c_hidden_msa_att,
        c_hidden_opm=ec.c_hidden_opm,
        c_hidden_mul=ec.c_hidden_mul,
        c_hidden_pair_att=ec.c_hidden_pair_att,
        c_s=gc_cfg.c_s,
        no_heads_msa=ec.no_heads_msa,
        no_heads_pair=ec.no_heads_pair,
        no_blocks=ec.no_blocks,
        transition_n=ec.transition_n,
        msa_dropout=0.0,
        pair_dropout=0.0,
        no_column_attention=ec.no_column_attention,
        opm_first=ec.opm_first,
        fuse_projection_weights=ec.fuse_projection_weights,
        blocks_per_ckpt=None,
        clear_cache_between_blocks=False,
        tune_chunk_size=False,   # KEY: disable tuning, always use chunk_size arg
        inf=ec.inf,
        eps=ec.eps,
    ).cuda().eval()
    if fp16:
        stack = stack.half()
    return stack


def run_length(stack, c_m, c_z, n_res, fp16=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.float16 if fp16 else torch.float32
    m         = torch.randn(N_SEQ, n_res, c_m, device="cuda", dtype=dtype)
    z         = torch.randn(n_res, n_res, c_z, device="cuda", dtype=dtype)
    msa_mask  = torch.ones(N_SEQ, n_res, device="cuda", dtype=dtype)
    pair_mask = torch.ones(n_res, n_res, device="cuda", dtype=dtype)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            m_out, z_out, s_out = stack(
                m, z,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                outputs=None,
                cycle_no=0,
                chunk_size=CHUNK_SIZE,
                inplace_safe=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        return "PASS", elapsed, peak_mb
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            torch.cuda.empty_cache()
            return "OOM", time.perf_counter() - t0, 0.0
        if "no kernel image" in msg:
            return "NO_KERNEL", time.perf_counter() - t0, 0.0
        raise
    finally:
        del m, z, msa_mask, pair_mask


def main():
    cfg = model_config("model_1_ptm")
    gc_cfg = cfg.globals
    c_m, c_z = gc_cfg.c_m, gc_cfg.c_z
    ec = cfg.model.evoformer_stack

    print(f"Config: c_m={c_m}, c_z={c_z}, no_blocks={ec.no_blocks}")
    print(f"n_seq={N_SEQ}, chunk_size={CHUNK_SIZE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")
    print()
    print(f"{'n_res':>6}  {'mode':>14}  {'status':>10}  {'time(s)':>8}  {'peak(GB)':>10}")
    print("-" * 60)

    # Test configurations: (label, fp16)
    configs = [
        ("attn_core fp32", False),
        ("attn_core fp16", True),
    ]

    for label, fp16 in configs:
        try:
            stack = make_stack(fp16=fp16)
        except Exception as e:
            print(f"{'---':>6}  {label:>14}  {'INIT_ERR':>10}  --  {e}")
            continue

        skip_remaining = False
        for n_res in LENGTHS:
            if skip_remaining:
                print(f"{n_res:>6}  {label:>14}  {'SKIP':>10}")
                continue

            status, elapsed, peak_mb = run_length(stack, c_m, c_z, n_res, fp16=fp16)
            peak_gb = peak_mb / 1e3
            if status in ("OOM", "NO_KERNEL"):
                marker = " <-- OOM" if status == "OOM" else " <-- kernel not supported on this GPU"
                print(f"{n_res:>6}  {label:>14}  {status:>10}  {elapsed:>8.1f}  {'---':>10}{marker}")
                skip_remaining = True
            else:
                print(f"{n_res:>6}  {label:>14}  {status:>10}  {elapsed:>8.1f}  {peak_gb:>10.2f}")

        del stack
        gc.collect()
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Isolated test script to diagnose and compare OpenFold attention kernel
approaches on V100 (Volta, compute capability 7.0) GPUs.

Kernel compatibility matrix (V100 / sm_70):
  DS4Sci_EvoformerAttention  — HANGS (kernel not compiled for sm_70)
  attn_core_inplace_cuda     — WORKS (fp32 only on V100)
  cuEquivariance triangle    — tested here

Each test reports GPU memory usage and wall-clock time.
Hang detection via signal.SIGALRM (2-minute timeout per test).

Usage:
    cd /home/gridsan/cou/openfold
    python tests/test_v100_kernels.py
"""

import importlib
import os
import signal
import sys
import time
import traceback

import torch


# ---------------------------------------------------------------------------
# Timeout helper (SIGALRM-based, Linux only)
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError("Test timed out (likely hung)")


TIMEOUT_SEC = 120
DS4SCI_TIMEOUT = 30   # short for DS4Sci since we know it hangs on V100


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

def print_gpu_info():
    print("=" * 70)
    print("GPU INFORMATION")
    print("=" * 70)
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)

    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cc_major, cc_minor = torch.cuda.get_device_capability(dev)
    mem_total = torch.cuda.get_device_properties(dev).total_memory / 1e9
    print(f"  Device:             {name}")
    print(f"  Compute capability: {cc_major}.{cc_minor}")
    print(f"  Total memory:       {mem_total:.1f} GB")
    print(f"  CUDA version:       {torch.version.cuda}")
    print(f"  PyTorch version:    {torch.__version__}")
    bf16_hw = cc_major >= 8
    print(f"  bf16 (hardware):    {bf16_hw} (requires CC >= 8.0 / Ampere)")
    print()
    return cc_major, cc_minor, bf16_hw


# ---------------------------------------------------------------------------
# Memory / timing helpers
# ---------------------------------------------------------------------------

def reset_memory_stats():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def get_peak_memory_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


# ---------------------------------------------------------------------------
# Test results collector
# ---------------------------------------------------------------------------

results = []


def run_test(name, fn, timeout=TIMEOUT_SEC):
    """Run a test function with timeout and memory/time tracking."""
    print("-" * 70)
    print(f"TEST: {name}")
    print("-" * 70)

    reset_memory_stats()
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)

    status = "PASS"
    mem_mb = 0.0
    elapsed = 0.0
    detail = ""

    try:
        t0 = time.perf_counter()
        detail = fn()
        elapsed = time.perf_counter() - t0
        mem_mb = get_peak_memory_mb()
        if detail is None:
            detail = ""
    except TimeoutError:
        status = "TIMEOUT (hung)"
        elapsed = timeout
    except Exception as e:
        status = f"FAIL ({type(e).__name__})"
        detail = str(e)
        traceback.print_exc()
    finally:
        signal.alarm(0)  # cancel alarm

    print(f"  Status:  {status}")
    print(f"  Time:    {elapsed:.3f}s")
    print(f"  Peak GPU memory: {mem_mb:.1f} MB")
    if detail:
        print(f"  Detail:  {detail}")
    print()

    results.append({
        "name": name,
        "status": status,
        "time_s": elapsed,
        "mem_mb": mem_mb,
        "detail": detail,
    })
    return status


# ---------------------------------------------------------------------------
# Helper: build EvoformerBlock from config
# ---------------------------------------------------------------------------

def _make_evoformer_block():
    from openfold.config import model_config
    from openfold.model.evoformer import EvoformerBlock

    cfg = model_config("model_1_ptm")
    bc = cfg.model.evoformer_stack
    gc = cfg.globals

    block = EvoformerBlock(
        c_m=gc.c_m,
        c_z=gc.c_z,
        c_hidden_msa_att=bc.c_hidden_msa_att,
        c_hidden_opm=bc.c_hidden_opm,
        c_hidden_mul=bc.c_hidden_mul,
        c_hidden_pair_att=bc.c_hidden_pair_att,
        no_heads_msa=bc.no_heads_msa,
        no_heads_pair=bc.no_heads_pair,
        transition_n=bc.transition_n,
        msa_dropout=0.0,
        pair_dropout=0.0,
        no_column_attention=bc.no_column_attention,
        opm_first=bc.opm_first,
        fuse_projection_weights=bc.fuse_projection_weights,
        inf=bc.inf,       # 1e9 — in evoformer_stack config, not globals
        eps=bc.eps,       # 1e-8
    ).cuda().eval()
    return block, gc.c_m, gc.c_z


# ---------------------------------------------------------------------------
# Test 1: bf16 basic sanity check
# ---------------------------------------------------------------------------

def test_bf16_basic():
    """Test if basic bf16 matmul works (emulated on V100, real on Ampere+)."""
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    assert c.dtype == torch.bfloat16
    return f"bf16 matmul OK, max={c.abs().max().item():.4f}"


# ---------------------------------------------------------------------------
# Test 2: fp16 basic sanity check
# ---------------------------------------------------------------------------

def test_fp16_basic():
    """Test basic fp16 matmul."""
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    assert c.dtype == torch.float16
    return f"fp16 matmul OK, max={c.abs().max().item():.4f}"


# ---------------------------------------------------------------------------
# Test 3: DS4Sci_EvoformerAttention with fp16
# ---------------------------------------------------------------------------

def test_ds4sci_fp16():
    """DS4Sci with fp16 — hangs on V100 (sm_70 not supported by the kernel)."""
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

    B, N, S, H, C = 1, 64, 32, 8, 32
    q = torch.randn(B, N, S, H, C, device="cuda", dtype=torch.float16)
    k = torch.randn(B, N, S, H, C, device="cuda", dtype=torch.float16)
    v = torch.randn(B, N, S, H, C, device="cuda", dtype=torch.float16)
    bias1 = torch.randn(B, N, 1, 1, S, device="cuda", dtype=torch.float16)
    bias2 = torch.randn(B, 1, H, S, S, device="cuda", dtype=torch.float16)

    o = DS4Sci_EvoformerAttention(q, k, v, [bias1, bias2])
    torch.cuda.synchronize()
    return f"Output shape={list(o.shape)}, max={o.abs().max().item():.4f}"


# ---------------------------------------------------------------------------
# Test 4: DS4Sci_EvoformerAttention with bf16
# ---------------------------------------------------------------------------

def test_ds4sci_bf16():
    """DS4Sci with bf16 — hangs on V100."""
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

    B, N, S, H, C = 1, 32, 32, 8, 32
    q = torch.randn(B, N, S, H, C, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, N, S, H, C, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, N, S, H, C, device="cuda", dtype=torch.bfloat16)
    bias1 = torch.randn(B, N, 1, 1, S, device="cuda", dtype=torch.bfloat16)
    bias2 = torch.randn(B, 1, H, S, S, device="cuda", dtype=torch.bfloat16)

    o = DS4Sci_EvoformerAttention(q, k, v, [bias1, bias2])
    torch.cuda.synchronize()
    return f"Output shape={list(o.shape)} (unexpectedly succeeded)"


# ---------------------------------------------------------------------------
# Test 5: attn_core_inplace_cuda with fp32
# ---------------------------------------------------------------------------

def test_attn_core_fp32():
    """Custom in-place softmax CUDA kernel with fp32 — expected to work."""
    from openfold.utils.kernel.attention_core import attention_core

    # [*, H, Q, C_hidden] — matching actual MSA row attention dims
    B, H, Q, C = 1, 8, 128, 32
    q = torch.randn(B, H, Q, C, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, Q, C, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, Q, C, device="cuda", dtype=torch.float32)
    bias1 = torch.randn(B, H, 1, Q, device="cuda", dtype=torch.float32)
    bias2 = torch.randn(B, 1, Q, Q, device="cuda", dtype=torch.float32)

    o = attention_core(q, k, v, bias1, bias2)
    torch.cuda.synchronize()
    assert o.shape == (B, H, Q, C)
    return f"Output shape={list(o.shape)}, max={o.abs().max().item():.4f}"


# ---------------------------------------------------------------------------
# Test 6: Full Attention module — DS4Sci path (expected to timeout on V100)
# ---------------------------------------------------------------------------

def test_attention_module_ds4sci():
    """Attention module with use_deepspeed_evo_attention=True — expected TIMEOUT on V100."""
    from openfold.model.primitives import Attention

    # c_q = c_k = c_v = c_m (input channel dim); c_hidden = per-head dim
    c_m, c_hidden, no_heads = 256, 32, 8
    n_seq, n_res = 32, 64

    attn = Attention(c_m, c_m, c_m, c_hidden, no_heads).cuda().eval()
    q_x  = torch.randn(1, n_seq, n_res, c_m, device="cuda", dtype=torch.float32)
    kv_x = q_x.clone()
    bias1 = torch.randn(1, n_seq, 1, 1, n_res, device="cuda", dtype=torch.float32)
    bias2 = torch.randn(1, 1, no_heads, n_res, n_res, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        o = attn(q_x, kv_x, biases=[bias1, bias2], use_deepspeed_evo_attention=True)
    torch.cuda.synchronize()
    assert o.shape == q_x.shape
    return f"Output shape={list(o.shape)}, max={o.abs().max().item():.4f}"


# ---------------------------------------------------------------------------
# Test 7: Full Attention module — attn_core fp32 path
# ---------------------------------------------------------------------------

def test_attention_module_memeff():
    """Attention module with use_memory_efficient_kernel=True (fp32)."""
    from openfold.model.primitives import Attention

    c_m, c_hidden, no_heads = 256, 32, 8
    n_seq, n_res = 32, 64

    attn = Attention(c_m, c_m, c_m, c_hidden, no_heads).cuda().eval()
    q_x  = torch.randn(1, n_seq, n_res, c_m, device="cuda", dtype=torch.float32)
    kv_x = q_x.clone()
    bias1 = torch.randn(1, n_seq, 1, 1, n_res, device="cuda", dtype=torch.float32)
    bias2 = torch.randn(1, 1, no_heads, n_res, n_res, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        o = attn(q_x, kv_x, biases=[bias1, bias2], use_memory_efficient_kernel=True)
    torch.cuda.synchronize()
    assert o.shape == q_x.shape
    return f"Output shape={list(o.shape)}, max={o.abs().max().item():.4f}"


# ---------------------------------------------------------------------------
# Test 8: EvoformerBlock — DS4Sci path (expected to timeout on V100)
# ---------------------------------------------------------------------------

def test_evoformer_block_ds4sci():
    """EvoformerBlock with use_deepspeed_evo_attention=True — expected TIMEOUT on V100."""
    block, c_m, c_z = _make_evoformer_block()
    n_seq, n_res = 32, 64

    m = torch.randn(n_seq, n_res, c_m, device="cuda", dtype=torch.float32)
    z = torch.randn(n_res, n_res, c_z, device="cuda", dtype=torch.float32)
    msa_mask  = torch.ones(n_seq, n_res, device="cuda", dtype=torch.float32)
    pair_mask = torch.ones(n_res, n_res, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        m_out, z_out = block(
            m, z,
            msa_mask=msa_mask, pair_mask=pair_mask,
            chunk_size=4,
            use_deepspeed_evo_attention=True,
            use_cuequivariance_attention=False,
            use_cuequivariance_multiplicative_update=False,
            use_lma=False, inplace_safe=False, _mask_trans=False,
        )
    torch.cuda.synchronize()
    return f"MSA={list(m_out.shape)}, Pair={list(z_out.shape)}"


# ---------------------------------------------------------------------------
# Test 9: EvoformerBlock — attn_core fp32 path
# ---------------------------------------------------------------------------

def test_evoformer_block_memeff():
    """EvoformerBlock with attn_core_inplace_cuda (fp32) — expected to work."""
    block, c_m, c_z = _make_evoformer_block()
    n_seq, n_res = 32, 64

    m = torch.randn(n_seq, n_res, c_m, device="cuda", dtype=torch.float32)
    z = torch.randn(n_res, n_res, c_z, device="cuda", dtype=torch.float32)
    msa_mask  = torch.ones(n_seq, n_res, device="cuda", dtype=torch.float32)
    pair_mask = torch.ones(n_res, n_res, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        m_out, z_out = block(
            m, z,
            msa_mask=msa_mask, pair_mask=pair_mask,
            chunk_size=4,
            use_deepspeed_evo_attention=False,
            use_cuequivariance_attention=False,
            use_cuequivariance_multiplicative_update=False,
            use_lma=False, inplace_safe=False, _mask_trans=False,
        )
    torch.cuda.synchronize()
    return f"MSA={list(m_out.shape)}, Pair={list(z_out.shape)}"


# ---------------------------------------------------------------------------
# Test 10: Numerical comparison — attn_core fp32 vs stock attention
# ---------------------------------------------------------------------------

def test_numerical_comparison():
    """Compare attn_core fp32 output against stock PyTorch attention."""
    from openfold.model.primitives import Attention, lecun_normal_init_

    c_m, c_hidden, no_heads = 256, 32, 8
    n_seq, n_res = 16, 32

    attn = Attention(c_m, c_m, c_m, c_hidden, no_heads).cuda().eval()
    with torch.no_grad():
        lecun_normal_init_(attn.linear_g.weight)
        lecun_normal_init_(attn.linear_o.weight)

    q_x  = torch.randn(1, n_seq, n_res, c_m, device="cuda", dtype=torch.float32)
    kv_x = q_x.clone()
    bias1 = torch.randn(1, n_seq, 1, 1, n_res, device="cuda", dtype=torch.float32)
    bias2 = torch.randn(1, 1, no_heads, n_res, n_res, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        out_me    = attn(q_x, kv_x, biases=[bias1, bias2], use_memory_efficient_kernel=True).cpu()
        out_stock = attn(q_x, kv_x, biases=[bias1, bias2]).cpu()

    err = torch.max(torch.abs(out_me - out_stock)).item()
    assert err < 1e-4, f"attn_core vs stock error too large: {err}"
    return f"Max abs error attn_core(fp32) vs stock(fp32): {err:.2e}  [WITHIN TOLERANCE]"


# ---------------------------------------------------------------------------
# Test 11: cuEquivariance triangle_multiplicative_update (fp32, isolated)
# ---------------------------------------------------------------------------

def test_cueq_triangle_mult_fp32():
    """cuet.triangle_multiplicative_update in isolated form (fp32).

    Requirement: c_z == c_hidden and both % 32 == 0.
    In the evoformer config: c_z=128, c_hidden_mul=128 — both match.

    Follows the same weight-extraction pattern as protein_transformer.py
    _run_triangle_mult_cuet().
    """
    import cuequivariance_torch as cuet
    from openfold.model.triangular_multiplicative_update import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
    )

    # c_z == c_hidden is required by cuEquivariance; use evoformer config values
    c_z = 128
    c_hidden = 128   # c_hidden_mul in config
    B, N = 1, 64    # batch, sequence length

    x = torch.randn(B, N, N, c_z, device="cuda", dtype=torch.float32)
    mask = torch.ones(B, N, N, device="cuda", dtype=torch.float32)

    results_detail = []
    for direction, ModuleCls in [("outgoing", TriangleMultiplicationOutgoing),
                                  ("incoming", TriangleMultiplicationIncoming)]:
        mod = ModuleCls(c_z=c_z, c_hidden=c_hidden).cuda().eval()

        # Extract weights following protein_transformer.py _run_triangle_mult_cuet()
        norm_in_weight = mod.layer_norm_in.weight
        norm_in_bias   = mod.layer_norm_in.bias
        p_in_weight = torch.cat([mod.linear_a_p.weight, mod.linear_b_p.weight], dim=0)
        p_in_bias   = torch.cat([mod.linear_a_p.bias,   mod.linear_b_p.bias],   dim=0)
        g_in_weight = torch.cat([mod.linear_a_g.weight, mod.linear_b_g.weight], dim=0)
        g_in_bias   = torch.cat([mod.linear_a_g.bias,   mod.linear_b_g.bias],   dim=0)
        norm_out_weight = mod.layer_norm_out.weight
        norm_out_bias   = mod.layer_norm_out.bias
        p_out_weight = mod.linear_z.weight
        p_out_bias   = mod.linear_z.bias
        g_out_weight = mod.linear_g.weight
        g_out_bias   = mod.linear_g.bias

        with torch.no_grad():
            out = cuet.triangle_multiplicative_update(
                x=x, direction=direction, mask=mask,
                norm_in_weight=norm_in_weight, norm_in_bias=norm_in_bias,
                p_in_weight=p_in_weight, p_in_bias=p_in_bias,
                g_in_weight=g_in_weight, g_in_bias=g_in_bias,
                norm_out_weight=norm_out_weight, norm_out_bias=norm_out_bias,
                p_out_weight=p_out_weight, p_out_bias=p_out_bias,
                g_out_weight=g_out_weight, g_out_bias=g_out_bias,
            )
        torch.cuda.synchronize()
        assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
        results_detail.append(f"{direction} shape={list(out.shape)} max={out.abs().max().item():.4f}")

    return "; ".join(results_detail)


# ---------------------------------------------------------------------------
# Test 12: EvoformerBlock — cuEquivariance path (fp32)
# ---------------------------------------------------------------------------

def test_evoformer_block_cueq():
    """EvoformerBlock with use_cuequivariance_attention=True (fp32)."""
    block, c_m, c_z = _make_evoformer_block()
    n_seq, n_res = 32, 64

    m = torch.randn(n_seq, n_res, c_m, device="cuda", dtype=torch.float32)
    z = torch.randn(n_res, n_res, c_z, device="cuda", dtype=torch.float32)
    msa_mask  = torch.ones(n_seq, n_res, device="cuda", dtype=torch.float32)
    pair_mask = torch.ones(n_res, n_res, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        m_out, z_out = block(
            m, z,
            msa_mask=msa_mask, pair_mask=pair_mask,
            chunk_size=4,
            use_deepspeed_evo_attention=False,
            use_cuequivariance_attention=True,
            use_cuequivariance_multiplicative_update=True,
            use_lma=False, inplace_safe=False, _mask_trans=False,
        )
    torch.cuda.synchronize()
    return f"MSA={list(m_out.shape)}, Pair={list(z_out.shape)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cc_major, cc_minor, bf16_hw = print_gpu_info()

    ds_installed   = importlib.util.find_spec("deepspeed") is not None
    ds4s_installed = ds_installed and importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
    cueq_installed = importlib.util.find_spec("cuequivariance_torch") is not None
    print(f"DeepSpeed installed:        {ds_installed}")
    print(f"DS4Sci (deepspeed4science): {ds4s_installed}")
    print(f"cuEquivariance installed:   {cueq_installed}")
    print()

    # --- Run tests ---
    print("=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)
    print()

    # Basic dtype
    run_test("1. bf16 basic matmul", test_bf16_basic)
    run_test("2. fp16 basic matmul", test_fp16_basic)

    # DS4Sci (short timeout — known to hang on V100)
    if ds4s_installed:
        run_test("3. DS4Sci fp16 (expect TIMEOUT on V100)", test_ds4sci_fp16, timeout=DS4SCI_TIMEOUT)
        run_test("4. DS4Sci bf16 (expect TIMEOUT on V100)", test_ds4sci_bf16, timeout=DS4SCI_TIMEOUT)
    else:
        print("SKIP tests 3-4: DS4Sci not installed\n")

    # attn_core (custom kernel, fp32) — our primary V100 kernel
    run_test("5. attn_core_inplace_cuda fp32", test_attn_core_fp32)

    # Attention module comparison
    if ds4s_installed:
        run_test("6. Attention module: DS4Sci (expect TIMEOUT on V100)",
                 test_attention_module_ds4sci, timeout=DS4SCI_TIMEOUT)
    run_test("7. Attention module: attn_core fp32", test_attention_module_memeff)

    # EvoformerBlock comparison
    if ds4s_installed:
        run_test("8. EvoformerBlock: DS4Sci (expect TIMEOUT on V100)",
                 test_evoformer_block_ds4sci, timeout=DS4SCI_TIMEOUT)
    run_test("9. EvoformerBlock: attn_core fp32", test_evoformer_block_memeff)

    # Numerical accuracy
    run_test("10. Numerical: attn_core fp32 vs stock fp32", test_numerical_comparison)

    # cuEquivariance
    if cueq_installed:
        run_test("11. cuEquivariance triangle_mult fp32 (isolated)", test_cueq_triangle_mult_fp32)
        run_test("12. EvoformerBlock: cuEquivariance fp32", test_evoformer_block_cueq)
    else:
        print("SKIP tests 11-12: cuEquivariance not installed\n")

    # --- Summary table ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    w = 60
    print(f"{'Test':<{w}} {'Status':<24} {'Time (s)':<10} {'Mem (MB)':<10}")
    print("-" * (w + 44))
    for r in results:
        print(f"{r['name']:<{w}} {r['status']:<24} {r['time_s']:<10.3f} {r['mem_mb']:<10.1f}")
    print()

    # --- EvoformerBlock memory comparison ---
    evo_results = {r["name"].split(":")[1].strip(): r
                   for r in results if "EvoformerBlock" in r["name"]}
    if len(evo_results) >= 2:
        print("EVOFORMER BLOCK MEMORY COMPARISON:")
        for label, r in evo_results.items():
            print(f"  {label:<35} {r['mem_mb']:.1f} MB, {r['time_s']:.3f}s  [{r['status']}]")
        print()

    # --- Verdict ---
    expected_timeouts = {"3.", "4.", "6.", "8."}
    critical_tests = [r for r in results
                      if not any(r["name"].startswith(p) for p in expected_timeouts)]
    critical_pass = all("PASS" in r["status"] for r in critical_tests)

    timeout_ok = all(
        "TIMEOUT" in r["status"]
        for r in results
        if any(r["name"].startswith(p) for p in expected_timeouts)
    )

    if critical_pass:
        print("VERDICT: Core kernel tests PASSED.")
        if timeout_ok:
            print("         DS4Sci correctly timed out — kernel is incompatible with this GPU.")
            print("         Use --no-use_deepspeed_evoformer_attention in the pipeline.")
        cueq_results = [r for r in results if "cuEquivariance" in r["name"]]
        if cueq_results and all("PASS" in r["status"] for r in cueq_results):
            print("         cuEquivariance PASSED — can use --use_cuequivariance_attention.")
        elif cueq_results:
            print("         cuEquivariance FAILED — use attn_core fp32 (no extra flags needed).")
    else:
        failed = [r["name"] for r in critical_tests if "PASS" not in r["status"]]
        print(f"VERDICT: Critical tests FAILED: {failed}")
    print()


if __name__ == "__main__":
    main()

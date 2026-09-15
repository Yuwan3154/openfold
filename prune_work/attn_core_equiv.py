"""Compare openfold's attn_core CUDA extension across (env, GPU arch) WITHOUT inventing a tolerance.

Replaces the absolute-accuracy assert that verify_sm120_gpu.sbatch used to make. That gate asked
bf16 to meet torch.testing's default rtol -- a bar this kernel has never met on ANY architecture,
because attn_softmax_inplace_grad_ accumulates the d_ov.values dot product in bf16 and then forms
(dy - warp_sum) * y, a cancellation. Measured identically on sm_90/cu126 and sm_120/cu128:
3192/13568 elements, max abs 0.1875, rel 334.0. Loosening that rtol would be picking a number.

The question that actually matters is whether a REBUILT env behaves like the env the project
already trusts, so this measures exactly that and carries no constant of its own:

  * SAME capability, two envs  -> BIT-FOR-BIT. Nothing may differ.
  * ACROSS capabilities        -> DRIFT-UNDER-NOISE. max|candidate - trusted| must not exceed
                                  max|trusted - float64 truth|, i.e. switching arch must move the
                                  answer LESS than the rounding error the trusted build already
                                  carries. The bar IS the trusted measurement.

⛔ An earlier version of the cross-arch leg compared the two builds' scalar max-errors directly
(`err_candidate <= err_trusted`). That is the wrong test: the two agreed to 5+ significant figures
(ratio 1.0000) and it still flagged 4 of 12 tensors, because comparing two nearly-equal scalars
with a strict inequality decides on their last bits. Drift-vs-noise asks the question that was
actually meant and is robust to that.

Two subjects are dumped, because the env swap changes both:
  kernel : attn_core_inplace_cuda.forward_/backward_ driven directly, so the compiled .so is
           isolated from cuBLAS. Logits come from a float64 CPU matmul, not from the GPU.
  full   : attention_core(), i.e. what training actually calls -- kernel AND cuBLAS together.

Inputs are drawn on the CPU from a seeded generator and then moved to the device, so every
(env, arch) sees byte-identical inputs. A CUDA generator would not guarantee that across builds.
"""

import argparse
import sys

import numpy as np
import torch

from openfold.utils.kernel.attention_core import attention_core

import attn_core_inplace_cuda

# Q == K is REQUIRED: attn_softmax_inplace_grad_ sets rows_values = cols_output and finds v's leaf
# with row_offset - row_offset % rows_values, which is the leaf start only for square attention.
B, H, Q, K, D = 2, 4, 53, 53, 32
SEED = 0
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def raw(t):
    """Exact bytes of a tensor, dtype-independent (numpy has no bfloat16)."""
    return t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy()


def make_inputs():
    g = torch.Generator().manual_seed(SEED)
    q = torch.randn(B, H, Q, D, generator=g)
    k = torch.randn(B, H, K, D, generator=g)
    v = torch.randn(B, H, K, D, generator=g)
    go = torch.randn(B, H, Q, D, generator=g)
    return q, k, v, go


def reference(q, k, v, go):
    """float64 CPU truth for both subjects. No GPU, no cuBLAS, no fast-math."""
    q64, k64, v64, go64 = (t.double() for t in (q, k, v, go))
    logits64 = q64 @ k64.transpose(-1, -2)
    p64 = torch.softmax(logits64, dim=-1)
    dp64 = go64 @ v64.transpose(-1, -2)
    ds64 = p64 * (dp64 - (dp64 * p64).sum(-1, keepdim=True))

    qa, ka, va = (t.clone().requires_grad_(True) for t in (q64, k64, v64))
    out64 = torch.softmax(qa @ ka.transpose(-1, -2), dim=-1) @ va
    out64.backward(go64)
    return {
        "kernel_p": p64, "kernel_ds": ds64, "logits": logits64,
        "full_out": out64.detach(), "full_gq": qa.grad, "full_gk": ka.grad, "full_gv": va.grad,
    }


def run_device(dtype, q, k, v, go, ref):
    """Both subjects on the current CUDA device, in `dtype`."""
    out = {}

    # --- kernel only: logits arrive from the float64 CPU matmul, so cuBLAS is not in this path
    logits = ref["logits"].to(dtype).cuda().contiguous()
    vv = v.to(dtype).cuda().contiguous()
    gg = go.to(dtype).cuda().contiguous()
    attn_core_inplace_cuda.forward_(logits, B * H * Q, K)
    out["kernel_p"] = logits.clone()
    attn_core_inplace_cuda.backward_(logits, gg, vv, B * H * Q, K, D)
    out["kernel_ds"] = logits

    # --- the whole function, which is what training calls
    qa, ka, va = (t.to(dtype).cuda().detach().requires_grad_(True) for t in (q, k, v))
    o = attention_core(qa, ka, va)
    o.backward(gg)
    out["full_out"] = o.detach()
    out["full_gq"], out["full_gk"], out["full_gv"] = qa.grad, ka.grad, va.grad

    torch.cuda.synchronize()
    return out


def dump(path):
    cap = torch.cuda.get_device_capability(0)
    payload = {
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
        "capability": f"sm_{cap[0]}{cap[1]}",
        "arch_list": ";".join(torch.cuda.get_arch_list()),
    }
    print(f"torch {payload['torch']} cuda {payload['cuda']}")
    print(f"device: {payload['device']}  {payload['capability']}")

    q, k, v, go = make_inputs()
    ref = reference(q, k, v, go)
    for name, dt in DTYPES.items():
        got = run_device(dt, q, k, v, go, ref)
        for key, t in got.items():
            payload[f"bytes/{name}/{key}"] = raw(t)
            payload[f"shape/{name}/{key}"] = np.asarray(tuple(t.shape), dtype=np.int64)
            err = (t.double().cpu() - ref[key]).abs().max().item()
            payload[f"err/{name}/{key}"] = np.float64(err)
            print(f"  {name:>9} {key:<10} max|err| vs float64 CPU = {err:.6e}")
    np.savez(path, **payload)
    print(f"wrote {path}")


def compare(trusted_path, candidate_path):
    a = np.load(trusted_path, allow_pickle=False)
    b = np.load(candidate_path, allow_pickle=False)
    ca, cb = str(a["capability"]), str(b["capability"])
    print(f"trusted  : {str(a['device'])}  {ca}  torch {str(a['torch'])} cuda {str(a['cuda'])}")
    print(f"candidate: {str(b['device'])}  {cb}  torch {str(b['torch'])} cuda {str(b['cuda'])}")

    keys = sorted(k for k in a.files if k.startswith("bytes/"))
    assert keys, f"{trusted_path} holds no dumped tensors"
    assert keys == sorted(k for k in b.files if k.startswith("bytes/")), "the two dumps differ in shape"

    bad = []
    if ca == cb:
        print(f"\nSAME capability ({ca}) -> BIT-FOR-BIT is the right test")
        for k in keys:
            same = np.array_equal(a[k], b[k])
            print(f"  {k[6:]:<22} {'identical' if same else 'DIFFERS'}  ({a[k].size} bytes)")
            if not same:
                bad.append(f"{k[6:]}: {int((a[k] != b[k]).sum())} of {a[k].size} bytes differ")
    else:
        print(f"\nDIFFERENT capabilities ({ca} vs {cb}) -> bit-for-bit is NOT a valid expectation")
        print("(cuBLAS picks different kernels and the fast-math intrinsics are per-arch)")
        print("-> DRIFT-UNDER-NOISE: max|candidate - trusted| must not exceed the rounding error")
        print("   the trusted build already carries, max|trusted - float64 truth|")
        for k in keys:
            name, key = k.split("/")[1:]
            ta = torch.from_numpy(a[k].copy()).view(DTYPES[name]).reshape(tuple(a["shape/" + name + "/" + key]))
            tb = torch.from_numpy(b[k].copy()).view(DTYPES[name]).reshape(tuple(b["shape/" + name + "/" + key]))
            drift = (tb.double() - ta.double()).abs().max().item()
            noise = float(a["err/" + name + "/" + key])
            ok = drift <= noise
            share = (drift / noise) if noise > 0 else (0.0 if drift == 0 else float("inf"))
            print(f"  {k[6:]:<22} drift {drift:.6e}  trusted noise {noise:.6e}  "
                  f"drift/noise {share:.4f}  {'OK' if ok else 'EXCEEDS'}")
            if not ok:
                bad.append(f"{k[6:]}: drift {drift:.6e} exceeds the trusted build's own "
                           f"rounding error {noise:.6e} (x{share:.3f})")

    if bad:
        print("\nEQUIV FAILED")
        for line in bad:
            print("  " + line)
        return 1
    print("\nEQUIV OK")
    return 0


def check(trusted_path, scratch):
    """Dump on THIS node and compare in one shot -- the pre-flight Run D runs on its real GPU."""
    dump(scratch)
    return compare(trusted_path, scratch)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dump", help="write this node's dump to PATH")
    p.add_argument("--compare", nargs=2, metavar=("TRUSTED", "CANDIDATE"))
    p.add_argument("--check", metavar="TRUSTED", help="dump here, then compare against TRUSTED")
    p.add_argument("--scratch", default="/tmp/attn_core_equiv_thisnode.npz")
    a = p.parse_args()
    if a.dump:
        dump(a.dump)
        sys.exit(0)
    if a.compare:
        sys.exit(compare(*a.compare))
    if a.check:
        sys.exit(check(a.check, a.scratch))
    p.error("one of --dump / --compare / --check is required")

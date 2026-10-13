"""Attention backend check (T2_KERNEL_OPT_PLAN.md direction (c)): scaled_dot_product_attention at the cc89 shapes (heads 8, head dim 32, fp16) for B = 64 and L = 86 / 207 / 350 / 450, no mask (all-ones seq_mask, the T2 case), with each SDPA backend
(flash, memory-efficient, cuDNN if available, math) and as one fused-by-hand alternative (q k^T softmax v via matmul + softmax). Prints ms per call (median of 50 after warm-up, CUDA events) and the max deviation of each backend from the math backend in fp32. Env: any with torch + a GPU.
Run: python t2_attn_bench.py
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

H, D = 8, 32


def timeit(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return sorted(a.elapsed_time(b) for a, b in ev)[n // 2]


def main():
    print(torch.__version__, torch.cuda.get_device_name())
    for L in (86, 207, 350, 450):
        B = 64
        q, k, v = (torch.randn(B, H, L, D, device="cuda", dtype=torch.float16) for _ in range(3))
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        row = {}
        for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("cudnn", getattr(SDPBackend, "CUDNN_ATTENTION", None)), ("math", SDPBackend.MATH)):
            if be is None:
                continue
            try:
                with sdpa_kernel(be):
                    out = F.scaled_dot_product_attention(q, k, v)
                    ms = timeit(lambda: F.scaled_dot_product_attention(q, k, v))
                row[name] = (ms, float((out.float() - ref).abs().max()))
            except RuntimeError as e:
                row[name] = (float("nan"), str(e)[:40])
        def manual():
            return torch.softmax((q @ k.transpose(-1, -2)) * D ** -0.5, dim=-1) @ v
        row["matmul+softmax"] = (timeit(manual), float((manual().float() - ref).abs().max()))
        print(f"B={B} H={H} L={L} D={D}: " + " | ".join(f"{k} {v[0]:.3f} ms (max dev {v[1] if isinstance(v[1], str) else format(v[1], '.1e')})" for k, v in row.items()))


if __name__ == "__main__":
    main()

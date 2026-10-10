"""Parity + timing harness for partial-diffusion speed variants (user 10-10: every optimisation gets a parity test and a time test). Env: protpardelle (GPU).
One batch of <n> edited variants of a chain (Stage A json) is run from a FIXED initial noisy state (xt_start per sample, as t2_verify.check_padding), so the ODE sampler is deterministic and a
variant's output can be compared to the baseline's: --save-ref stores the baseline coordinates, --ref compares against them. Variants are selected by the environment of the process
(T2_TF32, T2_AUTOCAST, T2_SDPA of the protpardelle patch 0003) and --compile (torch.compile mode of the coordinate denoiser). Reports: ms per call (median of --repeat after one untimed warm-up;
the warm-up time is the compile/first-call cost), ms per denoising step, peak memory, and parity = per-atom coordinate deviation (max / mean / 99th percentile, A, same frame, valid residues)
and the change of the sequence-independent TM to the native (mean |delta| over the samples).
Run: python t2_bench.py --stage-a-dir A --chain C --native-pdb P --n 64 --label NAME [--save-ref f.npz | --ref f.npz] [--compile reduce-overhead] [--pad-multiple 16]
"""
import argparse
import os
import time

import numpy as np
import torch

import t2_stage_b as sb
from t2_verify import get_model, initial_state, load_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-dir", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--native-pdb", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--rewind", type=int, default=250)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--pad-multiple", type=int, default=1)
    ap.add_argument("--compile", default=None, help="torch.compile mode for the coordinate denoiser (default | reduce-overhead | max-autotune-no-cudagraphs)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--profile", action="store_true", help="one extra call under torch.profiler: top kernels by GPU time, GPU busy share, launches per step")
    ap.add_argument("--graph-wrap", action="store_true", help="clone the compiled denoiser outputs and mark a CUDA-graph step per call (needed for reduce-overhead: the sampler reuses outputs across steps)")
    ap.add_argument("--save-ref", default=None)
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    model = get_model()
    if a.compile:
        compiled = torch.compile(model.struct_model, mode=None if a.compile == "default" else a.compile, dynamic=False)
        if a.graph_wrap:
            class Wrapped(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, *args, **kw):
                    torch.compiler.cudagraph_mark_step_begin()
                    return self.m(*args, **kw).clone()
            compiled = Wrapped(compiled)
        model.struct_model = compiled
    items, _, _ = load_items(a.stage_a_dir, a.chain, a.native_pdb, a.n)
    pb = sb.pad_batch(items, "cuda", a.pad_multiple)
    xt = initial_state(model, pb, a.rewind, list(range(100, 100 + len(items))))
    steps = int(sb.sampling_kwargs([a.rewind])["num_steps"])
    t0 = time.perf_counter()
    out = sb.run_pd(model, pb, a.rewind, xt_start=xt)["xt_traj"][-1]
    torch.cuda.synchronize()
    first = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        out2 = sb.run_pd(model, pb, a.rewind, xt_start=xt)["xt_traj"][-1]
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    rerun = float((out2 - out).abs().max())
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            sb.run_pd(model, pb, a.rewind, xt_start=xt)
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        ka = prof.key_averages()
        gpu_total = sum(e.device_time_total for e in ka if e.device_type == torch.autograd.DeviceType.CUDA) / 1e6
        n_launch = sum(e.count for e in ka if e.device_type == torch.autograd.DeviceType.CUDA)
        print(f"[{a.label}] PROFILE wall {wall:.2f}s (profiler-inflated), GPU kernel time {gpu_total:.2f}s, kernel launches {n_launch} = {n_launch / steps:.0f} per step")
        print(ka.table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=70))
    med = float(np.median(ts))
    lens = [len(it[1]) for it in items]
    x = out.numpy()
    msg = (f"[{a.label}] gpu={torch.cuda.get_device_name()} chain={a.chain} n={len(items)} Lmax={pb['aat'].shape[1]} T2_TF32={os.environ.get('T2_TF32')} "
           f"T2_AUTOCAST={os.environ.get('T2_AUTOCAST')} T2_SDPA={os.environ.get('T2_SDPA')} compile={a.compile} | first call {first:.1f}s, median {med:.3f}s "
           f"({1000 * med / steps:.2f} ms/step, {1e3 * med / len(items):.1f} ms per template), peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, rerun diff {rerun:.1e}")
    if a.save_ref:
        np.savez(a.save_ref, x=x, lens=lens)
        print(msg + " | saved as reference")
        return
    if a.ref:
        r = np.load(a.ref)
        devs = []
        for b, n in enumerate(lens):
            valid = np.abs(r["x"][b, :n]).sum(-1) > 0    # atoms that exist (absent atoms are exactly 0 in the output)
            devs.append(np.linalg.norm(x[b, :n] - r["x"][b, :n], axis=-1)[valid])
        d = np.concatenate(devs)
        ca = np.array([np.linalg.norm(x[b, :n, 1] - r["x"][b, :n, 1], axis=-1).mean() for b, n in enumerate(lens)])
        cb = [sb.score_one((x[b, :n][:, sb.BB_IDX].astype(np.float64), a.native_pdb, items[b][3], "A" * n, np.zeros(max(len(items[b][3]) - 1, 0), bool)))["tm_native"] for b, n in enumerate(lens)]
        cr = [sb.score_one((r["x"][b, :n][:, sb.BB_IDX].astype(np.float64), a.native_pdb, items[b][3], "A" * n, np.zeros(max(len(items[b][3]) - 1, 0), bool)))["tm_native"] for b, n in enumerate(lens)]
        msg += (f" | parity vs ref: atom dev max {d.max():.3e} A, p99 {np.percentile(d, 99):.3e}, mean-over-samples mean CA dev "
                f"{ca.mean():.3e} A (max sample {ca.max():.3e}); tm_native mean {np.mean(cb):.4f} vs ref {np.mean(cr):.4f}, mean |dTM| {np.mean(np.abs(np.array(cb) - np.array(cr))):.4f}")
    print(msg)


if __name__ == "__main__":
    main()

"""T-1: measure what NCCL_P2P_DISABLE=1 costs on the A6000's 4-GPU path, in the training env's own NCCL.

The flag is read once, at NCCL communicator init, so P2P-on vs P2P-off is two LAUNCHES of this
script, not two arguments. Launch each with torchrun and set the env outside:

  CUDA_VISIBLE_DEVICES=0,1,2,3 [NCCL_P2P_DISABLE=1] timeout <s> \\
    torchrun --standalone --nproc_per_node=4 prune_work/nccl_p2p_bench.py --sizes ... --out x.csv

Two modes:
  sweep    : each (op, size) timed separately -- the nccl-tests view (algbw / busbw per size).
  sequence : the --sizes list issued back-to-back as ONE iteration -- e.g. the run's DDP gradient
             buckets, so the time is the all-reduce cost of one optimizer step with no overlap.
             Because DDP overlaps bucket all-reduce with backward, this is an UPPER bound on the
             per-step exposed comm time, which is what makes it safe to divide by step time.

Timing follows nccl-tests: warmup iterations untimed, then n iterations inside one pair of CUDA
events per rank, and the reported time is the MAX over ranks (the slowest rank gates a collective).
busbw uses nccl-tests' PERFORMANCE.md factors: all_reduce 2(n-1)/n, broadcast 1, all_gather (n-1)/n.
Sizes are required on purpose: they must come from the run (bucket sizes) or a stated sweep, not
from a default in this file.
"""

import argparse
import csv
import os
import sys

import torch
import torch.distributed as dist

BUSBW_FACTOR = {
    "all_reduce": lambda n: 2.0 * (n - 1) / n,
    "broadcast": lambda n: 1.0,
    "all_gather": lambda n: (n - 1) / n,
}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def make_op(op, nbytes, dtype, world, device):
    elem = torch.tensor([], dtype=dtype).element_size()
    assert nbytes % elem == 0, f"{nbytes} bytes is not a whole number of {dtype} elements"
    buf = torch.ones(nbytes // elem, dtype=dtype, device=device)
    if op == "all_reduce":
        return lambda: dist.all_reduce(buf)
    if op == "broadcast":
        return lambda: dist.broadcast(buf, src=0)
    if op == "all_gather":
        assert buf.numel() % world == 0, f"{buf.numel()} elements do not split over {world} ranks"
        # separate input buffer: an in-place all-gather requires the input at THIS rank's offset
        shard = torch.ones(buf.numel() // world, dtype=dtype, device=device)
        return lambda: dist.all_gather_into_tensor(buf, shard)
    raise ValueError(op)


def time_iters(fns, warmup, iters, device):
    for _ in range(warmup):
        for f in fns:
            f()
    torch.cuda.synchronize(device)
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        for f in fns:
            f()
    end.record()
    torch.cuda.synchronize(device)
    per_iter_us = torch.tensor([start.elapsed_time(end) * 1e3 / iters], device=device)
    dist.all_reduce(per_iter_us, op=dist.ReduceOp.MAX)
    return per_iter_us.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "sequence"], required=True)
    ap.add_argument("--ops", default="all_reduce,broadcast")
    ap.add_argument("--sizes", required=True, help="comma-separated message sizes in BYTES")
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))  # nccl-tests default -d float
    ap.add_argument("--warmup", type=int, default=5)  # nccl-tests default -w 5
    ap.add_argument("--iters", type=int, default=20)  # nccl-tests default -n 20
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    dtype = DTYPES[args.dtype]
    sizes = [int(s) for s in args.sizes.split(",")]
    ops = args.ops.split(",")
    p2p_disable = os.environ.get("NCCL_P2P_DISABLE", "")
    nccl_env = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("NCCL_"))

    rows = []
    for op in ops:
        if args.mode == "sweep":
            for nbytes in sizes:
                t_us = time_iters([make_op(op, nbytes, dtype, world, device)], args.warmup, args.iters, device)
                algbw = nbytes / (t_us * 1e-6) / 1e9
                rows.append(dict(op=op, mode="sweep", bytes=nbytes, n_msgs=1, time_us=t_us,
                                 algbw_GBps=algbw, busbw_GBps=algbw * BUSBW_FACTOR[op](world)))
        else:
            fns = [make_op(op, nbytes, dtype, world, device) for nbytes in sizes]
            t_us = time_iters(fns, args.warmup, args.iters, device)
            total = sum(sizes)
            algbw = total / (t_us * 1e-6) / 1e9
            rows.append(dict(op=op, mode="sequence", bytes=total, n_msgs=len(sizes), time_us=t_us,
                             algbw_GBps=algbw, busbw_GBps=algbw * BUSBW_FACTOR[op](world)))

    if rank == 0:
        meta = dict(world=world, dtype=args.dtype, warmup=args.warmup, iters=args.iters,
                    nccl_p2p_disable=p2p_disable, torch=torch.__version__,
                    nccl=".".join(map(str, torch.cuda.nccl.version())), nccl_env=nccl_env)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) + list(meta))
            w.writeheader()
            for r in rows:
                w.writerow({**r, **meta})
        for r in rows:
            print(f"{r['op']:>10} {r['mode']:>8} {r['bytes']:>12d} B x{r['n_msgs']:<3d} "
                  f"{r['time_us']:12.1f} us  algbw {r['algbw_GBps']:7.2f}  busbw {r['busbw_GBps']:7.2f} GB/s",
                  flush=True)
        print(f"P2P_DISABLE={p2p_disable!r} {meta['nccl']} -> {args.out}", file=sys.stderr, flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

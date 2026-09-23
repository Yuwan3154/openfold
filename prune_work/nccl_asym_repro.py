"""T-1: reproduce Run C v2's 2026-09-21 hang (E4) outside Lightning -- a rank-0-only trainer.log_dir.

train_openfold.py's _flush_per_entry_records reads `trainer.log_dir` only inside
`if records and self.trainer.is_global_zero`, once per process. In Lightning 2.5.1 that property
ends in `strategy.broadcast(dirpath)` (trainer.py:1264), i.e. DDPStrategy.broadcast ->
broadcast_object_list(group=WORLD, src=0) = two NCCL BROADCASTs that ranks 1-3 never issue.
Lightning's own docstring: "You must call this on all processes. Failing to do so will cause your
program to stall forever."

This issues exactly that call on rank 0 only (--asym 1) or on every rank (--asym 0, the fix), then
the op ranks 1-3 issued next:
  --next rebuild : DDP Reducer::sync_bucket_indices -- broadcast int32[n_params + 1] from rank 0,
                   a blocking D2H copy, then an int32[n_buckets] bucket-sizes broadcast, then the
                   gradient-bucket all-reduces. This is E4's order (n_params = 4471, 15 buckets).
  --next buckets : straight to the gradient-bucket all-reduces -- the order of the Run C v2
                   processes that carried the same asymmetric pair and did NOT hang.
Every all-reduce and broadcast is checked against its exact expected value on every rank, so a run
that completes with corrupted data is reported as CORRUPT, never as passed.

Launch (the process-group timeout is the hang detector; a completed run takes well under a second):
  CUDA_VISIBLE_DEVICES=0,1,2,3 timeout <s> torchrun --standalone --nproc_per_node=4 \\
    prune_work/nccl_asym_repro.py --asym 1 --next rebuild --bucket_bytes ... --pg_timeout_s <s>
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def check(name, ok, rank):
    print(f"[rank {rank}] {name}: {'OK' if ok else 'CORRUPT'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asym", type=int, choices=[0, 1], required=True)
    ap.add_argument("--next", choices=["rebuild", "buckets"], required=True)
    ap.add_argument("--n_params", type=int, required=True, help="trainable parameter TENSORS (E4: 4471)")
    ap.add_argument("--bucket_bytes", required=True, help="comma-separated fp32 gradient-bucket sizes in bytes")
    ap.add_argument("--pg_timeout_s", type=int, required=True)
    ap.add_argument("--steps", type=int, default=1, help="training-like steps of bucket all-reduces after the pair")
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=args.pg_timeout_s))
    rank, world = dist.get_rank(), dist.get_world_size()
    buckets = [int(b) // 4 for b in args.bucket_bytes.split(",")]
    expected_sum = world * (world + 1) / 2  # every rank contributes rank+1
    ok = True

    # E4 had issued 350 symmetric collectives first, so the communicator was connected; do one of each kind
    warm = torch.full((1,), float(rank + 1), device=device)
    dist.all_reduce(warm)
    dist.broadcast(warm, src=0)
    torch.cuda.synchronize(device)

    if args.asym == 0 or rank == 0:
        obj = ["/home/jupyter-chenxi/runs/runC_v2/lightning_logs/version_3" if rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)  # what DDPStrategy.broadcast does (ddp.py:306-308)
        print(f"[rank {rank}] log_dir broadcast issued -> {obj[0]}", flush=True)

    if args.next == "rebuild":
        idx = (torch.arange(args.n_params + 1, dtype=torch.int32, device=device) if rank == 0
               else torch.zeros(args.n_params + 1, dtype=torch.int32, device=device))
        dist.broadcast(idx, src=0)
        idx_cpu = idx.cpu()  # the reducer's blocking D2H copy right after the broadcast
        ok &= check("rebuild indices broadcast", torch.equal(idx_cpu, torch.arange(args.n_params + 1, dtype=torch.int32)), rank)
        sizes = (torch.tensor(buckets, dtype=torch.int32, device=device) if rank == 0
                 else torch.zeros(len(buckets), dtype=torch.int32, device=device))
        dist.broadcast(sizes, src=0)
        ok &= check("bucket sizes broadcast", sizes.cpu().tolist() == buckets, rank)

    for step in range(args.steps):
        bad = 0
        for i, n in enumerate(buckets):
            g = torch.full((n,), float(rank + 1), device=device)
            dist.all_reduce(g)
            if args.steps == 1:
                ok &= check(f"bucket {i} all_reduce ({n * 4} B)", bool((g == expected_sum).all()), rank)
            else:
                bad += int(not bool((g == expected_sum).all()))
        if args.steps > 1:
            print(f"[rank {rank}] step {step}: {bad} of {len(buckets)} bucket all-reduces CORRUPT", flush=True)
            ok &= bad == 0

    dist.barrier()
    print(f"[rank {rank}] COMPLETED asym={args.asym} next={args.next} data={'OK' if ok else 'CORRUPT'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

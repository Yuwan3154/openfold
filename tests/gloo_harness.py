"""Run a function on N gloo ranks in forked CPU processes and collect each rank's return value.

Used by the all-rank metric tests to check that every rank issues the SAME collectives (the property whose
absence caused the val-metric cross-pairing and, in train_openfold's log_dir read, the E4 NCCL hang).
"""

import datetime
import multiprocessing

import torch
import torch.distributed as dist

COLLECTIVES = ("all_gather_object", "all_reduce", "all_gather", "broadcast", "broadcast_object_list", "barrier",
               "reduce", "gather", "gather_object", "scatter", "scatter_object_list", "reduce_scatter",
               "all_to_all")
# hang guard only: each simulation completes in about a second on the A6000; not a tuned value
TIMEOUT_S = 120


def record_collectives():
    """Log every torch.distributed collective called in THIS process, by name, in call order."""
    calls = []
    for name in COLLECTIVES:
        def wrapped(*args, _orig=getattr(dist, name), _name=name, **kwargs):
            calls.append(_name)
            return _orig(*args, **kwargs)
        setattr(dist, name, wrapped)
    return calls


def _rank_main(fn, rank, world, init_method, queue):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=TIMEOUT_S))
    queue.put((rank, fn(rank, world)))
    dist.destroy_process_group()


def run_ranks(fn, world, store_path):
    """{rank: fn(rank, world)} with a gloo process group initialised in every rank."""
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_rank_main, args=(fn, r, world, f"file://{store_path}", queue))
             for r in range(world)]
    for p in procs:
        p.start()
    results = dict(queue.get(timeout=TIMEOUT_S) for _ in range(world))
    for p in procs:
        p.join(TIMEOUT_S)
    assert [p.exitcode for p in procs] == [0] * world, [p.exitcode for p in procs]
    return results

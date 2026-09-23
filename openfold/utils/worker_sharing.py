"""DataLoader worker->parent tensor sharing: `file_descriptor`, with the soft fd limit raised to the hard one.

`file_system` (used until 2026-09-23) routes every shared tensor through one torch_shm_manager per rank whose
client waits a hard 1000 ms; on a timeout CPython's queue feeder drops the batch silently and the rank waits
forever (E3, 2026-09-08). `file_descriptor` has no manager. It holds one fd per live shared storage, and the
only reason it was not used is the A6000's soft RLIMIT_NOFILE of 1024 (hard 1048576, measured).
"""

import resource

import torch.multiprocessing as mp


def configure_worker_sharing():
    """Raise soft RLIMIT_NOFILE to the hard limit and select `file_descriptor`; returns what was done.

    Call before any DataLoader worker starts; forked workers inherit both settings.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = hard
    if hard == resource.RLIM_INFINITY:
        # Linux refuses a NOFILE limit above fs.nr_open, the kernel's own per-process cap
        with open("/proc/sys/fs/nr_open") as fh:
            target = int(fh.read())
    if soft != target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    mp.set_sharing_strategy("file_descriptor")
    return {
        "strategy": mp.get_sharing_strategy(),
        "soft_before": soft,
        "soft_after": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        "hard": hard,
    }

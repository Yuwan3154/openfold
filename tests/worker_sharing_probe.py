"""Run by tests/test_worker_sharing.py in a fresh interpreter, so fd limits and the sharing strategy never
leak into the pytest process. Prints one JSON line on success; a failure is the process's own traceback.

  configure   configure_worker_sharing() only
  new         configure_worker_sharing(), then the loader below
  old_fd1024  file_descriptor at the A6000's measured default soft limit (1024): the known-bad control

The loader is the codebase's OpenFoldDataLoader, and the consumer keeps every batch alive, so the parent holds
N_BATCHES x N_FEAT shared storages at once: the worst case of a real rank, whose parent can hold
num_workers x prefetch_factor outstanding batches (config.py 16 x torch default 2 = 32) of
len(config.data.common.feat) = 56 feature tensors.
"""

import json
import resource
import sys

import ml_collections as mlc
import torch
import torch.multiprocessing as mp

from openfold.data import data_modules
from openfold.data.data_modules import OpenFoldBatchCollator, OpenFoldDataLoader
from openfold.utils import worker_sharing

N_BATCHES = 32
N_FEAT = 56
OLD_SOFT_LIMIT = 1024
MAX_RECYCLING_ITERS = 3  # config.py data.common.max_recycling_iters


class _Features(torch.utils.data.Dataset):
    def __len__(self):
        return N_BATCHES

    def __getitem__(self, idx):
        # every feature carries the trailing recycling dim OpenFoldDataLoader._add_batch_properties slices
        feats = {"aatype": torch.full((1, MAX_RECYCLING_ITERS + 1), float(idx))}
        for j in range(N_FEAT - 1):
            feats[f"f{j}"] = torch.full((1, MAX_RECYCLING_ITERS + 1), float(idx))
        return feats


def _loader():
    config = mlc.ConfigDict({
        "train": {"uniform_recycling": True},
        "common": {"max_recycling_iters": MAX_RECYCLING_ITERS},
    })
    return OpenFoldDataLoader(_Features(), config=config, stage="train",
                              generator=torch.Generator().manual_seed(0), batch_size=1, num_workers=2,
                              collate_fn=OpenFoldBatchCollator())


def main(mode):
    out = {"mode": mode, "worker_sharing": worker_sharing.__file__, "data_modules": data_modules.__file__}
    print("probe imports:", json.dumps(out), flush=True)
    if mode in ("configure", "new"):
        out["configured"] = worker_sharing.configure_worker_sharing()
    elif mode == "old_fd1024":
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (OLD_SOFT_LIMIT, hard))
        mp.set_sharing_strategy("file_descriptor")
    out["strategy"] = mp.get_sharing_strategy()
    out["rlimit_nofile"] = list(resource.getrlimit(resource.RLIMIT_NOFILE))
    if mode != "configure":
        kept = list(_loader())
        out["batches"] = len(kept)
        out["live_shared_tensors"] = sum(
            1 for b in kept for v in b.values() if torch.is_tensor(v) and v.is_shared())
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

"""The TRAIN loader's generator is seeded from (batch_seed, epoch); eval/predict keep batch_seed.

⛔ THE BUG THIS PINS (T-3b, window audit C 2026-09-23): `_gen_dataloader` built
`torch.Generator().manual_seed(batch_seed)` on every call, so every epoch drew the same DataLoader base seed
(hence the same per-position worker seeds: crop, template count, MSA sampling) and the same per-step recycle
count schedule. Run C v2's rank-0 `t4/has_template` sat at exactly 587/750 in 88 of 104 epochs because of it.

Real OpenFoldDataModule._gen_dataloader / train_dataloader, OpenFoldDataset and OpenFoldDataLoader; only
setup() and the per-chain dataset are stubbed. Each item reports its worker's torch seed minus the worker id,
i.e. the loader's base seed, which every per-position augmentation draw derives from.
"""

import os
import types

import ml_collections as mlc
import torch

from openfold.data import data_modules
from openfold.data.data_modules import (
    OpenFoldBatchCollator,
    OpenFoldDataLoader,
    OpenFoldDataModule,
    OpenFoldDataset,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_SEED = 42           # --seed of the Run C v2 / Run D launchers
MAX_RECYCLING_ITERS = 3   # config.py data.common.max_recycling_iters
EPOCH_LEN = 8
N_CHAINS = 64


class _Items(torch.utils.data.Dataset):
    chain_data_cache = None

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def idx_to_chain_id(self, idx):
        return f"c{idx}"

    def __getitem__(self, idx):
        wid = torch.utils.data.get_worker_info().id
        base = torch.initial_seed() - wid  # torch seeds worker w with base_seed + w
        # trailing recycling dim, as OpenFoldDataLoader._add_batch_properties expects
        return {"aatype": torch.zeros(1, MAX_RECYCLING_ITERS + 1),
                "base_seed": torch.full((1, MAX_RECYCLING_ITERS + 1), base, dtype=torch.int64)}


class _DM(OpenFoldDataModule):
    """The real datamodule with only setup() replaced (the real one parses mmCIF/alignment dirs)."""

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(self.batch_seed + 1)
        self.train_dataset = OpenFoldDataset(
            datasets=[_Items(N_CHAINS)], probabilities=[1.], epoch_len=self.train_epoch_len,
            generator=generator, _roll_at_init=False)
        self.eval_dataset = _Items(EPOCH_LEN)
        self.predict_dataset = _Items(EPOCH_LEN)


def _config():
    return mlc.ConfigDict({
        "data_module": {"data_loaders": {"batch_size": 1, "num_workers": 2}},
        # production values: config.py train.uniform_recycling True, eval/predict False
        "train": {"uniform_recycling": True},
        "eval": {"uniform_recycling": False},
        "predict": {"uniform_recycling": False},
        "common": {"max_recycling_iters": MAX_RECYCLING_ITERS},
    })


def _dm():
    dm = _DM(config=_config(), template_mmcif_dir="unused", max_template_date="2018-04-30",
             train_data_dir="unused", train_alignment_dir="unused", batch_seed=BATCH_SEED,
             train_epoch_len=EPOCH_LEN)
    dm.setup()
    return dm


def _draws(loader):
    base_seeds, recycles = [], []
    for batch in loader:
        base_seeds.append(int(batch["base_seed"].flatten()[0]))
        recycles.append(int(batch["no_recycling_iters"].flatten()[0]))
    return {"generator_seed": loader.generator.initial_seed(), "base_seeds": base_seeds,
            "recycles": recycles}


def _epoch(dm, epoch, rank=0):
    dm.trainer = types.SimpleNamespace(current_epoch=epoch, global_rank=rank,
                                       reload_dataloaders_every_n_epochs=1)
    return _draws(dm.train_dataloader())


def test_imports_are_this_checkout():
    print("data_modules:", data_modules.__file__)
    assert data_modules.__file__.startswith(REPO_ROOT), (data_modules.__file__, REPO_ROOT)


def test_different_epochs_draw_different_worker_seeds_and_recycle_schedules():
    dm = _dm()
    e0, e1 = _epoch(dm, 0), _epoch(dm, 1)
    print("epoch 0:", e0, "\nepoch 1:", e1)
    assert e0["generator_seed"] != e1["generator_seed"]
    assert not set(e0["base_seeds"]) & set(e1["base_seeds"]), "an epoch replayed another's worker seeds"
    assert e0["recycles"] != e1["recycles"], "an epoch replayed another's per-step recycle counts"


def test_an_epoch_reproduces_on_a_fresh_datamodule_on_any_rank():
    dm = _dm()
    _epoch(dm, 0)
    original = _epoch(dm, 1)
    resumed = _epoch(_dm(), 1, rank=1)  # a new process, asked for epoch 1 directly, on another rank
    print("original:", original, "\nresumed:", resumed)
    assert resumed == original


def test_eval_and_predict_loader_generators_are_unchanged():
    legacy = torch.Generator().manual_seed(BATCH_SEED).get_state()  # what the old code built for every stage
    dm = _dm()
    for epoch in (0, 1, 7):
        dm.trainer = types.SimpleNamespace(current_epoch=epoch, global_rank=0)
        for stage in ("eval", "predict"):
            assert torch.equal(dm._gen_dataloader(stage).generator.get_state(), legacy), (stage, epoch)


def test_known_bad_control_a_fixed_loader_seed_replays_every_epoch():
    """The old seeding, rebuilt: proves this harness DETECTS a replay, so the tests above are not vacuous."""
    dm = _dm()

    def legacy_loader():
        dm.train_dataset.reroll()
        return OpenFoldDataLoader(dm.train_dataset, config=dm.config, stage="train",
                                  generator=torch.Generator().manual_seed(BATCH_SEED), batch_size=1,
                                  num_workers=2, collate_fn=OpenFoldBatchCollator())

    e0, e1 = _draws(legacy_loader()), _draws(legacy_loader())
    print("legacy epoch 0:", e0, "\nlegacy epoch 1:", e1)
    assert e0 == e1


def test_the_loader_seed_leaves_the_dataset_reroll_stream_unchanged():
    """The sampler draws live on the dataset's OWN generator (batch_seed + 1); one reroll per call, as before."""
    dm = _dm()
    ref = OpenFoldDataset(datasets=[_Items(N_CHAINS)], probabilities=[1.], epoch_len=EPOCH_LEN,
                          generator=torch.Generator().manual_seed(BATCH_SEED + 1), _roll_at_init=False)
    for epoch in range(3):
        dm.trainer = types.SimpleNamespace(current_epoch=epoch, global_rank=0,
                                           reload_dataloaders_every_n_epochs=1)
        dm.train_dataloader()
        ref.reroll()
        assert [(int(a), int(b)) for a, b in dm.train_dataset.datapoints] == \
               [(int(a), int(b)) for a, b in ref.datapoints]
        assert torch.equal(dm.train_dataset.generator.get_state(), ref.generator.get_state())

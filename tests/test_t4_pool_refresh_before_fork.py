"""T4: the workers that serve epoch N must see the promoted pool refreshed for epoch N -- including the
FIRST epoch of a process (fresh start and full ckpt_path resume).

Real CPU pl.Trainer, the real OpenFoldDataModule.train_dataloader/_gen_dataloader, OpenFoldDataset,
OpenFoldDataLoader and PromotedTemplatePool. Only setup() and the inner per-chain dataset are stubbed,
so train_openfold (whose import needs a GPU) is never imported.

Before the fix the refresh lived in OpenFoldWrapper.on_train_epoch_start. PL 2.5.1 forks the epoch's
workers in fit_loop.setup_data (fit_loop.py:275) BEFORE that hook (fit_loop.py:437-438), and only re-forks
after it when current_epoch > 0 and not restarting (training_epoch_loop.py:238-239) -- so the first epoch
of every process trained on an EMPTY pool (Run C v2 ep90: the whole epoch, all 4 ranks).
"""

import json
import os
import types

import ml_collections as mlc
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from openfold.data import data_modules
from openfold.data.data_modules import OpenFoldDataModule, OpenFoldDataset
from openfold.utils import t4_pool
from openfold.utils.t4_pool import PromotedTemplatePool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAIN = "c0"
N_SEED = 3      # records on disk before the first fit, so an unrefreshed (empty) snapshot reads 0, not 3
EPOCH_LEN = 4   # = optimizer steps per epoch at batch_size 1 on 1 device


def _append_record(pool_dir, epoch, step):
    d = os.path.join(pool_dir, "rank0")
    os.makedirs(d, exist_ok=True)
    rec = {"chain": CHAIN, "epoch": int(epoch), "step": int(step), "sample": 0, "tm_pred": 0.9,
           "tm_template": 0.5, "n_res": 1, "npz": f"{CHAIN}_e{epoch}_s{step}_k0.npz"}
    with open(os.path.join(d, "index.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")


class _OneChain(torch.utils.data.Dataset):
    """Stands in for OpenFoldSingleDataset; reports, from inside the worker, what its pool holds."""
    chain_data_cache = None

    def __init__(self, pool):
        self.pool = pool

    def __len__(self):
        return 1

    def idx_to_chain_id(self, idx):
        return CHAIN

    def __getitem__(self, idx):
        # every feature carries a trailing recycling dim, as OpenFoldDataLoader._add_batch_properties expects
        return {"aatype": torch.zeros(1, 1),
                "n_pool": torch.full((1, 1), float(self.pool.n_for_chain(CHAIN)))}


class _DM(OpenFoldDataModule):
    """The real datamodule with only setup() replaced (the real one parses mmCIF/alignment dirs)."""

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(self.batch_seed + 1)
        self.train_dataset = OpenFoldDataset(
            datasets=[_OneChain(self.t4_promoted_pool)], probabilities=[1.],
            epoch_len=self.train_epoch_len, generator=generator, _roll_at_init=False)
        self.eval_dataset = _OneChain(self.t4_promoted_pool)


class _LegacyDM(_DM):
    """The pre-fix train_dataloader (no refresh), for the known-bad control."""

    def train_dataloader(self):
        return self._gen_dataloader("train")


class _Recorder(pl.LightningModule):
    """Stands in for OpenFoldWrapper: records what each TRAINED batch saw, and writes one promotion per
    step synchronously (the real writer is a background thread; synchronous keeps the counts exact)."""

    def __init__(self, pool_dir, refresh_at_epoch_start=False):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        self.pool_dir = pool_dir
        self.refresh_at_epoch_start = refresh_at_epoch_start
        self.seen = {}

    def on_train_epoch_start(self):
        if self.refresh_at_epoch_start:
            self.trainer.datamodule.t4_promoted_pool.refresh()

    def training_step(self, batch, batch_idx):
        self.seen.setdefault(self.current_epoch, []).append(int(batch["n_pool"].flatten()[0]))
        _append_record(self.pool_dir, self.current_epoch, self.global_step)
        return self.lin(batch["aatype"].reshape(-1, 1)).sum()

    def validation_step(self, batch, batch_idx):
        return self.lin(batch["aatype"].reshape(-1, 1)).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def _config(num_workers):
    return mlc.ConfigDict({
        "data_module": {"data_loaders": {"batch_size": 1, "num_workers": num_workers}},
        "train": {"uniform_recycling": False},
        "eval": {"uniform_recycling": False},
        "common": {"max_recycling_iters": 0},
    })


def _dm(cls, pool, num_workers=2):
    return cls(config=_config(num_workers), template_mmcif_dir="unused", max_template_date="2018-04-30",
               train_data_dir="unused", train_alignment_dir="unused", batch_seed=42,
               train_epoch_len=EPOCH_LEN, t4_promoted_pool=pool, t4_n_promoted=1)


def _fit(ckpt_dir, dm, module, max_epochs, ckpt_path=None, every=EPOCH_LEN, max_steps=-1):
    # every == EPOCH_LEN puts last.ckpt on an epoch's LAST batch, the shape Run C v2's version_1
    # resumed from (and the shape that trained a whole epoch on an empty pool)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=max_epochs, max_steps=max_steps,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0, logger=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[ModelCheckpoint(dirpath=str(ckpt_dir), every_n_train_steps=every,
                                   save_top_k=0, save_last=True)],
        default_root_dir=str(ckpt_dir))
    trainer.fit(module, datamodule=dm, ckpt_path=ckpt_path)
    return module.seen


def _run_fresh_then_resume(tmp_path, dm_cls, refresh_at_epoch_start):
    pool_dir = str(tmp_path / "pool")
    for s in range(N_SEED):
        _append_record(pool_dir, -1, s)
    seen = _fit(tmp_path / "ck1", _dm(dm_cls, PromotedTemplatePool(pool_dir)),
                _Recorder(pool_dir, refresh_at_epoch_start), max_epochs=2)
    # a new process: a fresh pool object in its constructor state, full resume from last.ckpt
    seen_resumed = _fit(tmp_path / "ck2", _dm(dm_cls, PromotedTemplatePool(pool_dir)),
                        _Recorder(pool_dir, refresh_at_epoch_start), max_epochs=3,
                        ckpt_path=str(tmp_path / "ck1" / "last.ckpt"))
    return seen, seen_resumed


def test_imports_are_this_checkout():
    print("data_modules:", data_modules.__file__)
    print("t4_pool:", t4_pool.__file__)
    assert data_modules.__file__.startswith(REPO_ROOT), (data_modules.__file__, REPO_ROOT)
    assert t4_pool.__file__.startswith(REPO_ROOT), (t4_pool.__file__, REPO_ROOT)


def test_every_epoch_including_a_process_first_sees_its_own_refresh(tmp_path):
    seen, seen_resumed = _run_fresh_then_resume(tmp_path, _DM, refresh_at_epoch_start=False)
    print("fresh:", seen, "resumed:", seen_resumed)
    # epoch N's snapshot = the seed + every record the EPOCH_LEN steps of epochs 0..N-1 wrote
    assert seen == {0: [N_SEED] * EPOCH_LEN, 1: [N_SEED + EPOCH_LEN] * EPOCH_LEN}
    assert seen_resumed == {2: [N_SEED + 2 * EPOCH_LEN] * EPOCH_LEN}


def test_known_bad_control_epoch_start_refresh_misses_a_process_first_epoch(tmp_path):
    # The pre-fix wiring, reproduced: proves this harness DETECTS the bug, so the test above is not
    # passing vacuously. If PL's fork order ever changes, this control is what will say so.
    seen, seen_resumed = _run_fresh_then_resume(tmp_path, _LegacyDM, refresh_at_epoch_start=True)
    print("fresh:", seen, "resumed:", seen_resumed)
    assert seen == {0: [0] * EPOCH_LEN, 1: [N_SEED + EPOCH_LEN] * EPOCH_LEN}
    assert seen_resumed == {2: [0] * EPOCH_LEN}


MID_STOP = EPOCH_LEN + 2  # a last.ckpt 2 batches into epoch 1: the shape of Run C v2's version_4/last.ckpt


def _run_mid_epoch_resume(tmp_path, dm_cls, refresh_at_epoch_start):
    pool_dir = str(tmp_path / "pool")
    for s in range(N_SEED):
        _append_record(pool_dir, -1, s)
    _fit(tmp_path / "ck1", _dm(dm_cls, PromotedTemplatePool(pool_dir)),
         _Recorder(pool_dir, refresh_at_epoch_start), max_epochs=3, every=2, max_steps=MID_STOP)
    return _fit(tmp_path / "ck2", _dm(dm_cls, PromotedTemplatePool(pool_dir)),
                _Recorder(pool_dir, refresh_at_epoch_start), max_epochs=3,
                ckpt_path=str(tmp_path / "ck1" / "last.ckpt"))


def test_mid_epoch_resume_trains_the_rest_of_the_epoch_on_a_fresh_snapshot(tmp_path):
    seen = _run_mid_epoch_resume(tmp_path, _DM, refresh_at_epoch_start=False)
    print("mid-epoch resumed:", seen)
    # every step so far wrote one record, so the resumed process's first snapshot holds N_SEED + MID_STOP
    assert seen[1] and set(seen[1]) == {N_SEED + MID_STOP}, seen
    assert set(seen[2]) == {N_SEED + MID_STOP + len(seen[1])}, seen


def test_known_bad_control_mid_epoch_resume_trains_on_an_empty_pool(tmp_path):
    seen = _run_mid_epoch_resume(tmp_path, _LegacyDM, refresh_at_epoch_start=True)
    print("mid-epoch resumed (legacy):", seen)
    assert seen[1] and set(seen[1]) == {0}, seen


def _fake_trainer(reload_every):
    return types.SimpleNamespace(reload_dataloaders_every_n_epochs=reload_every, current_epoch=0)


@pytest.mark.parametrize("reload_every", [0, 2])
def test_pool_without_per_epoch_reload_fails_loudly(tmp_path, reload_every):
    dm = _dm(_DM, PromotedTemplatePool(str(tmp_path / "pool")), num_workers=0)
    dm.setup()
    dm.trainer = _fake_trainer(reload_every)
    with pytest.raises(AssertionError, match="reload_dataloaders_every_n_epochs"):
        dm.train_dataloader()


def test_refresh_leaves_the_sampler_draws_unchanged(tmp_path):
    pool_dir = str(tmp_path / "pool")
    for s in range(N_SEED):
        _append_record(pool_dir, -1, s)
    with_pool = _dm(_DM, PromotedTemplatePool(pool_dir), num_workers=0)
    without_pool = _dm(_DM, None, num_workers=0)
    for dm in (with_pool, without_pool):
        dm.setup()
        dm.trainer = _fake_trainer(1)
    for _ in range(3):
        with_pool.train_dataloader()
        without_pool.train_dataloader()
        assert ([(int(a), int(b)) for a, b in with_pool.train_dataset.datapoints]
                == [(int(a), int(b)) for a, b in without_pool.train_dataset.datapoints])
        assert torch.equal(with_pool.train_dataset.generator.get_state(),
                           without_pool.train_dataset.generator.get_state())
    assert with_pool.t4_promoted_pool.n_for_chain(CHAIN) == N_SEED

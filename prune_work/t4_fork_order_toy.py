"""Toy: which pool snapshot do an epoch's DataLoader workers carry, under PL 2.5.1?

Mirrors the T4 wiring of train_openfold.py / openfold/data/data_modules.py: the train dataset holds a
reference to a pool object that the MAIN process refreshes; persistent_workers=False;
reload_dataloaders_every_n_epochs=1; ModelCheckpoint(every_n_train_steps=K, save_top_k=0,
save_last=True); full resume via trainer.fit(ckpt_path=...). CPU only, DDP over gloo.

  --mode old : refresh in LightningModule.on_train_epoch_start (the code at 2b4ff6b)
  --mode new : refresh in LightningDataModule.train_dataloader(), before the DataLoader is built

The pool's value is the epoch whose refresh produced it (-1 = constructor state, i.e. EMPTY).
Every DataLoader.__iter__ bumps a generation counter on the dataset first, so each forked worker
carries (generation, snapshot) and stamps both into every sample it produces.

Run:       python t4_fork_order_toy.py --out DIR --mode old --max_epochs 3
Resume:    python t4_fork_order_toy.py --out DIR2 --mode old --max_epochs 6 --resume DIR/ckpt/last.ckpt
Summarize: python t4_fork_order_toy.py --summarize DIR
"""

import argparse
import collections
import glob
import json
import os
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy


class Pool:
    def __init__(self):
        self.snapshot_epoch = -1

    def refresh(self, epoch):
        self.snapshot_epoch = int(epoch)


class Events:
    def __init__(self, path):
        self.path = path

    def __call__(self, ev, **kw):
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"t": time.time(), "ev": ev, **kw}) + "\n")


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, pool, n, out_dir):
        self.pool = pool
        self.n = n
        self.out_dir = out_dir
        self.gen = -1
        self.rank = -1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rec = {"rank": self.rank, "gen": self.gen, "snap": self.pool.snapshot_epoch,
               "pid": os.getpid(), "idx": int(i)}
        with open(os.path.join(self.out_dir, f"getitem_pid{os.getpid()}.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return {"x": torch.ones(1), "gen": torch.tensor(self.gen), "snap": torch.tensor(rec["snap"])}


class ValDS(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, i):
        return {"x": torch.ones(1)}


class LoggingDL(torch.utils.data.DataLoader):
    """Mirrors OpenFoldDataLoader overriding __iter__; logs the moment each iterator (= worker fork) is made."""

    def __init__(self, *args, events=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events

    def __iter__(self):
        self.dataset.gen += 1
        self.events("dataloader_iter", gen=self.dataset.gen, snap_main=self.dataset.pool.snapshot_epoch)
        return super().__iter__()


class DM(pl.LightningDataModule):
    def __init__(self, mode, n_train, num_workers, out_dir):
        super().__init__()
        self.mode = mode
        self.n_train = n_train
        self.num_workers = num_workers
        self.out_dir = out_dir
        self.pool = Pool()

    def setup(self, stage=None):
        self.train_ds = TrainDS(self.pool, self.n_train, self.out_dir)

    def train_dataloader(self):
        ep = self.trainer.current_epoch
        if self.mode == "new":
            self.pool.refresh(ep)
        self.train_ds.rank = self.trainer.global_rank
        self.trainer.lightning_module.events("train_dataloader", epoch=ep, snap_main=self.pool.snapshot_epoch)
        return LoggingDL(self.train_ds, batch_size=1, num_workers=self.num_workers,
                         persistent_workers=False, events=self.trainer.lightning_module.events)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(ValDS(), batch_size=1, num_workers=0)


class M(pl.LightningModule):
    def __init__(self, mode, out_dir):
        super().__init__()
        self.mode = mode
        self.out_dir = out_dir
        self.lin = torch.nn.Linear(1, 1)
        self.events = None

    def setup(self, stage=None):
        self.events = Events(os.path.join(self.out_dir, f"events_rank{self.global_rank}.jsonl"))

    def on_train_epoch_start(self):
        pool = self.trainer.datamodule.pool
        if self.mode == "old":
            pool.refresh(self.current_epoch)
        self.events("on_train_epoch_start", epoch=self.current_epoch, snap_main=pool.snapshot_epoch)

    def training_step(self, batch, batch_idx):
        self.events("training_step", epoch=self.current_epoch, batch_idx=batch_idx,
                    gen=batch["gen"].tolist(), snap=batch["snap"].tolist())
        return self.lin(batch["x"]).sum()

    def validation_step(self, batch, batch_idx):
        return self.lin(batch["x"]).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def run(args):
    os.makedirs(args.out, exist_ok=True)
    trainer = pl.Trainer(
        accelerator="cpu", devices=args.devices,
        strategy=DDPStrategy(process_group_backend="gloo") if args.devices > 1 else "auto",
        max_epochs=args.max_epochs, max_steps=args.max_steps,
        reload_dataloaders_every_n_epochs=1, num_sanity_val_steps=0,
        callbacks=[ModelCheckpoint(dirpath=os.path.join(args.out, "ckpt"),
                                   every_n_train_steps=args.ckpt_every, save_top_k=0, save_last=True)],
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=args.out,
    )
    dm = DM(args.mode, args.n_train, args.num_workers, args.out)
    trainer.fit(M(args.mode, args.out), datamodule=dm, ckpt_path=args.resume)


def summarize(out):
    getitems = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(out, "getitem_pid*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            getitems[(r["rank"], r["gen"])].append(r)
    for ev_path in sorted(glob.glob(os.path.join(out, "events_rank*.jsonl"))):
        rank = int(ev_path.rsplit("rank", 1)[1].split(".")[0])
        evs = [json.loads(l) for l in open(ev_path)]
        print(f"=== rank {rank}: {ev_path} ({len(evs)} events)")
        print("  main-process trace (training_step runs collapsed):")
        prev = None
        for e in evs:
            if e["ev"] == "training_step":
                key = (e["epoch"], tuple(e["gen"]), tuple(e["snap"]))
                if prev == key:
                    continue
                prev = key
                print(f"    training_step       epoch={e['epoch']} first_batch_idx={e['batch_idx']} "
                      f"batch_gen={e['gen']} batch_snap={e['snap']}")
            else:
                prev = None
                print("    " + f"{e['ev']:<20}" + " ".join(f"{k}={v}" for k, v in e.items()
                                                          if k not in ("t", "ev")))
        trained = collections.defaultdict(lambda: collections.Counter())
        for e in evs:
            if e["ev"] == "training_step":
                for g, s in zip(e["gen"], e["snap"]):
                    trained[e["epoch"]][(g, s)] += 1
        print("  per trained epoch: Counter((worker_generation, pool_snapshot_epoch) -> n trained samples)")
        for ep in sorted(trained):
            snaps = sorted({s for (_, s) in trained[ep]})
            verdict = ("FRESH" if snaps == [ep] else "EMPTY(constructor)" if snaps == [-1]
                       else "STALE by %s" % [ep - s for s in snaps])
            print(f"    epoch {ep}: {dict(trained[ep])}  -> {verdict}")
        print("  per worker generation (all samples produced by workers, trained or not):")
        gens = sorted(g for (r, g) in getitems if r == rank)
        for g in gens:
            recs = getitems[(rank, g)]
            n_trained = sum(c for ep in trained for (gg, _), c in trained[ep].items() if gg == g)
            print(f"    gen {g}: snap={sorted({r['snap'] for r in recs})} pids={sorted({r['pid'] for r in recs})} "
                  f"produced={len(recs)} trained={n_trained}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--mode", choices=["old", "new"])
    ap.add_argument("--max_epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--devices", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--ckpt_every", type=int, default=4)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
    else:
        run(a)

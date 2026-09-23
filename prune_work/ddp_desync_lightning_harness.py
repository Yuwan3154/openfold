"""T-1 follow-up: a FAITHFUL Lightning 2.5.1 + DDP reproduction of Run C v2's rank-0-only trainer.log_dir desync.

Question: prune_work/nccl_asym_repro.py (bare torch.distributed) shows that the rank-0-only
broadcast_object_list pair followed by gradient all-reduces corrupts every all-reduce and then hangs, yet
Run C v2 processes carried the same +2 rank-0 SeqNum offset for up to 88 epochs without hanging and with
exact epoch-end metrics. Were their gradients corrupted after the first flush?

What is mirrored from the real run (train_openfold.py @ 2b4ff6b, ~/prune_work/run_C_v2_base200.sh):
  - parameters: the SAME 5051 tensors (4471 trainable, 87,953,280 fp32 elements) in registration order,
    rebuilt from version_2/last.ckpt's state_dict (prune_work/dump_trainable_shapes.py), so DDP's
    _verify_param_shape / _sync_module_states / bucket layout are the real ones; no buffers (as the run).
  - Trainer: DDPStrategy(find_unused_parameters=False, process_group_backend="nccl") (train_openfold.py:1578-1580),
    devices auto (4), precision "bf16", accumulate 1, gradient_clip_val 0.1 norm (:1587-1598),
    num_sanity_val_steps 0, reload_dataloaders_every_n_epochs 1, log_every_n_steps 20; Lightning's own
    subprocess launcher (python harness.py, no torchrun), as the real run.
  - callbacks: best ModelCheckpoint(monitor val/lddt_ca, max, top_k 5, save_last False) + periodic
    ModelCheckpoint(every_n_train_steps 20, save_top_k 0, save_last True) (:1488-1499) + LearningRateMonitor
    (step) with a TensorBoardLogger(save_dir, name="lightning_logs") (--log_lr, :1530-1545).
  - logging: 20 synced train epoch keys + the unsynced on_step keys; 58 synced val keys (the audit's counts).
  - collectives: pTM all_gather_object at validation-epoch end; per-entry records all_gather_object at
    on_train_epoch_start + on_fit_end, via the REAL _flush_per_entry_records compiled out of a given
    train_openfold.py with ast (never imported), so BUGGY = the Run C v2 file, FIXED = 2b4ff6b.
  - dataloaders: batch_size 1, 16 workers, persistent_workers False, no pin_memory (data_modules.py:1500-1507).
Instrumentation (does not add NCCL work): an exactly checkable gradient (loss = sum_i (p_i * C_i).sum() in fp32,
C_i small integers from (rank, global_step, element), so the DDP mean is exact and known on every rank without
communication) checked for EVERY trainable tensor in on_after_backward; a bitwise parameter checksum compared
across ranks over a separate gloo group; strategy.broadcast/reduce values; the NCCL flight recorder.
Bisection knobs: --pre_flush_bcasts N adds N symmetric 1-float broadcasts before each epoch>0 flush (the only
option that adds NCCL work); --resume_ckpt emulates a resumed Run C v2 process.
Result (2026-09-22, box A6000, NCCL 2.26.2): survivor order clean (NCCL Simple round-up absorbs the pair),
pair-before-rebuild = the E4 hang. Evidence: SOLab/_box_snapshot/t1_nccl_20260922/followups/ddp_survivor/
(notes.md, box_runs/) and RAW-ARCHIVE Phase M. S1..S2_pre3 ran a revision differing only in the metric-check
initialisation and this docstring.
"""

import argparse
import ast
import csv
import hashlib
import json
import os
import pickle
import time
from datetime import timedelta

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as _tmp
import torch.nn as nn
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

_tmp.set_sharing_strategy("file_system")  # train_openfold.py:19-20

LOSS_NAMES = ["distogram", "experimentally_resolved", "fape", "plddt_loss", "masked_msa", "supervised_chi",
              "violation", "tm", "unscaled_loss", "loss"]  # AlphaFoldLoss breakdown order, tm enabled
VAL_METRICS = ["lddt_ca", "drmsd_ca", "alignment_rmsd", "recall_2A", "gdt_ts", "gdt_ha"]
EXPLORE_UNSYNCED = ["conf_picks_loss_argmin", "loss_spread", "loss_gain_vs_mean", "regret_vs_best",
                    "conf_spread", "using_true_loss", "selected_rung", "selected_tau"]
T4_ON_EPOCH = ["tm_pred", "tm_template", "margin", "promote_rate", "has_template", "promoted_per_step"]
N_RUNGS = 4  # --explore_k 4 (run_C_v2_base200.sh:124-125)


def load_flush(src_path):
    """_flush_per_entry_records compiled from src_path's OpenFoldWrapper (line numbers kept for tracebacks)."""
    src = open(src_path).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == "OpenFoldWrapper":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "_flush_per_entry_records":
                    ns = {"os": os, "csv": csv, "dist": dist}
                    exec(compile(ast.Module(body=[fn], type_ignores=[]), src_path, "exec"), ns)
                    return ns["_flush_per_entry_records"], ast.get_source_segment(src, fn), \
                        hashlib.md5(src.encode()).hexdigest()
    raise ValueError(f"no OpenFoldWrapper._flush_per_entry_records in {src_path}")


def build_params(entries):
    """nn.Module tree whose named_parameters() equals the checkpoint's state_dict keys, in order."""
    root = nn.Module()
    gen = torch.Generator().manual_seed(0)
    for e in entries:
        path = e["name"].split(".")
        assert path[0] == "model", e["name"]
        mod = root
        for p in path[1:-1]:
            if p not in mod._modules:
                mod.add_module(p, nn.Module())
            mod = mod._modules[p]
        t = torch.empty(e["shape"], dtype=torch.float32).uniform_(-0.01, 0.01, generator=gen)
        mod.register_parameter(path[-1], nn.Parameter(t, requires_grad=e["trainable"]))
    return root


def train_val(r, key_idx, j, epoch):
    return float(((r * 5 + key_idx * 3 + j * 7 + epoch * 11) % 13) - 6)


def val_val(r, key_idx, j, epoch):
    return float(((r * 3 + key_idx * 5 + j * 11 + epoch * 7) % 17) - 8)


def val_groups(j):
    """Suffix groups batch j logs to; identical on every rank, all 7 present after 3 batches."""
    return ["train_overlap" if j % 2 else "held_out", "nonneural" if (j // 2) % 2 else "neural_gated",
            "src_" + ("pda", "easy", "hard")[j % 3]]


def val_keys_for(j):
    keys = [f"val/{n}" for n in LOSS_NAMES] + [f"val/{m}" for m in VAL_METRICS]
    for g in val_groups(j):
        keys += [f"val/{m}_{g}" for m in VAL_METRICS]
    return keys


ALL_VAL_KEYS = list(dict.fromkeys(k for j in range(3) for k in val_keys_for(j)))
TRAIN_SYNCED = ([f"explore/{p}_rung{r}" for r in range(N_RUNGS) for p in ("picked", "best_loss")]
                + [f"train/{n}_epoch" for n in LOSS_NAMES] + ["train/lddt_ca", "train/drmsd_ca"])
assert len(ALL_VAL_KEYS) == 58 and len(TRAIN_SYNCED) == 20, (len(ALL_VAL_KEYS), len(TRAIN_SYNCED))


def record(r, idx, epoch):
    """Per-entry record as the real (idx, lddt, rmsd, recall, gdt) tuple; floats pickle to 9 B like the run's."""
    return (int(idx), (idx % 97) / 97.0 + epoch, (idx % 13) / 3.0, float(idx % 2), (idx % 89) / 89.0)


class IndexDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"idx": torch.tensor([i], dtype=torch.int64)}


class HarnessData(pl.LightningDataModule):
    def __init__(self, steps_epoch0, steps, n_val, world, workers):
        super().__init__()
        self.steps_epoch0, self.steps, self.n_val, self.world, self.workers = steps_epoch0, steps, n_val, world, workers

    def train_dataloader(self):
        k = self.steps_epoch0 if self.trainer.current_epoch == 0 else self.steps
        return DataLoader(IndexDataset(k * self.world), batch_size=1, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(IndexDataset(self.n_val), batch_size=1, num_workers=self.workers)


class TracedDDPStrategy(DDPStrategy):
    """DDPStrategy with pass-through broadcast/reduce that log what each rank sent and got."""

    trace_path = None

    def _trace(self, msg):
        with open(self.trace_path, "a") as fh:
            fh.write(f"{time.time():.6f} {msg}\n")

    def broadcast(self, obj, src=0):
        out = super().broadcast(obj, src)
        self._trace(f"strategy.broadcast src={src} sent={obj!r:.120} got={out!r:.120}")
        return out

    def reduce(self, tensor, group=None, reduce_op="mean"):
        before = tensor.detach().flatten()[:4].tolist() if torch.is_tensor(tensor) else tensor
        out = super().reduce(tensor, group=group, reduce_op=reduce_op)
        after = out.detach().flatten()[:4].tolist() if torch.is_tensor(out) else out
        self._trace(f"strategy.reduce op={reduce_op} in={before} out={after}")
        return out


class Harness(pl.LightningModule):
    def __init__(self, entries, flush_fn, out_dir, param_check_every, pg_timeout_s, pre_flush_bcasts):
        super().__init__()
        self.pre_flush_bcasts = pre_flush_bcasts
        self.model = build_params(entries)
        self._flush_impl = flush_fn
        self.out_dir = out_dir
        self.param_check_every = param_check_every
        self.pg_timeout_s = pg_timeout_s
        self._is_distributed = None
        self._val_ptm_calib_pairs = []
        self._val_per_entry_records = []
        self._val_per_entry_epoch = 0
        self._val_per_entry_step = 0
        self._per_entry_csv_path = None
        self._reported_buckets = set()
        self._train_steps_last_epoch = None  # None until this process has trained an epoch (a resumed one starts mid-run)

    # ---- helpers -------------------------------------------------------------------------------
    def _rank_file(self, stem):
        return os.path.join(self.out_dir, f"{stem}_rank{self.global_rank}")

    def _trace(self, msg):
        with open(self._rank_file("trace") + ".log", "a") as fh:
            fh.write(f"{time.time():.6f} rank={self.global_rank} ep={self.current_epoch} gs={self.global_step} {msg}\n")

    def _csv(self, stem, header, row):
        path = self._rank_file(stem) + ".csv"
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(header)
            w.writerow(row)

    def _coeff(self, r, step):
        return (((self._g * 7 + (r * 3 + step * 5)) % 9) - 4).float()

    # ---- setup ---------------------------------------------------------------------------------
    def setup(self, stage):
        # store-based rendezvous only; issues no NCCL work (checked against the flight recorder)
        self.gloo = dist.new_group(backend="gloo", timeout=timedelta(seconds=self.pg_timeout_s))
        self.train_params = [p for p in self.model.parameters() if p.requires_grad]
        self.all_params = list(self.model.parameters())
        self._trace(f"setup stage={stage} n_params={len(self.all_params)} n_trainable={len(self.train_params)} "
                    f"numel_trainable={sum(p.numel() for p in self.train_params)}")

    def on_fit_start(self):
        dev = self.device
        n = sum(p.numel() for p in self.train_params)
        self._g = torch.arange(n, device=dev, dtype=torch.int32)
        sizes = torch.tensor([p.numel() for p in self.train_params], device=dev)
        self._tid = torch.repeat_interleave(torch.arange(len(self.train_params), device=dev, dtype=torch.int32), sizes)
        self._offsets = torch.cumsum(sizes, 0).tolist()
        self._offsets = [0] + self._offsets[:-1]
        wmax = max(p.numel() for p in self.all_params)
        self._w = (torch.arange(wmax, device=dev, dtype=torch.int64) % 1021) + 1
        self._trace(f"fit_start device={dev} world={self.trainer.world_size}")

    def on_train_start(self):
        self._param_check("train_start")

    # ---- training ------------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        r, ep = self.global_rank, self.current_epoch
        if self._is_distributed is None:  # train_openfold.py:167-168
            self._is_distributed = hasattr(self, "trainer") and self.trainer and self.trainer.world_size > 1
        sync = self._is_distributed
        # explore/* then t4/* then _log, the real first-insertion order (train_openfold.py:393-563, 166-208)
        for ki, k in enumerate(EXPLORE_UNSYNCED):
            self.log(f"explore/{k}", train_val(r, 100 + ki, batch_idx, ep), on_step=True, on_epoch=True, logger=True)
        for ki, k in enumerate(TRAIN_SYNCED[:2 * N_RUNGS]):
            v = train_val(r, ki, batch_idx, ep)
            self.log(k, v, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        for ki, k in enumerate(T4_ON_EPOCH):
            self.log(f"t4/{k}", train_val(r, 200 + ki, batch_idx, ep), on_step=True, on_epoch=True, logger=True)
        self.log("t4/pool_written", 0.0, on_step=True, on_epoch=False, logger=True)
        self.log("t4/pool_dropped", 0.0, on_step=True, on_epoch=False, logger=True)
        for ni, n in enumerate(LOSS_NAMES):
            v = torch.tensor(train_val(r, 2 * N_RUNGS + ni, batch_idx, ep), device=self.device)
            self.log(f"train/{n}", v, prog_bar=(n == "loss"), on_step=True, on_epoch=False, logger=True, sync_dist=False)
            self.log(f"train/{n}_epoch", v, on_step=False, on_epoch=True, logger=True, sync_dist=sync)
        for mi, m in enumerate(["lddt_ca", "drmsd_ca"]):
            v = torch.tensor(train_val(r, 2 * N_RUNGS + len(LOSS_NAMES) + mi, batch_idx, ep), device=self.device)
            self.log(f"train/{m}", v, on_step=False, on_epoch=True, logger=True, sync_dist=sync)

        with torch.autocast("cuda", enabled=False):
            c = self._coeff(r, self.global_step)
            loss = torch.zeros((), device=self.device)
            for p, off in zip(self.train_params, self._offsets):
                loss = loss + (p * c[off:off + p.numel()].view_as(p)).sum()
        return loss

    def on_after_backward(self):
        world, s = self.trainer.world_size, self.global_step
        expected = sum(self._coeff(r, s) / world for r in range(world))
        missing = sum(1 for p in self.train_params if p.grad is None)
        got = torch.cat([p.grad.reshape(-1) for p in self.train_params])
        diff = (got - expected).abs()
        bad = diff != 0
        n_bad = int(bad.sum())
        n_corrupt = int((torch.bincount(self._tid[bad].long(), minlength=len(self.train_params)) > 0).sum()) if n_bad else 0
        self._csv("grad_check", ["rank", "epoch", "global_step", "n_corrupt_grad_tensors", "n_bad_elements",
                                 "max_abs_err", "n_grad_none", "wall"],
                  [self.global_rank, self.current_epoch, s, n_corrupt, n_bad, float(diff.max()), missing,
                   f"{time.time():.6f}"])
        self._trace(f"after_backward n_corrupt={n_corrupt} n_bad={n_bad} max_abs_err={float(diff.max())}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step in (1, 2) and self.global_step not in self._reported_buckets:
            self._reported_buckets.add(self.global_step)
            ld = self.trainer.strategy.model._get_ddp_logging_data()
            self._trace("ddp_logging " + json.dumps({k: v for k, v in ld.items() if "bucket" in k}))
        if self.param_check_every and self.global_step % self.param_check_every == 0:
            self._param_check("batch_end")

    def _param_check(self, where):
        cs = torch.stack([torch.stack([p.detach().view(torch.int32).sum(dtype=torch.int64),
                                       (p.detach().reshape(-1).view(torch.int32).long() * self._w[:p.numel()]).sum()])
                          for p in self.all_params]).cpu()
        out = [torch.empty_like(cs) for _ in range(self.trainer.world_size)]
        dist.all_gather(out, cs, group=self.gloo)
        n_diff = int((out[self.global_rank] != out[0]).any(dim=1).sum())
        n_diff_any = int(sum((o != out[0]).any(dim=1) for o in out).clamp(max=1).sum())
        self._csv("param_check", ["rank", "epoch", "global_step", "where", "n_tensors_differ_vs_rank0",
                                  "n_tensors_differ_any_rank", "wall"],
                  [self.global_rank, self.current_epoch, self.global_step, where, n_diff, n_diff_any, f"{time.time():.6f}"])

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-5)  # --learning_rate 1e-4; eps :882
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step", "name": "AlphaFoldLRScheduler"}}

    # ---- validation ----------------------------------------------------------------------------
    def on_validation_epoch_start(self):
        self._val_ptm_calib_pairs = []
        self._val_per_entry_records = []
        self._trace("val_epoch_start")

    def validation_step(self, batch, batch_idx):
        r, ep = self.global_rank, self.current_epoch
        if self._is_distributed is None:
            self._is_distributed = hasattr(self, "trainer") and self.trainer and self.trainer.world_size > 1
        for ki, k in enumerate(val_keys_for(batch_idx)):
            v = torch.tensor(val_val(r, ALL_VAL_KEYS.index(k), batch_idx, ep), device=self.device)
            self.log(k, v, on_step=False, on_epoch=True, logger=True, sync_dist=self._is_distributed)
        idx = int(batch["idx"].flatten()[0])
        self._val_ptm_calib_pairs.append(((idx % 101) / 101.0, (idx % 103) / 103.0))
        self._val_per_entry_records.append(record(r, idx, ep))

    def on_validation_epoch_end(self):
        # train_openfold.py:698-712, minus the EMA weight swap (local)
        pairs = self._val_ptm_calib_pairs
        if self._is_distributed:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, pairs)
            pairs = [p for rank_pairs in gathered for p in rank_pairs]
        if pairs:
            a = torch.tensor([p[0] for p in pairs]).argsort().argsort().float()
            b = torch.tensor([p[1] for p in pairs]).argsort().argsort().float()
            rho = torch.corrcoef(torch.stack([a, b]))[0, 1]
            self.log("val/ptm_calibration_spearman", rho, logger=True, rank_zero_only=True)
        self._trace(f"val_epoch_end ptm_pairs_gathered={len(pairs)}")
        self._val_per_entry_epoch = self.current_epoch
        self._val_per_entry_step = self.global_step

    # ---- the flush -----------------------------------------------------------------------------
    def _flush_per_entry_records(self):
        return self._flush_impl(self)

    def _check_epoch_metrics(self, where):
        # read the plain dict: the logged_metrics property could trigger a metric compute (= a sync)
        logged = self.trainer._logger_connector._logged_metrics
        ep = self._val_per_entry_epoch
        rows = []
        world = self.trainer.world_size
        for k in ALL_VAL_KEYS:
            ki = ALL_VAL_KEYS.index(k)
            s = n = 0
            for r in range(world):
                for j in range(self._n_val_per_rank):
                    if k in val_keys_for(j):
                        s += val_val(r, ki, j, ep)
                        n += 1
            exp = torch.tensor(float(s)) / torch.tensor(float(n))
            got = logged.get(k)
            rows.append((k, float(exp), None if got is None else float(got)))
        for ki, k in enumerate(TRAIN_SYNCED):
            s = n = 0
            for r in range(world):
                for j in range(self._train_steps_last_epoch):
                    s += train_val(r, ki, j, ep)
                    n += 1
            exp = torch.tensor(float(s)) / torch.tensor(float(n))
            got = logged.get(k)
            rows.append((k, float(exp), None if got is None else float(got)))
        n_bad = sum(1 for _, e, g in rows if g is None or e != g)
        for k, e, g in rows:
            self._csv("metric_check", ["rank", "where", "epoch_checked", "key", "expected", "got", "ok"],
                      [self.global_rank, where, ep, k, e, g, int(g is not None and e == g)])
        self._trace(f"metric_check {where} epoch={ep} bad={n_bad} of {len(rows)}")

    def on_train_epoch_start(self):
        self._trace("train_epoch_start enter")
        if self._train_steps_last_epoch is not None:
            self._check_epoch_metrics("next_epoch_start")
        if self.current_epoch > 0:
            # bisection knob: each symmetric 4-byte broadcast is 1 LL step on ring edge 0->1, channel 0
            for _ in range(self.pre_flush_bcasts):
                dist.broadcast(torch.ones(1, device=self.device), src=0)
        self._trace("flush enter")
        self._flush_per_entry_records()
        self._trace(f"flush exit csv_path={self._per_entry_csv_path}")
        self._train_steps_last_epoch = self.trainer.num_training_batches

    def on_fit_end(self):
        self._check_epoch_metrics("fit_end")
        self._flush_per_entry_records()
        self._trace("fit_end flush done")
        raw = torch._C._distributed_c10d._dump_nccl_trace()
        with open(self._rank_file("fr_final") + ".pkl", "wb") as fh:
            fh.write(raw)
        self._trace(f"fr dumped {len(pickle.loads(raw).get('entries', []))} entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--flush_src", required=True, help="train_openfold.py whose _flush_per_entry_records is used")
    ap.add_argument("--flush_variant", choices=["buggy", "fixed"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps_epoch0", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--max_epochs", type=int, required=True)
    ap.add_argument("--n_val", type=int, required=True)
    ap.add_argument("--num_workers", type=int, required=True)
    ap.add_argument("--pg_timeout_s", type=int, required=True)
    ap.add_argument("--param_check_every", type=int, required=True)
    ap.add_argument("--resume_ckpt", default=None, help="Lightning ckpt_path, to emulate a resumed Run C v2 process")
    ap.add_argument("--pre_flush_bcasts", type=int, default=0,
                    help="symmetric 1-float broadcasts before each epoch>0 flush (shifts the NCCL step residue)")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("medium")  # train_openfold.py:982
    seed_everything(42, workers=True)  # --seed 42
    os.makedirs(args.out_dir, exist_ok=True)

    flush_fn, flush_src, md5 = load_flush(args.flush_src)
    fixed_marker = "if records and self._per_entry_csv_path is None:" in flush_src
    assert fixed_marker == (args.flush_variant == "fixed"), (args.flush_variant, md5)
    with open(args.shapes) as fh:
        shapes = json.load(fh)
    entries = shapes["params"]
    assert shapes["n_trainable"] == 4471 and shapes["numel_trainable"] == 87953280, shapes["n_trainable"]

    world = torch.cuda.device_count()
    model = Harness(entries, flush_fn, args.out_dir, args.param_check_every, args.pg_timeout_s, args.pre_flush_bcasts)
    model._n_val_per_rank = -(-args.n_val // world)
    data = HarnessData(args.steps_epoch0, args.steps, args.n_val, world, args.num_workers)

    strategy = TracedDDPStrategy(find_unused_parameters=False, cluster_environment=None,
                                 process_group_backend="nccl", timeout=timedelta(seconds=args.pg_timeout_s))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    TracedDDPStrategy.trace_path = os.path.join(args.out_dir, f"strategy_trace_local{local_rank}.log")
    if local_rank == 0:
        print(f"flush_src={args.flush_src} md5={md5} variant={args.flush_variant}\n{flush_src}", flush=True)
    callbacks = [
        ModelCheckpoint(monitor="val/lddt_ca", mode="max", save_top_k=5, save_last=False,
                        filename="best-{epoch:03d}-{step:06d}", auto_insert_metric_name=False),
        ModelCheckpoint(every_n_train_steps=20, save_top_k=0, save_last=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(
        num_nodes=1, precision="bf16", max_epochs=args.max_epochs, log_every_n_steps=20,
        num_sanity_val_steps=0, reload_dataloaders_every_n_epochs=1,
        default_root_dir=args.out_dir, strategy=strategy, callbacks=callbacks,
        logger=[TensorBoardLogger(save_dir=args.out_dir, name="lightning_logs")],
        accumulate_grad_batches=1, gradient_clip_val=0.1, gradient_clip_algorithm="norm",
    )
    trainer.fit(model, datamodule=data, ckpt_path=args.resume_ckpt)
    print(f"[local_rank {local_rank}] FIT RETURNED", flush=True)


if __name__ == "__main__":
    main()

import argparse
import csv
import logging
import os
import sys
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch
import torch.distributed as dist
import torch.multiprocessing as _tmp
_tmp.set_sharing_strategy("file_system")
import wandb
from deepspeed.utils import zero_to_fp32 

from openfold.config import model_config
from openfold.data.data_modules import OpenFoldDataModule, OpenFoldMultimerDataModule
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.np import residue_constants
from openfold.utils.callbacks import (
    EarlyStoppingVerbose,
)
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.loss import AlphaFoldLoss, lddt_ca
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.superimposition import superimpose
from openfold.utils.t4_self_distill import template_gate_metrics
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
    actual_tm_score,
    spearman_corr,
)
from openfold.utils.import_weights import (
    import_jax_weights_,
    import_openfold_weights_
)
from openfold.utils.logger import PerformanceLoggingCallback
from openfold.block_replacement_scripts.custom_evoformer_replacement import (
    replace_evoformer_block, 
    freeze_all_except_replaced_block
)
from openfold.block_replacement_scripts.pruned_evoformer import (
    prune_blocks,
    freeze_all_except_evoformer,
    freeze_all_except_heads,
)

# Import AdaptiveOpenFoldWrapper for adaptive training
try:
    from openfold.block_replacement_scripts.custom_openfold_wrapper import AdaptiveOpenFoldWrapper
    ADAPTIVE_WRAPPER_AVAILABLE = True
except ImportError:
    ADAPTIVE_WRAPPER_AVAILABLE = False


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config, replace_block_index=None, replacement_hidden_dim=None, learning_rate=1e-3, warmup_no_steps=1000):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        self.model = AlphaFold(config)
        self.is_multimer = self.config.globals.is_multimer
        self.replace_block_index = replace_block_index
        self.replacement_hidden_dim = replacement_hidden_dim
        self.learning_rate = learning_rate
        self.warmup_no_steps = warmup_no_steps

        # Apply block replacement if specified
        if replace_block_index is not None:
            self._apply_block_replacement()

        self.loss = AlphaFoldLoss(config.loss)

        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )

        self.cached_weights = None
        self.last_lr_step = -1
        self._is_distributed = None  # Cache for distributed detection
        self._val_ptm_calib_pairs = []  # (ptm_score, actual_tm) pairs, this rank, this val epoch
        # Per-entry validation records for T1's "which targets did we get better at" tracking
        # (see ESMFOLD2_RECYCLE_SCALING.md T1). Epoch-mean scalars in TensorBoard cannot answer
        # that question -- this keeps the identity of each entry alongside its own metrics.
        self._val_per_entry_records = []
        self._val_per_entry_epoch = 0
        self._val_per_entry_step = 0
        self._per_entry_csv_path = None
        self.save_hyperparameters()
    
    def _apply_block_replacement(self):
        """Apply the custom block replacement and freezing logic"""
        if self.replace_block_index is None:
            return
            
        # Get dimensions from config
        c_m = self.config.model.evoformer_stack.c_m
        c_z = self.config.model.evoformer_stack.c_z
        
        # Replace the specified block
        self.model = replace_evoformer_block(
            self.model, 
            self.replace_block_index, 
            c_m, 
            c_z, 
            self.replacement_hidden_dim
        )
        
        # Freeze all parameters except the replaced block
        trainable_params = freeze_all_except_replaced_block(
            self.model, 
            self.replace_block_index
        )
        
        rank_zero_info(f"Applied block replacement and freezing. Trainable parameters: {trainable_params:,}")

    def forward(self, batch):
        return self.model(batch)

    def on_before_optimizer_step(self, optimizer):
        # Opt-in gradient diagnostic (DEBUG_GRAD_CHECK=1): reports whether the evoformer, the
        # ESMFold2-inspired contractive pair-update module (if enabled), and any frozen
        # parameters have the gradient state they SHOULD have -- added specifically to verify
        # the contractive module's learnable Delta/A/B parameters actually receive gradients
        # (they live outside model.evoformer, so a freeze scheme scoped to "evoformer only"
        # would otherwise silently leave them frozen/dead).
        if os.environ.get("DEBUG_GRAD_CHECK") != "1":
            return
        groups = {"evoformer": list(self.model.evoformer.parameters())}
        contractive = getattr(self.model.recycling_embedder, "contractive_pair_update", None)
        if contractive is not None:
            groups["contractive_pair_update"] = list(contractive.parameters())
        groups["structure_module (expected frozen)"] = list(self.model.structure_module.parameters())
        for name, params in groups.items():
            grads = [p.grad for p in params if p.grad is not None]
            if not grads:
                rank_zero_info(f"[grad-check] {name}: 0/{len(params)} params have a gradient")
                continue
            finite = all(torch.isfinite(g).all() for g in grads)
            total_norm = sum(g.norm().item() ** 2 for g in grads) ** 0.5
            rank_zero_info(f"[grad-check] {name}: {len(grads)}/{len(params)} params have a "
                           f"gradient, finite={finite}, total_norm={total_norm:.3e}")

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
        
        # Detect if we're in a distributed setting (cache for efficiency)
        if self._is_distributed is None:
            self._is_distributed = hasattr(self, 'trainer') and self.trainer and self.trainer.world_size > 1
        
        sync_epoch_metrics = self._is_distributed
        
        for loss_name, indiv_loss in loss_breakdown.items():
            # Determine if this will be epoch-level logging
            is_epoch_level = (not train)  # Validation logs are epoch-level
            sync_for_this_call = sync_epoch_metrics if is_epoch_level else False
            
            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                prog_bar=(loss_name == 'loss'),
                on_step=train, on_epoch=(not train), logger=True, 
                sync_dist=sync_for_this_call,  # Sync for epoch-level (including validation)
            )

            if (train):
                # Additional epoch-level logging for training (sync in distributed settings)
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True, 
                    sync_dist=sync_epoch_metrics,  # Sync for epoch-level in distributed
                )

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch,
                outputs,
                superimposition_metrics=(not train)
            )

        for k, v in other_metrics.items():
            # Epoch-level validation metrics (sync in distributed settings)
            self.log(
                f"{phase}/{k}",
                torch.mean(v),
                prog_bar = (k == 'loss'),
                on_step=False, on_epoch=True, logger=True,
                sync_dist=sync_epoch_metrics,  # Sync for epoch-level in distributed
            )

        # Additionally split validation metrics by "is this PDA entry verbatim present in the
        # model's own training set" (see ESMFOLD2_RECYCLE_SCALING.md PDA investigation) -- these
        # entries stay IN the validation population (not filtered out), this just reports them
        # separately as a diagnostic marker for whether the model has actually learned its own
        # training data. Only active when PDASingleSeqDataset was built with
        # --pda_train_overlap_ids (batch then carries "is_train_overlap"); the full-population
        # val/{k} above is logged unconditionally either way.
        if (not train) and "is_train_overlap" in batch:
            suffix = "train_overlap" if bool(batch["is_train_overlap"].flatten()[0]) else "held_out"
            for k, v in other_metrics.items():
                self.log(
                    f"{phase}/{k}_{suffix}",
                    torch.mean(v),
                    on_step=False, on_epoch=True, logger=True,
                    sync_dist=sync_epoch_metrics,
                )

    def training_step(self, batch, batch_idx):
        if (self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)

        ground_truth = batch.pop('gt_features', None)

        # Direction-2 hybrid: frozen full-48 teacher forward (no grad) on the same batch -> single/pair targets.
        _teacher = getattr(self, "distill_teacher", None)
        _dw = getattr(self, "distill_weight", 0.0)
        _teacher_out = None
        if _teacher is not None and _dw > 0:
            _dev = batch["aatype"].device
            if next(_teacher.parameters()).device != _dev:
                _teacher.to(_dev)
            # NOTE: do NOT set return_representations on the shared batch -- it makes
            # AlphaFold.iteration early-return before the structure module (no outputs['sm'])
            # and would crash the student loss. Default forward already exposes outputs['single']/['pair'].
            _tb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                _teacher_out = _teacher(_tb)

        # Cached-teacher distillation: pop targets before the forward (they carry no recycling dim)
        _ck = ["single", "msa_row0", "pair", "distogram", "plddt", "pae"]
        _cache_t = {k: batch.pop("teacher_" + k) for k in _ck if "teacher_" + k in batch}
        batch.pop("num_res_crop_start", None)

        # Run the model
        outputs = self(batch)

        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        if self.is_multimer:
            batch = multi_chain_permutation_align(out=outputs,
                                                  features=batch,
                                                  ground_truth=ground_truth)

        # Compute loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        # Direction-2 hybrid: add teacher-distillation MSE on final single/pair representations.
        if _teacher_out is not None:
            _tg = getattr(self, "distill_targets", "s,z")
            _dl = outputs["single"].float().new_zeros(())
            if "s" in _tg and "single" in outputs and "single" in _teacher_out:
                _dl = _dl + torch.nn.functional.mse_loss(outputs["single"].float(), _teacher_out["single"].float())
            if "z" in _tg and "pair" in outputs and "pair" in _teacher_out:
                _dl = _dl + torch.nn.functional.mse_loss(outputs["pair"].float(), _teacher_out["pair"].float())
            loss = loss + _dw * _dl
            loss_breakdown["distill_mse"] = _dl.detach()

        # Cached-teacher distillation: embedding MSE (single/msa-row/pair) + KL (distogram/pLDDT/pAE), masked by seq_mask
        if _cache_t:
            import torch.nn.functional as _F
            _m = batch["seq_mask"].float()
            _m1 = _m.unsqueeze(-1)
            _m2d = _m[:, :, None] * _m[:, None, :]
            _m2 = _m2d.unsqueeze(-1)
            def _mse1(pp, tt):
                d = (pp.float() - tt.float()) ** 2
                return (d * _m1).sum() / (_m1.sum() * pp.shape[-1]).clamp_min(1.0)
            def _mse2(pp, tt):
                d = (pp.float() - tt.float()) ** 2
                return (d * _m2).sum() / (_m2.sum() * pp.shape[-1]).clamp_min(1.0)
            def _kl1(sl, tl):
                tp = _F.softmax(tl.float(), -1); sp = _F.log_softmax(sl.float(), -1)
                kl = (tp * (tp.clamp_min(1e-9).log() - sp)).sum(-1)
                return (kl * _m).sum() / _m.sum().clamp_min(1.0)
            def _kl2(sl, tl):
                tp = _F.softmax(tl.float(), -1); sp = _F.log_softmax(sl.float(), -1)
                kl = (tp * (tp.clamp_min(1e-9).log() - sp)).sum(-1)
                return (kl * _m2d).sum() / _m2d.sum().clamp_min(1.0)
            _cl = (_mse1(outputs["single"], _cache_t["single"])
                   + _mse1(outputs["msa"][:, 0], _cache_t["msa_row0"])
                   + _mse2(outputs["pair"], _cache_t["pair"])
                   + _kl2(outputs["distogram_logits"], _cache_t["distogram"])
                   + _kl1(outputs["lddt_logits"], _cache_t["plddt"])
                   + _kl2(outputs["tm_logits"], _cache_t["pae"]))
            _cw = float(getattr(self, "cache_distill_weight", 1.0))
            loss = loss + _cw * _cl
            loss_breakdown["cache_distill"] = _cl.detach()

        # T4 self-distillation gate: measure whether this prediction beat the template it was given.
        # Measurement only for now -- promotion I/O needs the chain->template index that T2's
        # template-consuming path will own. Costs ~0.03% of a step (openfold/utils/tm_score.py).
        if getattr(self, "t4_self_distill", False):
            m = template_gate_metrics(
                outputs, batch,
                delta=getattr(self, "t4_delta", 0.05),
                min_tm=getattr(self, "t4_min_tm", 0.0),
            )
            n_t = m["has_template"].sum().clamp_min(1.0)
            self.log("t4/tm_pred", m["tm_pred"].mean(), on_step=True, on_epoch=True, logger=True)
            self.log("t4/tm_template", (m["tm_template"] * m["has_template"]).sum() / n_t,
                     on_step=True, on_epoch=True, logger=True)
            self.log("t4/margin", ((m["tm_pred"] - m["tm_template"]) * m["has_template"]).sum() / n_t,
                     on_step=True, on_epoch=True, logger=True)
            self.log("t4/promote_rate", m["promote"].sum() / n_t,
                     on_step=True, on_epoch=True, logger=True)
            self.log("t4/has_template", m["has_template"].mean(), on_step=True, on_epoch=True,
                     logger=True)

        # Log it
        self._log(loss_breakdown, batch, outputs)

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def on_validation_epoch_start(self):
        self._val_ptm_calib_pairs = []
        self._val_per_entry_records = []

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if (self.cached_weights is None):
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling
            # load_state_dict().
            def clone_param(t): return t.detach().clone()
            self.cached_weights = tensor_tree_map(
                clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])

            # ESMFold2-tricks run: validate on TRUE single-sequence prediction (no templates),
            # even when templates are kept ON for training (--single_seq_keep_templates) -- makes
            # val/lddt_ca (checkpoint selection) reflect genuine single-sequence capability
            # rather than template-assisted performance. Restored in on_validation_epoch_end.
            if getattr(self, "validate_without_templates", False):
                self._orig_template_enabled = self.model.config.template.enabled
                self.model.config.template.enabled = False

        ground_truth = batch.pop('gt_features', None)

        # Run the model
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        batch["use_clamped_fape"] = 0.

        if self.is_multimer:
            batch = multi_chain_permutation_align(out=outputs,
                                                  features=batch,
                                                  ground_truth=ground_truth)

        # Compute loss and other metrics
        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown, batch, outputs, train=False)
        
    def on_validation_epoch_end(self):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None
        if hasattr(self, "_orig_template_enabled"):
            self.model.config.template.enabled = self._orig_template_enabled
            del self._orig_template_enabled

        # pTM calibration: gather (ptm_score, actual_tm) pairs from every DDP rank (correlation
        # needs the whole validation population, not a per-rank-shard estimate) and log a single
        # Spearman correlation for the epoch. Empty when the tm head is disabled (e.g. non-ptm
        # config presets never populate outputs["ptm_score"]) -- skip logging in that case rather
        # than crashing on zip(*[]).
        pairs = self._val_ptm_calib_pairs
        if self._is_distributed:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, pairs)
            pairs = [p for rank_pairs in gathered for p in rank_pairs]
        if pairs:
            ptm_vals, real_tm_vals = zip(*pairs)
            calibration = spearman_corr(
                torch.tensor(ptm_vals), torch.tensor(real_tm_vals)
            )
            self.log("val/ptm_calibration_spearman", calibration, logger=True, rank_zero_only=True)

        # Capture epoch/step HERE (still correct for "the epoch that just validated") since the
        # actual gather+write is deferred to on_train_epoch_start, by which point Lightning's own
        # epoch counter has already advanced to the NEXT epoch -- see _flush_per_entry_records.
        self._val_per_entry_epoch = self.current_epoch
        self._val_per_entry_step = self.global_step

    def on_train_epoch_start(self):
        # T1 per-entry tracking, deferred flush. A/B-tested 2026-08-11 (see
        # ESMFOLD2_RECYCLE_SCALING.md T1): calling dist.all_gather_object for this from
        # on_validation_epoch_end -- immediately before Lightning's own ModelCheckpoint
        # _monitor_candidates DDP-metric-sync (a known-fragile, unresolved-upstream race,
        # github.com/Lightning-AI/pytorch-lightning#19045) -- reliably deadlocked the run.
        # Deferring the gather to here (on_train_epoch_start of the NEXT epoch, which Lightning
        # only reaches after the previous epoch's checkpoint decision has fully completed, see
        # fit_loop.py on_advance_end's callback/module-hook ordering) moves it out of that race
        # window. on_fit_end below is the safety-net flush for the final epoch, which has no
        # "next" on_train_epoch_start to defer to.
        self._flush_per_entry_records()

    def on_fit_end(self):
        self._flush_per_entry_records()

    def _flush_per_entry_records(self):
        # T1 per-entry tracking: gather every rank's records (each rank only sees its own DDP
        # shard, so a per-rank file would silently hold a fraction of the val set) and append one
        # row per entry per epoch to a CSV next to the checkpoints. Appending rather than
        # overwriting keeps the full history, so "got better at" is answerable over time, and
        # survives the auto-versioning that puts each resume in a fresh lightning_logs dir.
        if os.environ.get("SKIP_PER_ENTRY_TRACKING"):
            return
        records = self._val_per_entry_records
        if self._is_distributed:
            gathered_recs = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_recs, records)
            records = [r for rank_recs in gathered_recs for r in rank_recs]
        self._val_per_entry_records = []  # consumed; avoids re-flushing the same epoch twice
        if records and self.trainer.is_global_zero:
            if self._per_entry_csv_path is None:
                log_dir = getattr(self.trainer, "log_dir", None) or "."
                self._per_entry_csv_path = os.path.join(log_dir, "per_entry_val_history.csv")
            write_header = not os.path.exists(self._per_entry_csv_path)
            with open(self._per_entry_csv_path, "a", newline="") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow(
                        ["epoch", "global_step", "batch_idx", "lddt_ca",
                         "alignment_rmsd", "recall_2A", "gdt_ts"]
                    )
                for idx, lddt, rmsd, recall, gdt in sorted(records):
                    writer.writerow(
                        [self._val_per_entry_epoch, self._val_per_entry_step, int(idx),
                         f"{lddt:.6f}", f"{rmsd:.6f}", f"{recall:.6f}", f"{gdt:.6f}"]
                    )

    def _compute_validation_metrics(self,
                                    batch,
                                    outputs,
                                    superimposition_metrics=False
                                    ):
        metrics = {}

        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]

        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]

        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=self.config.globals.eps,
            per_residue=False,
        )

        metrics["lddt_ca"] = lddt_ca_score

        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca,  # still required here to compute n
        )

        metrics["drmsd_ca"] = drmsd_ca_score

        if (superimposition_metrics):
            superimposed_pred, alignment_rmsd = superimpose(
                gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["alignment_rmsd"] = alignment_rmsd
            # Pass/fail recall at the standard self-consistency threshold -- Lightning's existing
            # epoch-mean aggregation over this 0/1 indicator gives the pass RATE directly.
            metrics["recall_2A"] = (alignment_rmsd < 2.0).float()
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score

            # pTM confidence calibration: the raw mean of pTM isn't informative on its own (we
            # already know from the standalone PDA baseline that both models' pTM sits low and
            # uniform regardless of actual quality) -- what matters is whether pTM actually
            # TRACKS real quality. actual_tm_score is the ground-truth analog of pTM (same d0
            # length-normalization as compute_tm, applied to real post-superposition distances),
            # so (ptm_score, actual_tm) pairs are directly comparable. Accumulated across the
            # whole validation epoch (and all DDP ranks, gathered in on_validation_epoch_end)
            # because a per-batch-item correlation is meaningless -- correlation needs the full
            # population, unlike the other metrics here which are plain epoch means.
            # outputs["ptm_score"] only exists when the tm head is enabled (config presets like
            # initial_training/model_1-5/finetuning leave it off by default) -- guard rather than
            # KeyError on those presets.
            if "ptm_score" in outputs:
                real_tm = actual_tm_score(
                    superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
                )
                self._val_ptm_calib_pairs.extend(
                    zip(outputs["ptm_score"].reshape(-1).tolist(), real_tm.reshape(-1).tolist())
                )

            # T1 per-entry tracking: keep each entry's IDENTITY next to its own metrics, so we can
            # answer "which specific targets did this run get better at" rather than only seeing
            # epoch means. "batch_idx" is the manifest index injected by PDASingleSeqDataset, so it
            # maps back to (pdb, chain_id) via the same manifest JSON -- no extra compute. Guarded:
            # only the PDA dataset provides it, so non-PDA validation paths are unaffected.
            if "batch_idx" in batch:
                idxs = batch["batch_idx"].reshape(-1).tolist()
                lddt_vals = metrics["lddt_ca"].reshape(-1).tolist()
                rmsd_vals = metrics["alignment_rmsd"].reshape(-1).tolist()
                recall_vals = metrics["recall_2A"].reshape(-1).tolist()
                gdt_vals = metrics["gdt_ts"].reshape(-1).tolist()
                # zip() truncates to the shortest, which would silently drop entries if any metric
                # were unexpectedly scalar rather than per-sample -- assert instead of losing rows.
                assert (
                    len(lddt_vals) == len(idxs)
                    and len(rmsd_vals) == len(idxs)
                    and len(recall_vals) == len(idxs)
                    and len(gdt_vals) == len(idxs)
                ), (
                    f"per-entry metric length mismatch: idx={len(idxs)} lddt={len(lddt_vals)} "
                    f"rmsd={len(rmsd_vals)} recall={len(recall_vals)} gdt={len(gdt_vals)}"
                )
                self._val_per_entry_records.extend(
                    zip(idxs, lddt_vals, rmsd_vals, recall_vals, gdt_vals)
                )

        return metrics

    def configure_optimizers(self, 
        learning_rate: float = None,
        eps: float = 1e-5,
    ) -> torch.optim.Adam:
        # Use learning rate from args if provided, otherwise use default
        if learning_rate is None:
            learning_rate = getattr(self, 'learning_rate', 1e-3)
        # Ignored as long as a DeepSpeed optimizer is configured
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            eps=eps
        )

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = learning_rate

        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
            last_epoch=self.last_lr_step,
            max_lr=learning_rate,
            warmup_no_steps=getattr(self, "warmup_no_steps", 1000),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            }
        }

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint["ema"]
        if (not self.model.template_config.enabled):
            ema["params"] = {k: v for k,
                             v in ema["params"].items() if not "template" in k}
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step

    def load_from_jax(self, jax_path):
        model_basename = os.path.splitext(
            os.path.basename(
                os.path.normpath(jax_path)
            )
        )[0]
        model_version = "_".join(model_basename.split("_")[1:])
        import_jax_weights_(
            self.model, jax_path, version=model_version
        )

def get_model_state_dict_from_ds_checkpoint(checkpoint_dir):
    latest_path = os.path.join(checkpoint_dir, 'latest')
    if os.path.isfile(latest_path):
        with open(latest_path, 'r') as fd:
            tag = fd.read().strip()
    else:
        raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    ds_checkpoint_dir = os.path.join(checkpoint_dir, tag)
    _DS_CHECKPOINT_VERSION = 2  # based on manual parsing of checkpoint files
    state_file = zero_to_fp32.get_model_state_file(ds_checkpoint_dir, _DS_CHECKPOINT_VERSION)
    return torch.load(state_file, weights_only=False)

def main(args):
    # Set float32 matmul precision for Tensor Cores
    torch.set_float32_matmul_precision("medium")
    
    if(args.seed is not None):
        seed_everything(args.seed, workers=True)

    is_low_precision = args.precision in [
        "bf16-mixed", "16", "bf16", "16-true", "16-mixed", "bf16-mixed"]

    config = model_config(
        args.config_preset, 
        train=True, 
        low_prec=is_low_precision,
    )
    _mri = os.environ.get("MAX_RECYCLING_ITERS")
    if _mri is not None:
        config.data.common.max_recycling_iters = int(_mri)   # student trains at the cache recycle count
    if args.experiment_config_json: 
        with open(args.experiment_config_json, 'r') as f:
            custom_config_dict = json.load(f)
        config.update_from_flattened_dict(custom_config_dict)

    # Configure for single sequence mode if requested
    if args.enable_single_seq_mode:
        rank_zero_info("Enabling single sequence mode - reducing MSA and template requirements")
        # Reduce MSA requirements for single sequence training
        config.data.common.max_extra_msa = 1
        config.data.common.max_msa_clusters = 1
        config.data.train.max_extra_msa = 1
        config.data.train.max_msa_clusters = 1
        # Disable templates entirely for single sequence mode, UNLESS --single_seq_keep_templates
        # (single-seq + templates: MSA-free query but keep the template channel, e.g. BindCraft-style design).
        if not getattr(args, "single_seq_keep_templates", False):
            config.model.template.enabled = False
            config.data.common.use_templates = False
            config.data.common.use_template_torsion_angles = False
        else:
            rank_zero_info("single_seq_keep_templates: templates KEPT enabled in single-seq mode")
        # Disable MSA-specific losses for single sequence training
        rank_zero_info("Disabling masked_msa loss for single sequence mode")
        config.loss.masked_msa.weight = 0.0
        # Reduce some computational requirements (override-able: pruned/single-seq models use much
        # less VRAM than full AF2, so longer crops may now fit -- default preserves prior behavior).
        _max_crop = int(os.environ.get("SINGLE_SEQ_MAX_CROP", "256"))
        config.data.train.crop_size = min(config.data.train.crop_size, _max_crop)

    # ESMFold2-inspired recycling opt-ins (Appendix A.2.5, arXiv:2604.12946) -- must be set
    # before the model is constructed, since RecyclingEmbedder.__init__ reads these at build time.
    if getattr(args, "contractive_recycling", False):
        rank_zero_info("contractive_recycling: replacing plain-additive z-recycling with the "
                       "ESMFold2-inspired contractive recurrence")
        config.model.recycling_embedder.use_contractive = True
    if getattr(args, "gaussian_pair_init", False):
        rank_zero_info("gaussian_pair_init: sampling the first cycle's pair state from "
                       "trunc_norm(0, 2/(5*c_z)) instead of zeros")
        config.model.recycling_embedder.use_gaussian_pair_init = True

    # Use AdaptiveOpenFoldWrapper if adaptive_config_path is provided
    adaptive_config_path = getattr(args, 'adaptive_config_path', None)
    
    if adaptive_config_path and ADAPTIVE_WRAPPER_AVAILABLE:
        model_module = AdaptiveOpenFoldWrapper(
            config,
            adaptive_config_path=adaptive_config_path,
            learning_rate=getattr(args, 'learning_rate', 1e-3),
            data_loading_strategy=getattr(args, 'data_loading_strategy', 'preload_gpu')
        )
    else:
        model_module = OpenFoldWrapper(
            config, 
            replace_block_index=getattr(args, 'replace_block_index', None),
            replacement_hidden_dim=getattr(args, 'replacement_hidden_dim', None),
            learning_rate=getattr(args, 'learning_rate', 1e-3)
        )

    # Handle checkpoint loading
    if args.resume_from_ckpt:
        if args.resume_model_weights_only and not getattr(args, "evoformer_keep_block_indices", None):
            # Load the checkpoint
            if os.path.isdir(args.resume_from_ckpt):
                sd = zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
                    args.resume_from_ckpt)
            else:
                sd = torch.load(args.resume_from_ckpt, weights_only=False)
            # Process the state dict
            # Use strict=False if we're doing block replacement, single sequence mode, or Evoformer
            # pruning (model structure changed). prune_evoformer deletes msa_att_col and replaces
            # tri_att_start/end with param-free no-ops (see pruned_evoformer.py) AFTER this load, so
            # a freshly-constructed (not-yet-pruned) model always has more keys than an
            # already-pruned checkpoint -- this holds regardless of enable_single_seq_mode, which
            # every prior launcher happened to always combine with prune_evoformer, masking the gap.
            strict_loading = not (
                (hasattr(args, 'replace_block_index') and args.replace_block_index is not None) or
                (hasattr(args, 'enable_single_seq_mode') and args.enable_single_seq_mode) or
                (hasattr(args, 'prune_evoformer') and args.prune_evoformer)
            )
            if not strict_loading:
                if hasattr(args, 'replace_block_index') and args.replace_block_index is not None:
                    rank_zero_info(f"Using strict=False for weight loading due to block replacement at index {args.replace_block_index}")
                elif hasattr(args, 'enable_single_seq_mode') and args.enable_single_seq_mode:
                    rank_zero_info(f"Using strict=False for weight loading due to single sequence mode (templates disabled)")
                elif hasattr(args, 'prune_evoformer') and args.prune_evoformer:
                    rank_zero_info(f"Using strict=False for weight loading due to prune_evoformer (column/triangle attention removed after load)")
            if 'module' in sd:
                sd = {k[len('module.'):]: v for k, v in sd['module'].items()}
                import_openfold_weights_(model=model_module, state_dict=sd, strict=strict_loading)
            elif 'state_dict' in sd:
                import_openfold_weights_(
                    model=model_module, state_dict=sd['state_dict'], strict=strict_loading)
            else:
                # Loading from pre-trained model
                sd = {'model.'+k: v for k, v in sd.items()}
                import_openfold_weights_(model=model_module, state_dict=sd, strict=strict_loading)
            logging.info("Successfully loaded model weights...")

        else:  # Loads a checkpoint to start from a specific time step
            if os.path.isdir(args.resume_from_ckpt):
                sd = get_model_state_dict_from_ds_checkpoint(args.resume_from_ckpt)
            else:
                sd = torch.load(args.resume_from_ckpt, weights_only=False)
            last_global_step = int(sd['global_step'])
            model_module.resume_last_lr_step(last_global_step)
            logging.info("Successfully loaded last lr step...")

    # Handle JAX weight loading with template workaround for single sequence mode
    if args.resume_from_jax_params:
        if args.enable_single_seq_mode and not getattr(args, "single_seq_keep_templates", False):
            rank_zero_info("JAX loading with template workaround for single sequence mode...")
            # Temporarily enable templates for JAX loading
            original_template_enabled = config.model.template.enabled
            config.model.template.enabled = True
            
            # Recreate model with templates enabled for JAX loading
            if adaptive_config_path and ADAPTIVE_WRAPPER_AVAILABLE:
                model_module = AdaptiveOpenFoldWrapper(
                    config,
                    adaptive_config_path=adaptive_config_path,
                    learning_rate=getattr(args, 'learning_rate', 1e-3),
                    data_loading_strategy=getattr(args, 'data_loading_strategy', 'preload_gpu')
                )
            else:
                model_module = OpenFoldWrapper(
                    config, 
                    replace_block_index=getattr(args, 'replace_block_index', None),
                    replacement_hidden_dim=getattr(args, 'replacement_hidden_dim', None),
                    learning_rate=getattr(args, 'learning_rate', 1e-3)
                )
            
            # Load JAX weights
            model_module.load_from_jax(args.resume_from_jax_params)
            logging.info(f"Successfully loaded JAX parameters at {args.resume_from_jax_params}...")
            
            # Disable templates again and remove template embedder
            config.model.template.enabled = False
            if hasattr(model_module.model, 'template_embedder'):
                delattr(model_module.model, 'template_embedder')
                rank_zero_info("Removed template_embedder from model after JAX loading")
            
            rank_zero_info("JAX loading completed - templates disabled for single sequence training")
        else:
            # Normal JAX loading when templates are enabled
            model_module.load_from_jax(args.resume_from_jax_params)
            logging.info(f"Successfully loaded JAX parameters at {args.resume_from_jax_params}...")

    # Apply requested LR-warmup length to the final wrapper (after any single-seq reconstruction).
    if hasattr(model_module, "warmup_no_steps"):
        model_module.warmup_no_steps = getattr(args, "warmup_no_steps", 1000)
    model_module.validate_without_templates = getattr(args, "validate_without_templates", False)
    model_module.t4_self_distill = getattr(args, "t4_self_distill", False)
    model_module.t4_delta = getattr(args, "t4_delta", 0.05)
    model_module.t4_min_tm = getattr(args, "t4_min_tm", 0.0)

    # Direction 2: keep only a SUBSET of the 48 Evoformer blocks (shallower full-block model),
    # warm-started from the matching AF2 block weights (full 48 loaded above, then sliced).
    if getattr(args, "evoformer_keep_block_indices", None):
        import torch.nn as _nn
        _keep = [int(x) for x in str(args.evoformer_keep_block_indices).split(",") if x.strip() != ""]
        _blocks = model_module.model.evoformer.blocks
        model_module.model.evoformer.blocks = _nn.ModuleList([_blocks[i] for i in _keep])
        model_module.ema = ExponentialMovingAverage(model=model_module.model, decay=config.ema.decay)
        rank_zero_info("Evoformer subset: kept %d of %d blocks: %s" % (len(_keep), len(_blocks), _keep))
        # Weights-only resume for the SLIM block-subset model: load AFTER the 48->24 slice so the
        # 24-block ckpt keys match (the early pre-slice load is skipped when keep_block_indices is set).
        if args.resume_from_ckpt and args.resume_model_weights_only:
            if os.path.isdir(args.resume_from_ckpt):
                _wsd = zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(args.resume_from_ckpt)
            else:
                _wsd = torch.load(args.resume_from_ckpt, weights_only=False)
            if 'module' in _wsd:
                _wsd = {k[len('module.'):]: v for k, v in _wsd['module'].items()}
            elif 'state_dict' in _wsd:
                _wsd = _wsd['state_dict']
            else:
                _wsd = {'model.' + k: v for k, v in _wsd.items()}
            import_openfold_weights_(model=model_module, state_dict=_wsd, strict=True)
            model_module.ema = ExponentialMovingAverage(model=model_module.model, decay=config.ema.decay)
            rank_zero_info("weights-only resume (post-slice): loaded sliced ckpt into %d-block model" % len(_keep))
    # Evoformer-only fine-tune for the SLIM block-subset model (freeze embedder/structure-module/heads; no within-block prune).
    if getattr(args, "freeze_non_evoformer", False):
        freeze_all_except_evoformer(model_module.model)
        model_module.ema = ExponentialMovingAverage(model=model_module.model, decay=config.ema.decay)
        _ntr = sum(p.numel() for p in model_module.model.parameters() if p.requires_grad)
        _nall = sum(p.numel() for p in model_module.model.parameters())
        rank_zero_info("freeze_non_evoformer: Evoformer-only trainable %d / %d" % (_ntr, _nall))

    # WS2: confidence-head-only fine-tune (freeze everything except the confidence heads).
    if getattr(args, "freeze_all_except_heads", False):
        _ntr, _nall = freeze_all_except_heads(
            model_module.model, train_distogram=getattr(args, "train_distogram_head", False))
        model_module.ema = ExponentialMovingAverage(model=model_module.model, decay=config.ema.decay)
        rank_zero_info("freeze_all_except_heads: heads-only trainable %d / %d" % (_ntr, _nall))

    # Prune the Evoformer AFTER warm-start weight load; fine-tune Evoformer-only (unless a
    # different freeze mode was already explicitly requested above -- e.g. --freeze_all_except_heads
    # for a confidence-head-only fine-tune of a pruned model, which prune_evoformer must not clobber).
    if getattr(args, "prune_evoformer", False):
        rank_zero_info("Pruning 48 EvoformerBlocks (drop column + triangle attention).")
        prune_blocks(model_module.model.evoformer)
        if not (getattr(args, "freeze_non_evoformer", False) or getattr(args, "freeze_all_except_heads", False)):
            rank_zero_info("prune_evoformer: no other freeze mode requested -- defaulting to Evoformer-only fine-tune.")
            freeze_all_except_evoformer(model_module.model)
        # EMA must be (re)built AFTER pruning regardless of freeze mode: prune_blocks() structurally
        # changes the model (deletes msa_att_col, replaces tri_att_start/end with no-ops), so an EMA
        # built beforehand (e.g. by the freeze_all_except_heads branch above) retains stale keys for
        # the removed modules, which then fails to load at validation time.
        model_module.ema = ExponentialMovingAverage(model=model_module.model, decay=config.ema.decay)
        n_tr = sum(prm.numel() for prm in model_module.model.parameters() if prm.requires_grad)
        n_all = sum(prm.numel() for prm in model_module.model.parameters())
        rank_zero_info("Post-prune trainable params %d / %d" % (n_tr, n_all))

    # Direction-2 hybrid: build a frozen FULL-48-block teacher for online representation distillation.
    if getattr(args, "distill_teacher_jax_params", None) and getattr(args, "distill_weight", 0.0) > 0:
        _teacher = AlphaFold(config)  # full 48 blocks (config unchanged; student was sliced above)
        _tb = os.path.splitext(os.path.basename(os.path.normpath(args.distill_teacher_jax_params)))[0]
        import_jax_weights_(_teacher, args.distill_teacher_jax_params, version="_".join(_tb.split("_")[1:]))
        _teacher.eval()
        for _p in _teacher.parameters():
            _p.requires_grad_(False)
        model_module.distill_teacher = _teacher
        model_module.distill_weight = float(args.distill_weight)
        model_module.distill_targets = args.distill_targets
        rank_zero_info("Distillation: frozen full-48 teacher loaded; weight=%s targets=%s"
                       % (args.distill_weight, args.distill_targets))

    # TorchScript components of the model
    if (args.script_modules):
        script_preset_(model_module)

    if "multimer" in args.config_preset:
        data_module = OpenFoldMultimerDataModule(
            config=config.data,
            batch_seed=args.seed,
            **vars(args)
        )
    else:
        data_module = OpenFoldDataModule(
            config=config.data,
            batch_seed=args.seed,
            **vars(args)
        )

    data_module.prepare_data()
    data_module.setup()

    if getattr(args, "pda_val_manifest", None):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "prune_work"))
        from pda_dataset import PDASingleSeqDataset

        def _set_pda_eval_dataset():
            data_module.eval_dataset = PDASingleSeqDataset(
                manifest_path=args.pda_val_manifest,
                cif_cache_dir=args.pda_cif_cache_dir,
                config=config.data,
                mode="eval",
                train_overlap_ids_path=getattr(args, "pda_train_overlap_ids", None),
            )
            rank_zero_info(
                f"pda_val_manifest: replacing standard eval_dataset with PDA-based single-sequence "
                f"validation ({len(data_module.eval_dataset)} de novo design entries)")

        # trainer.fit(datamodule=data_module)/trainer.validate(...) call data_module.setup()
        # AGAIN internally (pytorch_lightning/trainer/call.py's _call_setup_hook), and
        # OpenFoldDataModule.setup() unconditionally rebuilds self.eval_dataset from
        # val_data_dir/val_chain_list_path with no idempotency guard -- silently clobbering this
        # override back to the natural-protein WS5 val set right before training starts. Patch
        # setup() itself to reapply the PDA override after every call, so it survives Lightning's
        # own re-invocation regardless of how many times setup() runs.
        _orig_setup = data_module.setup
        def _setup_then_reapply_pda(stage=None):
            _orig_setup(stage)
            _set_pda_eval_dataset()
        data_module.setup = _setup_then_reapply_pda

        _set_pda_eval_dataset()

    callbacks = []
    
    # Checkpointing: BEST by validation loss (early-stopping target to mitigate single-seq
    # overfitting) + LAST (save_last) for resume + optional periodic for the recovery curve.
    monitor_metric = (getattr(args, 'checkpoint_monitor', None)
                      or ('val/loss' if (hasattr(args, 'val_data_dir') and args.val_data_dir) else 'train/loss'))
    # max for structural-quality metrics (val/lddt_ca, val/gdt_ts, val/tm); min for any *_loss (e.g. val/plddt_loss).
    monitor_mode = 'max' if (any(k in monitor_metric for k in ('lddt', 'gdt', 'tm')) and 'loss' not in monitor_metric) else 'min'
    best_ckpt = ModelCheckpoint(
        monitor=monitor_metric, mode=monitor_mode, save_top_k=1, save_last=False,
        filename='best-{epoch:03d}-{step:06d}', auto_insert_metric_name=False,
    )
    callbacks.append(best_ckpt)
    # Frequent rolling last.ckpt (every_n_train_steps) so an interruption loses <=~15 min of work.
    # save_top_k=0 -> NO monitored/accumulating files; save_last=True -> only last.ckpt (overwritten).
    _periodic_n = getattr(args, 'checkpoint_every_n_steps', None)
    if _periodic_n:
        callbacks.append(ModelCheckpoint(
            every_n_train_steps=_periodic_n, save_top_k=0, save_last=True,
        ))
    rank_zero_info(f"Checkpoint: best by {monitor_metric} ({monitor_mode}) [1] + last.ckpt every {_periodic_n} steps")

    if (args.early_stopping):
        # Use training metric for early stopping if no validation data is available
        early_stopping_metric = getattr(args, 'early_stopping_metric', 'val/lddt_ca')
        if args.enable_single_seq_mode:
            # In single sequence mode, we typically don't have validation data
            early_stopping_metric = 'train/lddt_ca'
            rank_zero_info(f"Using training metric for early stopping: {early_stopping_metric}")
        
        es = EarlyStoppingVerbose(
            monitor=early_stopping_metric,
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode="max",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)

    if (args.log_performance):
        global_batch_size = args.num_nodes * args.gpus
        perf = PerformanceLoggingCallback(
            log_file=os.path.join(args.output_dir, "performance_log.json"),
            global_batch_size=global_batch_size,
        )
        callbacks.append(perf)

    if (args.log_lr):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    loggers = []
    is_rank_zero = args.mpi_plugin and (int(os.environ.get("PMI_RANK")) == 0)
    
    # Add TensorBoard logger if log_lr is used but no wandb logger is configured
    if args.log_lr and not args.wandb:
        from pytorch_lightning.loggers import TensorBoardLogger
        tb_logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name="lightning_logs"
        )
        loggers.append(tb_logger)
    
    if(args.wandb):
        if args.mpi_plugin and is_rank_zero:
            wandb_init_dict = dict(
                name=args.experiment_name,
                project=args.wandb_project,
                id=args.wandb_id,
                dir=args.output_dir,
                resume="allow",
                anonymous=None,
                entity=args.wandb_entity
            )
            wandb.run = wandb.init(**wandb_init_dict)

        wdb_logger = WandbLogger(
            name=args.experiment_name,
            save_dir=args.output_dir,
            id=args.wandb_id,
            project=args.wandb_project,
            **{"entity": args.wandb_entity}
        )
        loggers.append(wdb_logger)

    cluster_environment = MPIEnvironment() if args.mpi_plugin else None
    if(args.deepspeed_config_path is not None):
        strategy = DeepSpeedStrategy(
            config=args.deepspeed_config_path,
            cluster_environment=cluster_environment,
        )
        if(args.wandb and is_rank_zero):
            wdb_logger.experiment.save(args.deepspeed_config_path)
            wdb_logger.experiment.save("openfold/config.py")
    else:
        rank_zero_info(f"Using distributed training with {args.distributed_backend} backend")
        strategy = DDPStrategy(find_unused_parameters=False,
                               cluster_environment=cluster_environment,
                               process_group_backend=args.distributed_backend)
 
    if(args.wandb and is_rank_zero):
        freeze_path = f"{wdb_logger.experiment.dir}/package_versions.txt"
        os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
        wdb_logger.experiment.save(f"{freeze_path}")

    trainer_kws = ['num_nodes', 'precision', 'max_epochs', 'log_every_n_steps',
                   'flush_logs_ever_n_steps', 'num_sanity_val_steps', 'reload_dataloaders_every_n_epochs']
    trainer_args = {k: v for k, v in vars(args).items() if k in trainer_kws}
    trainer_args.update({
        'default_root_dir': args.output_dir,
        'strategy': strategy,
        'callbacks': callbacks,
        'logger': loggers,
        'accumulate_grad_batches': args.grad_accum_steps,
        'gradient_clip_val': 0.1,
        'gradient_clip_algorithm': 'norm',
    })
    trainer = pl.Trainer(**trainer_args)


    if (args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt

    if getattr(args, "validate_only", False):
        rank_zero_info("validate_only: refreshing EMA from loaded weights and running trainer.validate.")
        model_module.ema = ExponentialMovingAverage(model=model_module.model, decay=config.ema.decay)
        trainer.validate(model_module, datamodule=data_module)
        return

    trainer.fit(
        model_module,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir", type=str,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "train_alignment_dir", type=str,
        help="Directory containing precomputed training alignments"
    )
    parser.add_argument(
        "template_mmcif_dir", type=str,
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    parser.add_argument(
        "max_template_date", type=str,
        help='''Cutoff for all templates. In training mode, templates are also 
                filtered by the release date of the target'''
    )
    parser.add_argument(
        "--train_mmcif_data_cache_path", type=str, default=None,
        help="Path to the json file which records all the information of mmcif structures used during training"
    )
    parser.add_argument(
        "--use_single_seq_mode", type=str, default=False,
        help="Use single sequence embeddings instead of MSAs."
    )
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--val_mmcif_data_cache_path", type=str, default=None,
        help="path to the json file which records all the information of mmcif structures used during validation"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default='/usr/bin/kalign',
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_filter_path", type=str, default=None,
        help='''Optional path to a text file containing names of training
                examples to include, one per line. Used to filter the training 
                set'''
    )
    parser.add_argument(
        "--distillation_filter_path", type=str, default=None,
        help="""See --train_filter_path"""
    )
    parser.add_argument(
        "--obsolete_pdbs_file_path", type=str, default=None,
        help="""Path to obsolete.dat file containing list of obsolete PDBs and 
             their replacements."""
    )
    parser.add_argument(
        "--template_release_dates_cache_path", type=str, default=None,
        help="""Output of scripts/generate_mmcif_cache.py run on template mmCIF
                files."""
    )
    parser.add_argument(
        "--use_small_bfd", type=bool_type, default=False,
        help="Whether to use a reduced version of the BFD database"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_every_epoch", action="store_true", default=False,
        help="""Whether to checkpoint at the end of every training epoch"""
    )
    parser.add_argument(
        "--checkpoint_every_n_steps", type=int, default=None,
        help="Save checkpoint every N training steps (overrides epoch-based saving)"
    )
    parser.add_argument(
        "--checkpoint_every_n_epochs", type=int, default=None,
        help="Save checkpoint every N epochs (alternative to every_epoch)"
    )
    parser.add_argument(
        "--checkpoint_save_top_k", type=int, default=None,
        help="Number of best checkpoints to keep (-1 for all, 1 for best only)"
    )
    parser.add_argument(
        "--checkpoint_monitor", type=str, default=None,
        help="Metric to monitor for best checkpoint (e.g., 'val/loss', 'train/lddt_ca')"
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--resume_from_jax_params", type=str, default=None,
        help="""Path to an .npz JAX parameter file with which to initialize the model"""
    )
    parser.add_argument(
        "--log_performance", type=bool_type, default=False,
        help="Measure performance"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None,
        help="ID of a previous run to be resumed"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project to which this run will belong"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="wandb username or team name to which runs are attributed"
    )
    parser.add_argument(
        "--script_modules", type=bool_type, default=False,
        help="Whether to TorchScript eligible components of them model"
    )
    parser.add_argument(
        "--train_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--distillation_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--train_epoch_len", type=int, default=10000,
        help=(
            "The virtual length of each training epoch. Stochastic filtering "
            "of training data means that training datasets have no "
            "well-defined length. This virtual length affects frequency of "
            "validation & checkpointing (by default, one of each per epoch)."
        )
    )
    parser.add_argument(
        "--log_lr", action="store_true", default=False,
        help="Whether to log the actual learning rate"
    )
    parser.add_argument(
        "--config_preset", type=str, default="initial_training",
        help=(
            'Config setting. Choose e.g. "initial_training", "finetuning", '
            '"model_1", etc. By default, the actual values in the config are '
            'used.'
        )
    )
    parser.add_argument(
        "--_distillation_structure_index_path", type=str, default=None,
    )
    parser.add_argument(
        "--alignment_index_path", type=str, default=None,
        help="Training alignment index. See the README for instructions."
    )
    parser.add_argument(
        "--distillation_alignment_index_path", type=str, default=None,
        help="Distillation alignment index. See the README for instructions."
    )
    parser.add_argument(
        "--experiment_config_json", default="", help="Path to a json file with custom config values to overwrite config setting",
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help='For determining optimal strategy and effective batch size.'
    )
    parser.add_argument("--mpi_plugin", action="store_true", default=False,
                        help="Whether to use MPI for parallele processing")
    parser.add_argument(
        "--distributed_backend", type=str, default="gloo", choices=["nccl", "gloo", "mpi"],
        help="Distributed backend for DDP training (gloo for CPU/compatibility, nccl for GPU performance)"
    )
    
    # Custom block replacement arguments
    parser.add_argument(
        "--replace_block_index", type=int, default=None,
        help="Index of evoformer block to replace with simple architecture (not first/last block)"
    )
    parser.add_argument(
        "--replacement_hidden_dim", type=int, default=None,
        help="Hidden dimension for replacement block (defaults to max(c_m, c_z))"
    )
    parser.add_argument(
        "--enable_single_seq_mode", action="store_true", default=False,
        help="Enable single sequence mode (no MSA/templates required)"
    )
    parser.add_argument(
        "--single_seq_keep_templates", action="store_true", default=False,
        help="In single-seq mode, KEEP templates enabled (MSA-free query + template channel; standard jax load)."
    )
    parser.add_argument(
        "--prune_evoformer", action="store_true", default=False,
        help="Prune all 48 EvoformerBlocks (drop column + triangle attention); fine-tune Evoformer-only."
    )
    parser.add_argument(
        "--contractive_recycling", action="store_true", default=False,
        help="ESMFold2-inspired (Appendix A.2.5, arXiv:2604.12946): replace the plain additive "
             "z-recycling combination with a contractive linear-SSM-style recurrence, which stays "
             "numerically bounded across arbitrarily many recycle iterations (unlike the plain "
             "additive update). Default off -- no behavior change unless set."
    )
    parser.add_argument(
        "--gaussian_pair_init", action="store_true", default=False,
        help="ESMFold2-inspired: sample the first cycle's recurrent pair state from "
             "trunc_norm(0, 2/(5*c_z)) instead of zeros, giving a seed-varying source of "
             "structural diversity that doesn't depend on MSA masking. Default off."
    )
    parser.add_argument(
        "--validate_without_templates", action="store_true", default=False,
        help="Disable template usage (model.config.template.enabled) during validation only, "
             "restoring it after -- makes val/lddt_ca (checkpoint selection) reflect TRUE "
             "single-sequence prediction capability instead of template-assisted performance, "
             "even when templates are kept ON for training (e.g. --single_seq_keep_templates). "
             "Default off (validation matches training's template setting, as before)."
    )
    parser.add_argument(
        "--t2_template_index", type=str, default=None,
        help="T2: path to index_all.npz from prune_work/build_template_index.py (per-chain TM and "
             "rewind for every generated Protpardelle template). Needs --t2_n_synthetic > 0 to "
             "take effect."
    )
    parser.add_argument(
        "--t2_templates_root", type=str, default=None,
        help="T2: root of the generated template shards (shardNNNN/<chain>.npz)."
    )
    parser.add_argument(
        "--t2_max_tm", type=float, default=0.9,
        help="T2: keep only synthetic templates with TM(template, native) BELOW this. Default 0.9 "
             "(user-set 2026-08-14: above it the task becomes trivial)."
    )
    parser.add_argument(
        "--t2_n_synthetic", type=int, default=0,
        help="T2: how many synthetic templates to concatenate onto each training example's natural "
             "hits. The existing train-mode subsampler then draws uniformly over the combined "
             "list, so this sets the mixing ratio: expected synthetic share is N/(4+N), because "
             "98.2%% of training chains carry exactly 4 natural hits (measured n=400). "
             "**Use 4** for the sanctioned 50/50 mixture. 0 = disabled (default), so existing "
             "launchers are untouched. NOTE it also shifts the delivered COUNT -- "
             "P(4 delivered) = (N+1)/(N+5), so zero-template steps fall from 20.6%% to 11%% at N=4."
    )
    parser.add_argument(
        "--t4_self_distill", action="store_true", default=False,
        help="T4: each training step, score the prediction and the best template it was given "
             "against the native (TM, in-loop on the crop) and log t4/{tm_pred,tm_template,"
             "margin,promote_rate,has_template}. Measurement only -- nothing is written and the "
             "loss is untouched, so this is safe to switch on mid-run. Default off."
    )
    parser.add_argument(
        "--t4_delta", type=float, default=0.05,
        help="T4 promotion margin: a prediction counts as beating its template when "
             "TM(pred,native) > TM(template,native) + this. Default 0.05 (user-set 2026-08-13)."
    )
    parser.add_argument(
        "--t4_min_tm", type=float, default=0.0,
        help="T4: absolute floor on TM(pred,native) before a prediction may be promoted. "
             "0.0 disables the floor (default) -- margin alone decides."
    )
    parser.add_argument(
        "--pda_val_manifest", type=str, default=None,
        help="Path to a JSON manifest (list of {pdb, chain_id, seq}) of PDA (Protein Design "
             "Archive) de novo design entries -- when set, REPLACES the standard validation "
             "dataset with PDASingleSeqDataset (true single-sequence, no-template prediction "
             "on real de novo designs, not natural-protein chains). --val_data_dir/"
             "--val_alignment_dir/--val_chain_list_path are still required (unused output "
             "discarded) since OpenFoldDataModule.setup() needs them to construct eval_dataset "
             "in the first place, before this override replaces it."
    )
    parser.add_argument(
        "--pda_cif_cache_dir", type=str, default=None,
        help="Directory of cached PDA mmCIF files ({pdb}.cif), required with --pda_val_manifest."
    )
    parser.add_argument(
        "--pda_train_overlap_ids", type=str, default=None,
        help="Path to a JSON list of {pdb, chain_id} PDA entries whose pdb_chain is verbatim "
             "present in the training set (e.g. classic pre-cutoff designs). These stay IN the "
             "validation population as-is; when this is set, val metrics are ADDITIONALLY logged "
             "split into val/{metric}_train_overlap and val/{metric}_held_out, alongside the "
             "existing full-population val/{metric} (unchanged). Default off (no split logging)."
    )
    parser.add_argument(
        "--evoformer_keep_block_indices", type=str, default=None,
        help="Direction 2: keep only this comma-separated subset of the 48 Evoformer blocks "
             "(e.g. '0,3,7,10,14,17,21,24,28,31,35,38,42,45,47'), warm-started from the matching "
             "AF2 block weights. Full blocks kept intact (no within-block pruning)."
    )
    parser.add_argument(
        "--freeze_non_evoformer", action="store_true", default=False,
        help="Freeze all params except the (sliced) Evoformer blocks -> Evoformer-only fine-tune."
    )
    parser.add_argument(
        "--freeze_all_except_heads", action="store_true", default=False,
        help="WS2: freeze everything except the confidence heads (plddt, experimentally_resolved, tm) -> heads-only fine-tune."
    )
    parser.add_argument(
        "--train_distogram_head", action="store_true", default=False,
        help="With --freeze_all_except_heads, also keep the distogram head trainable."
    )
    parser.add_argument(
        "--distill_teacher_jax_params", type=str, default=None,
        help="Direction 2 hybrid: JAX npz for a frozen FULL-48-block teacher; its final single/pair "
             "representations are matched (MSE) by the subset student each step."
    )
    parser.add_argument(
        "--distill_weight", type=float, default=0.0,
        help="Weight of the teacher-distillation MSE term (0 = off)."
    )
    parser.add_argument(
        "--distill_targets", type=str, default="s,z",
        help="Which teacher representations to distill: 's' (single), 'z' (pair), or 's,z'."
    )
    parser.add_argument(
        "--warmup_no_steps", type=int, default=1000,
        help="Linear LR warmup steps (default 1000; pruned single-seq fine-tune uses 3000)."
    )
    parser.add_argument(
        "--validate_only", action="store_true", default=False,
        help="Run trainer.validate on the provided data and exit (no training)."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Learning rate for training (default: 1e-3)"
    )
    
    # Enhanced data loading arguments
    parser.add_argument(
        "--train_chain_list_path", type=str, default=None,
        help="Path to text file containing training chains (e.g., '1abc_A' per line)"
    )
    parser.add_argument(
        "--distillation_chain_list_path", type=str, default=None,
        help="Path to text file containing distillation chains"
    )
    parser.add_argument(
        "--val_chain_list_path", type=str, default=None,
        help="Path to text file containing validation chains"
    )
    parser.add_argument(
        "--enable_recursive_search", action="store_true", default=True,
        help="Enable recursive search for structure files in subdirectories"
    )
    
    # Adaptive training arguments
    parser.add_argument(
        "--adaptive_config_path", type=str, default=None,
        help="Path to adaptive training configuration JSON file"
    )
    parser.add_argument(
        "--data_loading_strategy", type=str, default="on_demand",
        choices=["preload_gpu", "preload_cpu", "on_demand"],
        help="Data loading strategy for adaptive training: 'preload_gpu' (default), 'preload_cpu', or 'on_demand'"
    )

    trainer_group = parser.add_argument_group(
        'Arguments to pass to PyTorch Lightning Trainer')
    trainer_group.add_argument(
        "--num_nodes", type=int, default=1,
    )
    trainer_group.add_argument(
        "--precision", type=str, default='bf16',
        help='Sets precision, lower precision improves runtime performance.',
    )
    trainer_group.add_argument(
        "--max_epochs", type=int, default=1,
    )
    trainer_group.add_argument(
        "--log_every_n_steps", type=int, default=25,
    )
    trainer_group.add_argument(
        "--flush_logs_every_n_steps", type=int, default=5,
    )
    trainer_group.add_argument(
        "--num_sanity_val_steps", type=int, default=0,
    )
    trainer_group.add_argument(
        "--reload_dataloaders_every_n_epochs", type=int, default=1,
    )
    trainer_group.add_argument(
        "--grad_accum_steps", type=int, default=1,
        help="Accumulate gradients over k batches before next optimizer step.")

    args = parser.parse_args()

    if (args.seed is None and
        ((args.gpus is not None and args.gpus > 1) or
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    if (str(args.precision) == "16" and args.deepspeed_config_path is not None):
        raise ValueError("DeepSpeed and FP16 training are not compatible")

    if (args.resume_from_jax_params is not None and args.resume_from_ckpt is not None):
        raise ValueError(
            "Choose between loading pretrained Jax-weights and a checkpoint-path")

    # Validate block replacement arguments
    if args.replace_block_index is not None:
        if args.replace_block_index <= 0:
            raise ValueError("replace_block_index must be greater than 0 (not first block)")
        if args.config_preset == "initial_training":
            # Default OpenFold has 48 blocks
            max_block_index = 47  # Not the last block (47)
        else:
            max_block_index = 47  # Conservative estimate
        if args.replace_block_index >= max_block_index:
            raise ValueError(f"replace_block_index must be less than {max_block_index} (not last block)")
        
        rank_zero_info(f"Will replace evoformer block {args.replace_block_index} with simple architecture")
        if args.replacement_hidden_dim:
            rank_zero_info(f"Using replacement hidden dimension: {args.replacement_hidden_dim}")

    main(args)

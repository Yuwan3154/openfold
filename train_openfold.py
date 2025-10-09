import argparse
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
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
)
from openfold.utils.import_weights import (
    import_jax_weights_,
    import_openfold_weights_
)
from openfold.utils.logger import PerformanceLoggingCallback
from block_replacement_scripts.custom_evoformer_replacement import (
    replace_evoformer_block, 
    freeze_all_except_replaced_block
)

# Import AdaptiveOpenFoldWrapper for adaptive training
try:
    from block_replacement_scripts.custom_openfold_wrapper import AdaptiveOpenFoldWrapper
    ADAPTIVE_WRAPPER_AVAILABLE = True
except ImportError:
    ADAPTIVE_WRAPPER_AVAILABLE = False


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config, replace_block_index=None, replacement_hidden_dim=None, learning_rate=1e-3):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        self.model = AlphaFold(config)
        self.is_multimer = self.config.globals.is_multimer
        self.replace_block_index = replace_block_index
        self.replacement_hidden_dim = replacement_hidden_dim
        self.learning_rate = learning_rate

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

    def training_step(self, batch, batch_idx):
        if (self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)

        ground_truth = batch.pop('gt_features', None)

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

        # Log it
        self._log(loss_breakdown, batch, outputs)

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

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
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score

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
            last_epoch=self.last_lr_step
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
    return torch.load(state_file)

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
        # Disable templates entirely for single sequence mode
        # (Use finetuning_no_templ_ptm preset with .pt weights instead of JAX .npz)
        config.model.template.enabled = False
        config.data.common.use_templates = False
        config.data.common.use_template_torsion_angles = False
        # Disable MSA-specific losses for single sequence training
        rank_zero_info("Disabling masked_msa loss for single sequence mode")
        config.loss.masked_msa.weight = 0.0
        # Reduce some computational requirements
        config.data.train.crop_size = min(config.data.train.crop_size, 256)

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
        if args.resume_model_weights_only:
            # Load the checkpoint
            if os.path.isdir(args.resume_from_ckpt):
                sd = zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
                    args.resume_from_ckpt)
            else:
                sd = torch.load(args.resume_from_ckpt)
            # Process the state dict
            # Use strict=False if we're doing block replacement or single sequence mode (model structure changed)
            strict_loading = not (
                (hasattr(args, 'replace_block_index') and args.replace_block_index is not None) or
                (hasattr(args, 'enable_single_seq_mode') and args.enable_single_seq_mode)
            )
            if not strict_loading:
                if hasattr(args, 'replace_block_index') and args.replace_block_index is not None:
                    rank_zero_info(f"Using strict=False for weight loading due to block replacement at index {args.replace_block_index}")
                elif hasattr(args, 'enable_single_seq_mode') and args.enable_single_seq_mode:
                    rank_zero_info(f"Using strict=False for weight loading due to single sequence mode (templates disabled)")
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
                sd = torch.load(args.resume_from_ckpt)
            last_global_step = int(sd['global_step'])
            model_module.resume_last_lr_step(last_global_step)
            logging.info("Successfully loaded last lr step...")

    # Handle JAX weight loading with template workaround for single sequence mode
    if args.resume_from_jax_params:
        if args.enable_single_seq_mode:
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

    callbacks = []
    
    # Enhanced checkpoint saving configuration
    checkpoint_config = {}
    if hasattr(args, 'checkpoint_save_top_k') and args.checkpoint_save_top_k is not None:
        checkpoint_config['save_top_k'] = args.checkpoint_save_top_k
    else:
        checkpoint_config['save_top_k'] = -1 if args.checkpoint_every_epoch else 1
    
    if hasattr(args, 'checkpoint_monitor') and args.checkpoint_monitor:
        checkpoint_config['monitor'] = args.checkpoint_monitor
    elif not args.checkpoint_every_epoch:
        # Use validation loss for best checkpoint when not saving every epoch
        checkpoint_config['monitor'] = 'val/loss' if hasattr(args, 'val_data_dir') and args.val_data_dir else 'train/loss'
        checkpoint_config['mode'] = 'min'
    
    if args.checkpoint_every_epoch:
        checkpoint_config['every_n_epochs'] = 1
        checkpoint_config['auto_insert_metric_name'] = False
    elif hasattr(args, 'checkpoint_every_n_steps') and args.checkpoint_every_n_steps:
        checkpoint_config['every_n_train_steps'] = args.checkpoint_every_n_steps
        checkpoint_config['auto_insert_metric_name'] = False
    elif hasattr(args, 'checkpoint_every_n_epochs') and args.checkpoint_every_n_epochs:
        checkpoint_config['every_n_epochs'] = args.checkpoint_every_n_epochs
        checkpoint_config['auto_insert_metric_name'] = False
    
    # Always create a checkpoint callback
    mc = ModelCheckpoint(**checkpoint_config)
    callbacks.append(mc)

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
    })
    trainer = pl.Trainer(**trainer_args)


    if (args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt

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

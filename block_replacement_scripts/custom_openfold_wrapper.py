#!/usr/bin/env python3
"""
Custom OpenFoldWrapper that supports adaptive weighting training.

This extends the standard OpenFoldWrapper to support:
1. Loading pre-trained replacement blocks
2. Creating adaptive weight predictors
3. Training both weight predictors and replacement blocks
4. Adding replace loss that penalizes mean adaptive weights
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import wandb
import numpy as np

# Import base OpenFoldWrapper
sys.path.append(str(Path(__file__).parent.parent))
from train_openfold import OpenFoldWrapper
from openfold.utils.loss import AlphaFoldLoss
from openfold.model.model import AlphaFold
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import protein
from openfold.np import residue_constants
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler

# Import adaptive training utilities
from .adaptive_wrapper import (
    setup_adaptive_training_model,
    compute_adaptive_replace_loss,
    freeze_model_except_adaptive_components
)


class AdaptiveOpenFoldWrapper(OpenFoldWrapper):
    """Extended OpenFoldWrapper with adaptive weighting training support"""
    
    def __init__(self, config, 
                 adaptive_config_path=None,
                 learning_rate=1e-3,
                 **kwargs):
        """
        Initialize wrapper with adaptive training support.
        
        Args:
            config: Model configuration
            adaptive_config_path: Path to adaptive training config
            learning_rate: Learning rate for training
            
        Note: Adaptive blocks are applied AFTER weight loading in on_fit_start()
        """
        
        # Call super().__init__ to create base model normally
        # This ensures JAX/checkpoint loading works correctly
        super().__init__(
            config=config,
            replace_block_index=None,  # No block replacement
            replacement_hidden_dim=None,
            learning_rate=learning_rate
        )
        
        # Store adaptive config path for later
        self.adaptive_config_path = adaptive_config_path
        
        # Adaptive training attributes (will be set in on_fit_start)
        self.weight_predictors = {}
        self.replace_loss_scaler = 0.0
        self.is_adaptive_training = False
        self.adaptive_setup_done = False
        self.optimizer_configured = False
        
        # Structure logging attributes
        self.log_structure_every_k_epoch = 1  # Default to logging every epoch
        self.train_sample_batch = None
        self.val_sample_batch = None
        self.disable_per_block_logging = False  # Option to disable per-block logging for speed
    
    def on_fit_start(self):
        """
        Called when fit begins. This is where we apply adaptive blocks.
        
        This ensures:
        1. Base model is created normally
        2. JAX/checkpoint weights are loaded correctly
        3. THEN adaptive blocks replace the original Evoformer blocks
        """

        if self.adaptive_config_path and not self.adaptive_setup_done:
            if Path(self.adaptive_config_path).exists():
                rank_zero_info("\n" + "=" * 80)
                rank_zero_info("Applying Adaptive Training Modifications (AFTER weight loading)")
                rank_zero_info("=" * 80)
                self._setup_adaptive_training()
                self.adaptive_setup_done = True
            else:
                rank_zero_info(f"Warning: Adaptive config not found: {self.adaptive_config_path}")
    
    def _setup_adaptive_training(self):
        """Setup adaptive training by loading pre-trained blocks and creating weight predictors"""
        
        # Note: trainer is not available during __init__, rank_zero_infos will be done on_fit_start
        rank_zero_info("\n=== Adaptive Training Mode ===")
        
        # Setup adaptive training
        self.model, training_info = setup_adaptive_training_model(
            model=self.model,
            config_path=Path(self.adaptive_config_path),
            model_config=self.config,
        )
        
        # Store training info
        self.weight_predictors = training_info['weight_predictors']
        self.replace_loss_scaler = training_info['replace_loss_scaler']
        self.log_structure_every_k_epoch = training_info.get('log_structure_every_k_epoch', 1)
        self.disable_per_block_logging = training_info.get('disable_per_block_logging', False)
        self.is_adaptive_training = True
        
        # Freeze all parameters except adaptive components
        trainable_params = freeze_model_except_adaptive_components(self.model)
        
        # CRITICAL: Reinitialize EMA after model structure changed
        # The EMA was created with the original model structure, but now we have adaptive blocks
        # with different parameter names, so we need to recreate it
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=self.config.ema.decay
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        rank_zero_info(f"Successfully set up adaptive training with {len(self.weight_predictors)} blocks")
        rank_zero_info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # CRITICAL: Reconfigure optimizer after adaptive blocks are applied
        self._reconfigure_optimizer()
    
    def _reconfigure_optimizer(self):
        """Reconfigure optimizer after adaptive blocks are applied"""

        if not hasattr(self, 'trainer') or self.trainer is None:
            return
            
        # Get the current optimizer from trainer
        if hasattr(self.trainer, 'optimizers') and self.trainer.optimizers:
            # Create new optimizer with correct parameters
            params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
            new_optimizer = torch.optim.Adam(
                params_to_optimize,
                lr=self.learning_rate,
                eps=1e-5
            )
            
            # Replace optimizer in trainer
            self.trainer.optimizers = [new_optimizer]
    
    def training_step(self, batch, batch_idx):
        """Override training step to ensure optimizer is reconfigured for adaptive training"""

        # Ensure optimizer is reconfigured if needed
        if self.is_adaptive_training and not self.optimizer_configured:
            self._reconfigure_optimizer()
            self.optimizer_configured = True
        
        # Call parent training step
        return super().training_step(batch, batch_idx)
    
    def _compute_adaptive_metrics(self, batch: Dict[str, torch.Tensor], log_per_block: bool = False) -> Dict[str, float]:
        """
        Compute adaptive training metrics for logging.
        
        Args:
            batch: Input batch
            log_per_block: If True, log weights for each block (expensive). 
                          If False, only log summary statistics (default).
        """
        
        metrics = {}
        
        if not self.is_adaptive_training:
            return metrics
        
        # Collect weights from adaptive blocks (computed during forward pass)
        block_weights = {}
        for block_idx, adaptive_block in self._get_adaptive_blocks():
            if hasattr(adaptive_block, '_predicted_weights') and block_idx in adaptive_block._predicted_weights:
                weight = adaptive_block._predicted_weights[block_idx].mean().item()
                block_weights[block_idx] = weight
                
                # Only log per-block weights if explicitly requested (expensive for 46 blocks)
                if log_per_block:
                    metrics[f"adaptive_weight_block_{block_idx:02d}"] = weight
        
        # Compute summary statistics efficiently
        if block_weights:
            weights_list = list(block_weights.values())
            metrics["mean_adaptive_weight"] = np.mean(weights_list)
            metrics["std_adaptive_weight"] = np.std(weights_list)
            metrics["min_adaptive_weight"] = np.min(weights_list)
            metrics["max_adaptive_weight"] = np.max(weights_list)
        
        return metrics
    
    def _get_adaptive_blocks(self):
        """Get all adaptive blocks from the model"""

        adaptive_blocks = []
        for block_idx, block in enumerate(self.model.evoformer.blocks):
            if hasattr(block, 'weight_predictor'):  # This is an AdaptiveEvoformerBlock
                adaptive_blocks.append((block_idx, block))
        return adaptive_blocks
    
    
    def on_train_epoch_end(self):
        """Log training structure sample at end of epoch if enabled"""

        # Only log on rank 0
        if (self.trainer.is_global_zero and 
            self.log_structure_every_k_epoch > 0 and 
            self.trainer.current_epoch % self.log_structure_every_k_epoch == 0 and
            self.train_sample_batch is not None):
            
            # Use EMA weights for structure prediction (consistent with validation)
            cached_weights_temp = tensor_tree_map(
                lambda t: t.detach().clone(), self.model.state_dict()
            )
            self.model.load_state_dict(self.ema.state_dict()["params"])
            
            # Run inference on stored batch to get structure
            with torch.no_grad():
                outputs = self(self.train_sample_batch)
            
            # Restore training weights
            self.model.load_state_dict(cached_weights_temp)
            
            # Convert to PDB and log to wandb
            pdb_string = self._convert_to_pdb(self.train_sample_batch, outputs)
            # Note: step parameter is ignored, kept for API compatibility
            self._log_structure_to_wandb(pdb_string, "train_structure", None)
    
    def on_validation_epoch_end(self):
        """Log validation structure sample at end of epoch if enabled"""

        # Structure logging happens BEFORE restoring weights (while still using EMA)
        if (self.trainer.is_global_zero and 
            self.log_structure_every_k_epoch > 0 and 
            self.trainer.current_epoch % self.log_structure_every_k_epoch == 0 and
            self.val_sample_batch is not None):
            
            # Run inference on stored batch to get structure (using EMA weights)
            with torch.no_grad():
                outputs = self(self.val_sample_batch)
            
            # Convert to PDB and log to wandb
            pdb_string = self._convert_to_pdb(self.val_sample_batch, outputs)
            # Note: step parameter is ignored, kept for API compatibility
            self._log_structure_to_wandb(pdb_string, "val_structure", None)
        
        # CRITICAL: Restore the model weights to normal (from parent class)
        if self.cached_weights is not None:
            self.model.load_state_dict(self.cached_weights)
            self.cached_weights = None
    
    def _convert_to_pdb(self, batch, outputs):
        """Convert model outputs to PDB string using OpenFold's protein module"""
        # CRITICAL: Process batch and outputs correctly (same as official OpenFold)
        # 1. Remove recycling dimension from batch: x[..., -1]
        # 2. Remove batch dimension: x[0] (since batch_size=1 in training)
        # 3. Outputs don't have recycling dimension, just remove batch dimension

        def process_features(x):
            if torch.is_tensor(x):
                x_np = np.array(x.cpu())
                # Handle different tensor shapes
                if x.ndim >= 2:
                    # Has both batch and recycling dimensions
                    return x_np[..., -1][0]
                elif x.ndim == 1:
                    # Only batch dimension (no recycling)
                    return x_np[0]
                else:
                    # Scalar
                    return x_np
            else:
                return x
        
        def process_outputs(x):
            if torch.is_tensor(x):
                x_np = np.array(x.cpu())
                # Remove batch dimension only (no recycling dimension in outputs)
                if x.ndim >= 1:
                    return x_np[0]
                else:
                    return x_np
            else:
                return x
        
        processed_feature_dict = tensor_tree_map(process_features, batch)
        out = tensor_tree_map(process_outputs, outputs)
        
        # Get pLDDT from outputs
        if "plddt" in out:
            plddt = out["plddt"]
        else:
            # Fallback to dummy plddt if not available
            seq_len = processed_feature_dict["aatype"].shape[0]
            plddt = np.full(seq_len, 90.0)
        
        # Create pLDDT b-factors (same as official OpenFold)
        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )
        
        # Create protein object using official OpenFold approach
        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=out,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=False,
            remark="Adaptive training prediction",
        )
        
        # Convert to PDB
        pdb_string = protein.to_pdb(unrelaxed_protein)
        
        return pdb_string
    
    def _log_structure_to_wandb(self, pdb_string, name, step):
        """Log structure to wandb using temporary file"""

        # Create temporary PDB file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_string)
            temp_pdb_path = f.name
        
        try:
            # Log to wandb WITHOUT step parameter - let wandb use its internal counter
            # This avoids conflicts with training/validation steps
            wandb.log({name: wandb.Molecule(temp_pdb_path)})
        finally:
            # Clean up temporary file
            os.unlink(temp_pdb_path)
    
    def training_step(self, batch, batch_idx):
        """Extended training step that includes adaptive training loss"""
        
        # Store a different batch each epoch for structure logging (only on rank 0)
        # Use a more robust mechanism: store batch from a position based on epoch
        if (self.trainer.is_global_zero and 
            self.log_structure_every_k_epoch > 0 and 
            batch_idx == 0):  # Always use first batch, but dataloader shuffles each epoch
            self.train_sample_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items() if v is not None}
        
        # Run standard training step
        if self.ema.device != batch["aatype"].device:
            self.ema.to(batch["aatype"].device)
        
        ground_truth = batch.pop('gt_features', None)
        
        # Run the model
        outputs = self(batch)
        
        # Remove the recycling dimension
        batch = {k: v[..., -1] if torch.is_tensor(v) else v for k, v in batch.items()}
        
        if self.is_multimer:
            batch = multi_chain_permutation_align(
                out=outputs,
                features=batch,
                ground_truth=ground_truth
            )
        
        # Compute main loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )
        
        # Add adaptive training loss if applicable
        if self.is_adaptive_training:
            # Compute raw replace loss (unscaled)
            raw_replace_loss = compute_adaptive_replace_loss(
                model=self.model,
                replace_loss_scaler=1.0,  # Use raw loss
                device=loss.device,
            )
            
            # Add scaled replace loss to main loss
            scaled_replace_loss = raw_replace_loss * self.replace_loss_scaler
            loss = loss + scaled_replace_loss
            
            # Store both raw and scaled losses in breakdown
            loss_breakdown["replace_loss"] = raw_replace_loss  # Raw loss for logging
            loss_breakdown["replace_loss_scaled"] = scaled_replace_loss  # Scaled loss
            
            # Compute and log adaptive metrics
            # Only compute adaptive metrics every 50 steps to reduce overhead significantly
            if batch_idx % 50 == 0:
                # Disable per-block logging if requested for speed
                log_per_block = not self.disable_per_block_logging and (batch_idx % 200 == 0)
                adaptive_metrics = self._compute_adaptive_metrics(batch, log_per_block=log_per_block)
                for metric_name, value in adaptive_metrics.items():
                    # Use sync_dist=True for distributed logging to avoid warnings
                    sync_dist = self.trainer.world_size > 1 if hasattr(self.trainer, 'world_size') else False
                    self.log(f"train/{metric_name}", value, 
                            on_step=True, on_epoch=True, 
                            prog_bar=(metric_name == "mean_adaptive_weight"),
                            sync_dist=sync_dist)
        
        # Log losses
        self._log(loss_breakdown, batch, outputs, train=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Extended validation step with adaptive metrics and replace loss"""

        # Store a different batch each epoch for structure logging (only on rank 0)
        # Use a more robust cycling mechanism that works with any number of validation batches
        if (self.trainer.is_global_zero and 
            self.log_structure_every_k_epoch > 0 and 
            batch_idx == 0):  # Always use first batch, but dataloader should shuffle each epoch
            self.val_sample_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items() if v is not None}
        
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
            batch = multi_chain_permutation_align(
                out=outputs,
                features=batch,
                ground_truth=ground_truth
            )

        # Compute loss and other metrics
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        # Add adaptive training loss if applicable (same as training)
        if self.is_adaptive_training:
            # Compute raw replace loss (unscaled)
            raw_replace_loss = compute_adaptive_replace_loss(
                model=self.model,
                replace_loss_scaler=1.0,  # Use raw loss
                device=loss.device,
            )
            
            # Add scaled replace loss to main loss
            scaled_replace_loss = raw_replace_loss * self.replace_loss_scaler
            loss = loss + scaled_replace_loss
            
            # Store both raw and scaled losses in breakdown
            loss_breakdown["replace_loss"] = raw_replace_loss  # Raw loss for logging
            loss_breakdown["replace_loss_scaled"] = scaled_replace_loss  # Scaled loss
            
            # Compute and log adaptive metrics
            # For validation, only compute on first batch to minimize overhead
            if batch_idx == 0:
                # Disable per-block logging if requested for speed
                log_per_block = not self.disable_per_block_logging
                adaptive_metrics = self._compute_adaptive_metrics(batch, log_per_block=log_per_block)
                for metric_name, value in adaptive_metrics.items():
                    # Use sync_dist=True for distributed logging to avoid warnings
                    sync_dist = self.trainer.world_size > 1 if hasattr(self.trainer, 'world_size') else False
                    self.log(f"val/{metric_name}", value,
                            on_step=False, on_epoch=True,
                            prog_bar=(metric_name == "mean_adaptive_weight"),
                            sync_dist=sync_dist)

        # Log losses (same as training)
        self._log(loss_breakdown, batch, outputs, train=False)
    
    def configure_optimizers(self, learning_rate: float = None, eps: float = 1e-5):
        """Configure optimizer with special handling for adaptive training"""
        
        # For adaptive training, defer optimizer configuration until after adaptive blocks are applied
        if self.adaptive_config_path and not self.adaptive_setup_done:
            # Return a placeholder optimizer that will be replaced later
            placeholder_params = [p for p in self.model.parameters() if p.requires_grad]
            return torch.optim.Adam(placeholder_params, lr=1e-3)
        
        # Use learning rate from args if provided
        if learning_rate is None:
            learning_rate = getattr(self, 'learning_rate', 1e-3)
        
        # For adaptive training, optimize both weight predictors and replacement blocks
        if self.is_adaptive_training:
            # Collect trainable parameters
            params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        else:
            # Standard optimization
            params_to_optimize = self.model.parameters()
        
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=learning_rate,
            eps=eps
        )
        
        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = learning_rate
        
        # Use a simpler scheduler for adaptive training
        if self.is_adaptive_training:
            # Use cosine annealing for adaptive training
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=100,  # Will be adjusted based on max_epochs
                eta_min=learning_rate * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "name": "CosineAnnealingLR",
                }
            }
        else:
            # Use standard AlphaFold scheduler
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

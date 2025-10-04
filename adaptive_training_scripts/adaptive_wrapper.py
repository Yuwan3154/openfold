#!/usr/bin/env python3
"""
Adaptive Training Wrapper - PyTorch Lightning module for adaptive training.

This wrapper extends the base training logic to support:
1. Adaptive weighting between original and replacement blocks
2. Replace loss computation
3. Structure logging to wandb
4. Adaptive metrics tracking
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import wandb
import numpy as np

# Add openfold to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openfold.utils.loss import AlphaFoldLoss
from openfold.model.model import AlphaFold
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import protein
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.validation_metrics import drmsd, gdt_ts, gdt_ha
from openfold.utils.loss import lddt_ca
from openfold.utils.superimposition import superimpose
from openfold.np import residue_constants

# Import adaptive model components
from adaptive_training_scripts.adaptive_model import (
    compute_adaptive_replace_loss,
    freeze_model_except_adaptive_components,
)
from block_replacement_scripts.adaptive_evoformer_blocks import AdaptiveWeightPredictor


class AdaptiveTrainingWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for adaptive training."""
    
    def __init__(
        self,
        model: AlphaFold,
        config: Any,
        training_info: Dict[str, Any],
        learning_rate: float = 1e-3,
        log_structure_every_k_epoch: int = 0,
    ):
        """
        Initialize the training wrapper.
        
        Args:
            model: AlphaFold model with adaptive blocks
            config: Model configuration
            training_info: Dictionary with adaptive training info
            learning_rate: Learning rate for training
            log_structure_every_k_epoch: Frequency of structure logging (0=disabled)
        """
        super().__init__()
        
        self.model = model
        self.config = config
        self.weight_predictors = training_info['weight_predictors']
        self.replace_loss_scaler = training_info['replace_loss_scaler']
        self.learning_rate = learning_rate
        self.log_structure_every_k_epoch = log_structure_every_k_epoch
        
        # Setup loss
        self.loss_fn = AlphaFoldLoss(config.loss)
        
        # Setup EMA
        self.ema = ExponentialMovingAverage(
            model=self.model,
            decay=config.ema.decay
        )
        
        # Structure logging
        self.train_sample_batch = None
        self.val_sample_batch = None
        
        # Validation weights caching
        self.cached_weights = None
        
        self.save_hyperparameters(ignore=['model', 'config'])
        
        # Freeze parameters (will print in on_fit_start)
        self.trainable_params, self.total_params = freeze_model_except_adaptive_components(self.model)
    
    def on_fit_start(self):
        """Called when fit begins. Print parameter info."""
        if self.trainer and self.trainer.is_global_zero:
            print(f"\nParameter freezing:")
            print(f"  Trainable: {self.trainable_params:,} / {self.total_params:,} ({self.trainable_params/self.total_params*100:.2f}%)")
            print(f"  Frozen: {self.total_params - self.trainable_params:,}")
    
    def forward(self, batch):
        """Forward pass through the model."""
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        """Training step with adaptive loss."""
        
        # Store first batch for structure logging
        if (self.log_structure_every_k_epoch > 0 and 
            self.train_sample_batch is None and 
            batch_idx == 0 and
            self.trainer.is_global_zero):
            self.train_sample_batch = {
                k: v.clone() if torch.is_tensor(v) else v 
                for k, v in batch.items() if v is not None
            }
        
        # Update EMA device if needed
        if self.ema.device != batch["aatype"].device:
            self.ema.to(batch["aatype"].device)
        
        # Run model
        ground_truth = batch.pop('gt_features', None)
        outputs = self(batch)
        
        # Remove recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1] if torch.is_tensor(t) else t, batch)
        
        # Compute main loss
        loss, loss_breakdown = self.loss_fn(outputs, batch, _return_breakdown=True)
        
        # Add replace loss
        replace_loss_raw = compute_adaptive_replace_loss(
            model=self.model,
            replace_loss_scaler=1.0,  # Raw loss
            device=loss.device,
        )
        replace_loss_scaled = replace_loss_raw * self.replace_loss_scaler
        loss = loss + replace_loss_scaled
        
        # Add to breakdown
        loss_breakdown["replace_loss"] = replace_loss_raw
        loss_breakdown["replace_loss_scaled"] = replace_loss_scaled
        
        # Log losses
        for loss_name, loss_value in loss_breakdown.items():
            self.log(f"train/{loss_name}", loss_value,
                    on_step=True, on_epoch=True,
                    prog_bar=(loss_name == "loss"),
                    sync_dist=False)
        
        # Log adaptive metrics
        adaptive_metrics = self._compute_adaptive_metrics(batch)
        for metric_name, value in adaptive_metrics.items():
            self.log(f"train/{metric_name}", value,
                    on_step=True, on_epoch=True,
                    prog_bar=(metric_name == "mean_adaptive_weight"),
                    sync_dist=False)
        
        # Log validation metrics
        val_metrics = self._compute_validation_metrics(batch, outputs, superimposition_metrics=False)
        for metric_name, value in val_metrics.items():
            self.log(f"train/{metric_name}", torch.mean(value),
                    on_step=False, on_epoch=True,
                    sync_dist=False)
        
        return loss
    
    def on_before_zero_grad(self, *args, **kwargs):
        """Update EMA before zeroing gradients."""
        self.ema.update(self.model)
    
    def validation_step(self, batch, batch_idx):
        """Validation step with adaptive metrics."""
        
        # Store first batch for structure logging
        if (self.log_structure_every_k_epoch > 0 and 
            self.val_sample_batch is None and 
            batch_idx == 0 and
            self.trainer.is_global_zero):
            self.val_sample_batch = {
                k: v.clone() if torch.is_tensor(v) else v 
                for k, v in batch.items() if v is not None
            }
        
        # Load EMA weights at start of validation
        if self.cached_weights is None:
            def clone_param(t): return t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])
        
        # Run model
        ground_truth = batch.pop('gt_features', None)
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1] if torch.is_tensor(t) else t, batch)
        batch["use_clamped_fape"] = 0.
        
        # Compute loss
        loss, loss_breakdown = self.loss_fn(outputs, batch, _return_breakdown=True)
        
        # Add replace loss
        replace_loss_raw = compute_adaptive_replace_loss(
            model=self.model,
            replace_loss_scaler=1.0,
            device=loss.device,
        )
        replace_loss_scaled = replace_loss_raw * self.replace_loss_scaler
        loss = loss + replace_loss_scaled
        
        loss_breakdown["replace_loss"] = replace_loss_raw
        loss_breakdown["replace_loss_scaled"] = replace_loss_scaled
        
        # Log losses
        for loss_name, loss_value in loss_breakdown.items():
            self.log(f"val/{loss_name}", loss_value,
                    on_step=False, on_epoch=True,
                    prog_bar=(loss_name == "loss"),
                    sync_dist=True)
        
        # Log adaptive metrics
        adaptive_metrics = self._compute_adaptive_metrics(batch)
        for metric_name, value in adaptive_metrics.items():
            self.log(f"val/{metric_name}", value,
                    on_step=False, on_epoch=True,
                    sync_dist=True)
        
        # Log validation metrics
        val_metrics = self._compute_validation_metrics(batch, outputs, superimposition_metrics=True)
        for metric_name, value in val_metrics.items():
            self.log(f"val/{metric_name}", torch.mean(value),
                    on_step=False, on_epoch=True,
                    sync_dist=True)
    
    def on_validation_epoch_end(self):
        """Restore model weights and log structure if needed."""
        # Restore weights
        if self.cached_weights is not None:
            self.model.load_state_dict(self.cached_weights)
            self.cached_weights = None
        
        # Log structure
        if (self.trainer.is_global_zero and
            self.log_structure_every_k_epoch > 0 and
            self.trainer.current_epoch % self.log_structure_every_k_epoch == 0 and
            self.val_sample_batch is not None):
            self._log_structure_sample("val", self.val_sample_batch)
    
    def on_train_epoch_end(self):
        """Log training structure if needed."""
        if (self.trainer.is_global_zero and
            self.log_structure_every_k_epoch > 0 and
            self.trainer.current_epoch % self.log_structure_every_k_epoch == 0 and
            self.train_sample_batch is not None):
            self._log_structure_sample("train", self.train_sample_batch)
    
    def _compute_adaptive_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute adaptive training metrics."""
        metrics = {}
        
        # Get MSA from batch
        msa = batch.get("msa", None)
        if msa is None or len(self.weight_predictors) == 0:
            return metrics
        
        with torch.no_grad():
            block_weights = {}
            for block_idx, predictor in self.weight_predictors.items():
                weight = predictor(msa).mean().item()
                block_weights[block_idx] = weight
            
            if block_weights:
                mean_weight = sum(block_weights.values()) / len(block_weights)
                metrics["mean_adaptive_weight"] = mean_weight
                metrics["adaptive_weight_std"] = torch.std(torch.tensor(list(block_weights.values()))).item()
                metrics["adaptive_weight_min"] = min(block_weights.values())
                metrics["adaptive_weight_max"] = max(block_weights.values())
        
        return metrics
    
    def _compute_validation_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        superimposition_metrics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Compute validation metrics (lddt_ca, drmsd, etc.)."""
        metrics = {}
        
        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]
        
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
            mask=all_atom_mask_ca,
        )
        metrics["drmsd_ca"] = drmsd_ca_score
        
        if superimposition_metrics:
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
    
    def _log_structure_sample(self, phase: str, sample_batch: Dict[str, torch.Tensor]):
        """Log structure sample to wandb."""
        try:
            with torch.no_grad():
                # Filter valid tensors
                valid_batch = {
                    k: v for k, v in sample_batch.items()
                    if v is not None and torch.is_tensor(v) and v.numel() > 0
                }
                
                if not valid_batch:
                    return
                
                # Run model
                outputs = self(valid_batch)
                
                # Convert to PDB and log
                pdb_string = self._convert_to_pdb(valid_batch, outputs)
                
                # Write to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                    f.write(pdb_string)
                    temp_path = f.name
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        f"{phase}_predicted_structure": wandb.Molecule(temp_path),
                        "epoch": self.trainer.current_epoch,
                    })
                
                # Cleanup
                os.unlink(temp_path)
                
        except Exception as e:
            if self.trainer.is_global_zero:
                print(f"Warning: Failed to log {phase} structure: {e}")
    
    def _convert_to_pdb(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor]
    ) -> str:
        """Convert predictions to PDB string using OpenFold's protein module."""
        
        # Convert to numpy
        processed_feature_dict = tensor_tree_map(
            lambda x: np.array(x[..., -1].cpu()) if torch.is_tensor(x) else x,
            batch
        )
        out = tensor_tree_map(
            lambda x: np.array(x.cpu()) if torch.is_tensor(x) else x,
            outputs
        )
        
        # Ensure aatype and residue_index are 1D
        if hasattr(processed_feature_dict.get("aatype"), 'ndim'):
            if processed_feature_dict["aatype"].ndim == 0:
                processed_feature_dict["aatype"] = np.array([processed_feature_dict["aatype"]])
        if hasattr(processed_feature_dict.get("residue_index"), 'ndim'):
            if processed_feature_dict["residue_index"].ndim == 0:
                processed_feature_dict["residue_index"] = np.array([processed_feature_dict["residue_index"]])
        
        # Create dummy plddt
        seq_len = processed_feature_dict["aatype"].shape[0]
        plddt = np.full(seq_len, 90.0)
        plddt_b_factors = np.repeat(plddt[..., None], 37, axis=-1)
        
        # Create protein object
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
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        
        # Collect trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Adam optimizer
        optimizer = torch.optim.Adam(
            params,
            lr=self.learning_rate,
            eps=1e-5
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "name": "CosineAnnealingLR",
            }
        }


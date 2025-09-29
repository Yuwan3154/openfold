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
from pathlib import Path
from typing import Optional, Dict, Any

# Import base OpenFoldWrapper
sys.path.append(str(Path(__file__).parent.parent))
from train_openfold import OpenFoldWrapper
from openfold.utils.loss import AlphaFoldLoss
from openfold.model.model import AlphaFold
from openfold.utils.exponential_moving_average import ExponentialMovingAverage

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
        """
        
        # Don't call super().__init__ yet - we need to handle model creation differently
        pl.LightningModule.__init__(self)
        
        self.config = config
        self.model = AlphaFold(config)
        self.is_multimer = self.config.globals.is_multimer
        self.adaptive_config_path = adaptive_config_path
        self.learning_rate = learning_rate
        
        # Adaptive training attributes
        self.weight_predictors = {}
        self.replace_loss_scaler = 0.0
        self.is_adaptive_training = False
        
        # Apply adaptive training if config provided
        if adaptive_config_path and Path(adaptive_config_path).exists():
            self._setup_adaptive_training()
        
        # Initialize loss and EMA
        self.loss = AlphaFoldLoss(config.loss)
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )
        
        self.cached_weights = None
        self.last_lr_step = -1
        self._is_distributed = None
        
        # Save hyperparameters
        self.save_hyperparameters()
    
    def _setup_adaptive_training(self):
        """Setup adaptive training by loading pre-trained blocks and creating weight predictors"""
        
        print("\n=== Adaptive Training Mode ===")
        
        # Setup adaptive training
        self.model, training_info = setup_adaptive_training_model(
            model=self.model,
            config_path=Path(self.adaptive_config_path),
            model_config=self.config,
        )
        
        # Store training info
        self.weight_predictors = training_info['weight_predictors']
        self.replace_loss_scaler = training_info['replace_loss_scaler']
        self.is_adaptive_training = True
        
        # Freeze all parameters except adaptive components
        freeze_model_except_adaptive_components(self.model)
    
    def _compute_adaptive_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute adaptive training metrics for logging"""
        
        metrics = {}
        
        if not self.is_adaptive_training:
            return metrics
        
        # Get MSA from batch - but skip if only msa_feat is available
        # 'msa' has the full MSA representation (c_m=256), 'msa_feat' is just features (49 dims)
        msa = None
        if "msa" in batch:
            msa = batch["msa"]
        elif "msa_feat" in batch:
            # Skip adaptive metrics computation if only msa_feat is available
            return metrics
        
        if msa is None:
            return metrics
        
        if self.weight_predictors:
            with torch.no_grad():
                # Collect weights for each block
                block_weights = {}
                for block_idx, predictor in self.weight_predictors.items():
                    # Get weight prediction for this batch
                    weight = predictor(msa).mean().item()
                    block_weights[block_idx] = weight
                    metrics[f"adaptive_weight_block_{block_idx}"] = weight
                
                # Compute mean weight
                if block_weights:
                    mean_weight = sum(block_weights.values()) / len(block_weights)
                    metrics["mean_adaptive_weight"] = mean_weight
                    
                    # Compute weight statistics
                    weights_list = list(block_weights.values())
                    metrics["adaptive_weight_std"] = torch.std(torch.tensor(weights_list)).item()
                    metrics["adaptive_weight_min"] = min(weights_list)
                    metrics["adaptive_weight_max"] = max(weights_list)
        
        return metrics
    
    def training_step(self, batch, batch_idx):
        """Extended training step that includes adaptive training loss"""
        
        # Run standard training step
        if self.ema.device != batch["aatype"].device:
            self.ema.to(batch["aatype"].device)
        
        ground_truth = batch.pop('gt_features', None)
        
        # Run the model
        outputs = self(batch)
        
        # Remove the recycling dimension
        batch = {k: v[..., -1] if torch.is_tensor(v) else v for k, v in batch.items()}
        
        if self.is_multimer:
            from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
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
            replace_loss = compute_adaptive_replace_loss(
                weight_predictors=self.weight_predictors,
                replace_loss_scaler=self.replace_loss_scaler,
                c_m=self.config.model.evoformer_stack.c_m,
                device=loss.device,
            )
            loss = loss + replace_loss
            loss_breakdown["replace_loss"] = replace_loss
            
            # Compute and log adaptive metrics
            adaptive_metrics = self._compute_adaptive_metrics(batch)
            for metric_name, value in adaptive_metrics.items():
                self.log(f"train/{metric_name}", value, 
                        on_step=True, on_epoch=True, 
                        prog_bar=(metric_name == "mean_adaptive_weight"))
        
        # Log losses
        self._log(loss_breakdown, batch, outputs, train=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Extended validation step with adaptive metrics"""
        
        # Run standard validation step
        super().validation_step(batch, batch_idx)
        
        # Log adaptive metrics if applicable
        if self.is_adaptive_training:
            adaptive_metrics = self._compute_adaptive_metrics(batch)
            for metric_name, value in adaptive_metrics.items():
                self.log(f"val/{metric_name}", value,
                        on_step=False, on_epoch=True,
                        prog_bar=(metric_name == "mean_adaptive_weight"))
    
    def configure_optimizers(self, learning_rate: float = None, eps: float = 1e-5):
        """Configure optimizer with special handling for adaptive training"""
        
        # Use learning rate from args if provided
        if learning_rate is None:
            learning_rate = getattr(self, 'learning_rate', 1e-3)
        
        # For adaptive training, optimize both weight predictors and replacement blocks
        if self.is_adaptive_training:
            # Collect trainable parameters
            params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
            
            print(f"Optimizing {len(params_to_optimize)} parameter groups for adaptive training")
            print(f"Learning rate: {learning_rate}")
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
            from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
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
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
        
        # Structure logging attributes
        self.log_structure_every_k_epoch = 1  # Default to logging every epoch
        self.train_sample_batch = None
        self.val_sample_batch = None
    
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
                print("\n" + "=" * 80)
                print("Applying Adaptive Training Modifications (AFTER weight loading)")
                print("=" * 80)
                self._setup_adaptive_training()
                self.adaptive_setup_done = True
            else:
                print(f"Warning: Adaptive config not found: {self.adaptive_config_path}")
    
    def _setup_adaptive_training(self):
        """Setup adaptive training by loading pre-trained blocks and creating weight predictors"""
        
        # Note: trainer is not available during __init__, prints will be done on_fit_start
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
        self.log_structure_every_k_epoch = training_info.get('log_structure_every_k_epoch', 1)
        self.is_adaptive_training = True
        
        # Freeze all parameters except adaptive components
        trainable_params = freeze_model_except_adaptive_components(self.model)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Successfully set up adaptive training with {len(self.weight_predictors)} blocks")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
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
    
    def _convert_to_protein(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> str:
        """Convert batch and outputs to a PDB string using BioPython (AFdiffusion approach)"""
        
        # Debug: Check for None values
        if "final_atom_positions" not in outputs or outputs["final_atom_positions"] is None:
            raise ValueError("final_atom_positions is missing or None in outputs")
        if "all_atom_mask" not in batch or batch["all_atom_mask"] is None:
            raise ValueError("all_atom_mask is missing or None in batch")
        if "aatype" not in batch or batch["aatype"] is None:
            raise ValueError("aatype is missing or None in batch")
        
        # Import BioPython components
        from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
        import numpy as np
        
        # Extract atom positions and mask
        atom_positions = outputs["final_atom_positions"]  # [batch, N_res, 37, 3]
        atom_mask = batch["all_atom_mask"]  # [batch, N_res, 37]
        aatype = batch["aatype"]  # [batch, N_res]
        
        # Take first sample from batch
        atom_positions = atom_positions[0]  # [N_res, 37, 3]
        atom_mask = atom_mask[0]  # [N_res, 37]
        aatype = aatype[0]  # [N_res]
        
        # Convert to numpy
        atom_positions = atom_positions.detach().cpu().numpy()
        atom_mask = atom_mask.detach().cpu().numpy()
        aatype = aatype.detach().cpu().numpy()
        
        # Extract CA atoms only (like AFdiffusion does)
        ca_index = 1  # CA is at index 1 in the 37 atoms
        ca_positions = atom_positions[:, ca_index, :]  # [N_res, 3]
        
        # Create BioPython structure
        structure = Structure.Structure("predicted_structure")
        model = Model.Model(0)
        chain = Chain.Chain('A')
        
        for i, ca_coord in enumerate(ca_positions):
            res_id = (' ', i, ' ')
            residue = Residue.Residue(res_id, 'GLY', '')  # Using GLY as placeholder
            
            # Convert to numpy and flatten
            coord_np = ca_coord.flatten()
            
            if coord_np.shape[0] != 3:
                raise ValueError(f"Coordinate shape mismatch: expected (3,), got {coord_np.shape}")
            
            atom_CA = Atom.Atom('CA', coord_np, 1.0, 1.0, ' ', 'CA', i, 'C')
            residue.add(atom_CA)
            chain.add(residue)
        
        model.add(chain)
        structure.add(model)
        
        # Convert to PDB string
        io = PDBIO()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            io.set_structure(structure)
            io.save(f)
            temp_pdb_path = f.name
        
        # Read the PDB string
        with open(temp_pdb_path, 'r') as f:
            pdb_string = f.read()
        
        # Clean up
        os.unlink(temp_pdb_path)
        
        return pdb_string
    
    def _log_structure_to_wandb(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], 
                               phase: str, epoch: int):
        """Log predicted structure to wandb as PDB file"""
        
        try:
            # Convert to PDB string directly
            pdb_string = self._convert_to_protein(batch, outputs)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                f.write(pdb_string)
                temp_pdb_path = f.name
            
            # Log to wandb using wandb.Molecule (like AFdiffusion)
            if wandb.run is not None:
                wandb.log({
                    f"{phase}_predicted_structure": wandb.Molecule(temp_pdb_path)
                })
            
            # Clean up temporary file
            os.unlink(temp_pdb_path)
            
        except Exception as e:
            print(f"Warning: Failed to log {phase} structure to wandb: {e}")
    
    def on_train_epoch_end(self):
        """Log training structure sample at end of epoch if enabled"""
        
        # Only log on rank 0
        if (self.trainer.is_global_zero and 
            self.log_structure_every_k_epoch > 0 and 
            self.trainer.current_epoch % self.log_structure_every_k_epoch == 0 and
            self.train_sample_batch is not None):
            
            # Get sample outputs for the stored batch
            with torch.no_grad():
                # Filter out None values from the batch
                filtered_batch = {k: v for k, v in self.train_sample_batch.items() if v is not None}
                if filtered_batch:  # Only proceed if we have valid data
                    # Additional check: ensure all values are tensors
                    valid_batch = {}
                    for k, v in filtered_batch.items():
                        if torch.is_tensor(v) and v.numel() > 0:
                            valid_batch[k] = v
                    
                    if valid_batch:  # Only proceed if we have valid tensors
                        outputs = self(valid_batch)
                        self._log_structure_to_wandb(
                            valid_batch, outputs, "train", self.trainer.current_epoch
                        )
    
    def on_validation_epoch_end(self):
        """Log validation structure sample at end of epoch if enabled"""
        
        # Only log on rank 0
        if (self.trainer.is_global_zero and 
            self.log_structure_every_k_epoch > 0 and 
            self.trainer.current_epoch % self.log_structure_every_k_epoch == 0 and
            self.val_sample_batch is not None):
            
            # Get sample outputs for the stored batch
            with torch.no_grad():
                # Filter out None values from the batch
                filtered_batch = {k: v for k, v in self.val_sample_batch.items() if v is not None}
                if filtered_batch:  # Only proceed if we have valid data
                    # Additional check: ensure all values are tensors
                    valid_batch = {}
                    for k, v in filtered_batch.items():
                        if torch.is_tensor(v) and v.numel() > 0:
                            valid_batch[k] = v
                    
                    if valid_batch:  # Only proceed if we have valid tensors
                        outputs = self(valid_batch)
                        self._log_structure_to_wandb(
                            valid_batch, outputs, "val", self.trainer.current_epoch
                        )
    
    def training_step(self, batch, batch_idx):
        """Extended training step that includes adaptive training loss"""
        
        # Store first batch for structure logging (only on rank 0)
        if (self.trainer.is_global_zero and 
            self.log_structure_every_k_epoch > 0 and 
            self.train_sample_batch is None and 
            batch_idx == 0):
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
            adaptive_metrics = self._compute_adaptive_metrics(batch)
            for metric_name, value in adaptive_metrics.items():
                self.log(f"train/{metric_name}", value, 
                        on_step=True, on_epoch=True, 
                        prog_bar=(metric_name == "mean_adaptive_weight"))
        
        # Log losses
        self._log(loss_breakdown, batch, outputs, train=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Extended validation step with adaptive metrics and replace loss"""
        
        # Store first batch for structure logging (only on rank 0)
        if (self.trainer.is_global_zero and 
            self.log_structure_every_k_epoch > 0 and 
            self.val_sample_batch is None and 
            batch_idx == 0):
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
            from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
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
            adaptive_metrics = self._compute_adaptive_metrics(batch)
            for metric_name, value in adaptive_metrics.items():
                self.log(f"val/{metric_name}", value,
                        on_step=False, on_epoch=True,
                        prog_bar=(metric_name == "mean_adaptive_weight"))

        # Log losses (same as training)
        self._log(loss_breakdown, batch, outputs, train=False)
    
    def configure_optimizers(self, learning_rate: float = None, eps: float = 1e-5):
        """Configure optimizer with special handling for adaptive training"""
        
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
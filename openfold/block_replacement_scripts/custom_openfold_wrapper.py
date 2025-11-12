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
from openfold.block_replacement_scripts.adaptive_wrapper import (
    setup_adaptive_training_model,
    compute_adaptive_replace_loss,
    freeze_model_except_adaptive_components
)


class AdaptiveOpenFoldWrapper(OpenFoldWrapper):
    """Extended OpenFoldWrapper with adaptive weighting training support"""
    
    def __init__(self, config,
                 adaptive_config_path=None,
                 learning_rate=1e-3,
                 data_loading_strategy='preload_gpu',
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
        self.train_sample_chain_id = None
        self.val_sample_chain_id = None
        self.disable_per_block_logging = False  # Option to disable per-block logging for speed

        # Data loading strategy
        self.data_loading_strategy = data_loading_strategy
    
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
    
    def _log_structure_to_wandb(self, pdb_string, name, step, ground_truth_pdb_path=None, chain_id=None):
        """Log structure to wandb using temporary file"""

        # Create temporary PDB file for prediction
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_string)
            temp_pdb_path = f.name

        try:
            # Log predicted structure to wandb
            wandb.log({name: wandb.Molecule(temp_pdb_path)})

            # Also log ground truth structure if available
            if ground_truth_pdb_path and os.path.exists(ground_truth_pdb_path):
                # Extract C-alpha only atoms from ground truth
                ca_only_pdb = self._extrac_backbone_only_pdb(ground_truth_pdb_path, chain_id)
                
                if ca_only_pdb:
                    try:
                        # Create a unique name for ground truth
                        gt_name = name.replace("structure", "ground_truth")
                        wandb.log({gt_name: wandb.Molecule(ca_only_pdb)})
                    finally:
                        # Clean up temporary C-alpha file
                        if os.path.exists(ca_only_pdb):
                            os.unlink(ca_only_pdb)
        finally:
            # Clean up temporary prediction file
            os.unlink(temp_pdb_path)

    
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
            # Use crop_to_ground_truth=True to match ground truth residue range
            pdb_string = self._convert_to_pdb(self.train_sample_batch, outputs, crop_to_ground_truth=True)
            # Find ground truth PDB path and log both predicted and ground truth structures
            rank_zero_info(f"Train chain_id for structure logging: {self.train_sample_chain_id}")
            ground_truth_pdb_path = self._find_ground_truth_pdb(self.train_sample_chain_id) if self.train_sample_chain_id else None
            if ground_truth_pdb_path:
                rank_zero_info(f"Found ground truth PDB: {ground_truth_pdb_path}")
            else:
                rank_zero_info("No ground truth PDB found for training structure")
            # Note: step parameter is ignored, kept for API compatibility
            self._log_structure_to_wandb(pdb_string, "train_structure", None, ground_truth_pdb_path, self.train_sample_chain_id)
    
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
            # Use crop_to_ground_truth=True to match ground truth residue range
            pdb_string = self._convert_to_pdb(self.val_sample_batch, outputs, crop_to_ground_truth=True)
            # Find ground truth PDB path and log both predicted and ground truth structures
            rank_zero_info(f"Val chain_id for structure logging: {self.val_sample_chain_id}")
            ground_truth_pdb_path = self._find_ground_truth_pdb(self.val_sample_chain_id) if self.val_sample_chain_id else None
            if ground_truth_pdb_path:
                rank_zero_info(f"Found ground truth PDB: {ground_truth_pdb_path}")
            else:
                rank_zero_info("No ground truth PDB found for validation structure")
            # Note: step parameter is ignored, kept for API compatibility
            self._log_structure_to_wandb(pdb_string, "val_structure", None, ground_truth_pdb_path, self.val_sample_chain_id)

        # CRITICAL: Restore the model weights to normal (from parent class)
        if self.cached_weights is not None:
            self.model.load_state_dict(self.cached_weights)
            self.cached_weights = None

    def train_dataloader(self):
        """Override train dataloader to support different loading strategies"""
        # For now, return None to use the default OpenFold dataloader
        return None

    def val_dataloader(self):
        """Override val dataloader to support different loading strategies"""
        # For now, return None to use the default OpenFold dataloader
        return None

    def _convert_to_pdb(self, batch, outputs, crop_to_ground_truth=False):
        """Convert model outputs to PDB string using OpenFold's protein module
        
        Args:
            batch: Input batch
            outputs: Model outputs
            crop_to_ground_truth: If True, crop predicted structure to match ground truth mask
        """
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
        
        # If crop_to_ground_truth is True, mask out predicted residues that don't have ground truth
        if crop_to_ground_truth and 'all_atom_mask' in processed_feature_dict:
            # The all_atom_mask tells us which residues have resolved coordinates in the ground truth
            # If a residue has no atoms with mask=1, it's unresolved in the crystal structure
            gt_residue_mask = np.any(processed_feature_dict['all_atom_mask'] > 0, axis=-1)
            
            # Apply mask to outputs - zero out predictions for unresolved residues
            # This makes them appear as missing in the PDB
            if 'final_atom_positions' in out:
                out['final_atom_positions'] = out['final_atom_positions'] * gt_residue_mask[:, None, None]
                # Also mask the atom positions mask
                if 'atom14_atom_exists' in out:
                    out['atom14_atom_exists'] = out['atom14_atom_exists'] * gt_residue_mask[:, None]
        
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
    
    def _get_chain_id_from_batch_idx(self, batch_idx, split='train'):
        """Get chain ID from dataloader for a given batch index"""
        try:
            if not hasattr(self, 'trainer') or not self.trainer or not hasattr(self.trainer, 'datamodule'):
                return None
            
            datamodule = self.trainer.datamodule
            
            # Get the appropriate dataset
            if split == 'train':
                if hasattr(datamodule, 'train_dataset'):
                    dataset = datamodule.train_dataset
                else:
                    return None
            else:  # validation
                # Try different attribute names
                if hasattr(datamodule, 'valid_dataset'):
                    dataset = datamodule.valid_dataset
                elif hasattr(datamodule, 'val_dataset'):
                    dataset = datamodule.val_dataset
                elif hasattr(datamodule, 'eval_dataset'):
                    dataset = datamodule.eval_dataset
                else:
                    return None
            
            # OpenFoldDataset is a wrapper that loops/samples from underlying datasets
            # We need to get the actual underlying dataset that has the chain IDs
            if hasattr(dataset, 'datasets') and hasattr(dataset, 'datapoints'):
                # This is the looped OpenFoldDataset wrapper
                # Get the actual dataset index and datapoint index
                if batch_idx < len(dataset.datapoints):
                    dataset_idx, datapoint_idx = dataset.datapoints[batch_idx]
                    actual_dataset = dataset.datasets[dataset_idx]
                    
                    # Now try to get chain ID from the actual dataset
                    if hasattr(actual_dataset, 'idx_to_chain_id'):
                        return actual_dataset.idx_to_chain_id(datapoint_idx)
                    elif hasattr(actual_dataset, '_chain_ids') and datapoint_idx < len(actual_dataset._chain_ids):
                        return actual_dataset._chain_ids[datapoint_idx]
            
            # Fallback: try direct access (for non-wrapped datasets)
            if hasattr(dataset, 'idx_to_chain_id'):
                return dataset.idx_to_chain_id(batch_idx)
            elif hasattr(dataset, '_chain_ids') and batch_idx < len(dataset._chain_ids):
                return dataset._chain_ids[batch_idx]
            
            return None
                
        except Exception as e:
            rank_zero_info(f"Warning: Could not get chain_id from batch_idx {batch_idx}: {e}")
            return None

    def _find_ground_truth_pdb(self, chain_id):
        """Find the ground truth PDB file path for a given chain_id"""
        if not chain_id:
            return None

        try:
            # Import here to avoid circular imports
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent))
            from enhanced_data_utils import EnhancedStructureFinder

            # Get the data directory from the trainer or use a default
            # Try to get it from the trainer's datamodule
            pdb_dir = None
            if hasattr(self, 'trainer') and self.trainer and hasattr(self.trainer, 'datamodule'):
                datamodule = self.trainer.datamodule
                if hasattr(datamodule, 'data_dir'):
                    pdb_dir = datamodule.data_dir
                elif hasattr(datamodule, 'train_data_dir'):
                    pdb_dir = datamodule.train_data_dir

            # Fallback to standard location if not found
            if not pdb_dir:
                pdb_dir = "/home/jupyter-chenxi/data/af2rank_single/pdb"

            # Create structure finder
            structure_finder = EnhancedStructureFinder(
                pdb_dir,
                [".cif", ".pdb", ".core"],
                None
            )

            # Find the structure path
            structure_path, file_id, chain_id_only, ext = structure_finder.find_structure_path(chain_id)

            return structure_path

        except Exception as e:
            rank_zero_info(f"Warning: Could not find ground truth PDB for chain_id {chain_id}: {e}")
            return None

    def _extrac_backbone_only_pdb(self, structure_path, chain_id):
        """
        Extract backbone atoms (N, CA, C, CB, O) from a structure file and save as temporary PDB.
        
        Args:
            structure_path: Path to the structure file (CIF, PDB, etc.)
            chain_id: Chain ID to extract (e.g., "1abc_A" -> extract chain "A")
        
        Returns:
            Path to temporary PDB file with backbone atoms only
        """
        try:
            # Parse chain ID to get the actual chain letter
            if '_' in chain_id:
                target_chain = chain_id.split('_')[-1]
            else:
                target_chain = None
            
            # Define backbone atoms to extract
            backbone_atoms = ['N', 'CA', 'C', 'CB', 'O']
            
            # Read structure file
            if structure_path.endswith('.cif') or structure_path.endswith('.core'):
                # Parse mmCIF file
                from openfold.data.mmcif_parsing import parse as parse_mmcif
                with open(structure_path, 'r') as f:
                    mmcif_string = f.read()
                parsing_result = parse_mmcif(file_id=os.path.basename(structure_path), mmcif_string=mmcif_string)
                
                # Get the Bio.PDB structure from the parsing result
                # ParsingResult.mmcif_object.structure is a Bio.PDB.Model object
                bio_structure = parsing_result.mmcif_object.structure
                
                # Extract backbone atoms for the target chain
                lines = []
                lines.append("HEADER    GROUND TRUTH STRUCTURE (BACKBONE ATOMS)")
                lines.append(f"TITLE     {chain_id}")
                
                atom_index = 1
                for chain in bio_structure.get_chains():
                    # Match chain ID (try both auth_asym_id and label_asym_id)
                    if target_chain and chain.id != target_chain:
                        continue
                    
                    for residue in chain.get_residues():
                        # Skip water molecules (HOH) and other hetero atoms
                        if residue.id[0] != ' ':  # Heteroatom flag - ' ' means standard residue
                            continue
                        if residue.resname in ['HOH', 'WAT']:  # Skip water molecules
                            continue
                        
                        for atom_name in backbone_atoms:
                            if atom_name in residue:
                                atom = residue[atom_name]
                                coord = atom.coord
                                res_name = residue.resname
                                res_num = residue.id[1]
                                
                                pdb_line = (
                                    f"ATOM  {atom_index:5d}  {atom_name:<3s} {res_name:3s} {chain.id}{res_num:4d}    "
                                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                                    f"  1.00 50.00           {atom_name[0]}"
                                )
                                lines.append(pdb_line)
                                atom_index += 1
                    
                    # If we found the target chain, stop
                    if target_chain:
                        break
                
                lines.append("END")
                
            else:
                # Parse PDB file
                lines = []
                lines.append("HEADER    GROUND TRUTH STRUCTURE (BACKBONE ATOMS)")
                lines.append(f"TITLE     {chain_id}")
                
                atom_index = 1
                with open(structure_path, 'r') as f:
                    for line in f:
                        if line.startswith('ATOM'):  # Only process ATOM, not HETATM
                            # Extract fields from PDB line
                            atom_name = line[12:16].strip()
                            res_name = line[17:20].strip()
                            chain = line[21].strip()
                            
                            # Skip water molecules
                            if res_name in ['HOH', 'WAT']:
                                continue
                            
                            # Check if this is a backbone atom and matches target chain
                            if atom_name in backbone_atoms:
                                if target_chain is None or chain == target_chain:
                                    # Extract coordinates and residue info
                                    res_num = line[22:26].strip()
                                    x = float(line[30:38])
                                    y = float(line[38:46])
                                    z = float(line[46:54])
                                    
                                    pdb_line = (
                                        f"ATOM  {atom_index:5d}  {atom_name:<3s} {res_name:3s} {chain}{res_num:>4s}    "
                                        f"{x:8.3f}{y:8.3f}{z:8.3f}"
                                        f"  1.00 50.00           {atom_name[0]}"
                                    )
                                    lines.append(pdb_line)
                                    atom_index += 1
                
                lines.append("END")
            
            # Write to temporary file
            temp_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
            temp_pdb.write('\n'.join(lines))
            temp_pdb.close()
            
            return temp_pdb.name
            
        except Exception as e:
            rank_zero_info(f"Warning: Could not extract backbone atoms from {structure_path}: {e}")
            return None

    def training_step(self, batch, batch_idx):
        """Extended training step that includes adaptive training loss"""
        
        # Store a different batch each epoch for structure logging (only on rank 0)
        # Use a more robust mechanism: store batch from a position based on epoch
        if (self.trainer.is_global_zero and
            self.log_structure_every_k_epoch > 0 and
            batch_idx == 0):  # Always use first batch, but dataloader shuffles each epoch
            self.train_sample_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items() if v is not None}
            # Get chain_id from datamodule using batch index
            self.train_sample_chain_id = self._get_chain_id_from_batch_idx(batch_idx, split='train')
        
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
        # Cycle through validation batches by using epoch number modulo
        # Since validation typically doesn't shuffle, we need to actively select different batches
        target_batch_idx = self.trainer.current_epoch % max(1, self.trainer.num_val_batches[0] if hasattr(self.trainer, 'num_val_batches') and self.trainer.num_val_batches else 10)
        
        if (self.trainer.is_global_zero and
            self.log_structure_every_k_epoch > 0 and
            batch_idx == target_batch_idx):
            self.val_sample_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items() if v is not None}
            # Get chain_id from datamodule using batch index
            self.val_sample_chain_id = self._get_chain_id_from_batch_idx(batch_idx, split='val')
        
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
        """Configure optimizer - uses AlphaFoldLRScheduler with warmup and decay"""
        
        # Use learning rate from args if provided
        if learning_rate is None:
            learning_rate = getattr(self, 'learning_rate', 1e-3)
        
        # Collect trainable parameters (works for both standard and adaptive training)
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=learning_rate,
            eps=eps
        )
        
        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = learning_rate
        
        # Use standard AlphaFold scheduler (has warmup + plateau + exponential decay)
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

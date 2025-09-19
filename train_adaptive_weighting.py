#!/usr/bin/env python3
"""
Adaptive weighting training script for OpenFold.

This script trains a version of OpenFold where each Evoformer block output is a weighted
combination of the original Evoformer block and a trained replacement block. The weights
are predicted dynamically based on the single representation.
"""

import argparse
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add openfold to path
sys.path.append(str(Path.home() / 'openfold'))

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.data import feature_pipeline, data_pipeline, templates
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.loss import AlphaFoldLoss
from openfold.utils.tensor_utils import tensor_tree_map

from custom_evoformer_replacement import SimpleEvoformerReplacement
from enhanced_data_utils import EnhancedStructureFinder
from evaluation_features_utils import extract_sequence_from_structure
from train_replacement_blocks import ReplacementBlockTrainer


class WeightPredictor(nn.Module):
    """Predicts mixing weight from single representation"""
    
    def __init__(self, c_m: int):
        super().__init__()
        self.linear = nn.Linear(c_m, 1)
        
        # Initialize to favor Evoformer block initially
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, 2.0)  # sigmoid(2) ≈ 0.88
    
    def forward(self, single_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            single_repr: [batch, n_res, c_m] single representation
        Returns:
            weight: [batch] scalar weight for Evoformer block
        """
        # Mean pool over residues: [batch, n_res, c_m] -> [batch, c_m]
        pooled = single_repr.mean(dim=1)
        
        # Project to scalar and apply sigmoid: [batch, c_m] -> [batch, 1] -> [batch]
        weight = torch.sigmoid(self.linear(pooled)).squeeze(-1)
        
        return weight


class AdaptiveEvoformerBlock(nn.Module):
    """Evoformer block with adaptive weighting between original and replacement"""
    
    def __init__(self, original_block, replacement_block, weight_predictor):
        super().__init__()
        self.original_block = original_block
        self.replacement_block = replacement_block
        self.weight_predictor = weight_predictor
    
    def forward(self, m, z, msa_mask, pair_mask, **kwargs):
        """
        Forward pass with adaptive weighting
        
        Args:
            m: MSA representation [batch, n_seq, n_res, c_m]
            z: Pair representation [batch, n_res, n_res, c_z]
            msa_mask: MSA mask
            pair_mask: Pair mask
            
        Returns:
            Tuple of (weighted_m, weighted_z, evoformer_weight)
        """
        
        # Run both blocks
        m_orig, z_orig = self.original_block(m, z, msa_mask, pair_mask, **kwargs)
        m_repl, z_repl = self.replacement_block(m, z, msa_mask, pair_mask, **kwargs)
        
        # Predict weight from single representation (first sequence of MSA)
        single_repr = m[:, 0, :, :]  # [batch, n_res, c_m]
        evo_weight = self.weight_predictor(single_repr)  # [batch]
        
        # Reshape weight for broadcasting
        # [batch] -> [batch, 1, 1, 1] for MSA, [batch, 1, 1, 1] for pair
        evo_weight_m = evo_weight.view(-1, 1, 1, 1)
        evo_weight_z = evo_weight.view(-1, 1, 1, 1)
        
        # Compute weighted combination
        m_weighted = evo_weight_m * m_orig + (1 - evo_weight_m) * m_repl
        z_weighted = evo_weight_z * z_orig + (1 - evo_weight_z) * z_repl
        
        return m_weighted, z_weighted, evo_weight


class AdaptiveOpenFold(nn.Module):
    """OpenFold model with adaptive weighting for all Evoformer blocks"""
    
    def __init__(self, base_model: AlphaFold, replacement_blocks: Dict[int, SimpleEvoformerReplacement], 
                 c_m: int):
        super().__init__()
        
        # Store base model components
        self.input_embedder = base_model.input_embedder
        self.recycling_embedder = base_model.recycling_embedder
        self.template_embedder = base_model.template_embedder
        self.extra_msa_embedder = base_model.extra_msa_embedder
        self.extra_msa_stack = base_model.extra_msa_stack
        self.structure_module = base_model.structure_module
        self.aux_heads = base_model.aux_heads
        self.config = base_model.config
        
        # Create adaptive Evoformer blocks
        self.adaptive_blocks = nn.ModuleList()
        self.weight_predictors = nn.ModuleList()
        
        for i, original_block in enumerate(base_model.evoformer.blocks):
            # Create weight predictor for this block
            weight_predictor = WeightPredictor(c_m)
            self.weight_predictors.append(weight_predictor)
            
            # Get replacement block if available, otherwise use identity
            if i in replacement_blocks:
                replacement_block = replacement_blocks[i]
            else:
                # Create identity replacement that just passes through
                replacement_block = IdentityBlock()
            
            # Create adaptive block
            adaptive_block = AdaptiveEvoformerBlock(
                original_block, replacement_block, weight_predictor
            )
            self.adaptive_blocks.append(adaptive_block)
        
        # Copy the linear layer from original evoformer
        self.evoformer_linear = base_model.evoformer.linear
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through adaptive OpenFold model"""
        
        # Most of this follows the original AlphaFold forward pass
        # but replaces the evoformer stack with adaptive blocks
        
        # Input embedding
        m, z = self.input_embedder(batch)
        
        # Recycling
        if self.config.model.recycle_features:
            m, z = self.recycling_embedder(m, z, batch)
        
        # Template embedding
        if self.config.model.template.enabled:
            z = self.template_embedder(z, batch)
        
        # Extra MSA stack
        if self.config.model.extra_msa.enabled:
            a = self.extra_msa_embedder(batch)
            z = self.extra_msa_stack(a, z, 
                                   msa_mask=batch.get("extra_msa_mask"),
                                   pair_mask=batch.get("seq_mask"),
                                   chunk_size=None)
        
        # Main Evoformer stack with adaptive weighting
        outputs = {}
        msa_mask = batch.get("msa_mask")
        pair_mask = batch.get("seq_mask")
        
        # Store all evoformer weights for replace_loss
        all_evo_weights = []
        
        # Pass through adaptive blocks
        for i, adaptive_block in enumerate(self.adaptive_blocks):
            m, z, evo_weight = adaptive_block(m, z, msa_mask, pair_mask)
            all_evo_weights.append(evo_weight)
        
        # Final linear layer
        s = self.evoformer_linear(m[..., 0, :, :])
        
        # Store weights for loss calculation
        outputs["evoformer_weights"] = torch.stack(all_evo_weights, dim=1)  # [batch, 48]
        
        # Structure module
        outputs.update(self.structure_module(
            evoformer_output_dict={"single": s, "pair": z},
            aatype=batch["aatype"],
            mask=batch["seq_mask"]
        ))
        
        # Auxiliary heads
        if self.aux_heads is not None:
            outputs.update(self.aux_heads(outputs))
        
        return outputs


class IdentityBlock(nn.Module):
    """Identity block that just passes inputs through unchanged"""
    
    def forward(self, m, z, msa_mask, pair_mask, **kwargs):
        return m, z


class ProteinDataset(Dataset):
    """Dataset for loading protein features"""
    
    def __init__(self, chain_list: List[str], pdb_dir: Path, data_processor, feature_processor):
        self.chain_list = chain_list
        self.pdb_dir = pdb_dir
        self.data_processor = data_processor
        self.feature_processor = feature_processor
        
    def __len__(self):
        return len(self.chain_list)
    
    def __getitem__(self, idx):
        chain_id = self.chain_list[idx]
        
        # Create features
        features = self._create_features(chain_id)
        
        # Process features
        processed_features = self.feature_processor.process_features(
            features, mode='train', is_multimer=False
        )
        
        return processed_features
    
    def _create_features(self, chain_id: str) -> Dict[str, np.ndarray]:
        """Create features for a protein"""
        structure_finder = EnhancedStructureFinder(
            str(self.pdb_dir),
            [".cif", ".pdb", ".core"],
            None
        )
        
        structure_path, file_id, chain_id_only, ext = structure_finder.find_structure_path(chain_id)
        sequence = extract_sequence_from_structure(structure_path, chain_id_only)
                
        tmp_fasta_path = os.path.join(os.getcwd(), f"tmp_{os.getpid()}_{chain_id}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{chain_id}\n{sequence}")
        
        # Create local alignment directory
        temp_alignment_dir = tempfile.mkdtemp()
        local_alignment_dir = os.path.join(temp_alignment_dir, chain_id)
        os.makedirs(local_alignment_dir, exist_ok=True)
        
        # Create minimal MSA file
        msa_path = os.path.join(local_alignment_dir, "output.a3m")
        with open(msa_path, 'w') as f:
            f.write(f">{chain_id}\n{sequence}\n")
        
        # Process features
        feature_dict = self.data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=local_alignment_dir,
            seqemb_mode=True
        )
        
        # Cleanup
        os.remove(tmp_fasta_path)
        shutil.rmtree(temp_alignment_dir)
        
        return feature_dict


class AdaptiveWeightingTrainer(pl.LightningModule):
    """PyTorch Lightning module for adaptive weighting training"""
    
    def __init__(self, adaptive_model: AdaptiveOpenFold, learning_rate: float = 1e-4, 
                 weight_decay: float = 1e-4, replace_loss_weight: float = 0.1):
        super().__init__()
        self.save_hyperparameters(ignore=['adaptive_model'])
        
        self.adaptive_model = adaptive_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.replace_loss_weight = replace_loss_weight
        
        # Main loss function
        self.alphafold_loss = AlphaFoldLoss(adaptive_model.config.loss)
        
    def forward(self, batch):
        return self.adaptive_model(batch)
    
    def training_step(self, batch, batch_idx):
        # Remove recycling for simplicity during training
        if isinstance(batch, dict):
            batch = {k: v[..., -1] if v.ndim > 1 and v.shape[-1] > 1 else v 
                    for k, v in batch.items()}
        
        # Forward pass
        outputs = self.forward(batch)
        
        # Main AlphaFold loss
        main_loss, loss_breakdown = self.alphafold_loss(
            outputs, batch, _return_breakdown=True
        )
        
        # Replace loss: encourage using replacement blocks
        evoformer_weights = outputs["evoformer_weights"]  # [batch, 48]
        replace_loss = evoformer_weights.mean()  # Higher weight = higher loss
        
        # Total loss
        total_loss = main_loss + self.replace_loss_weight * replace_loss
        
        # Log metrics
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/main_loss', main_loss)
        self.log('train/replace_loss', replace_loss)
        self.log('train/mean_evo_weight', evoformer_weights.mean())
        
        # Log individual loss components
        for loss_name, loss_value in loss_breakdown.items():
            self.log(f'train/{loss_name}', loss_value)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Remove recycling for simplicity during validation
        if isinstance(batch, dict):
            batch = {k: v[..., -1] if v.ndim > 1 and v.shape[-1] > 1 else v 
                    for k, v in batch.items()}
        
        # Forward pass
        outputs = self.forward(batch)
        
        # Main AlphaFold loss
        main_loss, loss_breakdown = self.alphafold_loss(
            outputs, batch, _return_breakdown=True
        )
        
        # Replace loss
        evoformer_weights = outputs["evoformer_weights"]
        replace_loss = evoformer_weights.mean()
        
        # Total loss
        total_loss = main_loss + self.replace_loss_weight * replace_loss
        
        # Log metrics
        self.log('val/total_loss', total_loss, prog_bar=True)
        self.log('val/main_loss', main_loss)
        self.log('val/replace_loss', replace_loss)
        self.log('val/mean_evo_weight', evoformer_weights.mean())
        
        # Log individual loss components
        for loss_name, loss_value in loss_breakdown.items():
            self.log(f'val/{loss_name}', loss_value)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=3,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
            }
        }


class AdaptiveWeightingPipeline:
    """Main training pipeline for adaptive weighting"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup paths
        self.home_dir = Path.home()
        self.csv_path = self.home_dir / args.csv_path
        self.pdb_dir = self.home_dir / args.pdb_dir
        self.weights_path = self.home_dir / args.weights_path
        self.trained_models_dir = self.home_dir / args.trained_models_dir
        self.output_dir = self.home_dir / args.output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup data pipeline
        self.data_processor, self.feature_processor = self._setup_data_pipeline()
        
        # Model dimensions
        self.c_m = 256
        self.c_z = 128
        
        print(f"Initialized adaptive weighting pipeline:")
        print(f"  CSV path: {self.csv_path}")
        print(f"  PDB directory: {self.pdb_dir}")
        print(f"  Weights: {self.weights_path}")
        print(f"  Trained models: {self.trained_models_dir}")
        print(f"  Output directory: {self.output_dir}")
        print()

    def _setup_data_pipeline(self):
        """Setup the data and feature processing pipelines"""
        config = model_config("model_2_ptm", train=True, low_prec=False)
        
        # Configure for single sequence mode
        config.data.common.max_extra_msa = 1
        config.data.common.max_msa_clusters = 1
        config.data.train.max_extra_msa = 1
        config.data.train.max_msa_clusters = 1
        config.data.common.use_templates = False
        config.data.common.use_template_torsion_angles = False
        config.data.train.crop_size = min(config.data.train.crop_size, 256)
        
        # Create dummy template directory
        temp_template_dir = tempfile.mkdtemp()
        dummy_cif_path = os.path.join(temp_template_dir, "dummy.cif")
        
        with open(dummy_cif_path, 'w') as f:
            f.write("""data_dummy
_entry.id dummy
_atom_site.group_PDB ATOM
_atom_site.id 1
_atom_site.type_symbol C
_atom_site.label_atom_id CA
_atom_site.label_alt_id .
_atom_site.label_comp_id ALA
_atom_site.label_asym_id A
_atom_site.label_entity_id 1
_atom_site.label_seq_id 1
_atom_site.pdbx_PDB_ins_code ?
_atom_site.Cartn_x 0.000
_atom_site.Cartn_y 0.000
_atom_site.Cartn_z 0.000
_atom_site.occupancy 1.00
_atom_site.B_iso_or_equiv 50.00
_atom_site.auth_seq_id 1
_atom_site.auth_comp_id ALA
_atom_site.auth_asym_id A
_atom_site.auth_atom_id CA
_atom_site.pdbx_PDB_model_num 1
""")
        
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=temp_template_dir,
            max_template_date="2025-01-01",
            max_hits=0,
            kalign_binary_path="/usr/bin/kalign",
            release_dates_path=None,
            obsolete_pdbs_path=None
        )
        
        self.temp_template_dir = temp_template_dir
        
        data_processor = data_pipeline.DataPipeline(template_featurizer=template_featurizer)
        feature_processor = feature_pipeline.FeaturePipeline(config.data)
        
        return data_processor, feature_processor

    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_template_dir'):
            try:
                shutil.rmtree(self.temp_template_dir)
            except:
                pass

    def _load_best_replacement_blocks(self) -> Dict[int, SimpleEvoformerReplacement]:
        """Load the best trained replacement blocks"""
        
        print("Loading best replacement blocks...")
        
        # Load training results to find best models
        results_path = self.trained_models_dir / "training_summary.csv"
        if not results_path.exists():
            print(f"  Training summary not found at {results_path}")
            return {}
        
        df = pd.read_csv(results_path)
        replacement_blocks = {}
        
        # Find best linear type per block
        for block_idx in df['block_idx'].unique():
            block_results = df[df['block_idx'] == block_idx]
            best_model = block_results.loc[block_results['best_val_loss'].idxmin()]
            
            best_linear_type = best_model['linear_type']
            
            # Load the replacement block
            checkpoint_path = self.trained_models_dir / f"block_{block_idx:02d}" / best_linear_type / "best_model.ckpt"
            
            if checkpoint_path.exists():
                try:
                    # Load PyTorch Lightning checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    
                    # Create replacement block
                    replacement_block = SimpleEvoformerReplacement(
                        c_m=self.c_m,
                        c_z=self.c_z,
                        m_hidden_dim=self.args.hidden_dim,
                        z_hidden_dim=self.args.hidden_dim,
                        linear_type=best_linear_type,
                        gating=True,
                        residual=True
                    )
                    
                    # Extract replacement block state dict
                    lightning_state_dict = checkpoint['state_dict']
                    replacement_state_dict = {}
                    
                    for key, value in lightning_state_dict.items():
                        if key.startswith('replacement_block.'):
                            new_key = key[len('replacement_block.'):]
                            replacement_state_dict[new_key] = value
                    
                    replacement_block.load_state_dict(replacement_state_dict)
                    replacement_blocks[block_idx] = replacement_block
                    
                    print(f"  Block {block_idx:02d}: {best_linear_type} (loss: {best_model['best_val_loss']:.6f})")
                    
                except Exception as e:
                    print(f"  Error loading block {block_idx}: {e}")
        
        print(f"Loaded {len(replacement_blocks)} replacement blocks")
        return replacement_blocks

    def _create_adaptive_model(self) -> AdaptiveOpenFold:
        """Create the adaptive OpenFold model"""
        
        print("Creating adaptive OpenFold model...")
        
        # Load base model
        config = model_config("model_2_ptm", train=True, low_prec=False)
        base_model = AlphaFold(config)
        
        # Load weights
        if self.weights_path.suffix == ".npz":
            model_basename = self.weights_path.stem
            model_version = "_".join(model_basename.split("_")[1:])
            import_jax_weights_(base_model, str(self.weights_path), version=model_version)
        else:
            checkpoint = torch.load(self.weights_path, map_location="cpu")
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
            
            base_model.load_state_dict(state_dict, strict=False)
        
        # Load replacement blocks
        replacement_blocks = self._load_best_replacement_blocks()
        
        # Create adaptive model
        adaptive_model = AdaptiveOpenFold(base_model, replacement_blocks, self.c_m)
        
        print(f"Created adaptive model with {len(replacement_blocks)} replacement blocks")
        return adaptive_model

    def _load_chain_list(self) -> Tuple[List[str], List[str]]:
        """Load chain list and create train/val split"""
        print("Loading chain list from CSV...")
        
        df = pd.read_csv(self.csv_path)
        all_chains = df['natives_rcsb'].dropna().tolist()
        
        # Filter by available structure files
        structure_finder = EnhancedStructureFinder(
            str(self.pdb_dir),
            [".cif", ".pdb", ".core"],
            None
        )
        
        available_chains = []
        for chain in all_chains:
            try:
                structure_finder.find_structure_path(chain)
                available_chains.append(chain)
            except ValueError:
                continue
        
        print(f"  Total chains in CSV: {len(all_chains)}")
        print(f"  Chains with available structures: {len(available_chains)}")
        
        # Limit if specified
        if self.args.max_proteins and self.args.max_proteins < len(available_chains):
            available_chains = available_chains[:self.args.max_proteins]
            print(f"  Limited to: {len(available_chains)} proteins")
        
        # Create 80/20 train/val split
        np.random.seed(42)
        indices = np.random.permutation(len(available_chains))
        split_idx = int(0.8 * len(available_chains))
        
        train_chains = [available_chains[i] for i in indices[:split_idx]]
        val_chains = [available_chains[i] for i in indices[split_idx:]]
        
        print(f"  Train set: {len(train_chains)} proteins")
        print(f"  Validation set: {len(val_chains)} proteins")
        
        return train_chains, val_chains

    def collate_fn(self, batch):
        """Custom collate function for variable-length sequences"""
        
        # Convert list of dicts to dict of lists
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = [item[key] for item in batch]
        
        # Convert to tensors and pad/stack as needed
        processed_batch = {}
        for key, values in batch_dict.items():
            if key == "aatype":
                # Pad sequences to same length
                max_len = max(v.shape[0] for v in values)
                padded = []
                for v in values:
                    if len(v.shape) == 1:
                        pad_len = max_len - v.shape[0]
                        padded_v = F.pad(v, (0, pad_len), value=0)
                    else:
                        pad_len = max_len - v.shape[0]
                        pad_shape = [0, 0] * (len(v.shape) - 1) + [0, pad_len]
                        padded_v = F.pad(v, pad_shape, value=0)
                    padded.append(padded_v)
                processed_batch[key] = torch.stack(padded)
            else:
                # Try to stack, pad if necessary
                try:
                    if isinstance(values[0], torch.Tensor):
                        if len(values[0].shape) > 0:
                            # Need to pad to consistent shape
                            max_shape = tuple(max(v.shape[i] for v in values) for i in range(len(values[0].shape)))
                            padded = []
                            for v in values:
                                pad_sizes = []
                                for i in range(len(v.shape) - 1, -1, -1):
                                    pad_size = max_shape[i] - v.shape[i]
                                    pad_sizes.extend([0, pad_size])
                                padded_v = F.pad(v, pad_sizes, value=0)
                                padded.append(padded_v)
                            processed_batch[key] = torch.stack(padded)
                        else:
                            processed_batch[key] = torch.stack(values)
                    else:
                        processed_batch[key] = torch.tensor(values)
                except Exception as e:
                    # Skip problematic keys
                    print(f"Skipping key {key}: {e}")
        
        return processed_batch

    def train(self):
        """Main training function"""
        
        print("=== Adaptive Weighting Training ===")
        print()
        
        # Create adaptive model
        adaptive_model = self._create_adaptive_model()
        
        # Load data
        train_chains, val_chains = self._load_chain_list()
        
        # Create datasets
        train_dataset = ProteinDataset(train_chains, self.pdb_dir, self.data_processor, self.feature_processor)
        val_dataset = ProteinDataset(val_chains, self.pdb_dir, self.data_processor, self.feature_processor)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")
        
        # Create trainer module
        trainer_module = AdaptiveWeightingTrainer(
            adaptive_model,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            replace_loss_weight=self.args.replace_loss_weight
        )
        
        # Create callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / "checkpoints",
            filename="adaptive_openfold_best",
            monitor="val/total_loss",
            mode="min",
            save_top_k=1,
            save_last=True
        )
        
        early_stopping = EarlyStopping(
            monitor="val/total_loss",
            mode="min",
            patience=5,
            verbose=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        # Create logger
        logger = TensorBoardLogger(
            save_dir=str(self.output_dir / "logs"),
            name="adaptive_weighting"
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            callbacks=[checkpoint_callback, early_stopping, lr_monitor],
            logger=logger,
            log_every_n_steps=5,
            val_check_interval=1.0,
            accelerator="auto",
            devices=1,
            precision="32-true",
            gradient_clip_val=1.0
        )
        
        # Train
        print("Starting training...")
        trainer.fit(trainer_module, train_loader, val_loader)
        
        print(f"\nTraining completed!")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train adaptive weighting for OpenFold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to CSV file containing chain list (relative to home directory)"
    )
    parser.add_argument(
        "--pdb_dir", type=str, required=True,
        help="Directory containing PDB/mmCIF files (relative to home directory)"
    )
    parser.add_argument(
        "--weights_path", type=str, required=True,
        help="Path to pretrained weights (relative to home directory)"
    )
    parser.add_argument(
        "--trained_models_dir", type=str, required=True,
        help="Directory containing trained replacement models (relative to home directory)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for adaptive model (relative to home directory)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256,
        help="Hidden dimension used in replacement blocks"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--replace_loss_weight", type=float, default=0.1,
        help="Weight for replace_loss term"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--max_proteins", type=int, default=None,
        help="Maximum number of proteins to use (None for all)"
    )
    
    args = parser.parse_args()
    
    # Run training
    pipeline = AdaptiveWeightingPipeline(args)
    pipeline.train()


if __name__ == "__main__":
    main()

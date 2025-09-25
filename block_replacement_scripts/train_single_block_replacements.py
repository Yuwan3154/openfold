#!/usr/bin/env python3
"""
Part 1: Single Block Replacement Training Script

This script trains individual replacement blocks for specific Evoformer block positions.
It tests different linear layer types (full, diagonal, affine) to find the best configuration.

Usage:
    python train_single_block_replacements.py \
        --data_dir path/to/block_data \
        --output_dir path/to/output \
        --blocks 1 2 3 \
        --linear_types full diagonal affine \
        --wandb --wandb_project my_project
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from typing import List, Any, Dict
import warnings
warnings.filterwarnings('ignore')

# Add openfold to path
sys.path.append(str(Path.home() / 'openfold'))

from custom_evoformer_replacement import SimpleEvoformerReplacement


class PaddingCollator:
    """
    Custom collator for handling variable-length protein sequences by padding.
    Based on dense_padding_data_loader.py from ProteinFoundation.
    """
    
    def __init__(self, float_padding_value: float = 1e-8, non_float_padding_value: int = -1):
        self.float_padding_value = float_padding_value
        self.non_float_padding_value = non_float_padding_value
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples with padding for variable-length sequences.
        
        Args:
            batch: List of dictionaries with keys: m_in, z_in, m_out, z_out, chain_id
        
        Returns:
            Dictionary with padded tensors and masks
        """
        if len(batch) == 1:
            # No padding needed for single sample
            return {
                'm_in': batch[0]['m_in'].unsqueeze(0),
                'z_in': batch[0]['z_in'].unsqueeze(0),
                'm_out': batch[0]['m_out'].unsqueeze(0),
                'z_out': batch[0]['z_out'].unsqueeze(0),
                'chain_ids': [batch[0]['chain_id']]
            }
        
        # Find maximum sequence length in batch
        max_n_res = max(sample['m_in'].shape[1] for sample in batch)
        batch_size = len(batch)
        
        # Get dimensions from first sample
        seq_len = batch[0]['m_in'].shape[0]  # MSA sequence length (usually 516)
        c_m = batch[0]['m_in'].shape[2]      # MSA channel dimension (usually 256)
        c_z = batch[0]['z_in'].shape[2]      # Pair channel dimension (usually 128)
        
        # Initialize padded tensors
        m_in_padded = torch.full(
            (batch_size, seq_len, max_n_res, c_m), 
            self.float_padding_value, 
            dtype=batch[0]['m_in'].dtype
        )
        z_in_padded = torch.full(
            (batch_size, max_n_res, max_n_res, c_z), 
            self.float_padding_value, 
            dtype=batch[0]['z_in'].dtype
        )
        m_out_padded = torch.full(
            (batch_size, seq_len, max_n_res, c_m), 
            self.float_padding_value, 
            dtype=batch[0]['m_out'].dtype
        )
        z_out_padded = torch.full(
            (batch_size, max_n_res, max_n_res, c_z), 
            self.float_padding_value, 
            dtype=batch[0]['z_out'].dtype
        )
        
        # Create masks for valid positions
        m_mask = torch.zeros((batch_size, seq_len, max_n_res, c_m), dtype=torch.bool)
        z_mask = torch.zeros((batch_size, max_n_res, max_n_res, c_z), dtype=torch.bool)
        
        chain_ids = []
        
        # Fill in actual data
        for i, sample in enumerate(batch):
            n_res = sample['m_in'].shape[1]
            
            # Copy data
            m_in_padded[i, :, :n_res, :] = sample['m_in']
            z_in_padded[i, :n_res, :n_res, :] = sample['z_in']
            m_out_padded[i, :, :n_res, :] = sample['m_out']
            z_out_padded[i, :n_res, :n_res, :] = sample['z_out']
            
            # Set masks for valid positions
            m_mask[i, :, :n_res, :] = True
            z_mask[i, :n_res, :n_res, :] = True
            
            chain_ids.append(sample['chain_id'])
        
        return {
            'm_in': m_in_padded,
            'z_in': z_in_padded,
            'm_out': m_out_padded,
            'z_out': z_out_padded,
            'm_mask': m_mask,
            'z_mask': z_mask,
            'chain_ids': chain_ids
        }


class BlockDataset(Dataset):
    """Dataset for loading block input/output pairs"""
    
    def __init__(self, data_dir: Path, block_idx: int, split: str):
        self.data_dir = data_dir / f"block_{block_idx:02d}" / split
        self.files = list(self.data_dir.glob("*.pt")) if self.data_dir.exists() else []
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = torch.load(file_path, map_location='cpu')
        
        # Extract input and output tensors
        m_in = data["input"]["m"]  # MSA representation (now single sequence)
        z_in = data["input"]["z"]  # Pair representation
        m_out = data["output"]["m"]  # Target MSA output (now single sequence)
        z_out = data["output"]["z"]  # Target pair output
        
        return {
            "m_in": m_in,
            "z_in": z_in, 
            "m_out": m_out,
            "z_out": z_out,
            "chain_id": data["chain_id"]
        }


class ReplacementBlockTrainer(pl.LightningModule):
    """PyTorch Lightning module for training replacement blocks"""
    
    def __init__(self, c_m: int, c_z: int, linear_type: str, 
                 m_hidden_dim: int, z_hidden_dim: int,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.c_m = c_m
        self.c_z = c_z
        self.linear_type = linear_type
        self.m_hidden_dim = m_hidden_dim
        self.z_hidden_dim = z_hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Create replacement block
        self.replacement_block = SimpleEvoformerReplacement(
            c_m=c_m,
            c_z=c_z, 
            m_hidden_dim=m_hidden_dim,
            z_hidden_dim=z_hidden_dim,
            linear_type=linear_type,
            gating=True,
            residual=True
        )
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
    def forward(self, m_in, z_in, batch_size=None):
        """Forward pass through replacement block"""
        
        # Create dummy masks (all ones)
        if batch_size is None:
            batch_size = m_in.shape[0]
        
        seq_len = m_in.shape[-2]
        res_len = z_in.shape[-2]
        
        msa_mask = torch.ones(batch_size, m_in.shape[-3], seq_len, device=m_in.device)
        pair_mask = torch.ones(batch_size, res_len, res_len, device=z_in.device)
        
        # Run replacement block
        m_pred, z_pred = self.replacement_block(
            m_in, z_in, msa_mask, pair_mask
        )
        
        return m_pred, z_pred
    
    def _compute_loss(self, m_pred, z_pred, m_target, z_target, m_mask=None, z_mask=None):
        """Compute MSE loss between predictions and targets with optional masking"""
        
        if m_mask is not None:
            # Apply mask to MSA tensors (only compute loss on valid positions)
            m_pred_masked = m_pred * m_mask
            m_target_masked = m_target * m_mask
            
            # Compute MSE loss only on valid (non-padded) positions
            m_loss_raw = (m_pred_masked - m_target_masked) ** 2
            m_loss = m_loss_raw.sum() / m_mask.sum()  # Average over valid positions
        else:
            # Standard MSE loss (for backward compatibility)
            m_loss = self.mse_loss(m_pred, m_target)
        
        if z_mask is not None:
            # Apply mask to pair tensors (only compute loss on valid positions)
            z_pred_masked = z_pred * z_mask
            z_target_masked = z_target * z_mask
            
            # Compute MSE loss only on valid (non-padded) positions
            z_loss_raw = (z_pred_masked - z_target_masked) ** 2
            z_loss = z_loss_raw.sum() / z_mask.sum()  # Average over valid positions
        else:
            # Standard MSE loss (for backward compatibility)
            z_loss = self.mse_loss(z_pred, z_target)
        
        # Combined loss
        total_loss = m_loss + z_loss
        
        return total_loss, m_loss, z_loss
    
    def training_step(self, batch, batch_idx):
        m_in = batch["m_in"]
        z_in = batch["z_in"]
        m_target = batch["m_out"]
        z_target = batch["z_out"]
        
        # Get masks for padded positions (if available)
        m_mask = batch.get("m_mask", None)
        z_mask = batch.get("z_mask", None)
        
        # Forward pass
        m_pred, z_pred = self.forward(m_in, z_in, batch_size=m_in.shape[0])
        
        # Compute loss with masking
        total_loss, m_loss, z_loss = self._compute_loss(
            m_pred, z_pred, m_target, z_target, m_mask, z_mask
        )
        
        # Log metrics
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/m_loss', m_loss)
        self.log('train/z_loss', z_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        m_in = batch["m_in"] 
        z_in = batch["z_in"]
        m_target = batch["m_out"]
        z_target = batch["z_out"]
        
        # Get masks for padded positions (if available)
        m_mask = batch.get("m_mask", None)
        z_mask = batch.get("z_mask", None)
        
        # Forward pass
        m_pred, z_pred = self.forward(m_in, z_in, batch_size=m_in.shape[0])
        
        # Compute loss with masking
        total_loss, m_loss, z_loss = self._compute_loss(
            m_pred, z_pred, m_target, z_target, m_mask, z_mask
        )
        
        # Log metrics
        self.log('val/total_loss', total_loss, prog_bar=True)
        self.log('val/m_loss', m_loss)
        self.log('val/z_loss', z_loss)
        
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
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
            }
        }


class SingleBlockTrainingPipeline:
    """Training pipeline for single block replacements"""
    
    def __init__(self, args):
        self.args = args
        self.home_dir = Path.home()
        # Check if data_dir already contains block_data subdirectory
        if (self.home_dir / args.data_dir / "block_data").exists():
            self.data_dir = self.home_dir / args.data_dir / "block_data"
        else:
            self.data_dir = self.home_dir / args.data_dir
        self.output_dir = self.home_dir / args.output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            print(f"Warning: Metadata file not found at {metadata_path}")
            self.metadata = {}
        
        # Model dimensions (from OpenFold model_2_ptm config)
        self.c_m = 256  # MSA representation dimension
        self.c_z = 128  # Pair representation dimension
        
        print(f"Initialized single block training pipeline:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Model dimensions: c_m={self.c_m}, c_z={self.c_z}")
        print(f"  Blocks to train: {args.blocks}")
        print(f"  Linear types: {args.linear_types}")
        print()

    def create_data_loaders(self, block_idx: int) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders for a specific block"""
        
        train_dataset = BlockDataset(self.data_dir, block_idx, "train")
        val_dataset = BlockDataset(self.data_dir, block_idx, "val")
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            return None, None
        
        # Create padding collator for variable-length sequences
        collator = PaddingCollator()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collator
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collator
        )
        
        return train_loader, val_loader

    def _check_training_completion(self, block_idx: int, linear_type: str) -> Optional[Dict]:
        """Check if training for this block and linear type is already completed"""
        
        checkpoint_dir = self.output_dir / f"block_{block_idx:02d}" / linear_type
        
        # Check for completion markers
        completion_files = [
            checkpoint_dir / "training_results.json",
            checkpoint_dir / "best_model.ckpt"
        ]
        
        all_exist = all(f.exists() for f in completion_files)
        
        if all_exist:
            print(f"  ✅ Training already completed for Block {block_idx:02d} with {linear_type} - skipping")
            
            # Load and return existing results
            try:
                with open(checkpoint_dir / "training_results.json", 'r') as f:
                    return json.load(f)
            except:
                return {}
        else:
            print(f"  🔄 Training needed for Block {block_idx:02d} with {linear_type}")
            return None

    def train_single_block(self, block_idx: int, linear_type: str) -> Dict[str, float]:
        """Train a replacement block for a specific block index and linear type"""
        
        print(f"\nTraining Block {block_idx:02d} with {linear_type} linear layers...")
        
        # Check if training is already completed
        existing_results = self._check_training_completion(block_idx, linear_type)
        if existing_results is not None:
            return existing_results
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(block_idx)
        
        if train_loader is None:
            print(f"  No data available for block {block_idx}, skipping...")
            return {}
        
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        
        # Set hidden dimensions based on linear type
        if linear_type in ["diagonal", "affine"] or self.args.hidden_dim is None:
            # These require hidden_dim == input_dim
            m_hidden_dim = self.c_m  # 256
            z_hidden_dim = self.c_z  # 128
        else:
            # Full linear can use any hidden dimension
            m_hidden_dim = self.args.hidden_dim
            z_hidden_dim = self.args.hidden_dim
        
        print(f"  Using hidden dimensions: m_hidden={m_hidden_dim}, z_hidden={z_hidden_dim}")
        
        # Create model
        model = ReplacementBlockTrainer(
            c_m=self.c_m,
            c_z=self.c_z,
            linear_type=linear_type,
            m_hidden_dim=m_hidden_dim,
            z_hidden_dim=z_hidden_dim,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        if self.args.compile:
            model.replacement_block = torch.compile(model.replacement_block)
        
        # Create callbacks
        checkpoint_dir = self.output_dir / f"block_{block_idx:02d}" / linear_type
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_model",
            monitor="val/total_loss",
            mode="min",
            save_top_k=1,
            save_last=False,  # Don't save last checkpoint during training
            every_n_epochs=1,  # Only save best model, not every epoch
            verbose=True
        )
        
        early_stopping = EarlyStopping(
            monitor="val/total_loss",
            mode="min",
            patience=10,
            verbose=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Create loggers
        loggers = []
        
        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=str(self.output_dir / "logs"),
            name=f"block_{block_idx:02d}_{linear_type}"
        )
        loggers.append(tb_logger)
        
        # Wandb logger (if enabled)
        if self.args.wandb:
            experiment_name = f"{self.args.experiment_name}_block_{block_idx:02d}_{linear_type}"
            wandb_logger = WandbLogger(
                project=self.args.wandb_project,
                name=experiment_name,
                save_dir=str(self.output_dir / "logs"),
                entity=self.args.wandb_entity
            )
            loggers.append(wandb_logger)
        
        # Create trainer with optimized settings
        trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            callbacks=[checkpoint_callback, early_stopping, lr_monitor],
            logger=loggers,
            log_every_n_steps=10,
            val_check_interval=1.0,
            accelerator="auto",
            devices=1,
            precision="16-mixed",  # Use mixed precision to save memory
            gradient_clip_val=1.0,  # Gradient clipping for stability
        )
        
        # Train
        trainer.fit(model, train_loader, val_loader)
        
        # Get best metrics
        best_val_loss = checkpoint_callback.best_model_score.item()
        
        # Save final results
        results = {
            "block_idx": block_idx,
            "linear_type": linear_type,
            "best_val_loss": best_val_loss,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "epochs_trained": trainer.current_epoch,
            "checkpoint_path": str(checkpoint_callback.best_model_path)
        }
        
        with open(checkpoint_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Training completed!")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"  Epochs trained: {trainer.current_epoch}")
        print(f"  Checkpoint saved: {checkpoint_callback.best_model_path}")
        
        return results

    def train_blocks(self):
        """Train replacement blocks for specified blocks and linear types"""
        
        print("=== Training Single Block Replacements ===")
        print()
        
        all_results = []
        
        # Train each combination
        for block_idx in self.args.blocks:
            for linear_type in self.args.linear_types:
                try:
                    results = self.train_single_block(block_idx, linear_type)
                    if results:
                        all_results.append(results)
                        
                except Exception as e:
                    print(f"Error training block {block_idx} with {linear_type}: {e}")
                    continue
        
        # Save overall results
        self._save_overall_results(all_results)
        
        print(f"\n=== Training Complete ===")
        print(f"Successfully trained {len(all_results)} models")
        print(f"Results saved to: {self.output_dir}")

    def _save_overall_results(self, all_results: List[Dict]):
        """Save overall training results and analysis"""
        
        # Save detailed results
        with open(self.output_dir / "all_training_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Check if CSV already exists and append if needed
            csv_path = self.output_dir / "training_summary.csv"
            if csv_path.exists():
                # Load existing data
                existing_df = pd.read_csv(csv_path)
                # Remove duplicate entries (same block_idx and linear_type)
                mask = ~((existing_df['block_idx'].isin(df['block_idx'])) & 
                        (existing_df['linear_type'].isin(df['linear_type'])))
                existing_df = existing_df[mask]
                # Combine and save
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(csv_path, index=False)
                print(f"📝 Appended results to existing CSV. Total entries: {len(combined_df)}")
            else:
                # Create new file
                df.to_csv(csv_path, index=False)
                print(f"📝 Created new CSV with {len(df)} entries")
            
            # Create analysis by linear type
            print("\n=== Training Summary ===")
            
            for linear_type in self.args.linear_types:
                type_results = df[df['linear_type'] == linear_type]
                if len(type_results) > 0:
                    mean_loss = type_results['best_val_loss'].mean()
                    std_loss = type_results['best_val_loss'].std()
                    print(f"{linear_type:>10s}: {mean_loss:.6f} ± {std_loss:.6f} "
                          f"({len(type_results)} blocks)")
            
            # Find best performing configurations
            print("\nBest performing models per block:")
            for block_idx in df['block_idx'].unique():
                block_results = df[df['block_idx'] == block_idx]
                best_model = block_results.loc[block_results['best_val_loss'].idxmin()]
                print(f"  Block {best_model['block_idx']:02d}: "
                      f"{best_model['linear_type']} "
                      f"(loss: {best_model['best_val_loss']:.6f})")
        
        print(f"\nDetailed results saved to:")
        print(f"  {self.output_dir / 'all_training_results.json'}")
        print(f"  {self.output_dir / 'training_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train single block replacements for Evoformer blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and output
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing collected block data (relative to home directory)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for trained models (relative to home directory)")
    
    # Training configuration
    parser.add_argument("--blocks", type=int, nargs="+", required=True,
                       help="Block indices to train (e.g., 1 2 3)")
    parser.add_argument("--linear_types", type=str, nargs="+", 
                       default=["full", "diagonal", "affine"],
                       choices=["full", "diagonal", "affine"],
                       help="Linear layer types to test")
    
    # Training parameters
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension for replacement blocks")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training (increased from 4 due to padding support)")
    parser.add_argument("--max_epochs", type=int, default=50,
                       help="Maximum number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--distributed_backend", type=str, default="gloo",
                       choices=["nccl", "gloo", "mpi"],
                       help="Distributed training backend (use 'gloo' for nodes without NCCL support)")
    parser.add_argument("--compile", action="store_true", default=False,
                       help="Compile the model using torch.compile")
    
    # Wandb logging arguments
    parser.add_argument("--wandb", action="store_true", default=False,
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="af2distill",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, 
                       default="kryst3154-massachusetts-institute-of-technology",
                       help="Wandb entity (username or team)")
    parser.add_argument("--experiment_name", type=str, default="single_block_replacement",
                       help="Base experiment name for wandb logging")
    
    args = parser.parse_args()
    
    # Run training
    pipeline = SingleBlockTrainingPipeline(args)
    pipeline.train_blocks()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training script for Evoformer replacement blocks.

This script trains 48 separate replacement blocks (one for each Evoformer block position)
using the collected input/output data. It compares 3 linear layer types: full, diagonal, affine.
"""

import argparse
import os
import sys
import json
import pickle
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
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
warnings.filterwarnings('ignore')

# Add openfold to path
sys.path.append(str(Path.home() / 'openfold'))

from custom_evoformer_replacement import SimpleEvoformerReplacement


class BlockDataset(Dataset):
    """Dataset for loading block input/output pairs"""
    
    def __init__(self, data_dir: Path, block_idx: int, split: str):
        self.data_dir = data_dir / f"block_{block_idx:02d}" / split
        self.files = list(self.data_dir.glob("*.pkl")) if self.data_dir.exists() else []
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            data = pickle.load(f)
        
        # Extract input and output tensors
        m_in = data["input"]["m"]  # MSA representation
        z_in = data["input"]["z"]  # Pair representation
        m_out = data["output"]["m"]  # Target MSA output
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
    
    def __init__(self, c_m: int, c_z: int, linear_type: str, hidden_dim: int, 
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.c_m = c_m
        self.c_z = c_z
        self.linear_type = linear_type
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Create replacement block
        self.replacement_block = SimpleEvoformerReplacement(
            c_m=c_m,
            c_z=c_z, 
            m_hidden_dim=hidden_dim,
            z_hidden_dim=hidden_dim,
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
    
    def _compute_loss(self, m_pred, z_pred, m_target, z_target):
        """Compute MSE loss between predictions and targets"""
        
        # MSA loss
        m_loss = self.mse_loss(m_pred, m_target)
        
        # Pair loss
        z_loss = self.mse_loss(z_pred, z_target)
        
        # Combined loss
        total_loss = m_loss + z_loss
        
        return total_loss, m_loss, z_loss
    
    def training_step(self, batch, batch_idx):
        m_in = batch["m_in"]
        z_in = batch["z_in"]
        m_target = batch["m_out"]
        z_target = batch["z_out"]
        
        # Forward pass
        m_pred, z_pred = self.forward(m_in, z_in, batch_size=m_in.shape[0])
        
        # Compute loss
        total_loss, m_loss, z_loss = self._compute_loss(m_pred, z_pred, m_target, z_target)
        
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
        
        # Forward pass
        m_pred, z_pred = self.forward(m_in, z_in, batch_size=m_in.shape[0])
        
        # Compute loss
        total_loss, m_loss, z_loss = self._compute_loss(m_pred, z_pred, m_target, z_target)
        
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


class ReplacementBlockTrainingPipeline:
    """Main training pipeline for replacement blocks"""
    
    def __init__(self, args):
        self.args = args
        self.home_dir = Path.home()
        self.data_dir = self.home_dir / args.data_dir / "block_data"
        self.output_dir = self.home_dir / args.output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load metadata
        with open(self.data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Model dimensions (from OpenFold model_2_ptm config)
        self.c_m = 256  # MSA representation dimension
        self.c_z = 128  # Pair representation dimension
        
        # Linear layer types to test
        self.linear_types = ["full", "diagonal", "affine"]
        
        print(f"Initialized training pipeline:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Model dimensions: c_m={self.c_m}, c_z={self.c_z}")
        print(f"  Linear types: {self.linear_types}")
        print()

    def create_data_loaders(self, block_idx: int) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders for a specific block"""
        
        train_dataset = BlockDataset(self.data_dir, block_idx, "train")
        val_dataset = BlockDataset(self.data_dir, block_idx, "val")
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            return None, None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader

    def train_single_block(self, block_idx: int, linear_type: str) -> Dict[str, float]:
        """Train a replacement block for a specific block index and linear type"""
        
        print(f"\nTraining Block {block_idx:02d} with {linear_type} linear layers...")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(block_idx)
        
        if train_loader is None:
            print(f"  No data available for block {block_idx}, skipping...")
            return {}
        
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        
        # Create model
        model = ReplacementBlockTrainer(
            c_m=self.c_m,
            c_z=self.c_z,
            linear_type=linear_type,
            hidden_dim=self.args.hidden_dim,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Create callbacks
        checkpoint_dir = self.output_dir / f"block_{block_idx:02d}" / linear_type
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_model",
            monitor="val/total_loss",
            mode="min",
            save_top_k=1,
            save_last=True
        )
        
        early_stopping = EarlyStopping(
            monitor="val/total_loss",
            mode="min",
            patience=10,
            verbose=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Create logger
        logger = TensorBoardLogger(
            save_dir=str(self.output_dir / "logs"),
            name=f"block_{block_idx:02d}_{linear_type}"
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            callbacks=[checkpoint_callback, early_stopping, lr_monitor],
            logger=logger,
            log_every_n_steps=10,
            val_check_interval=1.0,
            accelerator="auto",
            devices=1,
            precision="32-true"  # Use full precision for stability
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

    def train_all_blocks(self):
        """Train replacement blocks for all blocks and linear types"""
        
        print("=== Training Replacement Blocks for All Evoformer Blocks ===")
        print()
        
        all_results = []
        
        # Determine which blocks to train
        if self.args.test_blocks:
            block_indices = self.args.test_blocks
        else:
            block_indices = list(range(48))
        
        print(f"Training blocks: {block_indices}")
        print(f"Linear types: {self.linear_types}")
        print(f"Total combinations: {len(block_indices) * len(self.linear_types)}")
        print()
        
        # Train each combination
        for block_idx in block_indices:
            for linear_type in self.linear_types:
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
            df.to_csv(self.output_dir / "training_summary.csv", index=False)
            
            # Create analysis by linear type
            print("\n=== Training Summary ===")
            
            for linear_type in self.linear_types:
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
        description="Train replacement blocks for all Evoformer blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing collected block data (relative to home directory)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for trained models (relative to home directory)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256,
        help="Hidden dimension for replacement blocks"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--test_blocks", type=int, nargs="+", default=None,
        help="Specific block indices to train (for testing, default: all 48 blocks)"
    )
    
    args = parser.parse_args()
    
    # Run training
    pipeline = ReplacementBlockTrainingPipeline(args)
    pipeline.train_all_blocks()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Main entry point for adaptive training.

This script orchestrates the complete adaptive training pipeline:
1. Load configuration
2. Prepare data (chain lists, alignments)
3. Build adaptive model (load weights, replace blocks)
4. Setup PyTorch Lightning trainer
5. Run training
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add openfold to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# Import local modules
from adaptive_training_scripts.utils import (
    load_config,
    extract_chain_list_from_csv,
    split_chains_for_validation,
    create_minimal_alignment_structure,
    count_available_replacement_blocks,
    print_config_summary,
)
from adaptive_training_scripts.adaptive_model import AdaptiveModelBuilder
from adaptive_training_scripts.adaptive_wrapper import AdaptiveTrainingWrapper

# Import OpenFold data module
from openfold.data.data_modules import OpenFoldDataModule
from openfold.config import model_config


def setup_data(config: dict, home_dir: Path) -> tuple:
    """
    Setup data: extract chains, create splits, setup alignments.
    
    Returns:
        Tuple of (train_chains, val_chains, train_list_path, val_list_path, alignment_dir)
    """
    csv_path = home_dir / config['csv_path']
    pdb_dir = home_dir / config['pdb_dir']
    output_dir = home_dir / config['output_dir']
    
    print("\n" + "=" * 80)
    print("Data Preparation")
    print("=" * 80)
    
    # Extract chain list
    print("\n1. Extracting chain list from CSV...")
    chain_list_dir = output_dir / "chain_lists"
    chain_list_path = chain_list_dir / "all_chains.txt"
    all_chains = extract_chain_list_from_csv(str(csv_path), str(pdb_dir), str(chain_list_path))
    
    # Split for validation
    print("\n2. Splitting data for validation...")
    val_fraction = config.get('validation_fraction', 0.1)
    train_chains, val_chains, train_list_path, val_list_path = split_chains_for_validation(
        all_chains, val_fraction, output_dir, seed=config.get('seed', 42)
    )
    
    # Create minimal alignment structure
    print("\n3. Setting up minimal alignment structure...")
    alignment_dir = output_dir / "alignments"
    create_minimal_alignment_structure(alignment_dir, all_chains)
    
    print("\n" + "=" * 80 + "\n")
    
    return train_chains, val_chains, train_list_path, val_list_path, alignment_dir


def build_adaptive_model(config: dict, home_dir: Path):
    """
    Build the adaptive model with proper weight loading.
    
    Returns:
        Tuple of (model, training_info, model_config)
    """
    weights_path = home_dir / config['weights_path']
    trained_models_dir = home_dir / config['trained_models_dir']
    
    # Create model builder
    builder = AdaptiveModelBuilder(
        weights_path=str(weights_path),
        trained_models_dir=str(trained_models_dir),
        linear_type=config['linear_type'],
        replace_loss_scaler=config['replace_loss_scaler'],
        c_m=256,  # Standard OpenFold dimensions
        c_z=128,
    )
    
    # Build model with proper config preset
    # Use finetuning_no_templ_ptm to disable templates in the model config
    model, training_info = builder.build_model(config_preset="finetuning_no_templ_ptm")
    
    return model, training_info


def setup_trainer(config: dict, home_dir: Path, val_chains: list):
    """
    Setup PyTorch Lightning trainer with callbacks and loggers.
    
    Returns:
        PyTorch Lightning Trainer
    """
    output_dir = home_dir / config['output_dir']
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best_model",
        save_top_k=config.get('checkpoint_save_top_k', 1),
        monitor="val/loss" if val_chains else "train/loss",
        mode="min",
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    if config.get('log_lr', True):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    
    # Setup loggers
    loggers = []
    if config.get('wandb', False):
        wandb_logger = WandbLogger(
            name=config.get('experiment_name', 'adaptive_training'),
            project=config.get('wandb_project', 'openfold'),
            entity=config.get('wandb_entity', None),
            save_dir=str(output_dir),
        )
        loggers.append(wandb_logger)
    
    # Setup strategy
    num_gpus = config.get('gpus', 1)
    if num_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            process_group_backend=config.get('distributed_backend', 'gloo')
        )
    else:
        strategy = "auto"
    
    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=config['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus,
        strategy=strategy,
        precision=config.get('precision', 'bf16-mixed'),
        callbacks=callbacks,
        logger=loggers if loggers else None,
        log_every_n_steps=config.get('log_every_n_steps', 10),
        accumulate_grad_batches=config.get('grad_accum_steps', 1),
        gradient_clip_val=config.get('gradient_clip_val', None),
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )
    
    return trainer


def setup_data_module(config: dict, home_dir: Path, train_list_path: Path, val_list_path: Path, alignment_dir: Path, model_config_obj):
    """
    Setup OpenFold data module for loading training data.
    
    Returns:
        OpenFold DataModule
    """
    pdb_dir = home_dir / config['pdb_dir']
    
    # Create data module arguments
    data_module_args = {
        'config': model_config_obj.data,
        'batch_seed': config.get('seed', 42),
        'train_data_dir': str(pdb_dir),
        'train_alignment_dir': str(alignment_dir),
        'template_mmcif_dir': str(pdb_dir),
        'max_template_date': config.get('max_template_date', '2025-01-01'),
        'train_chain_list_path': str(train_list_path),
        'kalign_binary_path': '/usr/bin/kalign',
        'train_epoch_len': config.get('train_epoch_len', 1000),
    }
    
    # Add validation data if available
    if val_list_path and val_list_path.exists():
        data_module_args.update({
            'val_data_dir': str(pdb_dir),
            'val_alignment_dir': str(alignment_dir),
            'val_chain_list_path': str(val_list_path),
        })
    
    # Create data module
    data_module = OpenFoldDataModule(**data_module_args)
    
    return data_module


def main():
    """Main entry point for adaptive training."""
    parser = argparse.ArgumentParser(
        description="Run adaptive training for Evoformer replacement blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--gpus", type=int, default=None,
        help="Override number of GPUs from config"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None,
        help="Override max epochs from config"
    )
    parser.add_argument(
        "--train_epoch_len", type=int, default=None,
        help="Override train epoch length from config"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Prepare everything but don't run training"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.gpus is not None:
        config['gpus'] = args.gpus
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    if args.train_epoch_len is not None:
        config['train_epoch_len'] = args.train_epoch_len
    
    # Resolve paths relative to home directory
    home_dir = Path.home()
    
    # Count available blocks
    trained_models_dir = home_dir / config['trained_models_dir']
    available_blocks = count_available_replacement_blocks(trained_models_dir, config['linear_type'])
    
    if not available_blocks:
        raise ValueError(f"No trained replacement blocks found in {trained_models_dir}")
    
    # Print config summary
    print_config_summary(config, available_blocks)
    
    # Setup data
    train_chains, val_chains, train_list_path, val_list_path, alignment_dir = setup_data(config, home_dir)
    
    # Build adaptive model
    model, training_info = build_adaptive_model(config, home_dir)
    model_config_obj = training_info['config']
    
    # Configure for single sequence mode (no templates in data)
    print("\nConfiguring for single sequence mode...")
    model_config_obj.data.common.max_extra_msa = 1
    model_config_obj.data.common.max_msa_clusters = 1
    model_config_obj.data.train.max_extra_msa = 1
    model_config_obj.data.train.max_msa_clusters = 1
    model_config_obj.data.common.use_templates = False
    model_config_obj.data.common.use_template_torsion_angles = False
    model_config_obj.loss.masked_msa.weight = 0.0  # Disable masked MSA loss
    print("  ✓ Single sequence mode configured")
    print(f"  ✓ Template data usage: {model_config_obj.data.common.use_templates}")
    print(f"  ✓ Template model enabled: {model_config_obj.model.template.enabled}")
    
    # Create training wrapper
    print("\nCreating training wrapper...")
    wrapper = AdaptiveTrainingWrapper(
        model=model,
        config=model_config_obj,
        training_info=training_info,
        learning_rate=config['learning_rate'],
        log_structure_every_k_epoch=config.get('log_structure_every_k_epoch', 0),
    )
    print("  ✓ Wrapper created")
    
    # Setup data module
    print("\nSetting up data module...")
    data_module = setup_data_module(config, home_dir, train_list_path, val_list_path, alignment_dir, model_config_obj)
    data_module.prepare_data()
    data_module.setup()
    print("  ✓ Data module ready")
    
    # Setup trainer
    print("\nSetting up trainer...")
    trainer = setup_trainer(config, home_dir, val_chains)
    print("  ✓ Trainer ready")
    
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - Training preparation complete (not running training)")
        print("=" * 80)
        return
    
    # Run training
    print("\n" + "=" * 80)
    print("Starting Adaptive Training")
    print("=" * 80 + "\n")
    
    trainer.fit(wrapper, datamodule=data_module)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest model saved to: {trainer.checkpoint_callback.best_model_path}")
    print(f"All outputs saved to: {home_dir / config['output_dir']}")


if __name__ == "__main__":
    main()



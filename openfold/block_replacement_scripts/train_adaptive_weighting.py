#!/usr/bin/env python3
"""
Part 2: Adaptive Weighting Training Script

This script implements the adaptive weighting mechanism where each Evoformer block
outputs a weighted sum: w * evoformer_output + (1-w) * replacement_output.

The weight w is predicted using: sigmoid(linear(mean_pool(m[..., 0, :, :])))
where m is the MSA representation (first sequence from MSA).

This script follows the same pattern as finetune_openfold_custom_layer.py,
leveraging the existing train_openfold.py infrastructure.
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path
import pandas as pd
import torch
import random
import json
from pytorch_lightning.utilities.rank_zero import rank_zero_info


def _is_torchrun():
    """Check if we're running under torchrun distributed training"""
    return 'TORCHRUN_PROC_RANK' in os.environ or 'RANK' in os.environ


def find_available_structure_files(pdb_dir):
    """Find all available structure files and extract their IDs"""
    import glob
    
    structure_files = {}
    supported_exts = [".cif", ".pdb", ".core"]
    
    for ext in supported_exts:
        pattern = os.path.join(pdb_dir, "**", f"*{ext}")
        matches = glob.glob(pattern, recursive=True)
        
        for file_path in matches:
            filename = os.path.basename(file_path)
            file_id = os.path.splitext(filename)[0]
            structure_files[file_id] = file_path
    
    return structure_files


def extract_chain_list_from_csv(csv_path, output_path, pdb_dir=None):
    """Extract chain list from CSV file's 'natives_rcsb' column and filter by available structure files"""
    rank_zero_info(f"Reading CSV file: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Extract the 'natives_rcsb' column, skip empty values
    all_chains = df['natives_rcsb'].dropna().tolist()
    
    rank_zero_info(f"Found {len(all_chains)} chains in CSV file")
    
    # Filter chains by available structure files if pdb_dir is provided
    if pdb_dir:
        rank_zero_info(f"Checking for available structure files in: {pdb_dir}")
        available_structures = find_available_structure_files(pdb_dir)
        
        available_chains = []
        missing_chains = []
        
        for chain in all_chains:
            # Extract file ID from chain (e.g., "1aaj_A" -> "1aaj")
            if '_' in chain:
                file_id = chain.rsplit('_', 1)[0]
            else:
                file_id = chain
            
            if file_id in available_structures:
                available_chains.append(chain)
            else:
                missing_chains.append(chain)
        
        rank_zero_info(f"Available structure files: {len(available_structures)}")
        rank_zero_info(f"Chains with available structures: {len(available_chains)}")
        rank_zero_info(f"Chains missing structures: {len(missing_chains)}")
        
        chain_list = available_chains
    else:
        chain_list = all_chains
    
    if not chain_list:
        raise ValueError("No chains with available structure files found!")
    
    # Write chain list to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# Chain list extracted from CSV file\n")
        f.write("# Only includes chains with available structure files\n")
        f.write("# Format: pdb_id_chain\n")
        for chain in chain_list:
            f.write(f"{chain}\n")
    
    rank_zero_info(f"Written filtered chain list to: {output_path}")
    
    return chain_list


def create_minimal_alignment_structure(alignment_dir, chain_list=None):
    """Create minimal alignment directory structure for single sequence mode"""
    os.makedirs(alignment_dir, exist_ok=True)
    
    # Create a dummy alignment file (required but not used in single seq mode)
    dummy_alignment_path = os.path.join(alignment_dir, "dummy.a3m")
    if not os.path.exists(dummy_alignment_path):
        with open(dummy_alignment_path, 'w') as f:
            f.write(">dummy\nA\n")
    
    # Create alignment files for each chain (required by OpenFold data loader)
    if chain_list:
        for chain in chain_list:
            # Extract PDB ID from chain (e.g., "1tg0_A" -> "1tg0")
            pdb_id = chain.split('_')[0]
            alignment_file = os.path.join(alignment_dir, f"{pdb_id}.a3m")
            if not os.path.exists(alignment_file):
                with open(alignment_file, 'w') as f:
                    f.write(f">{chain}\nA\n")  # Single sequence alignment
    
    rank_zero_info(f"Created minimal alignment structure at: {alignment_dir}")


def split_chains_for_validation(chain_list, val_fraction, output_dir, seed=42):
    """Split chain list into training and validation sets"""
    if val_fraction <= 0.0:
        # No validation split requested
        return chain_list, [], None, None
    
    if val_fraction >= 1.0:
        raise ValueError("Validation fraction must be less than 1.0")
    
    # Set random seed for reproducible splits
    random.seed(seed)
    
    # Shuffle chains and split
    shuffled_chains = chain_list.copy()
    random.shuffle(shuffled_chains)
    
    val_size = int(len(shuffled_chains) * val_fraction)
    val_chains = shuffled_chains[:val_size]
    train_chains = shuffled_chains[val_size:]
    
    # Ensure we have at least one training example
    if len(train_chains) == 0:
        train_chains = [val_chains.pop()]
    
    rank_zero_info(f"Chain split (seed={seed}):")
    rank_zero_info(f"  - Total chains: {len(chain_list)}")
    rank_zero_info(f"  - Training chains: {len(train_chains)}")
    rank_zero_info(f"  - Validation chains: {len(val_chains)}")
    
    # Save split chain lists
    chain_list_dir = output_dir / "chain_lists"
    os.makedirs(chain_list_dir, exist_ok=True)
    
    train_list_path = chain_list_dir / "training_chains.txt" 
    val_list_path = chain_list_dir / "validation_chains.txt"
    
    # Write training chains
    with open(train_list_path, 'w') as f:
        f.write("# Training chain list (after validation split)\n")
        f.write("# Format: pdb_id_chain\n")
        for chain in train_chains:
            f.write(f"{chain}\n")
    
    # Write validation chains (if any)
    if val_chains:
        with open(val_list_path, 'w') as f:
            f.write("# Validation chain list (after training split)\n")
            f.write("# Format: pdb_id_chain\n")
            for chain in val_chains:
                f.write(f"{chain}\n")
        rank_zero_info(f"  - Validation chains saved to: {val_list_path}")
    else:
        val_list_path = None
    
    rank_zero_info(f"  - Training chains saved to: {train_list_path}")
    
    return train_chains, val_chains, train_list_path, val_list_path


def create_adaptive_training_command_file(output_dir, trained_models_dir, linear_type, replace_loss_scaler, 
                                         log_structure_every_k_epoch=1, disable_per_block_logging=False):
    """Create the adaptive training command file for custom loading"""
    adaptive_cmd_file = output_dir / "adaptive_training_cmd.json"
    
    cmd_data = {
        "trained_models_dir": str(trained_models_dir),
        "linear_type": linear_type,
        "replace_loss_scaler": replace_loss_scaler,
        "log_structure_every_k_epoch": log_structure_every_k_epoch,
        "disable_per_block_logging": disable_per_block_logging,
        "adaptive_training": True
    }
    
    with open(adaptive_cmd_file, 'w') as f:
        json.dump(cmd_data, f, indent=2)
    
    return adaptive_cmd_file


def run_adaptive_training(args):
    """Run the adaptive weighting training with OpenFold infrastructure"""
    # Convert relative paths to absolute paths relative to home directory
    home_dir = Path.home()
    
    # Setup paths
    if args.config:
        # Load config file
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with command line arguments (only if explicitly provided)
        # Check if arguments were explicitly provided by looking at sys.argv
        provided_args = set()
        for i, arg in enumerate(sys.argv):
            if arg.startswith('--'):
                arg_name = arg[2:]  # Remove '--'
                provided_args.add(arg_name)
        
        if 'csv_path' in provided_args and args.csv_path is not None:
            config['csv_path'] = args.csv_path
        if 'pdb_dir' in provided_args and args.pdb_dir is not None:
            config['pdb_dir'] = args.pdb_dir
        if 'checkpoint_path' in provided_args and args.checkpoint_path is not None:
            config['weights_path'] = args.checkpoint_path
        if 'trained_models_dir' in provided_args and args.trained_models_dir is not None:
            config['trained_models_dir'] = args.trained_models_dir
        if 'output_dir' in provided_args and args.output_dir is not None:
            config['output_dir'] = args.output_dir
        if 'linear_type' in provided_args and args.linear_type is not None:
            config['linear_type'] = args.linear_type
        if 'replace_loss_scaler' in provided_args and args.replace_loss_scaler is not None:
            config['replace_loss_scaler'] = args.replace_loss_scaler
        if 'max_epochs' in provided_args and args.max_epochs is not None:
            config['max_epochs'] = args.max_epochs
        if 'batch_size' in provided_args and args.batch_size is not None:
            config['batch_size'] = args.batch_size
        if 'learning_rate' in provided_args and args.learning_rate is not None:
            config['learning_rate'] = args.learning_rate
        if 'train_epoch_len' in provided_args and args.train_epoch_len is not None:
            config['train_epoch_len'] = args.train_epoch_len
        if 'validation_fraction' in provided_args and args.validation_fraction is not None:
            config['validation_fraction'] = args.validation_fraction
        if 'grad_accum_steps' in provided_args and args.grad_accum_steps is not None:
            config['grad_accum_steps'] = args.grad_accum_steps
        if 'log_structure_every_k_epoch' in provided_args and args.log_structure_every_k_epoch is not None:
            config['log_structure_every_k_epoch'] = args.log_structure_every_k_epoch
        if 'disable_per_block_logging' in provided_args and args.disable_per_block_logging is not None:
            config['disable_per_block_logging'] = args.disable_per_block_logging
        if 'wandb' in provided_args and args.wandb is not None:
            config['wandb'] = args.wandb
        if 'wandb_project' in provided_args and args.wandb_project is not None:
            config['wandb_project'] = args.wandb_project
        if 'wandb_entity' in provided_args and args.wandb_entity is not None:
            config['wandb_entity'] = args.wandb_entity
        if 'experiment_name' in provided_args and args.experiment_name is not None:
            config['experiment_name'] = args.experiment_name
        if 'distributed_backend' in provided_args and args.distributed_backend is not None:
            config['distributed_backend'] = args.distributed_backend
    else:
        # Use command line arguments directly
        config = {
            'csv_path': args.csv_path,
            'pdb_dir': args.pdb_dir,
            'weights_path': args.checkpoint_path,
            'trained_models_dir': args.trained_models_dir,
            'output_dir': args.output_dir,
            'linear_type': args.linear_type,
            'replace_loss_scaler': args.replace_loss_scaler,
            'max_epochs': args.max_epochs,
            'train_epoch_len': args.train_epoch_len,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'validation_fraction': args.validation_fraction,
            'grad_accum_steps': args.grad_accum_steps,
            'log_structure_every_k_epoch': args.log_structure_every_k_epoch,
            'distributed_backend': args.distributed_backend,
            'wandb': args.wandb,
            'wandb_project': args.wandb_project,
            'wandb_entity': args.wandb_entity,
            'experiment_name': args.experiment_name,
        }
    
    csv_path = home_dir / config['csv_path']
    pdb_dir = home_dir / config['pdb_dir']
    checkpoint_path = home_dir / config['weights_path']
    trained_models_dir = home_dir / config['trained_models_dir']
    output_dir = home_dir / config['output_dir']
    
    # Handle resume_adaptive_checkpoint if provided
    resume_adaptive_checkpoint = None
    if args.resume_adaptive_checkpoint:
        resume_adaptive_checkpoint = home_dir / args.resume_adaptive_checkpoint if not Path(args.resume_adaptive_checkpoint).is_absolute() else Path(args.resume_adaptive_checkpoint)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    rank_zero_info("=== Adaptive Evoformer-Replacement Weighting Training ===")

    # Check if running under torchrun
    is_torchrun = _is_torchrun()
    if is_torchrun:
        rank = os.environ.get('RANK', 'unknown')
        local_rank = os.environ.get('LOCAL_RANK', 'unknown')
        world_size = os.environ.get('WORLD_SIZE', 'unknown')
        print(f"🔄 Running under torchrun distributed training:")
        print(f"  RANK: {rank}")
        print(f"  LOCAL_RANK: {local_rank}")
        print(f"  WORLD_SIZE: {world_size}")
    else:
        print("🔄 Running as single process (not distributed)")

    # Determine training mode
    if resume_adaptive_checkpoint:
        rank_zero_info(f"\n{'='*60}")
        rank_zero_info("Mode: Resuming from adaptive checkpoint")
        rank_zero_info(f"Checkpoint: {resume_adaptive_checkpoint}")
        rank_zero_info(f"{'='*60}\n")
    else:
        rank_zero_info(f"\n{'='*60}")
        rank_zero_info("Mode: Initial training from base weights")
        rank_zero_info(f"{'='*60}\n")

    rank_zero_info(f"Configuration:")
    rank_zero_info(f"  CSV path: {csv_path}")
    rank_zero_info(f"  PDB directory: {pdb_dir}")
    rank_zero_info(f"  Pre-trained weights: {checkpoint_path}")
    rank_zero_info(f"  Trained models directory: {trained_models_dir}")
    rank_zero_info(f"  Output directory: {output_dir}")
    rank_zero_info(f"  Linear type: {config['linear_type']}")
    rank_zero_info(f"  Replace loss scaler: {config['replace_loss_scaler']}")
    rank_zero_info(f"  Learning rate: {config['learning_rate']}")
    rank_zero_info(f"  Max epochs: {config['max_epochs']}")
    rank_zero_info(f"  Train epoch len: {config.get('train_epoch_len', 'NOT SET')}")
    rank_zero_info(f"  Batch size: {config['batch_size']}")
    rank_zero_info(f"  Validation fraction: {config.get('validation_fraction', 'NOT SET')}")
    rank_zero_info(f"  Grad accum steps: {config.get('grad_accum_steps', 'NOT SET')}")
    rank_zero_info(f"  Log structure every k epochs: {config.get('log_structure_every_k_epoch', 'NOT SET')}")
    rank_zero_info(f"  Wandb enabled: {config.get('wandb', 'NOT SET')}")
    rank_zero_info(f"  Wandb project: {config.get('wandb_project', 'NOT SET')}")
    rank_zero_info(f"  Experiment name: {config.get('experiment_name', 'NOT SET')}")
    
    # Check trained models directory
    if not trained_models_dir.exists():
        raise FileNotFoundError(f"Trained models directory not found: {trained_models_dir}")
    
    # Count available trained blocks
    available_blocks = []
    for block_dir in trained_models_dir.glob("block_*"):
        if block_dir.is_dir():
            checkpoint_file = block_dir / config['linear_type'] / "best_model.ckpt"
            if checkpoint_file.exists():
                block_idx = int(block_dir.name.split("_")[1])
                available_blocks.append(block_idx)
    
    available_blocks.sort()
    rank_zero_info(f"Found {len(available_blocks)} trained replacement blocks: {available_blocks[:10]}...")
    
    if len(available_blocks) == 0:
        raise ValueError(f"No trained replacement blocks found in {trained_models_dir}")
    
    # Extract chain list
    rank_zero_info("\n1. Extracting chain list from CSV...")
    chain_list_dir = output_dir / "chain_lists"
    chain_list_path = chain_list_dir / "all_chains.txt"
    all_chains = extract_chain_list_from_csv(csv_path, chain_list_path, pdb_dir)
    
    # Split chains for validation
    rank_zero_info("2. Splitting data for validation...")
    val_fraction = config.get('validation_fraction', 0.1)
    train_chains, val_chains, train_list_path, val_list_path = split_chains_for_validation(
        all_chains, val_fraction, output_dir, args.seed
    )
    
    # Create minimal alignment structure
    rank_zero_info("3. Setting up minimal alignment structure...")
    alignment_dir = output_dir / "alignments"
    create_minimal_alignment_structure(alignment_dir, all_chains)
    
    # Verify checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Create adaptive training command file FIRST (before wrapper script)
    rank_zero_info("4. Creating adaptive training configuration...")
    adaptive_cmd_file = create_adaptive_training_command_file(
        output_dir,
        trained_models_dir,
        config['linear_type'],
        config['replace_loss_scaler'],
        config.get('log_structure_every_k_epoch', 1),
        config.get('disable_per_block_logging', False)
    )
    rank_zero_info(f"Adaptive training config saved to: {adaptive_cmd_file}")
    
    # Build training command - use standard train_openfold.py directly
    train_script = home_dir / "openfold" / "train_openfold.py"

    # Set distributed backend if specified (from config or command line)
    if args.distributed_backend is not None:
        distributed_backend = args.distributed_backend
    else:
        distributed_backend = config.get('distributed_backend')

    cmd = [
        sys.executable, str(train_script),
        str(pdb_dir),                    # train_data_dir
        str(alignment_dir),              # train_alignment_dir  
        str(pdb_dir),                    # template_mmcif_dir
        str(output_dir),                 # output_dir
        args.max_template_date,          # max_template_date
        
        # Chain list and data loading
        "--train_chain_list_path", str(train_list_path),
        "--enable_recursive_search",     # Search subdirectories
        "--enable_single_seq_mode",      # No MSAs or templates
        
        # Adaptive training configuration
        "--adaptive_config_path", str(adaptive_cmd_file),
        "--apply_adaptive_blocks_immediately",  # Apply blocks after weight loading
        
        # Training configuration  
        "--config_preset", "finetuning_no_templ_ptm",  # Use no-template preset for single seq mode
        "--max_epochs", str(config['max_epochs']),
        "--train_epoch_len", str(config.get('train_epoch_len', 1000)),
        "--learning_rate", str(config['learning_rate']),
        
        # Resource settings - let torchrun handle GPU assignment
        "--precision", args.precision,
        "--gpus", "1",  # Each process uses 1 GPU when using torchrun
        "--seed", str(args.seed),

        # Gradient accumulation
        "--grad_accum_steps", str(config.get('grad_accum_steps', 1)),
        
        # Logging
        "--log_lr",
        "--log_every_n_steps", "10",
    ]
    
    # Load pre-trained weights with priority: resume_adaptive_checkpoint > checkpoint_path
    if resume_adaptive_checkpoint:
        cmd.extend(["--resume_from_ckpt", str(resume_adaptive_checkpoint)])
        rank_zero_info(f"  ✓ Resuming from adaptive checkpoint: {resume_adaptive_checkpoint}")
    elif checkpoint_path:
        if str(checkpoint_path).endswith(".npz"):
            cmd.extend(["--resume_from_jax_params", str(checkpoint_path)])
            rank_zero_info(f"  ✓ Loading JAX weights: {checkpoint_path}")
        else:
            cmd.extend(["--resume_from_ckpt", str(checkpoint_path)])
            rank_zero_info(f"  ✓ Loading PyTorch weights: {checkpoint_path}")
        cmd.extend(["--resume_model_weights_only", "True"])

    # Add distributed backend if specified
    if distributed_backend:
        cmd.extend(["--distributed_backend", distributed_backend])

    # Add validation data if available
    if val_list_path and val_chains:
        cmd.extend([
            "--val_data_dir", str(pdb_dir),
            "--val_alignment_dir", str(alignment_dir),
            "--val_chain_list_path", str(val_list_path),
        ])
    
    # Add checkpoint configuration
    if args.checkpoint_every_epoch:
        cmd.extend(["--checkpoint_every_epoch"])
    
    if args.checkpoint_save_top_k is not None:
        cmd.extend(["--checkpoint_save_top_k", str(args.checkpoint_save_top_k)])
    
    # Add wandb logging if specified
    if config.get('wandb', False):
        cmd.extend([
            "--wandb",
            "--experiment_name", config['experiment_name'],
            "--wandb_project", config['wandb_project'],
        ])
        if config.get('wandb_entity'):
            cmd.extend(["--wandb_entity", config['wandb_entity']])
    
    rank_zero_info("5. Training command:")
    rank_zero_info(" ".join(cmd))
    
    rank_zero_info("Key features:")
    rank_zero_info("  ✓ Adaptive weighting: w * evoformer_output + (1-w) * replacement_output")
    rank_zero_info(f"  ✓ Weight prediction: sigmoid(linear(mean_pool(m[..., 0, :, :]))) ")
    rank_zero_info(f"  ✓ {len(available_blocks)} replacement blocks loaded from {trained_models_dir}")
    rank_zero_info(f"  ✓ Replace loss scaler: {config['replace_loss_scaler']} (penalizes mean weights)")
    rank_zero_info("  ✓ Trainable parameters: weight predictors + replacement blocks")
    rank_zero_info("  ✓ Single sequence mode: No MSAs or templates required")
    rank_zero_info(f"  ✓ Learning rate: {config['learning_rate']} (for adaptive components)")
    rank_zero_info(f"  ✓ Gradient accumulation: {config.get('grad_accum_steps', 1)} steps")
    
    if val_chains:
        rank_zero_info(f"  ✓ Validation split: {len(val_chains)}/{len(all_chains)} chains")
    
    
    if args.dry_run:
        rank_zero_info("DRY RUN: Command would be executed but --dry_run flag is set")
        return cmd
    
    rank_zero_info("6. Starting adaptive training...")
    rank_zero_info("=" * 60)
    
    # Set environment for proper distributed training
    env = os.environ.copy()

    # Only set CUDA_VISIBLE_DEVICES for single GPU training
    # For multi-GPU, let torchrun handle GPU assignment
    if config.get('gpus', args.gpus) == 1:
        env['CUDA_VISIBLE_DEVICES'] = '0'
        rank_zero_info("Setting CUDA_VISIBLE_DEVICES=0 for single GPU training")

    # Don't override distributed backend - let torchrun handle it
    # Only set if explicitly needed for compatibility
    if 'PL_TORCH_DISTRIBUTED_BACKEND' not in env:
        env['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
        rank_zero_info("Setting PL_TORCH_DISTRIBUTED_BACKEND=gloo for compatibility")
    
    # Execute the training command
    result = subprocess.run(cmd, check=True, env=env)
    rank_zero_info("Adaptive training completed successfully!")
    return result


def create_default_config():
    """Create default adaptive training configuration"""
    return {
        # Data paths (relative to home directory)
        'csv_path': 'data/af2rank_single/af2rank_single_set_single_tms_07.csv',
        'pdb_dir': 'data/af2rank_single/pdb',
        'weights_path': 'openfold/openfold/resources/openfold_params/finetuning_ptm_2.pt',
        'trained_models_dir': 'AFdistill/pretrain_full',
        'output_dir': 'AFdistill/adaptive_block_1-46',
        
        # Model configuration
        'linear_type': 'full',  # Should match pre-trained blocks
        'replace_loss_scaler': 1.0,  # Weight for penalizing mean adaptive weights
        
        # Training parameters
        'batch_size': 1,
        'max_epochs': 1000,
        'train_epoch_len': 32,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'validation_fraction': 0.1,
        'num_workers': 2,
        'grad_accum_steps': 2,  # Gradient accumulation steps
        'log_structure_every_k_epoch': 1,  # Log structure every k epochs (0 = disabled)
        
        # Wandb logging
        'wandb': True,
        'wandb_project': 'af2distill',
        'wandb_entity': 'kryst3154-massachusetts-institute-of-technology',
        'experiment_name': 'adaptive_block_1-46',

        # Data loading strategy
        'data_loading_strategy': 'preload_gpu',  # Options: 'preload_gpu', 'preload_cpu', 'on_demand'
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train adaptive weighting for Evoformer replacement blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file option
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file")
    parser.add_argument("--create_config", type=str, nargs='?', 
                       const="adaptive_config.yaml",
                       help="Create default config file and exit")
    
    # Data paths (only for command-line override, config file takes precedence)
    parser.add_argument("--csv_path", type=str, default=None,
                       help="Override CSV path from config file")
    parser.add_argument("--pdb_dir", type=str, default=None,
                       help="Override PDB directory from config file")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Override checkpoint path from config file")
    parser.add_argument("--trained_models_dir", type=str, default=None,
                       help="Override trained models directory from config file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Override output directory from config file")
    
    # Model configuration (only for command-line override, config file takes precedence)
    parser.add_argument("--linear_type", type=str, default=None,
                       choices=["full", "diagonal", "affine"],
                       help="Override linear type from config file")
    parser.add_argument("--replace_loss_scaler", type=float, default=None,
                       help="Override replace loss scaler from config file")
    
    # Training parameters (only for command-line override, config file takes precedence)
    parser.add_argument("--max_epochs", type=int, default=None,
                       help="Override max epochs from config file")
    parser.add_argument("--train_epoch_len", type=int, default=None,
                       help="Override train epoch length from config file")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size from config file")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Override learning rate from config file")
    parser.add_argument("--validation_fraction", type=float, default=None,
                       help="Override validation fraction from config file")
    parser.add_argument("--grad_accum_steps", type=int, default=None,
                       help="Override gradient accumulation steps from config file")
    parser.add_argument("--log_structure_every_k_epoch", type=int, default=None,
                       help="Override structure logging frequency from config file")
    parser.add_argument("--disable_per_block_logging", action="store_true", default=False,
                       help="Disable per-block weight logging for faster training")
    parser.add_argument("--max_template_date", type=str, default="2025-01-01",
                       help="Cutoff date for templates (YYYY-MM-DD)")
    
    # Checkpoint resuming
    parser.add_argument("--resume_adaptive_checkpoint", type=str, default=None,
                       help="Path to Lightning checkpoint with adaptive blocks. Takes priority over --checkpoint_path.")
    
    # Checkpoint configuration
    parser.add_argument("--checkpoint_every_epoch", action="store_true", default=False,
                       help="Save checkpoint at the end of every training epoch")
    parser.add_argument("--checkpoint_save_top_k", type=int, default=1,
                       help="Number of best checkpoints to keep")
    
    # Resource settings (only for command-line override, config file takes precedence)
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                       choices=["16", "32", "bf16", "bf16-mixed"],
                       help="Training precision")
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--distributed_backend", type=str, default=None,
                       choices=["nccl", "gloo", "mpi"],
                       help="Override distributed backend from config file")
    
    # Logging and monitoring (only for command-line override, config file takes precedence)
    parser.add_argument("--wandb", action="store_true", default=False,
                       help="Override wandb setting from config file")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Override experiment name from config file")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="Override wandb project from config file")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Override wandb entity from config file")
    parser.add_argument("--data_loading_strategy", type=str, default=None,
                       choices=["preload_gpu", "preload_cpu", "on_demand"],
                       help="Override data loading strategy from config file")

    # Utility
    parser.add_argument("--dry_run", action="store_true", default=False,
                       help="rank_zero_info command but don't execute training")
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_default_config()
        config_path = args.create_config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        rank_zero_info(f"Created default config file: {config_path}")
        rank_zero_info("Edit this file to customize the adaptive training, then run:")
        rank_zero_info(f"python {sys.argv[0]} --config {config_path}")
        return
    
    # Require config file for adaptive training
    if not args.config:
        rank_zero_info("ERROR: Config file is required for adaptive training.")
        rank_zero_info("Use --create_config to generate a default config file, then:")
        rank_zero_info(f"python {sys.argv[0]} --config <config_file.yaml>")
        sys.exit(1)
    
    # Run adaptive training
    run_adaptive_training(args)


if __name__ == "__main__":
    main()

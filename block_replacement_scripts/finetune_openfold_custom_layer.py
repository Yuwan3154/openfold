#!/usr/bin/env python3
"""
Fine-tuning script for OpenFold with custom Evoformer layer replacement.
This script:
1. Extracts chain list from CSV file
2. Sets up training with no MSAs or templates (single sequence mode) 
3. Replaces a specified Evoformer layer and freezes all other parameters
4. Uses pre-trained weights for fine-tuning
5. Provides flexible checkpoint saving and learning rate configuration
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import torch
import random


def find_available_structure_files(pdb_dir):
    """
    Find all available structure files and extract their IDs
    """
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
    """
    Extract chain list from CSV file's 'natives_rcsb' column and filter by available structure files
    """
    print(f"Reading CSV file: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Extract the 'natives_rcsb' column, skip empty values
    all_chains = df['natives_rcsb'].dropna().tolist()
    
    print(f"Found {len(all_chains)} chains in CSV file")
    
    # Filter chains by available structure files if pdb_dir is provided
    if pdb_dir:
        print(f"Checking for available structure files in: {pdb_dir}")
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
        
        print(f"Available structure files: {len(available_structures)}")
        for file_id, path in list(available_structures.items())[:3]:
            print(f"  {file_id}: {path}")
        if len(available_structures) > 3:
            print(f"  ... and {len(available_structures) - 3} more")
        
        print(f"Chains with available structures: {len(available_chains)}")
        print(f"Chains missing structures: {len(missing_chains)}")
        if missing_chains:
            print("Missing chains (first 5):")
            for chain in missing_chains[:5]:
                print(f"  {chain}")
            if len(missing_chains) > 5:
                print(f"  ... and {len(missing_chains) - 5} more")
        
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
    
    print(f"Written filtered chain list to: {output_path}")
    print("Final chain list contents:")
    for i, chain in enumerate(chain_list[:5]):  # Show first 5
        print(f"  {chain}")
    if len(chain_list) > 5:
        print(f"  ... and {len(chain_list) - 5} more")
    
    return chain_list


def create_minimal_alignment_structure(alignment_dir):
    """
    Create minimal alignment directory structure for single sequence mode
    """
    os.makedirs(alignment_dir, exist_ok=True)
    
    # Create a dummy alignment file (required but not used in single seq mode)
    dummy_alignment_path = os.path.join(alignment_dir, "dummy.a3m")
    if not os.path.exists(dummy_alignment_path):
        with open(dummy_alignment_path, 'w') as f:
            f.write(">dummy\nA\n")
    
    print(f"Created minimal alignment structure at: {alignment_dir}")


def split_chains_for_validation(chain_list, val_fraction, output_dir, seed=42):
    """
    Split chain list into training and validation sets
    
    Args:
        chain_list: List of chains to split
        val_fraction: Fraction to use for validation (0.0 to 1.0)
        output_dir: Directory to save the split chain lists
        seed: Random seed for reproducible splits
    
    Returns:
        tuple: (train_chains, val_chains, train_list_path, val_list_path)
    """
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
    
    print(f"Chain split (seed={seed}):")
    print(f"  - Total chains: {len(chain_list)}")
    print(f"  - Training chains: {len(train_chains)}")
    print(f"  - Validation chains: {len(val_chains)}")
    print(f"  - Validation fraction: {len(val_chains)/len(chain_list):.3f}")
    
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
        print(f"  - Validation chains saved to: {val_list_path}")
    else:
        val_list_path = None
    
    print(f"  - Training chains saved to: {train_list_path}")
    
    return train_chains, val_chains, train_list_path, val_list_path


def run_openfold_training(args):
    """
    Run the OpenFold training with specified parameters
    """
    # Convert relative paths to absolute paths relative to home directory
    home_dir = Path.home()
    
    # Setup paths
    csv_path = home_dir / args.csv_path
    pdb_dir = home_dir / args.pdb_dir
    checkpoint_path = home_dir / args.checkpoint_path
    output_dir = home_dir / args.output_dir
    
    # Create chain list file
    chain_list_dir = output_dir / "chain_lists"
    chain_list_path = chain_list_dir / "training_chains.txt"
    
    print("=== OpenFold Fine-tuning with Custom Layer Replacement ===")
    print()
    
    # Extract chain list from CSV and filter by available structure files
    print("1. Extracting chain list from CSV...")
    all_chains = extract_chain_list_from_csv(csv_path, chain_list_path, pdb_dir)
    print()
    
    # Split chains for validation if requested
    print("2. Splitting data for validation...")
    train_chains, val_chains, train_list_path, val_list_path = split_chains_for_validation(
        all_chains, args.validation_fraction, output_dir, args.seed
    )
    print()
    
    # Create minimal alignment directory
    print("3. Setting up minimal alignment structure...")
    alignment_dir = output_dir / "alignments"
    create_minimal_alignment_structure(alignment_dir)
    print()
    
    # Create output directory
    print("4. Creating output directory...")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Verify checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"5. Using pre-trained weights: {checkpoint_path}")
    print(f"6. Replacing Evoformer block: {args.replace_block_index}")
    print(f"7. Replacement hidden dimension: {args.replacement_hidden_dim}")
    print(f"8. Learning rate: {args.learning_rate}")
    grad_accum_note = ""
    if args.grad_accum_steps > 1:
        effective_batch_size = args.gpus * args.grad_accum_steps
        grad_accum_note = f" (effective batch size: {effective_batch_size})"
    print(f"9. Gradient accumulation: {args.grad_accum_steps} batches{grad_accum_note}")
    print(f"10. Distributed backend: {args.distributed_backend}")
    
    # Print checkpoint configuration
    print("11. Checkpoint configuration:")
    if args.checkpoint_every_epoch:
        print("   - Save every epoch")
    elif args.checkpoint_every_n_steps:
        print(f"   - Save every {args.checkpoint_every_n_steps} steps")
    elif args.checkpoint_every_n_epochs:
        print(f"   - Save every {args.checkpoint_every_n_epochs} epochs")
    else:
        print("   - Save best checkpoint only")
    print(f"   - Keep top {args.checkpoint_save_top_k} checkpoints")
    print(f"   - Monitor metric: {args.checkpoint_monitor}")
    print()
    
    # Provide guidance on train_epoch_len vs dataset size
    estimated_samples_per_epoch = args.train_epoch_len
    chains_per_epoch_per_sample = estimated_samples_per_epoch / len(train_chains) if train_chains else 1
    print(f"12. Dataset and epoch configuration:")
    print(f"   - Training chains: {len(train_chains)}")
    if val_chains:
        print(f"   - Validation chains: {len(val_chains)}")
    print(f"   - Virtual epoch length: {args.train_epoch_len}")
    print(f"   - Expected iterations per GPU: {args.train_epoch_len // args.gpus}")
    print(f"   - Each training chain seen ~{chains_per_epoch_per_sample:.1f}x per epoch")
    if chains_per_epoch_per_sample > 5:
        print(f"   ⚠️  Consider reducing --train_epoch_len to {len(train_chains) * 2} for less repetition")
    elif chains_per_epoch_per_sample < 1:
        print(f"   ⚠️  Consider increasing --train_epoch_len to {len(train_chains)} to see all data")
    print()
    
    # Build training command
    train_script = home_dir / "openfold" / "train_openfold.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    
    cmd = [
        sys.executable, str(train_script),
        str(pdb_dir),                    # train_data_dir
        str(alignment_dir),              # train_alignment_dir  
        str(pdb_dir),                    # template_mmcif_dir (same as train_data_dir)
        str(output_dir),                 # output_dir
        args.max_template_date,          # max_template_date
        
        # Chain list and data loading
        "--train_chain_list_path", str(train_list_path),
        "--enable_recursive_search",     # Search subdirectories
        "--enable_single_seq_mode",      # No MSAs or templates
        
        # Custom block replacement
        "--replace_block_index", str(args.replace_block_index),
        "--replacement_hidden_dim", str(args.replacement_hidden_dim),
        
        # Load pre-trained weights
        "--resume_from_ckpt", str(checkpoint_path),
        "--resume_model_weights_only", "True",
        
        # Training configuration  
        "--config_preset", "finetuning_ptm",
        "--max_epochs", str(args.max_epochs),
        "--train_epoch_len", str(args.train_epoch_len),
        
        # Learning rate configuration
        "--learning_rate", str(args.learning_rate),
    ]
    
    # Add validation data if available
    if val_list_path and val_chains:
        cmd.extend([
            "--val_data_dir", str(pdb_dir),                    # Same data dir for validation
            "--val_alignment_dir", str(alignment_dir),         # Same alignment dir  
            "--val_chain_list_path", str(val_list_path),       # Validation chain list
        ])
        
    # Add checkpoint configuration options
    if args.checkpoint_every_epoch:
        cmd.extend(["--checkpoint_every_epoch"])
    
    if args.checkpoint_every_n_steps:
        cmd.extend(["--checkpoint_every_n_steps", str(args.checkpoint_every_n_steps)])
    
    if args.checkpoint_every_n_epochs:
        cmd.extend(["--checkpoint_every_n_epochs", str(args.checkpoint_every_n_epochs)])
    
    if args.checkpoint_save_top_k is not None:
        cmd.extend(["--checkpoint_save_top_k", str(args.checkpoint_save_top_k)])
    
    if args.checkpoint_monitor:
        cmd.extend(["--checkpoint_monitor", args.checkpoint_monitor])
    
    # Continue with remaining arguments
    cmd.extend([
        # Resource settings
        "--precision", args.precision,
        "--gpus", str(args.gpus),
        "--seed", str(args.seed),
        "--distributed_backend", args.distributed_backend,
        
        # Gradient accumulation
        "--grad_accum_steps", str(args.grad_accum_steps),
        
        # Logging
        "--log_lr",
        "--log_every_n_steps", "10",
        
        # Early stopping (optional) - disabled by default for single sequence mode
        "--early_stopping", str(args.enable_early_stopping).lower(),
        "--patience", str(args.patience),
        "--min_delta", str(args.min_delta),
    ])
    
    # Add optional wandb logging if specified
    if args.wandb:
        cmd.extend([
            "--wandb",
            "--experiment_name", args.experiment_name,
            "--wandb_project", args.wandb_project,
        ])
        if args.wandb_entity:
            cmd.extend(["--wandb_entity", args.wandb_entity])
    
    print("11. Training command:")
    print(" ".join(cmd))
    print()
    
    print("Key features:")
    print("  ✓ Single sequence mode: No MSAs or templates required")
    print("  ✓ Custom layer replacement: Only specified Evoformer block is trainable")
    print("  ✓ Pre-trained weights: Starting from fine-tuning checkpoint")
    print("  ✓ Chain-specific training: Using chains from CSV file")
    print("  ✓ Recursive search: Finding PDB files in subdirectories")
    print(f"  ✓ Distributed backend: {args.distributed_backend} (compatible with your setup)")
    print(f"  ✓ Learning rate: {args.learning_rate} (optimized for fine-tuning)")
    print(f"  ✓ Gradient accumulation: {args.grad_accum_steps} batches")
    
    # Validation status
    if val_chains:
        print(f"  ✓ Validation split: {len(val_chains)}/{len(all_chains)} chains ({args.validation_fraction:.1%})")
    else:
        print("  ✓ Validation split: None (training only)")
    
    # Enhanced checkpoint status
    if args.checkpoint_every_epoch:
        checkpoint_status = "every epoch"
    elif args.checkpoint_every_n_steps:
        checkpoint_status = f"every {args.checkpoint_every_n_steps} steps"
    elif args.checkpoint_every_n_epochs:
        checkpoint_status = f"every {args.checkpoint_every_n_epochs} epochs"
    else:
        checkpoint_status = "best only"
    print(f"  ✓ Checkpoint saving: {checkpoint_status} (keep top {args.checkpoint_save_top_k})")
    
    early_stopping_status = "enabled (train/lddt_ca)" if args.enable_early_stopping else "disabled"
    print(f"  ✓ Early stopping: {early_stopping_status}")
    print()
    
    if args.dry_run:
        print("DRY RUN: Command would be executed but --dry_run flag is set")
        return cmd
    
    print("12. Starting training...")
    print("=" * 60)
    
    # Execute the training command
    try:
        result = subprocess.run(cmd, check=True)
        print("Training completed successfully!")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune OpenFold with custom Evoformer layer replacement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input data
    parser.add_argument(
        "--csv_path", type=str, 
        default="data/af2rank_single/af2rank_single_set_single_tms_07.csv",
        help="Path to CSV file containing chain list (relative to home directory)"
    )
    parser.add_argument(
        "--pdb_dir", type=str,
        default="data/af2rank_single/pdb", 
        help="Directory containing PDB/mmCIF files (relative to home directory)"
    )
    
    # Model and checkpoint
    parser.add_argument(
        "--checkpoint_path", type=str,
        default="openfold/openfold/resources/openfold_params/finetuning_ptm_2.pt",
        help="Path to pre-trained checkpoint (relative to home directory)"
    )
    parser.add_argument(
        "--replace_block_index", type=int, default=23,
        help="Index of Evoformer block to replace (0-based, not first/last block)"
    )
    parser.add_argument(
        "--replacement_hidden_dim", type=int, default=256,
        help="Hidden dimension for replacement block"
    )
    
    # Output
    parser.add_argument(
        "--output_dir", type=str, default="openfold_finetuning_output",
        help="Output directory for checkpoints and logs (relative to home directory)"
    )
    
    # Training parameters
    parser.add_argument(
        "--max_epochs", type=int, default=10,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--train_epoch_len", type=int, default=1000,
        help="Virtual length of each training epoch"
    )
    parser.add_argument(
        "--max_template_date", type=str, default="2025-01-01",
        help="Cutoff date for templates (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate for fine-tuning (default: 1e-4, lower than default training)"
    )
    parser.add_argument(
        "--grad_accum_steps", type=int, default=1,
        help="Accumulate gradients over k batches before optimizer step (default: 1, no accumulation)"
    )
    parser.add_argument(
        "--validation_fraction", type=float, default=0.0,
        help="Fraction of data to use for validation (0.0 to 1.0, default: 0.0 - no validation)"
    )
    
    # Checkpoint configuration
    parser.add_argument(
        "--checkpoint_every_epoch", action="store_true", default=False,
        help="Save checkpoint at the end of every training epoch"
    )
    parser.add_argument(
        "--checkpoint_every_n_steps", type=int, default=None,
        help="Save checkpoint every N training steps (overrides epoch-based saving)"
    )
    parser.add_argument(
        "--checkpoint_every_n_epochs", type=int, default=None,
        help="Save checkpoint every N epochs (alternative to every_epoch)"
    )
    parser.add_argument(
        "--checkpoint_save_top_k", type=int, default=1,
        help="Number of best checkpoints to keep (-1 for all, 1 for best only, default: 1)"
    )
    parser.add_argument(
        "--checkpoint_monitor", type=str, default="train/lddt_ca",
        help="Metric to monitor for best checkpoint (default: train/lddt_ca for single sequence mode)"
    )
    
    # Resource settings
    parser.add_argument(
        "--precision", type=str, default="bf16-mixed", choices=["16", "32", "bf16", "bf16-mixed"],
        help="Training precision"
    )
    parser.add_argument(
        "--gpus", type=int, default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--distributed_backend", type=str, default="gloo", choices=["nccl", "gloo", "mpi"],
        help="Distributed backend for DDP training (gloo for CPU/compatibility, nccl for GPU performance)"
    )
    
    # Early stopping parameters
    parser.add_argument(
        "--enable_early_stopping", action="store_true", default=False,
        help="Enable early stopping (disabled by default for single sequence training)"
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience (number of epochs to wait)"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0.001,
        help="Minimum change in monitored metric to qualify as improvement"
    )
    
    # Logging and monitoring
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="openfold_custom_layer_finetuning",
        help="Experiment name for logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="af2distill",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="kryst3154-massachusetts-institute-of-technology",
        help="Wandb entity (username or team)"
    )
    
    # Utility
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="Print command but don't execute training"
    )
    
    args = parser.parse_args()
    
    # Validate block index
    if args.replace_block_index <= 0:
        raise ValueError("replace_block_index must be greater than 0 (not first block)")
    if args.replace_block_index >= 47:  # OpenFold typically has 48 blocks (0-47)
        raise ValueError("replace_block_index must be less than 47 (not last block)")
    
    # Validate validation fraction
    if args.validation_fraction < 0.0 or args.validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be between 0.0 and 1.0 (exclusive)")
    
    # Validate gradient accumulation
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")
    
    # Run training
    run_openfold_training(args)


if __name__ == "__main__":
    main()

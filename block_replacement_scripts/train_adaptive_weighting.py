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
        print(f"Chains with available structures: {len(available_chains)}")
        print(f"Chains missing structures: {len(missing_chains)}")
        
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
    
    return chain_list


def create_minimal_alignment_structure(alignment_dir):
    """Create minimal alignment directory structure for single sequence mode"""
    os.makedirs(alignment_dir, exist_ok=True)
    
    # Create a dummy alignment file (required but not used in single seq mode)
    dummy_alignment_path = os.path.join(alignment_dir, "dummy.a3m")
    if not os.path.exists(dummy_alignment_path):
        with open(dummy_alignment_path, 'w') as f:
            f.write(">dummy\nA\n")
    
    print(f"Created minimal alignment structure at: {alignment_dir}")


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
    
    print(f"Chain split (seed={seed}):")
    print(f"  - Total chains: {len(chain_list)}")
    print(f"  - Training chains: {len(train_chains)}")
    print(f"  - Validation chains: {len(val_chains)}")
    
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


def create_adaptive_training_command_file(output_dir, trained_models_dir, linear_type, replace_loss_scaler):
    """Create the adaptive training command file for custom loading"""
    adaptive_cmd_file = output_dir / "adaptive_training_cmd.json"
    
    cmd_data = {
        "trained_models_dir": str(trained_models_dir),
        "linear_type": linear_type,
        "replace_loss_scaler": replace_loss_scaler,
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
        
        # Override with command line arguments
        if args.csv_path:
            config['csv_path'] = args.csv_path
        if args.pdb_dir:
            config['pdb_dir'] = args.pdb_dir
        if args.checkpoint_path:
            config['weights_path'] = args.checkpoint_path
        if args.trained_models_dir:
            config['trained_models_dir'] = args.trained_models_dir
        if args.output_dir:
            config['output_dir'] = args.output_dir
        if args.linear_type:
            config['linear_type'] = args.linear_type
        if args.replace_loss_scaler is not None:
            config['replace_loss_scaler'] = args.replace_loss_scaler
        if args.max_epochs:
            config['max_epochs'] = args.max_epochs
        if args.batch_size:
            config['batch_size'] = args.batch_size
        if args.learning_rate:
            config['learning_rate'] = args.learning_rate
        if args.wandb is not None:
            config['wandb'] = args.wandb
        if args.wandb_project:
            config['wandb_project'] = args.wandb_project
        if args.wandb_entity:
            config['wandb_entity'] = args.wandb_entity
        if args.experiment_name:
            config['experiment_name'] = args.experiment_name
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Adaptive Evoformer-Replacement Weighting Training ===")
    print()
    print(f"Configuration:")
    print(f"  CSV path: {csv_path}")
    print(f"  PDB directory: {pdb_dir}")
    print(f"  Pre-trained weights: {checkpoint_path}")
    print(f"  Trained models directory: {trained_models_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Linear type: {config['linear_type']}")
    print(f"  Replace loss scaler: {config['replace_loss_scaler']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Max epochs: {config['max_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print()
    
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
    print(f"Found {len(available_blocks)} trained replacement blocks: {available_blocks[:10]}...")
    
    if len(available_blocks) == 0:
        raise ValueError(f"No trained replacement blocks found in {trained_models_dir}")
    
    # Extract chain list
    print("\n1. Extracting chain list from CSV...")
    chain_list_dir = output_dir / "chain_lists"
    chain_list_path = chain_list_dir / "all_chains.txt"
    all_chains = extract_chain_list_from_csv(csv_path, chain_list_path, pdb_dir)
    print()
    
    # Split chains for validation
    print("2. Splitting data for validation...")
    val_fraction = config.get('validation_fraction', 0.1)
    train_chains, val_chains, train_list_path, val_list_path = split_chains_for_validation(
        all_chains, val_fraction, output_dir, args.seed
    )
    print()
    
    # Create minimal alignment structure
    print("3. Setting up minimal alignment structure...")
    alignment_dir = output_dir / "alignments"
    create_minimal_alignment_structure(alignment_dir)
    print()
    
    # Verify checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Create adaptive training command file
    print("4. Creating adaptive training configuration...")
    adaptive_cmd_file = create_adaptive_training_command_file(
        output_dir,
        trained_models_dir,
        config['linear_type'],
        config['replace_loss_scaler']
    )
    print(f"Adaptive training config saved to: {adaptive_cmd_file}")
    print()
    
    # Build training command - use standard train_openfold.py
    train_script = home_dir / "openfold" / "train_openfold.py"
    
    # Create a Python wrapper script that imports our custom wrapper
    wrapper_script = output_dir / "run_adaptive_training.py"
    wrapper_content = f"""#!/usr/bin/env python3
# Auto-generated wrapper script for adaptive training
import sys
from pathlib import Path

# Add openfold to path
sys.path.insert(0, str(Path.home() / "openfold"))

# Import and replace the OpenFoldWrapper with our custom version
from custom_openfold_wrapper import CustomOpenFoldWrapper
import train_openfold
train_openfold.OpenFoldWrapper = CustomOpenFoldWrapper

# Add adaptive config to args
class Args:
    pass

# Run the original training script
if __name__ == "__main__":
    # Get original args
    import argparse
    args = train_openfold.parser.parse_args()
    
    # Add our adaptive config
    args.adaptive_config_path = "{adaptive_cmd_file}"
    
    # Run training
    train_openfold.main(args)
"""
    
    with open(wrapper_script, 'w') as f:
        f.write(wrapper_content)
    
    # Make wrapper script executable
    os.chmod(wrapper_script, 0o755)
    
    print(f"Created adaptive training wrapper: {wrapper_script}")
    
    # Update train_script to use our wrapper
    train_script = wrapper_script
    
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
        
        # Load pre-trained weights
        "--resume_from_ckpt", str(checkpoint_path),
        "--resume_model_weights_only", "True",
        
        # Training configuration  
        "--config_preset", "finetuning",
        "--max_epochs", str(config['max_epochs']),
        "--train_epoch_len", str(config.get('train_epoch_len', 1000)),
        "--learning_rate", str(config['learning_rate']),
        
        # Resource settings
        "--precision", args.precision,
        "--gpus", str(args.gpus),
        "--seed", str(args.seed),
        "--distributed_backend", args.distributed_backend,
        
        # Gradient accumulation
        "--accumulate_grad_batches", str(args.accumulate_grad_batches),
        
        # Logging
        "--log_lr",
        "--log_every_n_steps", "10",
    ]
    
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
    
    print("5. Training command:")
    print(" ".join(cmd))
    print()
    
    print("Key features:")
    print("  ✓ Adaptive weighting: w * evoformer_output + (1-w) * replacement_output")
    print(f"  ✓ Weight prediction: sigmoid(linear(mean_pool(m[..., 0, :, :]))) ")
    print(f"  ✓ {len(available_blocks)} replacement blocks loaded from {trained_models_dir}")
    print(f"  ✓ Replace loss scaler: {config['replace_loss_scaler']} (penalizes mean weights)")
    print("  ✓ Only new parameters trainable: weight predictors")
    print("  ✓ Single sequence mode: No MSAs or templates required")
    print(f"  ✓ Learning rate: {config['learning_rate']} (for adaptive weights)")
    
    if val_chains:
        print(f"  ✓ Validation split: {len(val_chains)}/{len(all_chains)} chains")
    
    print()
    
    if args.dry_run:
        print("DRY RUN: Command would be executed but --dry_run flag is set")
        return cmd
    
    print("6. Starting adaptive training...")
    print("=" * 60)
    
    # Execute the training command
    try:
        result = subprocess.run(cmd, check=True)
        print("Adaptive training completed successfully!")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(1)


def create_default_config():
    """Create default adaptive training configuration"""
    return {
        # Data paths (relative to home directory)
        'csv_path': 'data/af2rank_single/af2rank_single_set_single_tms_07.csv',
        'pdb_dir': 'data/af2rank_single/pdb',
        'weights_path': 'openfold/openfold/resources/openfold_params/finetuning_ptm_2.pt',
        'trained_models_dir': 'AFdistill/pretrain_full',
        'output_dir': 'adaptive_training_output',
        
        # Model configuration
        'linear_type': 'full',  # Should match pre-trained blocks
        'replace_loss_scaler': 0.1,  # Weight for penalizing mean adaptive weights
        
        # Training parameters
        'batch_size': 1,
        'max_epochs': 10,
        'train_epoch_len': 1000,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'validation_fraction': 0.1,
        'num_workers': 2,
        
        # Wandb logging
        'wandb': False,
        'wandb_project': 'af2distill',
        'wandb_entity': 'kryst3154-massachusetts-institute-of-technology',
        'experiment_name': 'adaptive_weighting',
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
    
    # Data paths
    parser.add_argument("--csv_path", type=str, default=None,
                       help="Path to CSV file containing chain list (relative to home)")
    parser.add_argument("--pdb_dir", type=str, default=None,
                       help="Directory containing PDB/mmCIF files (relative to home)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to pre-trained checkpoint (relative to home)")
    parser.add_argument("--trained_models_dir", type=str, default=None,
                       help="Directory containing trained replacement blocks (relative to home)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for checkpoints and logs (relative to home)")
    
    # Model configuration
    parser.add_argument("--linear_type", type=str, default=None,
                       choices=["full", "diagonal", "affine"],
                       help="Linear type used in pre-trained replacement blocks")
    parser.add_argument("--replace_loss_scaler", type=float, default=None,
                       help="Scaler for replace loss that penalizes mean adaptive weights")
    
    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=10,
                       help="Maximum number of training epochs")
    parser.add_argument("--train_epoch_len", type=int, default=1000,
                       help="Virtual length of each training epoch")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for adaptive weight training")
    parser.add_argument("--validation_fraction", type=float, default=0.1,
                       help="Fraction of data to use for validation")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                       help="Accumulate gradients over k batches")
    parser.add_argument("--max_template_date", type=str, default="2025-01-01",
                       help="Cutoff date for templates (YYYY-MM-DD)")
    
    # Checkpoint configuration
    parser.add_argument("--checkpoint_every_epoch", action="store_true", default=False,
                       help="Save checkpoint at the end of every training epoch")
    parser.add_argument("--checkpoint_save_top_k", type=int, default=1,
                       help="Number of best checkpoints to keep")
    
    # Resource settings
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                       choices=["16", "32", "bf16", "bf16-mixed"],
                       help="Training precision")
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--distributed_backend", type=str, default="gloo",
                       choices=["nccl", "gloo", "mpi"],
                       help="Distributed backend for DDP training")
    
    # Logging and monitoring
    parser.add_argument("--wandb", action="store_true", default=False,
                       help="Enable Weights & Biases logging")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name for logging")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Wandb entity (username or team)")
    
    # Utility
    parser.add_argument("--dry_run", action="store_true", default=False,
                       help="Print command but don't execute training")
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_default_config()
        config_path = args.create_config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Created default config file: {config_path}")
        print("Edit this file to customize the adaptive training, then run:")
        print(f"python {sys.argv[0]} --config {config_path}")
        return
    
    # Run adaptive training
    run_adaptive_training(args)


if __name__ == "__main__":
    main()

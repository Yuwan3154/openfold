#!/usr/bin/env python3
"""
Utilities for adaptive training - data preparation, config parsing, etc.
"""

import os
import sys
import yaml
import random
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import glob

# Add openfold to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_available_structure_files(pdb_dir: str) -> Dict[str, str]:
    """Find all available structure files and extract their IDs."""
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


def extract_chain_list_from_csv(
    csv_path: str,
    pdb_dir: str,
    output_path: Optional[str] = None
) -> List[str]:
    """
    Extract chain list from CSV file's 'natives_rcsb' column.
    
    Args:
        csv_path: Path to CSV file
        pdb_dir: Directory containing structure files
        output_path: Optional path to save filtered chain list
        
    Returns:
        List of chains with available structures
    """
    print(f"Reading CSV file: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    all_chains = df['natives_rcsb'].dropna().tolist()
    
    print(f"  Found {len(all_chains)} chains in CSV")
    
    # Filter by available structures
    print(f"  Checking for available structures in: {pdb_dir}")
    available_structures = find_available_structure_files(pdb_dir)
    
    available_chains = []
    for chain in all_chains:
        # Extract file ID (e.g., "1aaj_A" -> "1aaj")
        if '_' in chain:
            file_id = chain.rsplit('_', 1)[0]
        else:
            file_id = chain
        
        if file_id in available_structures:
            available_chains.append(chain)
    
    print(f"  Available structure files: {len(available_structures)}")
    print(f"  Chains with available structures: {len(available_chains)}")
    print(f"  Chains missing structures: {len(all_chains) - len(available_chains)}")
    
    if not available_chains:
        raise ValueError("No chains with available structure files found!")
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("# Chain list extracted from CSV file\n")
            f.write("# Only includes chains with available structure files\n")
            f.write("# Format: pdb_id_chain\n")
            for chain in available_chains:
                f.write(f"{chain}\n")
        print(f"  Written filtered chain list to: {output_path}")
    
    return available_chains


def split_chains_for_validation(
    chain_list: List[str],
    val_fraction: float,
    output_dir: Path,
    seed: int = 42
) -> Tuple[List[str], List[str], Path, Optional[Path]]:
    """
    Split chain list into training and validation sets.
    
    Args:
        chain_list: List of chains
        val_fraction: Fraction for validation (0.0 to 1.0)
        output_dir: Directory to save split lists
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_chains, val_chains, train_list_path, val_list_path)
    """
    if val_fraction <= 0.0:
        return chain_list, [], output_dir / "chain_lists" / "training_chains.txt", None
    
    if val_fraction >= 1.0:
        raise ValueError("Validation fraction must be less than 1.0")
    
    # Set random seed
    random.seed(seed)
    
    # Shuffle and split
    shuffled_chains = chain_list.copy()
    random.shuffle(shuffled_chains)
    
    val_size = int(len(shuffled_chains) * val_fraction)
    val_chains = shuffled_chains[:val_size]
    train_chains = shuffled_chains[val_size:]
    
    # Ensure at least one training example
    if len(train_chains) == 0:
        train_chains = [val_chains.pop()]
    
    print(f"\nChain split (seed={seed}):")
    print(f"  Total chains: {len(chain_list)}")
    print(f"  Training chains: {len(train_chains)}")
    print(f"  Validation chains: {len(val_chains)}")
    
    # Save split lists
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
    
    # Write validation chains
    if val_chains:
        with open(val_list_path, 'w') as f:
            f.write("# Validation chain list (after training split)\n")
            f.write("# Format: pdb_id_chain\n")
            for chain in val_chains:
                f.write(f"{chain}\n")
        print(f"  Validation chains saved to: {val_list_path}")
    else:
        val_list_path = None
    
    print(f"  Training chains saved to: {train_list_path}")
    
    return train_chains, val_chains, train_list_path, val_list_path


def create_minimal_alignment_structure(alignment_dir: Path, chain_list: List[str]):
    """
    Create minimal alignment directory structure for single sequence mode.
    
    Args:
        alignment_dir: Directory to create alignments in
        chain_list: List of chains to create alignments for
    """
    os.makedirs(alignment_dir, exist_ok=True)
    
    # Create alignment files for each chain
    for chain in chain_list:
        # Extract PDB ID (e.g., "1tg0_A" -> "1tg0")
        pdb_id = chain.split('_')[0]
        alignment_file = alignment_dir / f"{pdb_id}.a3m"
        
        if not alignment_file.exists():
            with open(alignment_file, 'w') as f:
                f.write(f">{chain}\nA\n")  # Single sequence alignment
    
    print(f"  Created minimal alignment structure at: {alignment_dir}")


def count_available_replacement_blocks(trained_models_dir: Path, linear_type: str) -> List[int]:
    """
    Count available trained replacement blocks.
    
    Args:
        trained_models_dir: Directory containing trained blocks
        linear_type: Type of linear layer
        
    Returns:
        List of available block indices
    """
    available_blocks = []
    for block_dir in trained_models_dir.glob("block_*"):
        if block_dir.is_dir():
            checkpoint_path = block_dir / linear_type / "best_model.ckpt"
            if checkpoint_path.exists():
                block_idx = int(block_dir.name.split("_")[1])
                available_blocks.append(block_idx)
    
    available_blocks.sort()
    return available_blocks


def print_config_summary(config: Dict[str, Any], available_blocks: List[int]):
    """Print a summary of the training configuration."""
    print("\n" + "=" * 80)
    print("Adaptive Evoformer Training Configuration")
    print("=" * 80)
    print(f"\nData:")
    print(f"  CSV path: {config['csv_path']}")
    print(f"  PDB directory: {config['pdb_dir']}")
    print(f"  Validation fraction: {config.get('validation_fraction', 0.1)}")
    
    print(f"\nModel:")
    print(f"  Pre-trained weights: {config['weights_path']}")
    print(f"  Trained models directory: {config['trained_models_dir']}")
    print(f"  Linear type: {config['linear_type']}")
    print(f"  Available replacement blocks: {len(available_blocks)}")
    if available_blocks:
        print(f"    Blocks: {available_blocks[:10]}{'...' if len(available_blocks) > 10 else ''}")
    
    print(f"\nTraining:")
    print(f"  Max epochs: {config['max_epochs']}")
    print(f"  Train epoch length: {config.get('train_epoch_len', 1000)}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Gradient accumulation steps: {config.get('grad_accum_steps', 1)}")
    print(f"  Replace loss scaler: {config['replace_loss_scaler']}")
    
    print(f"\nLogging:")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  Experiment name: {config.get('experiment_name', 'adaptive_training')}")
    print(f"  Wandb enabled: {config.get('wandb', False)}")
    if config.get('wandb'):
        print(f"  Wandb project: {config.get('wandb_project', 'N/A')}")
    print(f"  Log structure every k epochs: {config.get('log_structure_every_k_epoch', 0)}")
    print("=" * 80 + "\n")



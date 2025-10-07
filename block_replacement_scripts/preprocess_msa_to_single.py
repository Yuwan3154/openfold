#!/usr/bin/env python3
"""
Pre-process MSA data to single sequence format

This script converts the collected MSA data to single sequence format by extracting
only the first sequence (index 0) from each MSA representation, as the other sequences
are not meaningfully updated due to proper masking in the original model.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List
import torch
from tqdm import tqdm


def process_block_data(input_dir: Path, output_dir: Path, block_idx: int):
    """Process data for a specific block"""
    
    block_input_dir = input_dir / "block_data" / f"block_{block_idx:02d}"
    block_output_dir = output_dir / f"block_{block_idx:02d}"
    
    if not block_input_dir.exists():
        print(f"⚠️  Block {block_idx:02d} directory not found: {block_input_dir}")
        return
    
    # Create output directories
    train_output_dir = block_output_dir / "train"
    val_output_dir = block_output_dir / "val"
    train_output_dir.mkdir(parents=True, exist_ok=True)
    val_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train data
    train_input_dir = block_input_dir / "train"
    if train_input_dir.exists():
        train_files = list(train_input_dir.glob("*.pt"))
        print(f"  Processing {len(train_files)} training samples...")
        
        for file_path in tqdm(train_files, desc=f"Block {block_idx:02d} train"):
            process_single_file(file_path, train_output_dir)
    
    # Process validation data
    val_input_dir = block_input_dir / "val"
    if val_input_dir.exists():
        val_files = list(val_input_dir.glob("*.pt"))
        print(f"  Processing {len(val_files)} validation samples...")
        
        for file_path in tqdm(val_files, desc=f"Block {block_idx:02d} val"):
            process_single_file(file_path, val_output_dir)


def process_single_file(input_file: Path, output_dir: Path):
    """Process a single data file to extract single sequence"""
    
    # Load original data
    data = torch.load(input_file, map_location='cpu')
    # Extract single sequence (first sequence only)
    # CRITICAL: Use .clone() to create new tensors, not views that reference original data
    processed_data = {
        'input': {
            'm': data['input']['m'][0:1].clone(),  # Take only first sequence [1, n_res, c_m]
            'z': data['input']['z'].clone()        # Pair representation is already single sequence
        },
        'output': {
            'm': data['output']['m'][0:1].clone(),  # Take only first sequence [1, n_res, c_m]
            'z': data['output']['z'].clone()        # Pair representation is already single sequence
        },
        'chain_id': data['chain_id'],
        'block_idx': data['block_idx']
    }
    # Save processed data
    output_file = output_dir / input_file.name
    torch.save(processed_data, output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process MSA data to single sequence format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory containing MSA data (relative to home directory)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for single sequence data (relative to home directory)"
    )
    parser.add_argument(
        "--blocks", type=int, nargs="+", 
        default=list(range(1, 47)),
        help="Block indices to process (default: all blocks 1-46)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    home_dir = Path.home()
    input_dir = home_dir / args.input_dir
    output_dir = home_dir / args.output_dir
    
    print(f"Pre-processing MSA data to single sequence format:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Blocks to process: {args.blocks}")
    print()
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each block
    total_blocks = len(args.blocks)
    for i, block_idx in enumerate(args.blocks):
        print(f"Processing block {block_idx:02d} ({i+1}/{total_blocks})...")
        process_block_data(input_dir, output_dir, block_idx)
        print(f"✅ Completed block {block_idx:02d}")
        print()
    
    print(f"🎉 Pre-processing complete!")
    print(f"Single sequence data saved to: {output_dir}")
    
    # Show size comparison
    print(f"\nSize comparison:")
    input_size = sum(f.stat().st_size for f in input_dir.rglob("*.pt"))
    output_size = sum(f.stat().st_size for f in output_dir.rglob("*.pt"))
    print(f"  Original MSA data: {input_size / (1024**3):.2f} GB")
    print(f"  Single sequence data: {output_size / (1024**3):.2f} GB")
    print(f"  Size reduction: {(1 - output_size/input_size)*100:.1f}%")


if __name__ == "__main__":
    main()

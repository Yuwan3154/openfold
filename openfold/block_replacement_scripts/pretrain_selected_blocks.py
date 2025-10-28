#!/usr/bin/env python3
"""
Quick Pre-training Script for Selected Linear Types

This script efficiently pre-trains replacement blocks with more epochs
for only selected linear types (e.g., "full" which performed best).

Features:
- Focus on specific linear types only
- Higher epoch counts for better convergence
- Optimized batch sizes and learning schedules
- Smart resume capability
- Comprehensive progress tracking

Usage:
    # Train full linear type for all blocks with more epochs
    python pretrain_selected_blocks.py \
        --data_dir data/af2rank_single/af2_block_data/ \
        --output_dir AFdistill/pretrained_full \
        --blocks 1 2 3 4 5 6 7 8 9 10 \
        --linear_types full \
        --max_epochs 100 \
        --batch_size 16
        
    # Train all blocks with full linear type
    python pretrain_selected_blocks.py \
        --data_dir data/af2rank_single/af2_block_data/ \
        --output_dir AFdistill/pretrained_all_full \
        --blocks $(seq 1 46) \
        --linear_types full \
        --max_epochs 100
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict
import pandas as pd
import subprocess

# Add openfold to path
sys.path.append(str(Path.home() / 'openfold'))


class SelectedBlockPretrainer:
    """Efficient pre-training for selected linear types"""
    
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
        
        print(f"Selected Block Pre-training Pipeline:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Blocks to train: {args.blocks}")
        print(f"  Selected linear types: {args.linear_types}")
        print(f"  Max epochs: {args.max_epochs}")
        print(f"  Batch size: {args.batch_size}")
        print()
    
    def check_completion(self, block_idx: int, linear_type: str) -> bool:
        """Check if training for this block and linear type is completed"""
        checkpoint_dir = self.output_dir / f"block_{block_idx:02d}" / linear_type
        checkpoint_file = checkpoint_dir / "best_model.ckpt"
        return checkpoint_file.exists()
    
    def get_training_tasks(self) -> List[tuple]:
        """Get list of training tasks (block, linear_type) that need training"""
        tasks = []
        completed_tasks = []
        
        for block_idx in self.args.blocks:
            for linear_type in self.args.linear_types:
                if self.check_completion(block_idx, linear_type):
                    completed_tasks.append((block_idx, linear_type))
                    print(f"  ✅ Block {block_idx:02d}-{linear_type} already completed")
                else:
                    tasks.append((block_idx, linear_type))
        
        print(f"\nTraining Status:")
        print(f"  Already completed: {len(completed_tasks)}")
        print(f"  Need training: {len(tasks)}")
        print()
        
        return tasks
    
    def build_training_command(self, blocks: List[int], linear_types: List[str]) -> List[str]:
        """Build command for training multiple blocks with specific linear types"""
        cmd = [
            sys.executable, "openfold/block_replacement_scripts/train_single_block_replacements.py",
            "--data_dir", str(self.args.data_dir),
            "--output_dir", str(self.args.output_dir),
            "--blocks"] + [str(b) for b in blocks] + [
            "--linear_types"] + linear_types + [
            "--batch_size", str(self.args.batch_size),
            "--max_epochs", str(self.args.max_epochs),
            "--learning_rate", str(self.args.learning_rate),
            "--weight_decay", str(self.args.weight_decay),
            "--num_workers", str(self.args.num_workers),
            "--distributed_backend", str(self.args.distributed_backend)
        ]
        
        if self.args.hidden_dim is not None:
            cmd.extend(["--hidden_dim", str(self.args.hidden_dim)])
        
        # Add force_multi_gpu if set
        if self.args.force_multi_gpu:
            cmd.append("--force_multi_gpu")
        
        # Add wandb if enabled
        if self.args.wandb:
            cmd.extend(["--wandb"])
            cmd.extend(["--wandb_project", self.args.wandb_project])
            cmd.extend(["--wandb_entity", self.args.wandb_entity])
            cmd.extend(["--experiment_name", f"{self.args.experiment_name}_pretraining"])
        
        return cmd
    
    def run_training_batch(self, blocks: List[int], linear_types: List[str]) -> bool:
        """Run training for a batch of blocks and linear types"""
        print(f"🚀 Training blocks {blocks} with linear types {linear_types}")
        
        cmd = self.build_training_command(blocks, linear_types)
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Training batch completed successfully!")
            
            # Show brief summary of stdout
            if result.stdout:
                lines = result.stdout.split('\n')
                summary_lines = [l for l in lines if 'Training Summary' in l or 'Best performing' in l or 'completed' in l]
                for line in summary_lines[-10:]:  # Show last 10 relevant lines
                    print(f"  {line}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Training batch failed with return code {e.returncode}")
            if e.stderr:
                print(f"Error output: {e.stderr[:500]}...")
            return False
    
    def run_optimized_pretraining(self):
        """Run optimized pre-training with batching and smart scheduling"""
        print("=== Starting Optimized Block Pre-training ===")
        
        # Get tasks that need training
        training_tasks = self.get_training_tasks()
        
        if not training_tasks:
            print("🎉 All requested training already completed!")
            return
        
        # Group tasks by linear type for efficient batch processing
        tasks_by_linear_type = {}
        for block_idx, linear_type in training_tasks:
            if linear_type not in tasks_by_linear_type:
                tasks_by_linear_type[linear_type] = []
            tasks_by_linear_type[linear_type].append(block_idx)
        
        print(f"Training plan:")
        for linear_type, blocks in tasks_by_linear_type.items():
            print(f"  {linear_type}: {len(blocks)} blocks - {blocks}")
        print()
        
        # Train each linear type with all its blocks at once
        all_success = True
        start_time = time.time()
        
        for linear_type, blocks in tasks_by_linear_type.items():
            print(f"\n🔧 Training {linear_type} linear type for {len(blocks)} blocks...")
            
            # Split into reasonable batches to avoid memory issues
            batch_size = min(10, len(blocks))  # Process up to 10 blocks at a time
            
            for i in range(0, len(blocks), batch_size):
                batch_blocks = blocks[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(blocks) + batch_size - 1) // batch_size
                
                print(f"  Batch {batch_num}/{total_batches}: blocks {batch_blocks}")
                
                success = self.run_training_batch(batch_blocks, [linear_type])
                if not success:
                    print(f"❌ Failed training batch {batch_num} for {linear_type}")
                    all_success = False
                else:
                    print(f"✅ Completed batch {batch_num}/{total_batches} for {linear_type}")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n=== Pre-training Summary ===")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Overall success: {'✅ Yes' if all_success else '❌ No'}")
        
        # Show final training results if available
        self.show_final_summary()
    
    def show_final_summary(self):
        """Show summary of all completed training"""
        results_file = self.output_dir / "training_summary.csv"
        
        if results_file.exists():
            print(f"\n📊 Final Training Results:")
            try:
                df = pd.read_csv(results_file)
                
                print(f"Total trained models: {len(df)}")
                
                # Summary by linear type
                for linear_type in df['linear_type'].unique():
                    type_data = df[df['linear_type'] == linear_type]
                    mean_loss = type_data['best_val_loss'].mean()
                    std_loss = type_data['best_val_loss'].std()
                    print(f"  {linear_type}: {mean_loss:.6f} ± {std_loss:.6f} ({len(type_data)} blocks)")
                
                # Best models per linear type
                print(f"\nBest models:")
                for linear_type in df['linear_type'].unique():
                    type_data = df[df['linear_type'] == linear_type]
                    best_model = type_data.loc[type_data['best_val_loss'].idxmin()]
                    print(f"  {linear_type}: Block {best_model['block_idx']:02d} "
                          f"(loss: {best_model['best_val_loss']:.6f})")
                
            except Exception as e:
                print(f"Warning: Could not read training summary: {e}")
        
        print(f"\nResults saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick Pre-training for Selected Linear Types",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and output
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing collected block data (relative to home directory)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for trained models (relative to home directory)")
    
    # Training configuration
    parser.add_argument("--blocks", type=int, nargs="+", required=True,
                       help="Block indices to train (e.g., 1 2 3 or $(seq 1 46) for all)")
    parser.add_argument("--linear_types", type=str, nargs="+", 
                       default=["full"],
                       choices=["full", "diagonal", "affine"],
                       help="Linear layer types to train (default: full only)")
    
    # Training parameters - optimized for longer training
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension for replacement blocks")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training (increased for better convergence)")
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Maximum number of training epochs (increased for better convergence)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--distributed_backend", type=str, default="gloo",
                       choices=["nccl", "gloo", "mpi"],
                       help="Distributed training backend")
    parser.add_argument("--force_multi_gpu", action="store_true", default=False,
                       help="Force multi-GPU training even for small datasets")
    
    # Wandb logging arguments
    parser.add_argument("--wandb", action="store_true", default=False,
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="af2distill",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, 
                       default="kryst3154-massachusetts-institute-of-technology",
                       help="Wandb entity (username or team)")
    parser.add_argument("--experiment_name", type=str, default="selected_block_pretraining",
                       help="Base experiment name for wandb logging")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.max_epochs < 50:
        print("Warning: Consider using more epochs (≥50) for better convergence in pre-training")
    
    # Run pre-training
    pretrainer = SelectedBlockPretrainer(args)
    pretrainer.run_optimized_pretraining()


if __name__ == "__main__":
    main()

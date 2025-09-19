#!/usr/bin/env python3
"""
Complete pipeline for training and evaluating Evoformer replacement blocks.

This script orchestrates the entire process:
1. Collect block input/output data
2. Train replacement blocks with different linear types
3. Evaluate replacement blocks in full model
4. Train adaptive weighting model

Usage:
    python run_replacement_pipeline.py --config config.yaml
"""

import argparse
import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any


class ReplacementBlockPipeline:
    """Orchestrates the complete replacement block training and evaluation pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.home_dir = Path.home()
        
        # Setup paths
        self.base_dir = self.home_dir / config['base_dir']
        self.data_dir = self.base_dir / "block_data"
        self.training_dir = self.base_dir / "trained_models"
        self.evaluation_dir = self.base_dir / "evaluation_results"
        self.adaptive_dir = self.base_dir / "adaptive_weighting"
        
        print(f"Pipeline initialized:")
        print(f"  Base directory: {self.base_dir}")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Training directory: {self.training_dir}")
        print(f"  Evaluation directory: {self.evaluation_dir}")
        print(f"  Adaptive directory: {self.adaptive_dir}")
        print()
    
    def run_step_1_data_collection(self):
        """Step 1: Collect input/output pairs for all Evoformer blocks"""
        
        print("=" * 60)
        print("STEP 1: Collecting Block Input/Output Data")
        print("=" * 60)
        
        cmd = [
            "python", "openfold/collect_block_data.py",
            "--csv_path", self.config['csv_path'],
            "--pdb_dir", self.config['pdb_dir'],
            "--weights_path", self.config['weights_path'],
            "--output_dir", str(self.base_dir),
        ]
        
        if self.config.get('max_proteins'):
            cmd.extend(["--max_proteins", str(self.config['max_proteins'])])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        print("Data collection completed successfully!")
        print()
    
    def run_step_2_replacement_training(self):
        """Step 2: Train replacement blocks with different linear types"""
        
        print("=" * 60)
        print("STEP 2: Training Replacement Blocks")
        print("=" * 60)
        
        cmd = [
            "python", "openfold/train_replacement_blocks.py",
            "--data_dir", str(self.base_dir),
            "--output_dir", str(self.training_dir),
            "--hidden_dim", str(self.config.get('hidden_dim', 256)),
            "--batch_size", str(self.config.get('batch_size', 4)),
            "--max_epochs", str(self.config.get('max_epochs', 50)),
            "--learning_rate", str(self.config.get('learning_rate', 1e-3)),
            "--weight_decay", str(self.config.get('weight_decay', 1e-4)),
            "--num_workers", str(self.config.get('num_workers', 4)),
        ]
        
        if self.config.get('test_blocks'):
            cmd.extend(["--test_blocks"] + [str(b) for b in self.config['test_blocks']])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        print("Replacement block training completed successfully!")
        print()
    
    def run_step_3_model_evaluation(self):
        """Step 3: Evaluate replacement blocks in full model"""
        
        print("=" * 60)
        print("STEP 3: Evaluating Replacement Blocks in Full Model")
        print("=" * 60)
        
        cmd = [
            "python", "openfold/evaluate_replacement_blocks.py",
            "--csv_path", self.config['csv_path'],
            "--pdb_dir", self.config['pdb_dir'],
            "--weights_path", self.config['weights_path'],
            "--trained_models_dir", str(self.training_dir),
            "--output_dir", str(self.evaluation_dir),
            "--hidden_dim", str(self.config.get('hidden_dim', 256)),
        ]
        
        if self.config.get('max_proteins'):
            cmd.extend(["--max_proteins", str(self.config['max_proteins'])])
        
        if self.config.get('test_blocks'):
            cmd.extend(["--test_blocks"] + [str(b) for b in self.config['test_blocks']])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        print("Model evaluation completed successfully!")
        print()
    
    def run_step_4_adaptive_training(self):
        """Step 4: Train adaptive weighting model"""
        
        print("=" * 60)
        print("STEP 4: Training Adaptive Weighting Model")
        print("=" * 60)
        
        cmd = [
            "python", "openfold/train_adaptive_weighting.py",
            "--csv_path", self.config['csv_path'],
            "--pdb_dir", self.config['pdb_dir'],
            "--weights_path", self.config['weights_path'],
            "--trained_models_dir", str(self.training_dir),
            "--output_dir", str(self.adaptive_dir),
            "--hidden_dim", str(self.config.get('hidden_dim', 256)),
            "--batch_size", str(self.config.get('adaptive_batch_size', 1)),
            "--max_epochs", str(self.config.get('adaptive_max_epochs', 10)),
            "--learning_rate", str(self.config.get('adaptive_learning_rate', 1e-4)),
            "--weight_decay", str(self.config.get('weight_decay', 1e-4)),
            "--replace_loss_weight", str(self.config.get('replace_loss_weight', 0.1)),
            "--num_workers", str(self.config.get('num_workers', 2)),
        ]
        
        if self.config.get('max_proteins'):
            cmd.extend(["--max_proteins", str(self.config['max_proteins'])])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        print("Adaptive weighting training completed successfully!")
        print()
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        
        print("=" * 70)
        print("EVOFORMER REPLACEMENT BLOCK TRAINING PIPELINE")
        print("=" * 70)
        print()
        
        steps = self.config.get('steps', [1, 2, 3, 4])
        
        if 1 in steps:
            self.run_step_1_data_collection()
        
        if 2 in steps:
            self.run_step_2_replacement_training()
        
        if 3 in steps:
            self.run_step_3_model_evaluation()
        
        if 4 in steps:
            self.run_step_4_adaptive_training()
        
        print("=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print("Results summary:")
        print(f"  Block data: {self.data_dir}")
        print(f"  Trained models: {self.training_dir}")
        print(f"  Evaluation results: {self.evaluation_dir}")
        print(f"  Adaptive weighting: {self.adaptive_dir}")
        print()
        
        # Print next steps
        print("Next steps:")
        print("1. Review training results in:")
        print(f"   {self.training_dir}/training_summary.csv")
        print("2. Review evaluation results in:")
        print(f"   {self.evaluation_dir}/analysis/evaluation_results.csv")
        print("3. Review adaptive weighting results in:")
        print(f"   {self.adaptive_dir}/logs/")
        print()


def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    
    return {
        # Required paths (relative to home directory)
        'csv_path': 'data/af2rank_single/af2rank_single_set_single_tms_07.csv',
        'pdb_dir': 'data/af2rank_single/pdb',
        'weights_path': 'params/params_model_2_ptm.npz',
        'base_dir': 'replacement_block_pipeline',
        
        # Training parameters
        'hidden_dim': 256,
        'batch_size': 4,
        'max_epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_workers': 4,
        
        # Adaptive weighting parameters
        'adaptive_batch_size': 1,
        'adaptive_max_epochs': 10,
        'adaptive_learning_rate': 1e-4,
        'replace_loss_weight': 0.1,
        
        # Optional limitations for testing
        'max_proteins': None,  # Set to small number for testing
        'test_blocks': None,   # Set to [23, 24] for testing specific blocks
        
        # Which steps to run
        'steps': [1, 2, 3, 4],  # Run all steps
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run complete Evoformer replacement block pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (will create default if not provided)"
    )
    parser.add_argument(
        "--create_config", action="store_true",
        help="Create default config file and exit"
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=None,
        help="Which steps to run (1=data collection, 2=training, 3=evaluation, 4=adaptive)"
    )
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_default_config()
        config_path = "replacement_pipeline_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Created default config file: {config_path}")
        print("Edit this file to customize the pipeline, then run:")
        print(f"python {sys.argv[0]} --config {config_path}")
        return
    
    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("Using default configuration (no config file provided)")
        config = create_default_config()
    
    # Override steps if specified
    if args.steps:
        config['steps'] = args.steps
    
    # Run pipeline
    pipeline = ReplacementBlockPipeline(config)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()

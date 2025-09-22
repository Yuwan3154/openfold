#!/usr/bin/env python3
"""
Testing Script for Replacement Linear Types

This script automates testing of the three different linear types (full, diagonal, affine)
using the Part 1 script (train_single_block_replacements.py).

It runs comprehensive tests and generates comparison reports to help identify the best
linear type configuration for replacement blocks.

Usage:
    python test_replacement_linear_types.py \
        --data_dir path/to/block_data \
        --output_dir path/to/output \
        --test_blocks 1 2 3 \
        --wandb --wandb_project my_project
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LinearTypeComparison:
    """Automates testing and comparison of different linear types"""
    
    def __init__(self, args):
        self.args = args
        self.home_dir = Path.home()
        self.data_dir = self.home_dir / args.data_dir
        self.output_dir = self.home_dir / args.output_dir
        self.results_dir = self.output_dir / "comparison_results"
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Linear types to test
        self.linear_types = ["full", "diagonal", "affine"]
        
        print(f"Linear Type Testing Pipeline:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Test blocks: {args.test_blocks}")
        print(f"  Linear types: {self.linear_types}")
        print()
    
    def run_single_block_training(self, blocks: List[int], linear_types: List[str]) -> bool:
        """Run the single block replacement training script"""
        
        print(f"Running single block training for blocks {blocks} with linear types {linear_types}...")
        
        # Build command
        cmd = [
            sys.executable, "openfold/train_single_block_replacements.py",
            "--data_dir", str(self.args.data_dir),
            "--output_dir", str(self.args.output_dir),
            "--blocks"] + [str(b) for b in blocks] + [
            "--linear_types"] + linear_types + [
            "--hidden_dim", str(self.args.hidden_dim),
            "--batch_size", str(self.args.batch_size),
            "--max_epochs", str(self.args.max_epochs),
            "--learning_rate", str(self.args.learning_rate),
            "--weight_decay", str(self.args.weight_decay),
            "--num_workers", str(self.args.num_workers)
        ]
        
        # Add wandb if enabled
        if self.args.wandb:
            cmd.extend(["--wandb"])
            cmd.extend(["--wandb_project", self.args.wandb_project])
            cmd.extend(["--wandb_entity", self.args.wandb_entity])
            cmd.extend(["--experiment_name", f"{self.args.experiment_name}_linear_comparison"])
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Training completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Training failed with return code {e.returncode}")
            print(f"Error output: {e.stderr}")
            return False
    
    def load_training_results(self) -> Optional[pd.DataFrame]:
        """Load training results from the single block training"""
        
        results_file = self.output_dir / "training_summary.csv"
        
        if not results_file.exists():
            print(f"No training results found at {results_file}")
            return None
        
        df = pd.read_csv(results_file)
        print(f"Loaded {len(df)} training results")
        return df
    
    def analyze_linear_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance of different linear types"""
        
        print("\n=== Linear Type Analysis ===")
        
        analysis = {}
        
        # Overall statistics by linear type
        linear_stats = df.groupby('linear_type').agg({
            'best_val_loss': ['mean', 'std', 'min', 'max', 'count'],
            'epochs_trained': ['mean', 'std'],
            'train_samples': 'first',
            'val_samples': 'first'
        }).round(6)
        
        analysis['linear_type_stats'] = linear_stats
        
        print("Performance by Linear Type:")
        for linear_type in self.linear_types:
            type_data = df[df['linear_type'] == linear_type]
            if len(type_data) > 0:
                mean_loss = type_data['best_val_loss'].mean()
                std_loss = type_data['best_val_loss'].std()
                count = len(type_data)
                print(f"  {linear_type:>8s}: {mean_loss:.6f} ± {std_loss:.6f} ({count} blocks)")
        
        # Best model per block
        print("\nBest Linear Type per Block:")
        best_per_block = {}
        for block_idx in df['block_idx'].unique():
            block_data = df[df['block_idx'] == block_idx]
            best_model = block_data.loc[block_data['best_val_loss'].idxmin()]
            best_per_block[block_idx] = {
                'linear_type': best_model['linear_type'],
                'loss': best_model['best_val_loss']
            }
            print(f"  Block {block_idx:02d}: {best_model['linear_type']} (loss: {best_model['best_val_loss']:.6f})")
        
        analysis['best_per_block'] = best_per_block
        
        # Overall best linear type (most frequently best)
        best_counts = pd.Series([v['linear_type'] for v in best_per_block.values()]).value_counts()
        overall_best = best_counts.index[0]
        analysis['overall_best_linear_type'] = overall_best
        
        print(f"\nOverall Best Linear Type: {overall_best}")
        print("Frequency of being best:")
        for linear_type, count in best_counts.items():
            print(f"  {linear_type}: {count} blocks ({count/len(best_per_block)*100:.1f}%)")
        
        # Convergence analysis
        print("\nConvergence Analysis:")
        convergence_stats = df.groupby('linear_type')['epochs_trained'].agg(['mean', 'std']).round(2)
        for linear_type, stats in convergence_stats.iterrows():
            print(f"  {linear_type:>8s}: {stats['mean']} ± {stats['std']} epochs")
        
        analysis['convergence_stats'] = convergence_stats
        
        return analysis
    
    def create_visualizations(self, df: pd.DataFrame, analysis: Dict[str, Any]):
        """Create comparison visualizations"""
        
        print("\nCreating comparison visualizations...")
        
        # Set up plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Linear Type Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Box plot of validation losses
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='linear_type', y='best_val_loss', ax=ax1)
        ax1.set_title('Validation Loss Distribution by Linear Type')
        ax1.set_xlabel('Linear Type')
        ax1.set_ylabel('Best Validation Loss')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Bar plot of mean losses with error bars
        ax2 = axes[0, 1]
        linear_stats = df.groupby('linear_type')['best_val_loss'].agg(['mean', 'std'])
        linear_stats.plot(kind='bar', y='mean', yerr='std', ax=ax2, legend=False)
        ax2.set_title('Mean Validation Loss by Linear Type')
        ax2.set_xlabel('Linear Type')
        ax2.set_ylabel('Mean Best Validation Loss')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Convergence comparison
        ax3 = axes[1, 0]
        convergence_stats = df.groupby('linear_type')['epochs_trained'].agg(['mean', 'std'])
        convergence_stats.plot(kind='bar', y='mean', yerr='std', ax=ax3, legend=False)
        ax3.set_title('Training Convergence by Linear Type')
        ax3.set_xlabel('Linear Type')
        ax3.set_ylabel('Mean Epochs to Convergence')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Heatmap of performance per block
        ax4 = axes[1, 1]
        pivot_data = df.pivot(index='block_idx', columns='linear_type', values='best_val_loss')
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis_r', ax=ax4)
        ax4.set_title('Validation Loss Heatmap\n(Block vs Linear Type)')
        ax4.set_xlabel('Linear Type')
        ax4.set_ylabel('Block Index')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "linear_type_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: {plot_path}")
        
        plt.close()
        
        # Create additional detailed plots if many blocks
        if len(df['block_idx'].unique()) > 5:
            self._create_detailed_block_comparison(df)
    
    def _create_detailed_block_comparison(self, df: pd.DataFrame):
        """Create detailed per-block comparison"""
        
        blocks = sorted(df['block_idx'].unique())
        n_blocks = len(blocks)
        
        # Create grid for per-block comparison
        n_cols = min(4, n_blocks)
        n_rows = (n_blocks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle('Per-Block Linear Type Comparison', fontsize=16, fontweight='bold')
        
        for i, block_idx in enumerate(blocks):
            ax = axes[i]
            block_data = df[df['block_idx'] == block_idx]
            
            # Bar plot for this block
            block_stats = block_data.set_index('linear_type')['best_val_loss']
            block_stats.plot(kind='bar', ax=ax)
            ax.set_title(f'Block {block_idx:02d}')
            ax.set_ylabel('Validation Loss')
            ax.tick_params(axis='x', rotation=45)
            
            # Highlight best
            best_idx = block_stats.idxmin()
            best_val = block_stats.min()
            ax.axhline(y=best_val, color='red', linestyle='--', alpha=0.7)
            ax.text(0.5, 0.95, f'Best: {best_idx}', transform=ax.transAxes, 
                   ha='center', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Hide unused subplots
        for i in range(n_blocks, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save detailed plot
        plot_path = self.results_dir / "detailed_block_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved detailed comparison plot: {plot_path}")
        
        plt.close()
    
    def generate_report(self, analysis: Dict[str, Any]):
        """Generate comprehensive comparison report"""
        
        print("\nGenerating comparison report...")
        
        report_path = self.results_dir / "linear_type_comparison_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Linear Type Comparison Report\n\n")
            f.write(f"**Test Configuration:**\n")
            f.write(f"- Test blocks: {self.args.test_blocks}\n")
            f.write(f"- Linear types tested: {self.linear_types}\n")
            f.write(f"- Max epochs: {self.args.max_epochs}\n")
            f.write(f"- Learning rate: {self.args.learning_rate}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"**Overall Best Linear Type: {analysis['overall_best_linear_type']}**\n\n")
            
            f.write("### Performance Statistics\n\n")
            f.write("| Linear Type | Mean Loss | Std Loss | Count | Mean Epochs |\n")
            f.write("|-------------|-----------|----------|-------|-------------|\n")
            
            # Get stats from analysis
            for linear_type in self.linear_types:
                stats = analysis.get('linear_type_stats', {})
                conv_stats = analysis.get('convergence_stats', {})
                if hasattr(stats, 'loc') and linear_type in stats.index:
                    mean_loss = stats.loc[linear_type, ('best_val_loss', 'mean')]
                    std_loss = stats.loc[linear_type, ('best_val_loss', 'std')]
                    count = stats.loc[linear_type, ('best_val_loss', 'count')]
                    if hasattr(conv_stats, 'loc') and linear_type in conv_stats.index:
                        mean_epochs = conv_stats.loc[linear_type, 'mean']
                    else:
                        mean_epochs = "N/A"
                    f.write(f"| {linear_type} | {mean_loss:.6f} | {std_loss:.6f} | {count} | {mean_epochs} |\n")
            
            f.write("\n### Best Linear Type per Block\n\n")
            f.write("| Block | Best Linear Type | Validation Loss |\n")
            f.write("|-------|------------------|------------------|\n")
            
            for block_idx, info in analysis['best_per_block'].items():
                f.write(f"| {block_idx:02d} | {info['linear_type']} | {info['loss']:.6f} |\n")
            
            f.write("\n### Recommendations\n\n")
            best_type = analysis['overall_best_linear_type']
            f.write(f"Based on the analysis, **{best_type}** linear type shows the best overall performance.\n")
            f.write("This recommendation is based on:\n")
            f.write("1. Frequency of being the best performer across blocks\n")
            f.write("2. Mean validation loss across all blocks\n")
            f.write("3. Training convergence characteristics\n\n")
            
            f.write("### Usage for Part 2\n\n")
            f.write("For the adaptive weighting training (Part 2), use the following configuration:\n\n")
            f.write("```yaml\n")
            f.write(f"linear_type: '{best_type}'\n")
            f.write("```\n\n")
            f.write("This will initialize all replacement blocks with the best performing linear type.\n")
        
        print(f"Saved comparison report: {report_path}")
        
        # Save analysis as JSON
        json_path = self.results_dir / "analysis_results.json"
        
        # Convert non-serializable objects to serializable format
        serializable_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, pd.DataFrame):
                serializable_analysis[key] = value.to_dict()
            elif isinstance(value, pd.Series):
                serializable_analysis[key] = value.to_dict()
            else:
                serializable_analysis[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2, default=str)
        
        print(f"Saved analysis data: {json_path}")
    
    def run_complete_analysis(self):
        """Run the complete linear type comparison analysis"""
        
        print("=== Starting Linear Type Comparison Analysis ===")
        print()
        
        # Step 1: Run training for all linear types
        success = self.run_single_block_training(
            blocks=self.args.test_blocks,
            linear_types=self.linear_types
        )
        
        if not success:
            print("Training failed. Aborting analysis.")
            return
        
        # Step 2: Load and analyze results
        df = self.load_training_results()
        if df is None:
            print("No training results to analyze. Aborting.")
            return
        
        # Step 3: Perform analysis
        analysis = self.analyze_linear_types(df)
        
        # Step 4: Create visualizations
        try:
            self.create_visualizations(df, analysis)
        except Exception as e:
            print(f"Warning: Failed to create visualizations: {e}")
        
        # Step 5: Generate report
        self.generate_report(analysis)
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved to: {self.results_dir}")
        print(f"Recommended linear type for Part 2: {analysis['overall_best_linear_type']}")


def main():
    parser = argparse.ArgumentParser(
        description="Test and compare different linear types for replacement blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and output
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing collected block data (relative to home directory)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for testing results (relative to home directory)")
    
    # Testing configuration
    parser.add_argument("--test_blocks", type=int, nargs="+", required=True,
                       help="Block indices to test (e.g., 1 2 3)")
    
    # Training parameters (passed to single block training)
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension for replacement blocks")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=20,
                       help="Maximum number of training epochs (reduced for testing)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Wandb logging arguments
    parser.add_argument("--wandb", action="store_true", default=False,
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="af2distill",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, 
                       default="kryst3154-massachusetts-institute-of-technology",
                       help="Wandb entity (username or team)")
    parser.add_argument("--experiment_name", type=str, default="linear_type_test",
                       help="Base experiment name for wandb logging")
    
    args = parser.parse_args()
    
    # Run analysis
    comparison = LinearTypeComparison(args)
    comparison.run_complete_analysis()


if __name__ == "__main__":
    main()

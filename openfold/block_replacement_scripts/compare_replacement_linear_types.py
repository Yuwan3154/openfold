#!/usr/bin/env python3
"""
Comprehensive Linear Type Analysis for Replacement Blocks

This script automates testing of the three different linear types (full, diagonal, affine)
across ALL Evoformer blocks (1-46) using statistical analysis to determine the best
performing linear type for adaptive training.

Features:
- Tests all three linear types on all blocks (1-46 by default)
- Comprehensive statistical analysis (ANOVA, t-tests, effect sizes)
- Multi-criteria ranking (frequency, rank, normalized performance)
- Detailed report generation with recommendations
- Visualizations and consistency analysis

Usage:
    # Test all blocks with all linear types (recommended)
    python test_replacement_linear_types.py \
        --data_dir replacement_block_pipeline \
        --output_dir linear_type_analysis \
        --wandb --wandb_project af2distill
    
    # Test specific blocks only
    python test_replacement_linear_types.py \
        --data_dir replacement_block_pipeline \
        --output_dir linear_type_analysis \
        --test_blocks 1 2 3 5 10
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
from scipy import stats
from scipy.stats import f_oneway, ttest_ind


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
            sys.executable, "openfold/block_replacement_scripts/train_single_block_replacements.py",
            "--data_dir", str(self.args.data_dir),
            "--output_dir", str(self.args.output_dir),
            "--blocks"] + [str(b) for b in blocks] + [
            "--linear_types"] + linear_types + [
            "--hidden_dim", str(self.args.hidden_dim),
            "--batch_size", str(self.args.batch_size),
            "--max_epochs", str(self.args.max_epochs),
            "--learning_rate", str(self.args.learning_rate),
            "--weight_decay", str(self.args.weight_decay),
            "--num_workers", str(self.args.num_workers),
            "--distributed_backend", str(self.args.distributed_backend)
        ]
        
        # Add force_multi_gpu flag if set
        if self.args.force_multi_gpu:
            cmd.append("--force_multi_gpu")
        
        # Add wandb if enabled
        if self.args.wandb:
            cmd.extend(["--wandb"])
            cmd.extend(["--wandb_project", self.args.wandb_project])
            cmd.extend(["--wandb_entity", self.args.wandb_entity])
            cmd.extend(["--experiment_name", f"{self.args.experiment_name}_linear_comparison"])
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
            return True
    
    def load_training_results(self) -> Optional[pd.DataFrame]:
        """Load training results from the single block training"""
        
        results_file = self.output_dir / "training_summary.csv"
        
        if not results_file.exists():
            print(f"No training results found at {results_file}")
            return None
        
        df = pd.read_csv(results_file)
        print(f"Loaded {len(df)} training results")
        print(f"Linear types found: {sorted(df['linear_type'].unique())}")
        return df
    
    def analyze_linear_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive statistical analysis of linear type performance"""
        
        print("\n=== Comprehensive Linear Type Analysis ===")
        
        analysis = {}
        
        # Basic statistics by linear type
        linear_stats = df.groupby('linear_type').agg({
            'best_val_loss': ['mean', 'std', 'min', 'max', 'count', 'median'],
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
                median_loss = type_data['best_val_loss'].median()
                count = len(type_data)
                print(f"  {linear_type:>8s}: {mean_loss:.6f} ± {std_loss:.6f} (median: {median_loss:.6f}, n={count})")
        
        # Statistical significance testing
        print("\n=== Statistical Analysis ===")
        analysis['statistical_tests'] = self._perform_statistical_tests(df)
        
        # Best model per block analysis  
        print("\nBest Linear Type per Block:")
        best_per_block = {}
        relative_performance = {}
        
        for block_idx in sorted(df['block_idx'].unique()):
            block_data = df[df['block_idx'] == block_idx]
            best_model = block_data.loc[block_data['best_val_loss'].idxmin()]
            worst_model = block_data.loc[block_data['best_val_loss'].idxmax()]
            
            # Calculate relative improvement
            improvement = (worst_model['best_val_loss'] - best_model['best_val_loss']) / worst_model['best_val_loss'] * 100
            
            best_per_block[block_idx] = {
                'linear_type': best_model['linear_type'],
                'loss': best_model['best_val_loss'],
                'improvement_percent': improvement
            }
            
            # Store all performances for this block
            relative_performance[block_idx] = {
                row['linear_type']: row['best_val_loss'] 
                for _, row in block_data.iterrows()
            }
            
            print(f"  Block {block_idx:02d}: {best_model['linear_type']} (loss: {best_model['best_val_loss']:.6f}, "
                  f"{improvement:.1f}% better than worst)")
        
        analysis['best_per_block'] = best_per_block
        analysis['relative_performance'] = relative_performance
        
        # Statistical ranking of linear types
        print("\n=== Linear Type Ranking ===")
        ranking_analysis = self._compute_linear_type_ranking(df)
        analysis.update(ranking_analysis)
        
        # Convergence analysis
        print("\nConvergence Analysis:")
        convergence_stats = df.groupby('linear_type')['epochs_trained'].agg(['mean', 'std']).round(2)
        for linear_type, stats in convergence_stats.iterrows():
            print(f"  {linear_type:>8s}: {stats['mean']} ± {stats['std']} epochs")
        
        analysis['convergence_stats'] = convergence_stats
        
        # Block-wise consistency analysis
        print("\n=== Consistency Analysis ===")
        consistency_analysis = self._analyze_consistency(df)
        analysis['consistency_analysis'] = consistency_analysis
        
        return analysis
    
    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive statistical tests"""
        
        test_results = {}
        
        # Extract data for each linear type
        linear_data = {}
        for linear_type in self.linear_types:
            linear_data[linear_type] = df[df['linear_type'] == linear_type]['best_val_loss'].values
        
        # ANOVA test
        if len(self.linear_types) > 2:
            f_stat, p_value = f_oneway(*linear_data.values())
            test_results['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': 'Significant differences between linear types' if p_value < 0.05 
                               else 'No significant differences between linear types'
            }
            print(f"ANOVA Test: F={f_stat:.4f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        
        # Pairwise t-tests and effect sizes
        print("\nPairwise Comparisons:")
        pairwise_tests = {}
        
        for i, type1 in enumerate(self.linear_types):
            for j, type2 in enumerate(self.linear_types):
                if i < j:  # Avoid duplicate comparisons
                    data1 = linear_data[type1]
                    data2 = linear_data[type2]
                    
                    # T-test
                    t_stat, p_val = ttest_ind(data1, data2)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                         (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                        (len(data1) + len(data2) - 2))
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                    
                    # Effect size interpretation
                    if abs(cohens_d) < 0.2:
                        effect_size = "negligible"
                    elif abs(cohens_d) < 0.5:
                        effect_size = "small"
                    elif abs(cohens_d) < 0.8:
                        effect_size = "medium"
                    else:
                        effect_size = "large"
                    
                    comparison_key = f"{type1}_vs_{type2}"
                    pairwise_tests[comparison_key] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'cohens_d': float(cohens_d),
                        'effect_size': effect_size,
                        'significant': p_val < 0.05,
                        'better_type': type1 if np.mean(data1) < np.mean(data2) else type2
                    }
                    
                    significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                    better = type1 if np.mean(data1) < np.mean(data2) else type2
                    print(f"  {type1} vs {type2}: t={t_stat:.3f}, p={p_val:.6f} {significance}, "
                          f"d={cohens_d:.3f} ({effect_size}), better: {better}")
        
        test_results['pairwise_tests'] = pairwise_tests
        return test_results
    
    def _compute_linear_type_ranking(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive ranking of linear types"""
        
        ranking_results = {}
        
        # Method 1: Frequency of being best
        best_counts = pd.Series([
            df[df['block_idx'] == block]['best_val_loss'].idxmin() 
            for block in df['block_idx'].unique()
        ]).map(lambda idx: df.loc[idx, 'linear_type']).value_counts()
        
        ranking_results['best_frequency'] = best_counts.to_dict()
        
        # Method 2: Average rank per block
        rank_data = {}
        for linear_type in self.linear_types:
            ranks = []
            for block_idx in df['block_idx'].unique():
                block_data = df[df['block_idx'] == block_idx].sort_values('best_val_loss')
                block_data_reset = block_data.reset_index(drop=True)
                type_mask = block_data_reset['linear_type'] == linear_type
                type_indices = block_data_reset[type_mask].index
                
                if len(type_indices) > 0:
                    rank = type_indices[0] + 1  # 1-based ranking
                    ranks.append(rank)
                else:
                    # Linear type not present for this block (training failed)
                    # Assign worst possible rank (number of types + 1)
                    max_rank = len(block_data) + 1
                    ranks.append(max_rank)
                    print(f"⚠️  {linear_type} not found for block {block_idx}, assigning worst rank ({max_rank})")
            
            rank_data[linear_type] = {
                'mean_rank': np.mean(ranks),
                'std_rank': np.std(ranks),
                'ranks': ranks
            }
        
        ranking_results['average_ranks'] = rank_data
        
        # Method 3: Score-based ranking (inverse of normalized loss)
        score_data = {}
        for linear_type in self.linear_types:
            scores = []
            for block_idx in df['block_idx'].unique():
                block_data = df[df['block_idx'] == block_idx]
                type_data = block_data[block_data['linear_type'] == linear_type]['best_val_loss']
                
                if len(type_data) > 0:
                    type_loss = type_data.iloc[0]
                    min_loss = block_data['best_val_loss'].min()
                    max_loss = block_data['best_val_loss'].max()
                    
                    # Normalized score (1 = best, 0 = worst)
                    if max_loss > min_loss:
                        score = 1 - (type_loss - min_loss) / (max_loss - min_loss)
                    else:
                        score = 1.0  # All equal
                else:
                    # Linear type not present for this block (training failed)
                    # Assign worst possible score (0)
                    score = 0.0
                    print(f"⚠️  {linear_type} not found for block {block_idx}, assigning worst score (0.0)")
                
                scores.append(score)
            
            score_data[linear_type] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores
            }
        
        ranking_results['normalized_scores'] = score_data
        
        # Overall recommendation based on multiple criteria
        criteria_scores = {}
        for linear_type in self.linear_types:
            # Frequency score (higher is better)
            freq_score = best_counts.get(linear_type, 0) / len(df['block_idx'].unique())
            
            # Rank score (lower rank is better, so invert)
            rank_score = (4 - rank_data[linear_type]['mean_rank']) / 3  # Normalize to 0-1
            
            # Normalized score (higher is better)
            norm_score = score_data[linear_type]['mean_score']
            
            # Combined score (equal weighting)
            combined_score = (freq_score + rank_score + norm_score) / 3
            
            criteria_scores[linear_type] = {
                'frequency_score': freq_score,
                'rank_score': rank_score,
                'normalized_score': norm_score,
                'combined_score': combined_score
            }
        
        # Final recommendation
        best_linear_type = max(criteria_scores.keys(), key=lambda x: criteria_scores[x]['combined_score'])
        ranking_results['criteria_scores'] = criteria_scores
        ranking_results['recommended_linear_type'] = best_linear_type
        
        print(f"Ranking Analysis:")
        print(f"  Best frequency: {dict(best_counts)}")
        avg_ranks_str = [(k, f'{v["mean_rank"]:.2f}') for k, v in rank_data.items()]
        print(f"  Average ranks: {avg_ranks_str}")
        combined_scores_str = [(k, f'{v["combined_score"]:.3f}') for k, v in criteria_scores.items()]
        print(f"  Combined scores: {combined_scores_str}")
        print(f"  Recommended: {best_linear_type}")
        
        return ranking_results
    
    def _analyze_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consistency of linear type performance across blocks"""
        
        consistency_results = {}
        
        # Coefficient of variation for each linear type
        cv_data = {}
        for linear_type in self.linear_types:
            type_data = df[df['linear_type'] == linear_type]['best_val_loss']
            cv = type_data.std() / type_data.mean()
            cv_data[linear_type] = {
                'coefficient_of_variation': cv,
                'interpretation': 'consistent' if cv < 0.1 else 'moderately_consistent' if cv < 0.2 else 'variable'
            }
        
        consistency_results['coefficient_of_variation'] = cv_data
        
        # Win rate analysis (how often each type wins)
        win_rates = {}
        total_blocks = len(df['block_idx'].unique())
        
        for linear_type in self.linear_types:
            wins = 0
            for block_idx in df['block_idx'].unique():
                block_data = df[df['block_idx'] == block_idx]
                best_idx = block_data['best_val_loss'].idxmin()
                if df.loc[best_idx, 'linear_type'] == linear_type:
                    wins += 1
            
            win_rates[linear_type] = {
                'wins': wins,
                'win_rate': wins / total_blocks,
                'confidence_level': 'high' if wins / total_blocks > 0.5 else 'medium' if wins / total_blocks > 0.3 else 'low'
            }
        
        consistency_results['win_rates'] = win_rates
        
        print(f"Consistency Analysis:")
        for linear_type in self.linear_types:
            cv = cv_data[linear_type]['coefficient_of_variation']
            cv_interp = cv_data[linear_type]['interpretation']
            win_rate = win_rates[linear_type]['win_rate']
            confidence = win_rates[linear_type]['confidence_level']
            print(f"  {linear_type}: CV={cv:.3f} ({cv_interp}), "
                  f"Win rate={win_rate:.1%} ({confidence} confidence)")
        
        return consistency_results
    
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
        """Generate comprehensive comparison report with statistical analysis"""
        
        print("\nGenerating comprehensive comparison report...")
        
        report_path = self.results_dir / "comprehensive_linear_type_analysis.md"
        
        with open(report_path, 'w') as f:
            f.write("# Comprehensive Linear Type Analysis Report\n\n")
            f.write(f"**Test Configuration:**\n")
            f.write(f"- Test blocks: {len(self.args.test_blocks)} blocks ({min(self.args.test_blocks)}-{max(self.args.test_blocks)})\n")
            f.write(f"- Linear types tested: {', '.join(self.linear_types)}\n")
            f.write(f"- Max epochs: {self.args.max_epochs}\n")
            f.write(f"- Learning rate: {self.args.learning_rate}\n")
            f.write(f"- Batch size: {self.args.batch_size}\n\n")
            
            # Executive Summary
            recommended_type = analysis.get('recommended_linear_type', analysis.get('overall_best_linear_type', 'full'))
            f.write("## 🎯 Executive Summary\n\n")
            f.write(f"**Recommended Linear Type: `{recommended_type}`**\n\n")
            
            if 'statistical_tests' in analysis and 'anova' in analysis['statistical_tests']:
                anova = analysis['statistical_tests']['anova']
                if anova['significant']:
                    f.write(f"✅ **Statistically significant differences found** (ANOVA p={anova['p_value']:.6f})\n\n")
                else:
                    f.write(f"⚠️ **No statistically significant differences found** (ANOVA p={anova['p_value']:.6f})\n\n")
            
            # Detailed Performance Statistics
            f.write("## 📊 Performance Statistics\n\n")
            f.write("| Linear Type | Mean Loss | Median Loss | Std Loss | Min Loss | Max Loss | Count |\n")
            f.write("|-------------|-----------|-------------|----------|----------|----------|-------|\n")
            
            for linear_type in self.linear_types:
                stats = analysis.get('linear_type_stats', {})
                if hasattr(stats, 'loc') and linear_type in stats.index:
                    mean_loss = stats.loc[linear_type, ('best_val_loss', 'mean')]
                    median_loss = stats.loc[linear_type, ('best_val_loss', 'median')]
                    std_loss = stats.loc[linear_type, ('best_val_loss', 'std')]
                    min_loss = stats.loc[linear_type, ('best_val_loss', 'min')]
                    max_loss = stats.loc[linear_type, ('best_val_loss', 'max')]
                    count = stats.loc[linear_type, ('best_val_loss', 'count')]
                    f.write(f"| {linear_type} | {mean_loss:.6f} | {median_loss:.6f} | {std_loss:.6f} | {min_loss:.6f} | {max_loss:.6f} | {count} |\n")
            
            # Statistical Analysis
            f.write("\n## 🔬 Statistical Analysis\n\n")
            
            if 'statistical_tests' in analysis:
                stats_tests = analysis['statistical_tests']
                
                # ANOVA Results
                if 'anova' in stats_tests:
                    anova = stats_tests['anova']
                    f.write("### ANOVA Test\n\n")
                    f.write(f"- **F-statistic:** {anova['f_statistic']:.4f}\n")
                    f.write(f"- **p-value:** {anova['p_value']:.6f}\n")
                    f.write(f"- **Significant:** {'Yes' if anova['significant'] else 'No'}\n")
                    f.write(f"- **Interpretation:** {anova['interpretation']}\n\n")
                
                # Pairwise Comparisons
                if 'pairwise_tests' in stats_tests:
                    f.write("### Pairwise Comparisons\n\n")
                    f.write("| Comparison | t-statistic | p-value | Effect Size (Cohen's d) | Magnitude | Better Type |\n")
                    f.write("|------------|-------------|---------|-------------------------|-----------|-------------|\n")
                    
                    for comparison, results in stats_tests['pairwise_tests'].items():
                        sig_marker = "***" if results['p_value'] < 0.001 else "**" if results['p_value'] < 0.01 else "*" if results['p_value'] < 0.05 else ""
                        f.write(f"| {comparison.replace('_vs_', ' vs ')} | {results['t_statistic']:.3f} | {results['p_value']:.6f}{sig_marker} | {results['cohens_d']:.3f} | {results['effect_size']} | {results['better_type']} |\n")
                    
                    f.write("\n*Significance levels: *** p<0.001, ** p<0.01, * p<0.05*\n\n")
            
            # Ranking Analysis
            f.write("## 🏆 Comprehensive Ranking\n\n")
            
            if 'criteria_scores' in analysis:
                f.write("### Multi-Criteria Scoring\n\n")
                f.write("| Linear Type | Frequency Score | Rank Score | Normalized Score | **Combined Score** |\n")
                f.write("|-------------|-----------------|------------|------------------|--------------------|\n")
                
                for linear_type in self.linear_types:
                    scores = analysis['criteria_scores'][linear_type]
                    f.write(f"| {linear_type} | {scores['frequency_score']:.3f} | {scores['rank_score']:.3f} | {scores['normalized_score']:.3f} | **{scores['combined_score']:.3f}** |\n")
                
                f.write("\n")
            
            if 'best_frequency' in analysis:
                f.write("### Win Rate Analysis\n\n")
                total_blocks = len(analysis['best_per_block'])
                f.write("| Linear Type | Wins | Win Rate | Confidence |\n")
                f.write("|-------------|------|----------|------------|\n")
                
                for linear_type in self.linear_types:
                    wins = analysis['best_frequency'].get(linear_type, 0)
                    win_rate = wins / total_blocks
                    confidence = 'High' if win_rate > 0.5 else 'Medium' if win_rate > 0.3 else 'Low'
                    f.write(f"| {linear_type} | {wins} | {win_rate:.1%} | {confidence} |\n")
                
                f.write("\n")
            
            # Consistency Analysis
            if 'consistency_analysis' in analysis:
                f.write("## 📈 Consistency Analysis\n\n")
                cv_data = analysis['consistency_analysis']['coefficient_of_variation']
                
                f.write("| Linear Type | Coefficient of Variation | Interpretation |\n")
                f.write("|-------------|--------------------------|----------------|\n")
                
                for linear_type in self.linear_types:
                    cv = cv_data[linear_type]['coefficient_of_variation']
                    interp = cv_data[linear_type]['interpretation']
                    f.write(f"| {linear_type} | {cv:.3f} | {interp} |\n")
                
                f.write("\n*Lower CV indicates more consistent performance across blocks*\n\n")
            
            # Block-by-Block Results
            f.write("## 🔍 Block-by-Block Analysis\n\n")
            f.write("| Block | Best Type | Loss | Improvement vs Worst | All Results |\n")
            f.write("|-------|-----------|------|---------------------|-------------|\n")
            
            for block_idx in sorted(analysis['best_per_block'].keys()):
                best_info = analysis['best_per_block'][block_idx]
                block_results = analysis['relative_performance'][block_idx]
                
                # Format all results for this block
                all_results = ", ".join([f"{k}:{v:.4f}" for k, v in sorted(block_results.items())])
                
                f.write(f"| {block_idx:02d} | {best_info['linear_type']} | {best_info['loss']:.6f} | {best_info['improvement_percent']:.1f}% | {all_results} |\n")
            
            # Training Efficiency
            f.write("\n## ⏱️ Training Efficiency\n\n")
            f.write("| Linear Type | Mean Epochs | Std Epochs | Convergence Quality |\n")
            f.write("|-------------|-------------|------------|---------------------|\n")
            
            if 'convergence_stats' in analysis:
                for linear_type, stats in analysis['convergence_stats'].iterrows():
                    quality = "Fast" if stats['mean'] < self.args.max_epochs * 0.6 else "Normal" if stats['mean'] < self.args.max_epochs * 0.8 else "Slow"
                    f.write(f"| {linear_type} | {stats['mean']:.1f} | {stats['std']:.1f} | {quality} |\n")
            
            # Recommendations and Next Steps
            f.write(f"\n## 🎯 Recommendations\n\n")
            f.write(f"### Primary Recommendation: **{recommended_type}**\n\n")
            f.write("**Justification:**\n")
            
            if 'criteria_scores' in analysis:
                best_scores = analysis['criteria_scores'][recommended_type]
                f.write(f"- **Combined Score:** {best_scores['combined_score']:.3f} (highest among all linear types)\n")
                f.write(f"- **Win Rate:** {best_scores['frequency_score']:.1%}\n")
                f.write(f"- **Average Rank:** {4 - best_scores['rank_score'] * 3:.2f} out of 3\n")
                f.write(f"- **Normalized Performance:** {best_scores['normalized_score']:.3f}\n\n")
            
            if 'statistical_tests' in analysis and 'pairwise_tests' in analysis['statistical_tests']:
                # Find significant advantages
                significant_advantages = []
                for comparison, results in analysis['statistical_tests']['pairwise_tests'].items():
                    if results['significant'] and results['better_type'] == recommended_type:
                        other_type = comparison.replace(f"{recommended_type}_vs_", "").replace(f"_vs_{recommended_type}", "")
                        significant_advantages.append(f"{other_type} (p={results['p_value']:.4f}, d={results['cohens_d']:.3f})")
                
                if significant_advantages:
                    f.write(f"**Statistical Advantages:**\n")
                    f.write(f"- Significantly better than: {', '.join(significant_advantages)}\n\n")
            
            # Implementation Instructions
            f.write("### 🚀 Implementation for Adaptive Training\n\n")
            f.write("To use these results for the adaptive weighting training (Part 2), follow these steps:\n\n")
            f.write("1. **Update your adaptive training config:**\n")
            f.write("```yaml\n")
            f.write(f"linear_type: '{recommended_type}'\n")
            f.write(f"trained_models_dir: '{self.args.output_dir}'\n")
            f.write("```\n\n")
            f.write("2. **Run the adaptive training:**\n")
            f.write("```bash\n")
            f.write("python openfold/block_replacement_scripts/train_adaptive_weighting.py \\\n")
            f.write("    --config adaptive_config.yaml \\\n")
            f.write("    --output_dir adaptive_training_output\n")
            f.write("```\n\n")
            
            f.write("3. **Expected Benefits:**\n")
            if 'best_frequency' in analysis:
                total_wins = analysis['best_frequency'].get(recommended_type, 0)
                total_blocks = len(analysis['best_per_block'])
                f.write(f"   - Uses the best-performing linear type for {total_wins}/{total_blocks} blocks ({total_wins/total_blocks:.1%})\n")
            f.write(f"   - Initializes replacement blocks with proven architecture\n")
            f.write(f"   - Provides optimal starting point for adaptive weighting\n\n")
            
            # Technical Notes
            f.write("## 🔧 Technical Notes\n\n")
            f.write("- **Evaluation Metric:** Best validation loss (MSE) achieved during training\n")
            f.write("- **Statistical Tests:** ANOVA for overall differences, t-tests for pairwise comparisons\n")
            f.write("- **Effect Sizes:** Cohen's d for practical significance assessment\n")
            f.write("- **Ranking Method:** Multi-criteria scoring combining frequency, rank, and normalized performance\n")
            f.write(f"- **Total Comparisons:** {len(self.args.test_blocks) * len(self.linear_types)} individual training runs\n\n")
            
            f.write("---\n")
            f.write(f"*Report generated automatically by Linear Type Comparison Pipeline*\n")
        
        print(f"Saved comprehensive report: {report_path}")
        
        # Save analysis as JSON
        json_path = self.results_dir / "complete_analysis_results.json"
        
        # Convert non-serializable objects to serializable format
        def make_serializable(obj):
            if isinstance(obj, pd.DataFrame):
                # Convert DataFrame to dict and ensure all keys are strings
                df_dict = obj.to_dict()
                # Convert tuple keys to strings
                serializable_dict = {}
                for col_key, col_data in df_dict.items():
                    if isinstance(col_key, tuple):
                        col_key = str(col_key)
                    serializable_dict[str(col_key)] = {str(k): v for k, v in col_data.items()}
                return serializable_dict
            elif isinstance(obj, pd.Series):
                series_dict = obj.to_dict()
                # Convert any tuple keys to strings
                return {str(k): v for k, v in series_dict.items()}
            elif isinstance(obj, dict):
                # Recursively handle nested dictionaries
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                # Handle lists and tuples
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_analysis = make_serializable(analysis)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2, default=str)
        
        print(f"Saved complete analysis data: {json_path}")
        
        # Create a simple summary file for quick reference
        summary_path = self.results_dir / "RECOMMENDATION.txt"
        with open(summary_path, 'w') as f:
            f.write("RECOMMENDED LINEAR TYPE FOR ADAPTIVE TRAINING\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Linear Type: {recommended_type}\n\n")
            f.write("Usage in adaptive_config.yaml:\n")
            f.write(f"linear_type: '{recommended_type}'\n\n")
            f.write(f"See comprehensive_linear_type_analysis.md for full details.\n")
        
        print(f"Saved quick reference: {summary_path}")
    
    def run_complete_analysis(self):
        """Run the complete linear type comparison analysis"""
        
        print("=== Starting Linear Type Comparison Analysis ===")
        print()
        
        # Step 1: Check existing results and determine what needs training
        results_file = self.output_dir / "training_summary.csv"
        missing_combinations = []
        
        if results_file.exists():
            print("✅ Training results found - checking for missing combinations")
            df = self.load_training_results()
            if df is not None and len(df) > 0:
                print(f"📊 Found {len(df)} existing training results")
                
                # Determine missing combinations
                missing_combinations = self._find_missing_combinations(df)
                
                if len(missing_combinations) == 0:
                    print("🎉 All training combinations complete - proceeding to analysis")
                    self._run_analysis_only(df)
                    return
                else:
                    print(f"⚠️  Found {len(missing_combinations)} missing combinations:")
                    for block_idx, linear_type in missing_combinations:
                        print(f"    Block {block_idx:02d} - {linear_type}")
            else:
                print("⚠️ Training results file exists but is empty - running all training")
                missing_combinations = [(block, linear_type) 
                                      for block in self.args.test_blocks 
                                      for linear_type in self.linear_types]
        else:
            print("📋 No training results found - running all training")
            missing_combinations = [(block, linear_type) 
                                  for block in self.args.test_blocks 
                                  for linear_type in self.linear_types]
        
        # Step 2: Train missing combinations
        if missing_combinations:
            print(f"🚀 Training {len(missing_combinations)} missing combinations...")
            success = self._train_missing_combinations(missing_combinations)
        
        if not success:
                print("❌ Training failed. Aborting analysis.")
            return
        
        # Step 3: Load final results and analyze
        df = self.load_training_results()
        if df is None:
            print("❌ No training results to analyze. Aborting.")
            return
        
        print(f"✅ Final dataset contains {len(df)} training results")
        self._run_analysis_only(df)
    
    def _find_missing_combinations(self, df: pd.DataFrame) -> List[Tuple[int, str]]:
        """Find missing (block_idx, linear_type) combinations from existing results"""
        
        # Get existing combinations
        existing_combinations = set()
        for _, row in df.iterrows():
            existing_combinations.add((row['block_idx'], row['linear_type']))
        
        # Generate all expected combinations
        expected_combinations = set()
        for block_idx in self.args.test_blocks:
            for linear_type in self.linear_types:
                expected_combinations.add((block_idx, linear_type))
        
        # Find missing combinations
        missing_combinations = expected_combinations - existing_combinations
        return sorted(list(missing_combinations))
    
    def _train_missing_combinations(self, missing_combinations: List[Tuple[int, str]]) -> bool:
        """Train only the missing (block_idx, linear_type) combinations"""
        
        # Group missing combinations by linear type for efficiency
        linear_type_to_blocks = {}
        for block_idx, linear_type in missing_combinations:
            if linear_type not in linear_type_to_blocks:
                linear_type_to_blocks[linear_type] = []
            linear_type_to_blocks[linear_type].append(block_idx)
        
        # Train each linear type separately
        for linear_type, blocks in linear_type_to_blocks.items():
            print(f"🔧 Training {linear_type} for blocks: {blocks}")
            
            success = self.run_single_block_training(
                blocks=blocks,
                linear_types=[linear_type]
            )
            
            if not success:
                print(f"❌ Failed to train {linear_type} for blocks {blocks}")
                return False
                
        return True
    
    def _run_analysis_only(self, df: pd.DataFrame):
        """Run only the analysis part (after training is complete)"""
        
        # Step 3: Perform analysis
        analysis = self.analyze_linear_types(df)
        
        # Step 4: Create visualizations
            self.create_visualizations(df, analysis)
        
        # Step 5: Generate report
        self.generate_report(analysis)
        
        print("\n=== Comprehensive Analysis Complete ===")
        print(f"Results saved to: {self.results_dir}")
        
        recommended_type = analysis.get('recommended_linear_type', analysis.get('overall_best_linear_type', 'full'))
        print(f"🎯 RECOMMENDED LINEAR TYPE: {recommended_type}")
        
        if 'criteria_scores' in analysis:
            best_score = analysis['criteria_scores'][recommended_type]['combined_score']
            print(f"📊 Combined Score: {best_score:.3f}")
        
        if 'best_frequency' in analysis:
            wins = analysis['best_frequency'].get(recommended_type, 0)
            total = len(analysis['best_per_block'])
            print(f"🏆 Win Rate: {wins}/{total} blocks ({wins/total:.1%})")
        
        if 'statistical_tests' in analysis and 'anova' in analysis['statistical_tests']:
            anova = analysis['statistical_tests']['anova']
            print(f"🔬 Statistical Significance: {'Yes' if anova['significant'] else 'No'} (p={anova['p_value']:.6f})")
        
        print(f"\n📋 Key Files Generated:")
        print(f"  - comprehensive_linear_type_analysis.md (detailed report)")
        print(f"  - RECOMMENDATION.txt (quick reference)")
        print(f"  - complete_analysis_results.json (raw data)")
        print(f"  - linear_type_comparison.png (visualizations)")
        
        print(f"\n🚀 Next Steps:")
        print(f"  1. Review the comprehensive report for detailed analysis")
        print(f"  2. Use '{recommended_type}' as linear_type in adaptive training config")
        print(f"  3. Run adaptive weighting training with the recommended weights")


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
    parser.add_argument("--test_blocks", type=int, nargs="+", default=list(range(1, 47)),
                       help="Block indices to test (default: all blocks 1-46, excluding first/last)")
    
    # Training parameters (passed to single block training)
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension for replacement blocks")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training (increased from 1 due to padding support for variable protein lengths)")
    parser.add_argument("--max_epochs", type=int, default=30,
                       help="Maximum number of training epochs (balanced for comprehensive testing)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--distributed_backend", type=str, default="gloo",
                       choices=["nccl", "gloo", "mpi"], 
                       help="Distributed training backend (use 'gloo' for nodes without NCCL support)")
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
    parser.add_argument("--experiment_name", type=str, default="linear_type_test",
                       help="Base experiment name for wandb logging")
    
    args = parser.parse_args()
    
    # Run analysis
    comparison = LinearTypeComparison(args)
    comparison.run_complete_analysis()


if __name__ == "__main__":
    main()

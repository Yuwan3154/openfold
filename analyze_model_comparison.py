#!/usr/bin/env python3
"""
Analysis script for OpenFold model architecture comparison results.

This script analyzes the distribution of TM-score differences between different model variants
and tests whether they significantly deviate from zero.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
from pathlib import Path


def load_results(results_path):
    """Load results from CSV file"""
    df = pd.read_csv(results_path)
    print(f"Loaded results for {len(df)} proteins")
    print(f"Columns: {list(df.columns)}")
    return df


def calculate_differences(df):
    """Calculate TM-score differences"""
    # Calculate differences
    df['tm_diff_removed'] = df['removed_block_tm'] - df['original_tm']
    df['tm_diff_replaced'] = df['replaced_block_tm'] - df['original_tm']
    
    # Also calculate pTM differences for additional analysis
    df['ptm_diff_removed'] = df['removed_block_ptm'] - df['original_ptm']
    df['ptm_diff_replaced'] = df['replaced_block_ptm'] - df['original_ptm']
    
    return df


def statistical_tests(values, name):
    """Perform statistical tests on the differences"""
    print(f"\n=== Statistical Analysis for {name} ===")
    print(f"N = {len(values)}")
    print(f"Mean = {np.mean(values):.6f}")
    print(f"Std = {np.std(values, ddof=1):.6f}")
    print(f"Median = {np.median(values):.6f}")
    print(f"Min = {np.min(values):.6f}")
    print(f"Max = {np.max(values):.6f}")
    
    # One-sample t-test against 0
    t_stat, t_pval = stats.ttest_1samp(values, 0)
    print(f"One-sample t-test (H0: mean = 0):")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value = {t_pval:.6f}")
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, w_pval = stats.wilcoxon(values)
    print(f"Wilcoxon signed-rank test (H0: median = 0):")
    print(f"  statistic = {w_stat:.4f}")
    print(f"  p-value = {w_pval:.6f}")
    
    # Check normality
    shapiro_stat, shapiro_pval = stats.shapiro(values)
    print(f"Shapiro-Wilk normality test:")
    print(f"  statistic = {shapiro_stat:.4f}")
    print(f"  p-value = {shapiro_pval:.6f}")
    
    # Effect size (Cohen's d)
    cohens_d = np.mean(values) / np.std(values, ddof=1)
    print(f"Cohen's d (effect size) = {cohens_d:.4f}")
    
    return {
        'name': name,
        'n': len(values),
        'mean': np.mean(values),
        'std': np.std(values, ddof=1),
        'median': np.median(values),
        't_stat': t_stat,
        't_pval': t_pval,
        'w_stat': w_stat,
        'w_pval': w_pval,
        'shapiro_pval': shapiro_pval,
        'cohens_d': cohens_d
    }


def create_plots(df, output_dir):
    """Create visualization plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Distribution comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # TM-score differences
    ax1 = axes[0, 0]
    ax1.hist(df['tm_diff_removed'], bins=15, alpha=0.7, label='Block Removed', density=True)
    ax1.hist(df['tm_diff_replaced'], bins=15, alpha=0.7, label='Block Replaced', density=True)
    ax1.axvline(0, color='red', linestyle='--', alpha=0.8, label='No Change')
    ax1.set_xlabel('TM-score Difference')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of TM-score Differences')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # pTM differences
    ax2 = axes[0, 1]
    ax2.hist(df['ptm_diff_removed'], bins=15, alpha=0.7, label='Block Removed', density=True)
    ax2.hist(df['ptm_diff_replaced'], bins=15, alpha=0.7, label='Block Replaced', density=True)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.8, label='No Change')
    ax2.set_xlabel('pTM Difference')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of pTM Differences')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Scatter plot: removed vs replaced differences
    ax3 = axes[1, 0]
    ax3.scatter(df['tm_diff_removed'], df['tm_diff_replaced'], alpha=0.6, s=50)
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('TM Difference (Block Removed)')
    ax3.set_ylabel('TM Difference (Block Replaced)')
    ax3.set_title('TM Differences: Removed vs Replaced')
    ax3.grid(True, alpha=0.3)
    
    # Add diagonal line
    lims = [np.min([ax3.get_xlim(), ax3.get_ylim()]),
            np.max([ax3.get_xlim(), ax3.get_ylim()])]
    ax3.plot(lims, lims, 'k-', alpha=0.3, zorder=0)
    
    # Box plot comparison
    ax4 = axes[1, 1]
    data_to_plot = [df['tm_diff_removed'], df['tm_diff_replaced']]
    box_plot = ax4.boxplot(data_to_plot, tick_labels=['Block Removed', 'Block Replaced'], patch_artist=True)
    ax4.axhline(0, color='red', linestyle='--', alpha=0.8, label='No Change')
    ax4.set_ylabel('TM-score Difference')
    ax4.set_title('TM-score Differences Box Plot')
    ax4.grid(True, alpha=0.3)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'model_comparison_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. Individual distribution plots with statistics
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # TM difference for removed block
    ax1 = axes[0]
    ax1.hist(df['tm_diff_removed'], bins=15, alpha=0.7, color='skyblue', density=True)
    ax1.axvline(0, color='red', linestyle='--', alpha=0.8, label='No Change')
    ax1.axvline(np.mean(df['tm_diff_removed']), color='blue', linestyle='-', alpha=0.8, label=f'Mean = {np.mean(df["tm_diff_removed"]):.4f}')
    ax1.set_xlabel('TM-score Difference (Block Removed - Original)')
    ax1.set_ylabel('Density')
    ax1.set_title('Block Removed: TM-score Change Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # TM difference for replaced block
    ax2 = axes[1]
    ax2.hist(df['tm_diff_replaced'], bins=15, alpha=0.7, color='lightgreen', density=True)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.8, label='No Change')
    ax2.axvline(np.mean(df['tm_diff_replaced']), color='green', linestyle='-', alpha=0.8, label=f'Mean = {np.mean(df["tm_diff_replaced"]):.4f}')
    ax2.set_xlabel('TM-score Difference (Block Replaced - Original)')
    ax2.set_ylabel('Density')
    ax2.set_title('Block Replaced: TM-score Change Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'individual_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'individual_distributions.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze OpenFold model comparison results")
    parser.add_argument("results_path", help="Path to summary_results.csv file")
    parser.add_argument("--output_dir", default="analysis_output", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load and process data
    df = load_results(args.results_path)
    df = calculate_differences(df)
    
    # Statistical analysis
    print("=" * 60)
    print("STATISTICAL ANALYSIS OF MODEL ARCHITECTURE DIFFERENCES")
    print("=" * 60)
    
    # Test TM-score differences
    removed_stats = statistical_tests(df['tm_diff_removed'], "TM-score (Block Removed - Original)")
    replaced_stats = statistical_tests(df['tm_diff_replaced'], "TM-score (Block Replaced - Original)")
    
    # Test pTM differences for completeness
    removed_ptm_stats = statistical_tests(df['ptm_diff_removed'], "pTM (Block Removed - Original)")
    replaced_ptm_stats = statistical_tests(df['ptm_diff_replaced'], "pTM (Block Replaced - Original)")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    alpha = 0.05
    print(f"Statistical significance threshold: α = {alpha}")
    print(f"\nTM-score Results:")
    print(f"  Block Removed: mean = {removed_stats['mean']:.6f}, p = {removed_stats['t_pval']:.6f} {'***' if removed_stats['t_pval'] < alpha else 'ns'}")
    print(f"  Block Replaced: mean = {replaced_stats['mean']:.6f}, p = {replaced_stats['t_pval']:.6f} {'***' if replaced_stats['t_pval'] < alpha else 'ns'}")
    
    print(f"\npTM Results:")
    print(f"  Block Removed: mean = {removed_ptm_stats['mean']:.6f}, p = {removed_ptm_stats['t_pval']:.6f} {'***' if removed_ptm_stats['t_pval'] < alpha else 'ns'}")
    print(f"  Block Replaced: mean = {replaced_ptm_stats['mean']:.6f}, p = {replaced_ptm_stats['t_pval']:.6f} {'***' if replaced_ptm_stats['t_pval'] < alpha else 'ns'}")
    
    # Create plots
    create_plots(df, args.output_dir)
    
    # Save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save statistical results
    stats_df = pd.DataFrame([removed_stats, replaced_stats, removed_ptm_stats, replaced_ptm_stats])
    stats_df.to_csv(output_dir / 'statistical_analysis.csv', index=False)
    
    # Save processed data
    df.to_csv(output_dir / 'processed_results.csv', index=False)
    
    print(f"\nDetailed results saved to: {output_dir}")


if __name__ == "__main__":
    main()

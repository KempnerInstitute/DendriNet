#!/usr/bin/env python3
"""
Analyze and create plots from existing multi-run correlation analysis results.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results(results_dir: str) -> pd.DataFrame:
    """Load results from a multi-run experiment directory."""
    results_file = os.path.join(results_dir, 'all_results.csv')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    return pd.read_csv(results_file)


def create_correlation_scatter_matrix(results_df: pd.DataFrame, save_path: str):
    """Create a scatter matrix of all correlations and tuning vs accuracy."""
    # Select correlation and tuning columns
    corr_cols = [col for col in results_df.columns if 
                 ('correlation' in col or 'tuning' in col) and 'mean_' in col]
    
    if not corr_cols:
        print("No correlation/tuning columns found")
        return
    
    # Create scatter matrix
    fig, axes = plt.subplots(len(corr_cols), len(corr_cols), 
                            figsize=(3*len(corr_cols), 3*len(corr_cols)))
    
    for i, col1 in enumerate(corr_cols):
        for j, col2 in enumerate(corr_cols):
            ax = axes[i, j] if len(corr_cols) > 1 else axes
            
            if i == j:
                # Diagonal: histogram
                ax.hist(results_df[col1].dropna(), bins=20, alpha=0.7)
                ax.set_xlabel(col1.replace('mean_', '').replace('_', ' '))
            else:
                # Off-diagonal: scatter plot
                valid_mask = ~(results_df[col1].isna() | results_df[col2].isna())
                if valid_mask.sum() > 2:
                    ax.scatter(results_df.loc[valid_mask, col2], 
                              results_df.loc[valid_mask, col1], alpha=0.5)
                    
                    # Add correlation coefficient
                    r, p = pearsonr(results_df.loc[valid_mask, col2], 
                                   results_df.loc[valid_mask, col1])
                    ax.text(0.05, 0.95, f'r={r:.2f}', transform=ax.transAxes,
                           verticalalignment='top', fontsize=8)
                
                if i == len(corr_cols) - 1:
                    ax.set_xlabel(col2.replace('mean_', '').replace('_', ' '))
                if j == 0:
                    ax.set_ylabel(col1.replace('mean_', '').replace('_', ' '))
    
    plt.suptitle('Correlation Matrix of Synaptic Properties', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_accuracy_prediction_plot(results_df: pd.DataFrame, save_path: str):
    """Create plot showing which features best predict accuracy."""
    # Get all correlation/tuning columns
    feature_cols = [col for col in results_df.columns if 
                   ('correlation' in col or 'tuning' in col) and 'mean_' in col]
    
    if not feature_cols:
        return
    
    # Compute correlations with accuracy
    correlations = []
    for col in feature_cols:
        valid_mask = ~(results_df[col].isna() | results_df['test_accuracy'].isna())
        if valid_mask.sum() > 2:
            r, p = pearsonr(results_df.loc[valid_mask, col], 
                           results_df.loc[valid_mask, 'test_accuracy'])
            correlations.append({
                'feature': col.replace('mean_', '').replace('_', ' '),
                'correlation': r,
                'p_value': p,
                'significant': p < 0.05
            })
    
    if not correlations:
        return
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(corr_df)), corr_df['correlation'], 
                   color=['red' if sig else 'gray' for sig in corr_df['significant']])
    
    # Add significance stars
    for i, (idx, row) in enumerate(corr_df.iterrows()):
        if row['significant']:
            ax.text(i, row['correlation'] + 0.01 * np.sign(row['correlation']), 
                   '*', ha='center', va='bottom' if row['correlation'] > 0 else 'top')
    
    ax.set_xticks(range(len(corr_df)))
    ax.set_xticklabels(corr_df['feature'], rotation=45, ha='right')
    ax.set_ylabel('Correlation with Test Accuracy')
    ax.set_title('Feature Importance for Predicting Network Accuracy')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='p < 0.05'),
                      Patch(facecolor='gray', label='p ≥ 0.05')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_noise_vs_total_correlation_plot(results_df: pd.DataFrame, save_path: str):
    """Compare noise correlations vs total correlations in predicting accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    correlation_pairs = [
        ('mean_ee_correlation_total', 'mean_ee_correlation_noise', 'E-E', axes[0]),
        ('mean_ei_correlation_total', 'mean_ei_correlation_noise', 'E-I', axes[1]),
        ('mean_ii_correlation_total', 'mean_ii_correlation_noise', 'I-I', axes[2])
    ]
    
    for total_col, noise_col, label, ax in correlation_pairs:
        if total_col in results_df.columns and noise_col in results_df.columns:
            # Plot total correlation
            valid_mask = ~(results_df[total_col].isna() | results_df['test_accuracy'].isna())
            if valid_mask.sum() > 2:
                ax.scatter(results_df.loc[valid_mask, total_col], 
                          results_df.loc[valid_mask, 'test_accuracy'],
                          alpha=0.6, label='Total', color='blue', s=50)
                
                # Fit line
                z = np.polyfit(results_df.loc[valid_mask, total_col], 
                              results_df.loc[valid_mask, 'test_accuracy'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(results_df.loc[valid_mask, total_col].min(),
                                   results_df.loc[valid_mask, total_col].max(), 100)
                ax.plot(x_line, p(x_line), 'b--', alpha=0.8)
            
            # Plot noise correlation
            valid_mask = ~(results_df[noise_col].isna() | results_df['test_accuracy'].isna())
            if valid_mask.sum() > 2:
                ax.scatter(results_df.loc[valid_mask, noise_col], 
                          results_df.loc[valid_mask, 'test_accuracy'],
                          alpha=0.6, label='Noise', color='red', s=50)
                
                # Fit line
                z = np.polyfit(results_df.loc[valid_mask, noise_col], 
                              results_df.loc[valid_mask, 'test_accuracy'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(results_df.loc[valid_mask, noise_col].min(),
                                   results_df.loc[valid_mask, noise_col].max(), 100)
                ax.plot(x_line, p(x_line), 'r--', alpha=0.8)
            
            ax.set_xlabel(f'{label} Correlation')
            ax.set_ylabel('Test Accuracy')
            ax.set_title(f'{label} Correlations vs Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Noise vs Total Correlations in Predicting Accuracy', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_combined_summary_plot(results_df: pd.DataFrame, save_path: str):
    """Create a combined summary plot with key findings."""
    fig = plt.figure(figsize=(16, 10))
    
    # Define grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results_df['test_accuracy'].dropna(), bins=20, alpha=0.7, color='green')
    ax1.axvline(results_df['test_accuracy'].mean(), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Test Accuracy')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Accuracy Distribution (μ={results_df["test_accuracy"].mean():.3f})')
    
    # 2. Best predictor scatter
    ax2 = fig.add_subplot(gs[0, 1:])
    # Find best predictor
    best_predictor = None
    best_r = 0
    for col in results_df.columns:
        if 'mean_' in col and col != 'test_accuracy':
            valid_mask = ~(results_df[col].isna() | results_df['test_accuracy'].isna())
            if valid_mask.sum() > 2:
                r, _ = pearsonr(results_df.loc[valid_mask, col], 
                               results_df.loc[valid_mask, 'test_accuracy'])
                if abs(r) > abs(best_r):
                    best_r = r
                    best_predictor = col
    
    if best_predictor:
        valid_mask = ~(results_df[best_predictor].isna() | results_df['test_accuracy'].isna())
        ax2.scatter(results_df.loc[valid_mask, best_predictor], 
                   results_df.loc[valid_mask, 'test_accuracy'], alpha=0.6, s=50)
        z = np.polyfit(results_df.loc[valid_mask, best_predictor], 
                      results_df.loc[valid_mask, 'test_accuracy'], 1)
        p = np.poly1d(z)
        ax2.plot(results_df.loc[valid_mask, best_predictor], 
                p(results_df.loc[valid_mask, best_predictor]), 'r--', alpha=0.8)
        ax2.set_xlabel(best_predictor.replace('mean_', '').replace('_', ' ').title())
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title(f'Best Predictor of Accuracy (r={best_r:.3f})')
    
    # 3. Correlation heatmap
    ax3 = fig.add_subplot(gs[1:, :])
    # Select key variables
    key_vars = ['test_accuracy']
    for col in ['mean_exc_tuning', 'mean_inh_tuning', 
                'mean_ee_correlation_total', 'mean_ei_correlation_total', 
                'mean_ii_correlation_total']:
        if col in results_df.columns:
            key_vars.append(col)
    
    if len(key_vars) > 1:
        corr_matrix = results_df[key_vars].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, ax=ax3,
                   xticklabels=[v.replace('mean_', '').replace('_', ' ') for v in key_vars],
                   yticklabels=[v.replace('mean_', '').replace('_', ' ') for v in key_vars])
        ax3.set_title('Correlation Matrix of Key Variables')
    
    plt.suptitle('Multi-Run Experiment Summary', fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-run correlation results')
    parser.add_argument('results_dir', type=str,
                       help='Directory containing multi-run results (with all_results.csv)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as results_dir)')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results_df = load_results(args.results_dir)
    print(f"Loaded {len(results_df)} runs")
    
    # Set output directory
    output_dir = args.output_dir or args.results_dir
    
    # Create additional plots
    print("\nCreating additional analysis plots...")
    
    # 1. Correlation scatter matrix
    print("- Creating correlation scatter matrix...")
    create_correlation_scatter_matrix(
        results_df, 
        os.path.join(output_dir, 'correlation_scatter_matrix.png')
    )
    
    # 2. Feature importance plot
    print("- Creating feature importance plot...")
    create_accuracy_prediction_plot(
        results_df,
        os.path.join(output_dir, 'feature_importance.png')
    )
    
    # 3. Noise vs total correlation comparison
    print("- Creating noise vs total correlation plot...")
    create_noise_vs_total_correlation_plot(
        results_df,
        os.path.join(output_dir, 'noise_vs_total_correlations.png')
    )
    
    # 4. Combined summary plot
    print("- Creating combined summary plot...")
    create_combined_summary_plot(
        results_df,
        os.path.join(output_dir, 'combined_summary.png')
    )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTest Accuracy: {results_df['test_accuracy'].mean():.4f} ± {results_df['test_accuracy'].std():.4f}")
    
    # Find significant predictors
    print("\nSignificant Predictors of Accuracy (p < 0.05):")
    for col in results_df.columns:
        if 'mean_' in col and col != 'test_accuracy':
            valid_mask = ~(results_df[col].isna() | results_df['test_accuracy'].isna())
            if valid_mask.sum() > 2:
                r, p = pearsonr(results_df.loc[valid_mask, col], 
                               results_df.loc[valid_mask, 'test_accuracy'])
                if p < 0.05:
                    print(f"  {col}: r = {r:.3f} (p = {p:.3f})")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main() 
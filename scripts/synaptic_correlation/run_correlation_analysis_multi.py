#!/usr/bin/env python3
"""
Run multiple experiments with different data realizations and analyze the relationship
between network accuracy and synaptic correlations/tuning.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def create_experiment_config(base_config_path: str, seed: int, output_dir: str, run_name: str) -> str:
    """Create a config for a specific random seed (generates new data)."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update seed - this will generate new data with new correlation matrix
    config['train']['seed'] = seed
    
    # Update output directory
    config['outputs']['run_name'] = run_name
    config['outputs']['dir'] = output_dir
    
    # Save modified config
    config_path = os.path.join(output_dir, 'config.yaml')
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def run_single_experiment(config_path: str) -> Dict:
    """Run a single experiment and return results."""
    # Run the training script
    cmd = [
        sys.executable,
        str(project_root / 'src' / 'dendritic_modeling' / 'scripts' / 'train_experiments.py'),
        config_path  # Config path as positional argument, not --config
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running experiment: {result.stderr}")
        return None
    
    # Extract results from output directory
    output_dir = os.path.dirname(config_path)
    
    # Find the actual run directory (it's nested)
    run_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('run_')]
    if run_dirs:
        actual_output_dir = os.path.join(output_dir, run_dirs[0])
    else:
        actual_output_dir = output_dir
    
    # Read performance results (JSON format)
    perf_file = os.path.join(actual_output_dir, 'performance', 'final')
    if os.path.exists(perf_file):
        import json
        with open(perf_file, 'r') as f:
            perf_data = json.load(f)
        test_accuracy = perf_data.get('accuracy', {}).get('test', None)
    else:
        test_accuracy = None
    
    # Read correlation analysis results
    corr_summary_file = os.path.join(actual_output_dir, 'correlation_analysis', 'correlation_summary.csv')
    if os.path.exists(corr_summary_file):
        corr_summary = pd.read_csv(corr_summary_file).to_dict('records')[0]
    else:
        corr_summary = {}
    
    return {
        'test_accuracy': test_accuracy,
        **corr_summary
    }


def create_comprehensive_plots(results_df: pd.DataFrame, save_dir: str):
    """Create comprehensive analysis plots."""
    # 1. Accuracy vs Synaptic Correlations (Total and Noise)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    plot_data = [
        ('mean_ee_correlation_total', 'E-E Total', axes[0, 0], 'red'),
        ('mean_ei_correlation_total', 'E-I Total', axes[0, 1], 'green'),
        ('mean_ii_correlation_total', 'I-I Total', axes[0, 2], 'blue'),
        ('mean_ee_correlation_noise', 'E-E Noise', axes[1, 0], 'darkred'),
        ('mean_ei_correlation_noise', 'E-I Noise', axes[1, 1], 'darkgreen'),
        ('mean_ii_correlation_noise', 'I-I Noise', axes[1, 2], 'darkblue')
    ]
    
    for corr_col, label, ax, color in plot_data:
        if corr_col in results_df.columns:
            valid_mask = ~(results_df[corr_col].isna() | results_df['test_accuracy'].isna())
            x_data = results_df.loc[valid_mask, corr_col]
            y_data = results_df.loc[valid_mask, 'test_accuracy']
            
            ax.scatter(x_data, y_data, alpha=0.6, s=50, color=color)
            
            if len(x_data) > 2:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(x_data, p(x_data), "k--", alpha=0.8)
                
                pearson_r, pearson_p = pearsonr(x_data, y_data)
                ax.text(0.05, 0.95, f'r = {pearson_r:.3f}\np = {pearson_p:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax.set_xlabel(f'{label} Correlation')
            ax.set_ylabel('Test Accuracy')
            ax.set_title(f'Accuracy vs {label} Correlation')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Network Accuracy vs Synaptic Correlations', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_vs_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy vs Tuning
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Excitatory tuning
    if 'mean_exc_tuning' in results_df.columns:
        valid_mask = ~(results_df['mean_exc_tuning'].isna() | results_df['test_accuracy'].isna())
        x_data = results_df.loc[valid_mask, 'mean_exc_tuning']
        y_data = results_df.loc[valid_mask, 'test_accuracy']
        
        axes[0].scatter(x_data, y_data, alpha=0.6, s=50, color='red')
        if len(x_data) > 2:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            axes[0].plot(x_data, p(x_data), "k--", alpha=0.8)
            pearson_r, pearson_p = pearsonr(x_data, y_data)
            axes[0].text(0.05, 0.95, f'r = {pearson_r:.3f}\np = {pearson_p:.3f}', 
                        transform=axes[0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        axes[0].set_xlabel('Mean Excitatory Tuning')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Accuracy vs Excitatory Tuning')
        axes[0].grid(True, alpha=0.3)
    
    # Inhibitory tuning
    if 'mean_inh_tuning' in results_df.columns:
        valid_mask = ~(results_df['mean_inh_tuning'].isna() | results_df['test_accuracy'].isna())
        x_data = results_df.loc[valid_mask, 'mean_inh_tuning']
        y_data = results_df.loc[valid_mask, 'test_accuracy']
        
        axes[1].scatter(x_data, y_data, alpha=0.6, s=50, color='blue')
        if len(x_data) > 2:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            axes[1].plot(x_data, p(x_data), "k--", alpha=0.8)
            pearson_r, pearson_p = pearsonr(x_data, y_data)
            axes[1].text(0.05, 0.95, f'r = {pearson_r:.3f}\np = {pearson_p:.3f}', 
                        transform=axes[1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        axes[1].set_xlabel('Mean Inhibitory Tuning')
        axes[1].set_ylabel('Test Accuracy')
        axes[1].set_title('Accuracy vs Inhibitory Tuning')
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Network Accuracy vs Synaptic Tuning', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_vs_tuning.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_information_correlation_plots(results_df: pd.DataFrame, save_dir: str):
    """Create comprehensive plots showing information vs correlation relationships."""
    
    # Define the metrics we want to plot
    correlation_metrics = [
        'mean_ee_correlation_total', 'mean_ei_correlation_total', 'mean_ii_correlation_total',
        'mean_ee_correlation_noise', 'mean_ei_correlation_noise', 'mean_ii_correlation_noise'
    ]
    
    information_metrics = [
        'I_E_C', 'I_I_C', 'I_Vout_C', 'I_EI_C', 
        'I_E_I', 'I_E_Vout', 'I_I_Vout',
        'integration_gain', 'ei_balance'
    ]
    
    # Check which metrics are available
    available_corr = [m for m in correlation_metrics if m in results_df.columns]
    available_info = [m for m in information_metrics if m in results_df.columns]
    
    if not available_corr or not available_info:
        print("Not enough correlation or information metrics available for plotting")
        return
    
    # 1. Correlation vs Information Scatter Matrix
    fig, axes = plt.subplots(len(available_info), len(available_corr), 
                            figsize=(4*len(available_corr), 3*len(available_info)))
    
    if len(available_info) == 1:
        axes = axes.reshape(1, -1)
    if len(available_corr) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, info_metric in enumerate(available_info):
        for j, corr_metric in enumerate(available_corr):
            ax = axes[i, j]
            
            # Get valid data points
            valid_mask = ~(results_df[info_metric].isna() | results_df[corr_metric].isna() | results_df['test_accuracy'].isna())
            
            if valid_mask.sum() > 2:
                x_data = results_df.loc[valid_mask, corr_metric]
                y_data = results_df.loc[valid_mask, info_metric]
                c_data = results_df.loc[valid_mask, 'test_accuracy']
                
                # Create scatter plot colored by accuracy
                scatter = ax.scatter(x_data, y_data, c=c_data, cmap='viridis', alpha=0.7, s=50)
                
                # Add correlation coefficient
                try:
                    r, p = pearsonr(x_data, y_data)
                    ax.text(0.05, 0.95, f'r={r:.3f}\np={p:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=8)
                except:
                    pass
                
                # Add trend line
                try:
                    z = np.polyfit(x_data, y_data, 1)
                    p_trend = np.poly1d(z)
                    ax.plot(x_data, p_trend(x_data), "r--", alpha=0.8, linewidth=1)
                except:
                    pass
            
            ax.set_xlabel(corr_metric.replace('mean_', '').replace('_', ' ').title())
            ax.set_ylabel(info_metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
    
    # Add colorbar
    if valid_mask.sum() > 0:
        cbar = plt.colorbar(scatter, ax=axes, shrink=0.6)
        cbar.set_label('Test Accuracy')
    
    plt.suptitle('Information Metrics vs Correlation Metrics\n(colored by accuracy)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'information_vs_correlation_matrix.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Key relationships plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: E-I correlation vs E-I information
    if 'mean_ei_correlation_total' in results_df.columns and 'I_E_I' in results_df.columns:
        valid_mask = ~(results_df['mean_ei_correlation_total'].isna() | 
                      results_df['I_E_I'].isna() | 
                      results_df['test_accuracy'].isna())
        if valid_mask.sum() > 2:
            scatter = axes[0, 0].scatter(
                results_df.loc[valid_mask, 'mean_ei_correlation_total'],
                results_df.loc[valid_mask, 'I_E_I'],
                c=results_df.loc[valid_mask, 'test_accuracy'],
                cmap='viridis', alpha=0.7, s=60
            )
            axes[0, 0].set_xlabel('Mean |E-I Correlation|')
            axes[0, 0].set_ylabel('I(E;I) [bits]')
            axes[0, 0].set_title('E-I Synaptic Correlation vs Information')
            
            # Add trend line and correlation
            try:
                x_data = results_df.loc[valid_mask, 'mean_ei_correlation_total']
                y_data = results_df.loc[valid_mask, 'I_E_I']
                z = np.polyfit(x_data, y_data, 1)
                p_trend = np.poly1d(z)
                axes[0, 0].plot(x_data, p_trend(x_data), "r--", alpha=0.8)
                r, p = pearsonr(x_data, y_data)
                axes[0, 0].text(0.05, 0.95, f'r={r:.3f}, p={p:.3f}', 
                               transform=axes[0, 0].transAxes, va='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass
    
    # Plot 2: Integration gain vs accuracy
    if 'integration_gain' in results_df.columns:
        valid_mask = ~(results_df['integration_gain'].isna() | results_df['test_accuracy'].isna())
        if valid_mask.sum() > 2:
            axes[0, 1].scatter(
                results_df.loc[valid_mask, 'integration_gain'],
                results_df.loc[valid_mask, 'test_accuracy'],
                alpha=0.7, s=60, color='green'
            )
            axes[0, 1].set_xlabel('Integration Gain [bits]')
            axes[0, 1].set_ylabel('Test Accuracy')
            axes[0, 1].set_title('Information Integration vs Performance')
            
            # Add trend line and correlation
            try:
                x_data = results_df.loc[valid_mask, 'integration_gain']
                y_data = results_df.loc[valid_mask, 'test_accuracy']
                z = np.polyfit(x_data, y_data, 1)
                p_trend = np.poly1d(z)
                axes[0, 1].plot(x_data, p_trend(x_data), "r--", alpha=0.8)
                r, p = pearsonr(x_data, y_data)
                axes[0, 1].text(0.05, 0.95, f'r={r:.3f}, p={p:.3f}', 
                               transform=axes[0, 1].transAxes, va='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass
    
    # Plot 3: E/I balance vs accuracy
    if 'ei_balance' in results_df.columns:
        valid_mask = ~(results_df['ei_balance'].isna() | results_df['test_accuracy'].isna())
        if valid_mask.sum() > 2:
            axes[0, 2].scatter(
                results_df.loc[valid_mask, 'ei_balance'],
                results_df.loc[valid_mask, 'test_accuracy'],
                alpha=0.7, s=60, color='purple'
            )
            axes[0, 2].set_xlabel('E/I Information Balance')
            axes[0, 2].set_ylabel('Test Accuracy')
            axes[0, 2].set_title('E/I Balance vs Performance')
            
            # Add trend line and correlation
            try:
                x_data = results_df.loc[valid_mask, 'ei_balance']
                y_data = results_df.loc[valid_mask, 'test_accuracy']
                z = np.polyfit(x_data, y_data, 1)
                p_trend = np.poly1d(z)
                axes[0, 2].plot(x_data, p_trend(x_data), "r--", alpha=0.8)
                r, p = pearsonr(x_data, y_data)
                axes[0, 2].text(0.05, 0.95, f'r={r:.3f}, p={p:.3f}', 
                               transform=axes[0, 2].transAxes, va='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass
    
    # Plot 4: Total input information vs output information
    if 'I_E_C' in results_df.columns and 'I_I_C' in results_df.columns and 'I_Vout_C' in results_df.columns:
        valid_mask = ~(results_df['I_E_C'].isna() | results_df['I_I_C'].isna() | 
                      results_df['I_Vout_C'].isna() | results_df['test_accuracy'].isna())
        if valid_mask.sum() > 2:
            input_info = results_df.loc[valid_mask, 'I_E_C'] + results_df.loc[valid_mask, 'I_I_C']
            output_info = results_df.loc[valid_mask, 'I_Vout_C']
            
            scatter = axes[1, 0].scatter(
                input_info, output_info,
                c=results_df.loc[valid_mask, 'test_accuracy'],
                cmap='viridis', alpha=0.7, s=60
            )
            axes[1, 0].set_xlabel('Input Information I(E;C)+I(I;C) [bits]')
            axes[1, 0].set_ylabel('Output Information I(Vout;C) [bits]')
            axes[1, 0].set_title('Input vs Output Information')
            
            # Add diagonal line
            min_val = min(input_info.min(), output_info.min())
            max_val = max(input_info.max(), output_info.max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Unity')
            axes[1, 0].legend()
    
    # Plot 5: Correlation strength vs information strength
    if available_corr and available_info:
        # Calculate mean correlation and information strength for each run
        corr_strength = results_df[available_corr].abs().mean(axis=1)
        info_strength = results_df[available_info].abs().mean(axis=1)
        
        valid_mask = ~(corr_strength.isna() | info_strength.isna() | results_df['test_accuracy'].isna())
        if valid_mask.sum() > 2:
            scatter = axes[1, 1].scatter(
                corr_strength[valid_mask], info_strength[valid_mask],
                c=results_df.loc[valid_mask, 'test_accuracy'],
                cmap='viridis', alpha=0.7, s=60
            )
            axes[1, 1].set_xlabel('Mean Correlation Strength')
            axes[1, 1].set_ylabel('Mean Information Strength')
            axes[1, 1].set_title('Overall Correlation vs Information')
    
    # Plot 6: Summary statistics
    axes[1, 2].axis('off')
    
    # Calculate summary statistics
    summary_text = f"""
Summary Statistics (N={len(results_df)})

Test Accuracy:
  Mean: {results_df['test_accuracy'].mean():.3f} ± {results_df['test_accuracy'].std():.3f}
  Range: [{results_df['test_accuracy'].min():.3f}, {results_df['test_accuracy'].max():.3f}]

Key Correlations with Accuracy:
"""
    
    # Find top predictors
    predictors = []
    for col in available_corr + available_info:
        valid_mask = ~(results_df[col].isna() | results_df['test_accuracy'].isna())
        if valid_mask.sum() > 2:
            r, p = pearsonr(results_df.loc[valid_mask, col], 
                           results_df.loc[valid_mask, 'test_accuracy'])
            if p < 0.05:
                predictors.append((col, r, p))
    
    # Sort by correlation magnitude
    predictors.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for col, r, p in predictors[:5]:  # Top 5
        summary_text += f"\n  {col}: r={r:.3f} (p={p:.3f})"
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=10, va='top', fontfamily='monospace')
    
    for ax in axes.flat:
        if ax != axes[1, 2]:
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'key_information_relationships.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created comprehensive information vs correlation plots in {save_dir}")


def create_correlation_bar_plots(results_df: pd.DataFrame, save_dir: str):
    """Create bar plots showing correlation coefficients from scatter plot analyses."""
    
    # Define categories of metrics
    synaptic_correlation_metrics = [
        'mean_ee_correlation_total', 'mean_ei_correlation_total', 'mean_ii_correlation_total',
        'mean_ee_correlation_noise', 'mean_ei_correlation_noise', 'mean_ii_correlation_noise'
    ]
    
    tuning_metrics = [
        'mean_exc_tuning', 'mean_inh_tuning'
    ]
    
    information_metrics = [
        'I_E_C', 'I_I_C', 'I_Vout_C', 'I_EI_C', 
        'I_E_I', 'I_E_Vout', 'I_I_Vout',
        'integration_gain', 'ei_balance'
    ]
    
    all_metrics = synaptic_correlation_metrics + tuning_metrics + information_metrics
    
    # Compute correlations with test accuracy
    correlation_data = []
    
    for metric in all_metrics:
        if metric in results_df.columns:
            valid_mask = ~(results_df[metric].isna() | results_df['test_accuracy'].isna())
            if valid_mask.sum() > 2:  # Need at least 3 points for correlation
                x_data = results_df.loc[valid_mask, metric]
                y_data = results_df.loc[valid_mask, 'test_accuracy']
                
                try:
                    pearson_r, pearson_p = pearsonr(x_data, y_data)
                    spearman_r, spearman_p = spearmanr(x_data, y_data)
                    
                    # Determine category for coloring
                    if metric in synaptic_correlation_metrics:
                        category = 'Synaptic Correlation'
                        if 'total' in metric:
                            subcategory = 'Total'
                        else:
                            subcategory = 'Noise'
                    elif metric in tuning_metrics:
                        category = 'Synaptic Tuning'
                        subcategory = 'Tuning'
                    elif metric in information_metrics:
                        category = 'Information'
                        subcategory = 'Information'
                    else:
                        category = 'Other'
                        subcategory = 'Other'
                    
                    correlation_data.append({
                        'metric': metric,
                        'metric_clean': metric.replace('mean_', '').replace('_', ' ').title(),
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'n_samples': valid_mask.sum(),
                        'category': category,
                        'subcategory': subcategory,
                        'significant': pearson_p < 0.05
                    })
                except:
                    continue
    
    if not correlation_data:
        print("No valid correlation data available for bar plots")
        return
    
    corr_df = pd.DataFrame(correlation_data)
    
    # Create figure with subplots for different metric categories
    categories = corr_df['category'].unique()
    n_categories = len(categories)
    
    # 1. Overall correlation bar plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All correlations sorted by magnitude
    ax = axes[0, 0]
    sorted_corr = corr_df.sort_values('pearson_r', key=abs, ascending=False)
    
    colors = []
    for _, row in sorted_corr.iterrows():
        if row['significant']:
            if row['pearson_r'] > 0:
                colors.append('darkgreen')
            else:
                colors.append('darkred')
        else:
            colors.append('lightgray')
    
    bars = ax.barh(range(len(sorted_corr)), sorted_corr['pearson_r'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_corr)))
    ax.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                        for name in sorted_corr['metric_clean']], fontsize=8)
    ax.set_xlabel('Pearson Correlation with Test Accuracy')
    ax.set_title('All Correlations with Test Accuracy\n(Dark colors: p<0.05, Gray: p≥0.05)')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Add significance markers
    for i, (_, row) in enumerate(sorted_corr.iterrows()):
        if row['significant']:
            significance_marker = '***' if row['pearson_p'] < 0.001 else '**' if row['pearson_p'] < 0.01 else '*'
            x_pos = row['pearson_r'] + (0.02 if row['pearson_r'] >= 0 else -0.02)
            ax.text(x_pos, i, significance_marker, va='center', fontweight='bold', fontsize=8)
    
    # Plot 2: Correlations by category
    ax = axes[0, 1]
    category_colors = {'Synaptic Correlation': 'blue', 'Synaptic Tuning': 'red', 'Information': 'green', 'Other': 'gray'}
    
    for i, category in enumerate(categories):
        cat_data = corr_df[corr_df['category'] == category]
        significant_data = cat_data[cat_data['significant']]
        
        if len(significant_data) > 0:
            mean_corr = significant_data['pearson_r'].mean()
            std_corr = significant_data['pearson_r'].std() if len(significant_data) > 1 else 0
            
            color = category_colors.get(category, 'gray')
            bar = ax.bar(i, mean_corr, yerr=std_corr, color=color, alpha=0.7, 
                        capsize=5, label=f'{category} (n={len(significant_data)})')
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Mean Pearson Correlation (Significant Only)')
    ax.set_title('Mean Correlations by Metric Category')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: P-values vs correlation strength
    ax = axes[1, 0]
    scatter = ax.scatter(corr_df['pearson_r'], -np.log10(corr_df['pearson_p']), 
                        c=[category_colors.get(cat, 'gray') for cat in corr_df['category']], 
                        alpha=0.7, s=60)
    
    # Add significance line
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax.axhline(y=-np.log10(0.01), color='red', linestyle='--', alpha=0.5, label='p=0.01')
    
    ax.set_xlabel('Pearson Correlation with Test Accuracy')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Correlation Strength vs Statistical Significance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations for most significant correlations
    top_corr = corr_df.nlargest(3, 'pearson_r')
    for _, row in top_corr.iterrows():
        if row['significant']:
            ax.annotate(row['metric_clean'][:15], 
                       (row['pearson_r'], -np.log10(row['pearson_p'])),
                       xytext=(5, 5), textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Plot 4: Effect sizes and sample sizes
    ax = axes[1, 1]
    bubble_sizes = (corr_df['n_samples'] - corr_df['n_samples'].min() + 1) * 100 / corr_df['n_samples'].max()
    
    for category in categories:
        cat_data = corr_df[corr_df['category'] == category]
        if len(cat_data) > 0:
            ax.scatter(cat_data['pearson_r'], cat_data['pearson_r']**2, 
                      s=[bubble_sizes[i] for i in cat_data.index],
                      color=category_colors.get(category, 'gray'), 
                      alpha=0.6, label=category)
    
    ax.set_xlabel('Pearson Correlation (r)')
    ax.set_ylabel('Effect Size (r²)')
    ax.set_title('Effect Size vs Correlation\n(Bubble size ∝ sample size)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle(f'Correlation Analysis Summary (N={len(results_df)} experiments)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_bar_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed bar plots by category
    fig, axes = plt.subplots(len(categories), 1, figsize=(12, 4*len(categories)))
    if len(categories) == 1:
        axes = [axes]
    
    for i, category in enumerate(categories):
        ax = axes[i]
        cat_data = corr_df[corr_df['category'] == category].sort_values('pearson_r', key=abs, ascending=False)
        
        if len(cat_data) > 0:
            colors = ['darkgreen' if (row['pearson_r'] > 0 and row['significant']) 
                     else 'darkred' if (row['pearson_r'] < 0 and row['significant'])
                     else 'lightgray' for _, row in cat_data.iterrows()]
            
            bars = ax.bar(range(len(cat_data)), cat_data['pearson_r'], color=colors, alpha=0.7)
            ax.set_xticks(range(len(cat_data)))
            ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                               for name in cat_data['metric_clean']], rotation=45, ha='right')
            ax.set_ylabel('Pearson Correlation')
            ax.set_title(f'{category} Correlations with Test Accuracy')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Add correlation values on bars
            for j, (_, row) in enumerate(cat_data.iterrows()):
                height = row['pearson_r']
                ax.text(j, height + (0.01 if height >= 0 else -0.01), 
                       f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=8, fontweight='bold')
    
    plt.suptitle('Detailed Correlation Analysis by Category', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_bars_by_category.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save correlation summary table
    summary_table = corr_df[['metric_clean', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 
                            'n_samples', 'category', 'significant']].round(4)
    summary_table.to_csv(os.path.join(save_dir, 'correlation_summary_table.csv'), index=False)
    
    print(f"Created correlation bar plots in {save_dir}")
    
    # Print summary of significant correlations
    significant_corr = corr_df[corr_df['significant']].sort_values('pearson_r', key=abs, ascending=False)
    print(f"\nSignificant correlations with test accuracy (p < 0.05):")
    for _, row in significant_corr.iterrows():
        print(f"  {row['metric_clean']}: r = {row['pearson_r']:.3f} (p = {row['pearson_p']:.3f})")


def main():
    parser = argparse.ArgumentParser(description='Run multiple correlation analysis experiments')
    parser.add_argument('--config', type=str, 
                       default='configs/synthetic_correlated_experiment.yaml',
                       help='Base configuration file')
    parser.add_argument('--output-dir', type=str, 
                       default='results/correlation_multi_run',
                       help='Output directory for results')
    parser.add_argument('--n-runs', type=int, default=30,
                       help='Number of runs with different data realizations')
    parser.add_argument('--start-seed', type=int, default=1000,
                       help='Starting random seed')
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(args.output_dir, f'multi_run_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment parameters
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    exp_params = {
        'n_runs': args.n_runs,
        'start_seed': args.start_seed,
        'base_config': args.config,
        'task_params': base_config['task']['parameters']
    }
    
    with open(os.path.join(experiment_dir, 'experiment_params.yaml'), 'w') as f:
        yaml.dump(exp_params, f)
    
    # Run experiments
    all_results = []
    for i in range(args.n_runs):
        seed = args.start_seed + i
        run_name = f'run_{i:03d}_seed_{seed}'
        run_dir = os.path.join(experiment_dir, run_name)
        
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{args.n_runs}")
        print(f"Seed: {seed}")
        print(f"{'='*60}\n")
        
        # Create config
        config_path = create_experiment_config(
            args.config,
            seed=seed,
            output_dir=run_dir,
            run_name=run_name
        )
        
        # Run experiment
        results = run_single_experiment(config_path)
        
        if results and results.get('test_accuracy') is not None:
            results['run_id'] = i
            results['seed'] = seed
            all_results.append(results)
            print(f"Run {i+1} completed. Test accuracy: {results['test_accuracy']:.4f}")
        else:
            print(f"Run {i+1} failed or incomplete!")
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(experiment_dir, 'all_results.csv'), index=False)
    
    # Create analysis plots
    print("\nCreating analysis plots...")
    create_comprehensive_plots(results_df, experiment_dir)
    
    # Create information vs correlation plots
    print("\nCreating information vs correlation plots...")
    create_information_correlation_plots(results_df, experiment_dir)
    
    # Create correlation bar plots
    print("\nCreating correlation bar plots...")
    create_correlation_bar_plots(results_df, experiment_dir)
    
    # Compute summary statistics
    summary_stats = results_df.describe()
    summary_stats.to_csv(os.path.join(experiment_dir, 'summary_statistics.csv'))
    
    # Compute correlation analysis
    print("\nComputing correlation analysis...")
    correlation_results = {}
    
    # Correlations with accuracy
    for col in results_df.columns:
        if col not in ['run_id', 'seed', 'test_accuracy'] and 'mean_' in col:
            valid_mask = ~(results_df[col].isna() | results_df['test_accuracy'].isna())
            if valid_mask.sum() > 2:
                x_data = results_df.loc[valid_mask, col]
                y_data = results_df.loc[valid_mask, 'test_accuracy']
                pearson_r, pearson_p = pearsonr(x_data, y_data)
                spearman_r, spearman_p = spearmanr(x_data, y_data)
                
                correlation_results[col] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n_samples': valid_mask.sum()
                }
    
    # Save correlation results
    corr_results_df = pd.DataFrame(correlation_results).T
    corr_results_df.to_csv(os.path.join(experiment_dir, 'accuracy_correlations.csv'))
    
    print(f"\n{'='*60}")
    print(f"Multi-run analysis completed!")
    print(f"Results saved to: {experiment_dir}")
    print(f"{'='*60}\n")
    
    # Print summary
    print("\nSummary Results:")
    print(f"Mean test accuracy: {results_df['test_accuracy'].mean():.4f} ± {results_df['test_accuracy'].std():.4f}")
    print("\nSignificant correlations with accuracy (p < 0.05):")
    for var, stats in correlation_results.items():
        if stats['pearson_p'] < 0.05:
            print(f"  {var}: r = {stats['pearson_r']:.3f} (p = {stats['pearson_p']:.3f})")


if __name__ == '__main__':
    main()

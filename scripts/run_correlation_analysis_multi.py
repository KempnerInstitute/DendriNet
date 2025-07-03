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
        str(project_root / 'src' / 'dendritic_modeling' / 'models' / 'train.py'),
        '--config', config_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running experiment: {result.stderr}")
        return None
    
    # Extract results from output directory
    output_dir = os.path.dirname(config_path)
    
    # Read performance results
    perf_file = os.path.join(output_dir, 'performance', 'performance_final.csv')
    if os.path.exists(perf_file):
        perf_df = pd.read_csv(perf_file)
        test_accuracy = perf_df[perf_df['dataset'] == 'test']['accuracy'].values[0]
    else:
        test_accuracy = None
    
    # Read correlation analysis results
    corr_summary_file = os.path.join(output_dir, 'correlation_analysis', 'correlation_summary.csv')
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
        
        if results:
            results['run_id'] = i
            results['seed'] = seed
            all_results.append(results)
            print(f"Run {i+1} completed. Test accuracy: {results['test_accuracy']:.4f}")
        else:
            print(f"Run {i+1} failed!")
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(experiment_dir, 'all_results.csv'), index=False)
    
    # Create analysis plots
    print("\nCreating analysis plots...")
    create_comprehensive_plots(results_df, experiment_dir)
    
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

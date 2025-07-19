#!/usr/bin/env python3
"""
Run sweep experiments on synthetic correlated dataset with varying correlation ranges.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
from pathlib import Path
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def create_sweep_config(base_config_path: str, corr_min: float, corr_max: float,
                       output_dir: str, run_name: str) -> str:
    """Create a modified config for a specific correlation range."""
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Update correlation range
    config['task']['parameters']['corr_min'] = corr_min
    config['task']['parameters']['corr_max'] = corr_max

    # Update output directory
    config['outputs']['run_name'] = run_name
    config['outputs']['dir'] = output_dir

    # Save modified config
    config_path = os.path.join(output_dir, 'config.yaml')
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return config_path


def run_single_experiment(config_path: str) -> dict:
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


def plot_sweep_results(results_df: pd.DataFrame, save_dir: str):
    """Create plots summarizing sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Accuracy vs correlation range
    ax = axes[0, 0]
    ax.plot(results_df['corr_range_size'], results_df['test_accuracy'], 'o-')
    ax.set_xlabel('Correlation Range Size')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Performance vs Input Correlation Strength')
    ax.grid(True, alpha=0.3)

    # Plot 2: Learned correlations vs input correlations
    ax = axes[0, 1]
    if 'mean_ee_correlation_total' in results_df.columns:
        ax.scatter(results_df['corr_range_size'], results_df['mean_ee_correlation_total'],
                   label='E-E Total', alpha=0.7)
    if 'mean_ei_correlation_total' in results_df.columns:
        ax.scatter(results_df['corr_range_size'], results_df['mean_ei_correlation_total'],
                   label='E-I Total', alpha=0.7)
    if 'mean_ii_correlation_total' in results_df.columns:
        ax.scatter(results_df['corr_range_size'], results_df['mean_ii_correlation_total'],
                   label='I-I Total', alpha=0.7)
    ax.set_xlabel('Input Correlation Range Size')
    ax.set_ylabel('Mean Learned Correlation')
    ax.set_title('Learned Synaptic Correlations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Tuning strength vs correlation
    ax = axes[1, 0]
    ax.scatter(results_df['corr_range_size'], results_df['mean_exc_tuning'],
               label='Excitatory', color='red', alpha=0.7)
    ax.scatter(results_df['corr_range_size'], results_df['mean_inh_tuning'],
               label='Inhibitory', color='blue', alpha=0.7)
    ax.set_xlabel('Input Correlation Range Size')
    ax.set_ylabel('Mean Tuning Strength')
    ax.set_title('Synaptic Tuning vs Input Correlations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary heatmap
    ax = axes[1, 1]
    # Select available columns for heatmap
    heatmap_cols = ['test_accuracy']
    for col in ['mean_ee_correlation_total', 'mean_exc_tuning', 'mean_inh_tuning']:
        if col in results_df.columns:
            heatmap_cols.append(col)
    summary_data = results_df[heatmap_cols].T
    sns.heatmap(summary_data, ax=ax, cmap='viridis',
                xticklabels=[f"±{r}" for r in results_df['corr_range_size']],
                yticklabels=['Accuracy', 'E-E Corr', 'E Tuning', 'I Tuning'])
    ax.set_title('Summary Heatmap')
    ax.set_xlabel('Correlation Range')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sweep_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run correlation sweep experiments')
    parser.add_argument('--base-config', type=str,
                       default='configs/synthetic_correlated_experiment.yaml',
                       help='Base configuration file')
    parser.add_argument('--output-dir', type=str,
                       default='results/correlation_sweep',
                       help='Output directory for sweep results')
    parser.add_argument('--correlation-ranges', type=str,
                       default='0.0,0.05,0.1,0.15,0.2',
                       help='Comma-separated list of correlation range sizes')
    parser.add_argument('--seeds', type=str, default='42,43,44',
                       help='Comma-separated list of random seeds')
    args = parser.parse_args()

    # Parse correlation ranges
    corr_ranges = [float(x) for x in args.correlation_ranges.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = os.path.join(args.output_dir, f'sweep_{timestamp}')
    os.makedirs(sweep_dir, exist_ok=True)

    # Run experiments
    all_results = []
    for corr_range in corr_ranges:
        for seed in seeds:
            # Create experiment name
            exp_name = f'corr_{corr_range}_seed_{seed}'
            exp_dir = os.path.join(sweep_dir, exp_name)

            print(f"\n{'='*60}")
            print(f"Running experiment: {exp_name}")
            print(f"Correlation range: [-{corr_range}, {corr_range}]")
            print(f"Random seed: {seed}")
            print(f"{'='*60}\n")

            # Create config
            config_path = create_sweep_config(
                args.base_config,
                corr_min=-corr_range,
                corr_max=corr_range,
                output_dir=exp_dir,
                run_name=exp_name
            )

            # Update seed in config
            with open(config_path) as f:
                config = yaml.safe_load(f)
            config['train']['seed'] = seed
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            # Run experiment
            results = run_single_experiment(config_path)

            if results:
                results['corr_range_size'] = corr_range
                results['seed'] = seed
                results['exp_name'] = exp_name
                all_results.append(results)
                print(f"Experiment completed. Test accuracy: {results['test_accuracy']:.4f}")
            else:
                print("Experiment failed!")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(sweep_dir, 'sweep_results.csv'), index=False)

    # Compute mean and std across seeds
    agg_dict = {'test_accuracy': ['mean', 'std']}

    # Add available correlation columns
    for col in ['mean_ee_correlation_total', 'mean_ei_correlation_total', 'mean_ii_correlation_total',
                'mean_ee_correlation_noise', 'mean_ei_correlation_noise', 'mean_ii_correlation_noise',
                'mean_exc_tuning', 'mean_inh_tuning']:
        if col in results_df.columns:
            agg_dict[col] = ['mean', 'std']

    summary_df = results_df.groupby('corr_range_size').agg(agg_dict).round(4)

    summary_df.to_csv(os.path.join(sweep_dir, 'sweep_summary.csv'))

    # Create plots
    mean_results = results_df.groupby('corr_range_size').mean().reset_index()
    plot_sweep_results(mean_results, sweep_dir)

    print(f"\n{'='*60}")
    print(f"Sweep completed! Results saved to: {sweep_dir}")
    print(f"{'='*60}\n")

    # Print summary
    print("\nSummary Results:")
    print(summary_df)


if __name__ == '__main__':
    main()

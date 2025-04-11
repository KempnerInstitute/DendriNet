#!/usr/bin/env python
"""
analyze_ei_sweep.py
------------------
Generate plots from the results of an EI sweep experiment.

This script:
1. Takes a timestamp ID or folder path as input
2. Analyzes all the results for different IE and EE parameter values
3. Generates summary plots in the plots directory

Usage:
    python scripts/analyze_ei_sweep.py --timestamp 20250411144518
    python scripts/analyze_ei_sweep.py --folder /path/to/results/ei_20250411144518
    python scripts/analyze_ei_sweep.py --all     # Analyze all ei_ folders
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze EI sweep results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--timestamp",
        type=str,
        help="Timestamp ID of the experiment (e.g. 20250411144518)",
    )
    group.add_argument("--folder", type=str, help="Full path to the experiment folder")
    group.add_argument(
        "--all",
        action="store_true",
        help="Analyze all ei_ folders in the output directory",
    )
    group.add_argument(
        "--multiple",
        type=str,
        nargs="+",
        help="List of timestamp IDs to analyze and compare",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="/n/holylabs/LABS/kempner_dev/Users/hsafaai/results/dendritic-modeling/outputs",
        help="Base directory for outputs",
    )
    parser.add_argument(
        "--prefix", type=str, default="ei_", help="Prefix used for the output folder"
    )
    parser.add_argument(
        "--save_plots", action="store_true", help="Save plots to the plots directory"
    )
    parser.add_argument(
        "--show_plots", action="store_true", help="Display plots (not just save them)"
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        help="Directory to save summary plots when analyzing multiple folders",
    )
    return parser.parse_args()


def find_experiment_directory(timestamp, output_base, prefix):
    """Find the experiment directory based on the timestamp ID"""
    if timestamp:
        folder_name = f"{prefix}{timestamp}"
        folder_path = os.path.join(output_base, folder_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Experiment folder {folder_path} not found")
        return folder_path
    return None


def find_all_experiment_directories(output_base, prefix):
    """Find all experiment directories with the given prefix"""
    pattern = os.path.join(output_base, f"{prefix}*")
    folders = glob.glob(pattern)

    # Filter to only include directories
    folders = [f for f in folders if os.path.isdir(f)]

    if not folders:
        raise FileNotFoundError(
            f"No experiment folders found matching pattern {pattern}"
        )

    # Sort by timestamp (newest first)
    folders.sort(reverse=True)

    return folders


def extract_parameter_values_from_configs(configs_dir):
    """Extract the parameter values from config files"""
    config_files = glob.glob(os.path.join(configs_dir, "*.yaml"))

    # Exclude the base sweep_params.yaml file
    config_files = [
        f for f in config_files if os.path.basename(f) != "sweep_params.yaml"
    ]

    if not config_files:
        raise FileNotFoundError(f"No config files found in {configs_dir}")

    params_data = []

    for config_file in config_files:
        try:
            # Extract task ID from filename
            match = re.search(r"_(\d+)\.yaml$", config_file)
            if match:
                task_id = int(match.group(1))
            else:
                continue

            # Load config
            config = OmegaConf.load(config_file)

            # Extract parameters of interest (EE and IE values)
            ee_values = (
                config.get("model", {})
                .get("network", {})
                .get("parameters", {})
                .get("ee_synapses_per_branch_per_layer", [])
            )
            ie_values = (
                config.get("model", {})
                .get("network", {})
                .get("parameters", {})
                .get("ie_synapses_per_branch_per_layer", [])
            )

            # Get seed value
            seed = config.get("train", {}).get("seed", 0)

            # Extract as single values if they're lists with one element
            if isinstance(ee_values, list) and len(ee_values) == 1:
                ee_value = ee_values[0]
            else:
                ee_value = ee_values

            if isinstance(ie_values, list) and len(ie_values) == 1:
                ie_value = ie_values[0]
            else:
                ie_value = ie_values

            params_data.append(
                {
                    "task_id": task_id,
                    "ee_value": ee_value,
                    "ie_value": ie_value,
                    "seed": seed,
                }
            )

        except Exception as e:
            print(f"Error processing config file {config_file}: {e}")

    return pd.DataFrame(params_data)


def collect_results(exp_dir, params_df):
    """Collect results from the experiment directory"""
    results_dir = os.path.join(exp_dir, "results")
    exp_timestamp = os.path.basename(exp_dir)  # Get folder name for identification

    # Find all job directories
    job_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    # Initialize results storage
    results_data = []

    for job_dir in job_dirs:
        job_path = os.path.join(results_dir, job_dir)

        # Look for task directories
        task_dirs = glob.glob(os.path.join(job_path, "task_*"))

        for task_dir in task_dirs:
            # Extract task ID
            task_match = re.search(r"task_(\d+)$", task_dir)
            if not task_match:
                continue

            task_id = int(task_match.group(1))

            # Look for results files
            performance_dir = os.path.join(task_dir, "performance")
            if not os.path.exists(performance_dir):
                continue

            # Look for final performance file
            final_perf_file = os.path.join(performance_dir, "final.json")
            if not os.path.exists(final_perf_file):
                continue

            try:
                with open(final_perf_file, "r") as f:
                    perf_data = json.load(f)

                # Get parameter values for this task
                task_params = params_df[params_df["task_id"] == task_id]
                if task_params.empty:
                    print(f"Warning: No parameter info found for task {task_id}")
                    continue

                # Extract values
                ee_value = task_params["ee_value"].values[0]
                ie_value = task_params["ie_value"].values[0]
                seed = task_params["seed"].values[0]

                # Add to results
                results_data.append(
                    {
                        "task_id": task_id,
                        "job_id": job_dir,
                        "ee_value": ee_value,
                        "ie_value": ie_value,
                        "seed": seed,
                        "train_acc": perf_data.get("train_acc", 0),
                        "valid_acc": perf_data.get("valid_acc", 0),
                        "test_acc": perf_data.get("test_acc", 0),
                        "experiment": exp_timestamp,  # Add experiment identifier
                    }
                )

            except Exception as e:
                print(f"Error processing results for task {task_id}: {e}")

    return pd.DataFrame(results_data)


def generate_plots(results_df, plots_dir, show_plots=False, experiment_name=None):
    """Generate summary plots from the results"""
    if results_df.empty:
        print("No results to plot")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Set a consistent style
    sns.set(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    # Add title prefix if multiple experiments
    title_prefix = f"{experiment_name} - " if experiment_name else ""

    # 1. Plot heatmap of test accuracy based on EE and IE values
    plt.figure(figsize=(12, 10))

    # Group by EE and IE values and calculate mean test accuracy
    heatmap_data = (
        results_df.groupby(["ee_value", "ie_value"])["test_acc"].mean().reset_index()
    )
    heatmap_pivot = heatmap_data.pivot(
        index="ie_value", columns="ee_value", values="test_acc"
    )

    ax = sns.heatmap(
        heatmap_pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Test Accuracy"},
    )
    plt.title(f"{title_prefix}Effect of EE and IE Synapses on Test Accuracy")
    plt.xlabel("EE Synapses Per Branch")
    plt.ylabel("IE Synapses Per Branch")

    # Save the plot
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.savefig(os.path.join(plots_dir, "ee_ie_heatmap.png"), dpi=300)
    plt.close()

    # 2. Line plot: EE values vs test accuracy for different IE values
    plt.figure(figsize=(12, 8))

    # Group by EE and IE values and calculate mean and std of test accuracy
    line_data = (
        results_df.groupby(["ee_value", "ie_value"])[["test_acc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    line_data.columns = ["ee_value", "ie_value", "test_acc_mean", "test_acc_std"]

    # Get unique IE values
    ie_values = sorted(results_df["ie_value"].unique())

    # Plot line for each IE value
    for ie_val in ie_values:
        subset = line_data[line_data["ie_value"] == ie_val]
        plt.errorbar(
            subset["ee_value"],
            subset["test_acc_mean"],
            yerr=subset["test_acc_std"],
            marker="o",
            label=f"IE={ie_val}",
        )

    plt.title(f"{title_prefix}Test Accuracy vs EE Synapses for Different IE Values")
    plt.xlabel("EE Synapses Per Branch")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.legend(title="IE Synapses")

    # Save the plot
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.savefig(os.path.join(plots_dir, "ee_vs_accuracy.png"), dpi=300)
    plt.close()

    # 3. Line plot: IE values vs test accuracy for different EE values
    plt.figure(figsize=(12, 8))

    # Get unique EE values
    ee_values = sorted(results_df["ee_value"].unique())

    # Plot line for each EE value
    for ee_val in ee_values:
        subset = line_data[line_data["ee_value"] == ee_val]
        plt.errorbar(
            subset["ie_value"],
            subset["test_acc_mean"],
            yerr=subset["test_acc_std"],
            marker="o",
            label=f"EE={ee_val}",
        )

    plt.title(f"{title_prefix}Test Accuracy vs IE Synapses for Different EE Values")
    plt.xlabel("IE Synapses Per Branch")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.legend(title="EE Synapses")

    # Save the plot
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.savefig(os.path.join(plots_dir, "ie_vs_accuracy.png"), dpi=300)
    plt.close()

    # 4. Bar plot: Comparison of validation and test accuracy
    plt.figure(figsize=(14, 8))

    # Prepare data for grouped bar chart
    # Create a unique identifier for each parameter combination
    results_df["param_combo"] = results_df.apply(
        lambda row: f"EE={row['ee_value']},IE={row['ie_value']}", axis=1
    )

    # Group by parameter combination
    bar_data = (
        results_df.groupby("param_combo")[["valid_acc", "test_acc"]]
        .mean()
        .reset_index()
    )

    # Sort by test accuracy
    bar_data = bar_data.sort_values("test_acc", ascending=False)

    # Take top N combinations
    top_n = min(15, len(bar_data))
    bar_data = bar_data.head(top_n)

    # Plot
    bar_width = 0.35
    x = np.arange(len(bar_data))

    plt.bar(x - bar_width / 2, bar_data["valid_acc"], bar_width, label="Validation")
    plt.bar(x + bar_width / 2, bar_data["test_acc"], bar_width, label="Test")

    plt.xlabel("Parameter Combination")
    plt.ylabel("Accuracy")
    plt.title(
        f"{title_prefix}Top {top_n} Parameter Combinations: Validation vs Test Accuracy"
    )
    plt.xticks(x, bar_data["param_combo"], rotation=45, ha="right")
    plt.legend()

    # Save the plot
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.savefig(os.path.join(plots_dir, "top_combinations.png"), dpi=300)
    plt.close()

    # 5. Save the summarized results to CSV
    summary_file = os.path.join(plots_dir, "ei_sweep_results.csv")
    results_df.to_csv(summary_file, index=False)

    # 6. Create a summary text file with the best parameters
    best_params = results_df.loc[results_df["test_acc"].idxmax()]
    summary_text = f"""
    EI Sweep Results Summary
    -----------------------

    Best Parameter Combination:
      EE Synapses: {best_params['ee_value']}
      IE Synapses: {best_params['ie_value']}

    Performance:
      Training Accuracy: {best_params['train_acc']:.4f}
      Validation Accuracy: {best_params['valid_acc']:.4f}
      Test Accuracy: {best_params['test_acc']:.4f}

    Task ID: {best_params['task_id']}
    Job ID: {best_params['job_id']}
    Seed: {best_params['seed']}
    Experiment: {best_params.get('experiment', 'N/A')}

    Total parameter combinations tested: {len(results_df['param_combo'].unique())}
    Total runs: {len(results_df)}
    """

    with open(os.path.join(plots_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    print("Plot generation completed.")
    print(f"Summary plots saved to: {plots_dir}")


def generate_comparison_plots(all_results_df, summary_dir):
    """Generate plots comparing results across multiple experiments"""
    if all_results_df.empty:
        print("No results to plot")
        return

    os.makedirs(summary_dir, exist_ok=True)

    # Set a consistent style
    sns.set(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    # Extract experiment names for labeling
    experiments = sorted(all_results_df["experiment"].unique())

    # 1. Box plot of test accuracy by experiment
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x="experiment", y="test_acc", data=all_results_df)
    plt.title("Test Accuracy Comparison Across Experiments")
    plt.xlabel("Experiment")
    plt.ylabel("Test Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "experiment_comparison_boxplot.png"), dpi=300)
    plt.close()

    # 2. Heatmap comparison of best performing parameter combinations
    # Get the best parameter combination for each experiment
    best_by_exp = (
        all_results_df.sort_values("test_acc", ascending=False)
        .groupby("experiment")
        .first()
    )
    best_by_exp = best_by_exp.reset_index()[
        ["experiment", "ee_value", "ie_value", "test_acc"]
    ]

    # Create a pivot table for the heatmap
    pivot_data = all_results_df.pivot_table(
        index=["ee_value", "ie_value"],
        columns="experiment",
        values="test_acc",
        aggfunc="mean",
    )

    # Plot a heatmap for each experiment
    for exp in experiments:
        plt.figure(figsize=(10, 8))
        exp_data = pivot_data[exp].unstack()
        sns.heatmap(exp_data, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"Test Accuracy for {exp}")
        plt.xlabel("EE Synapses Per Branch")
        plt.ylabel("IE Synapses Per Branch")
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, f"heatmap_{exp}.png"), dpi=300)
        plt.close()

    # 3. Bar chart comparing the best parameter combinations
    plt.figure(figsize=(14, 8))

    # Prepare data for the bar chart
    bar_data = best_by_exp.copy()
    bar_data["param_combo"] = bar_data.apply(
        lambda row: f"EE={row['ee_value']},IE={row['ie_value']}", axis=1
    )

    # Plot
    x = np.arange(len(bar_data))
    plt.bar(x, bar_data["test_acc"], color=sns.color_palette("viridis", len(bar_data)))

    plt.xlabel("Experiment")
    plt.ylabel("Best Test Accuracy")
    plt.title("Best Performing Parameter Combination by Experiment")
    plt.xticks(x, bar_data["experiment"], rotation=45, ha="right")

    # Add parameter combo as text on each bar
    for i, row in enumerate(bar_data.itertuples()):
        plt.text(
            i,
            row.test_acc - 0.05,
            row.param_combo,
            ha="center",
            va="top",
            rotation=90,
            color="white",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "best_params_by_experiment.png"), dpi=300)
    plt.close()

    # 4. Line plot showing the best test accuracy for each (EE, IE) combination across experiments
    # First, get the unique parameter combinations
    all_results_df["param_combo"] = all_results_df.apply(
        lambda row: f"EE={row['ee_value']},IE={row['ie_value']}", axis=1
    )
    param_combos = all_results_df["param_combo"].unique()

    # Then create a pivot table with parameter combinations as rows and experiments as columns
    pivot_test_acc = all_results_df.pivot_table(
        index="param_combo", columns="experiment", values="test_acc", aggfunc="mean"
    )

    # Plot the top N parameter combinations
    top_n = min(10, len(pivot_test_acc))
    top_combos = (
        pivot_test_acc.mean(axis=1).sort_values(ascending=False).head(top_n).index
    )

    plt.figure(figsize=(14, 10))
    for combo in top_combos:
        plt.plot(pivot_test_acc.loc[combo], marker="o", label=combo)

    plt.xlabel("Experiment")
    plt.ylabel("Test Accuracy")
    plt.title(f"Top {top_n} Parameter Combinations Across Experiments")
    plt.legend(title="Parameter Combination")
    plt.grid(True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(summary_dir, "param_comparison_across_experiments.png"), dpi=300
    )
    plt.close()

    # 5. Save the summarized combined results to CSV
    summary_file = os.path.join(summary_dir, "all_experiments_results.csv")
    all_results_df.to_csv(summary_file, index=False)

    # 6. Create a summary text file
    summary_text = f"""
    Multiple EI Sweep Experiments Summary
    ------------------------------------

    Experiments analyzed: {len(experiments)}
    Experiment IDs: {', '.join(experiments)}

    Best performing experiment: {best_by_exp.iloc[0]['experiment']}
    Best parameter combination: EE={best_by_exp.iloc[0]['ee_value']}, IE={best_by_exp.iloc[0]['ie_value']}
    Best test accuracy: {best_by_exp.iloc[0]['test_acc']:.4f}

    Summary of best parameter combinations by experiment:

    """

    for _, row in best_by_exp.iterrows():
        summary_text += f"    {row['experiment']}: EE={row['ee_value']}, IE={row['ie_value']}, Acc={row['test_acc']:.4f}\n"

    summary_text += f"""
    Total parameter combinations tested across all experiments: {len(param_combos)}
    Total runs analyzed: {len(all_results_df)}
    """

    with open(os.path.join(summary_dir, "multi_experiment_summary.txt"), "w") as f:
        f.write(summary_text)

    print("Comparison plot generation completed.")
    print(f"Summary comparison plots saved to: {summary_dir}")


def process_single_experiment(exp_dir, show_plots=False):
    """Process a single experiment directory"""
    # Extract experiment name from the directory path
    exp_name = os.path.basename(exp_dir)

    # Ensure required directories exist
    configs_dir = os.path.join(exp_dir, "configs")
    results_dir = os.path.join(exp_dir, "results")
    plots_dir = os.path.join(exp_dir, "plots")

    if not os.path.exists(configs_dir) or not os.path.exists(results_dir):
        print(f"Warning: Required directories not found in {exp_dir}. Skipping.")
        return None

    try:
        # Extract parameter values from config files
        print(f"Extracting parameter values from config files in {exp_name}...")
        params_df = extract_parameter_values_from_configs(configs_dir)
        print(f"Found {len(params_df)} parameter combinations")

        # Collect results
        print(f"Collecting results from {exp_name}...")
        results_df = collect_results(exp_dir, params_df)
        print(f"Collected results for {len(results_df)} runs")

        # Generate plots
        print(f"Generating plots for {exp_name}...")
        generate_plots(
            results_df, plots_dir, show_plots=show_plots, experiment_name=exp_name
        )

        return results_df

    except Exception as e:
        print(f"Error processing experiment {exp_name}: {e}")
        return None


def main():
    args = parse_args()

    # Create timestamp for summary directory if analyzing multiple experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If analyzing a single experiment
    if args.timestamp or args.folder:
        # Determine the experiment directory
        if args.folder:
            exp_dir = args.folder
        else:
            exp_dir = find_experiment_directory(
                args.timestamp, args.output_base, args.prefix
            )

        print(f"Analyzing experiment in: {exp_dir}")
        process_single_experiment(exp_dir, show_plots=args.show_plots)

    # If analyzing multiple experiments
    elif args.all or args.multiple:
        # Find all experiment directories to analyze
        if args.all:
            exp_dirs = find_all_experiment_directories(args.output_base, args.prefix)
            print(f"Found {len(exp_dirs)} experiment directories to analyze")
        else:
            exp_dirs = [
                find_experiment_directory(ts, args.output_base, args.prefix)
                for ts in args.multiple
            ]
            print(f"Analyzing {len(exp_dirs)} specified experiment directories")

        # Create a summary directory for multi-experiment plots
        summary_dir = args.summary_dir
        if not summary_dir:
            summary_dir = os.path.join(
                args.output_base, f"multi_exp_summary_{timestamp}"
            )
        os.makedirs(summary_dir, exist_ok=True)

        # Process each experiment and collect results
        all_results = []
        for exp_dir in exp_dirs:
            results_df = process_single_experiment(exp_dir, show_plots=args.show_plots)
            if results_df is not None and not results_df.empty:
                all_results.append(results_df)

        # Combine all results for comparison plots
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            print("\nGenerating comparison plots across all experiments...")
            generate_comparison_plots(combined_results, summary_dir)
            print(f"Saved comparison plots to: {summary_dir}")
        else:
            print(
                "No valid results found across experiments. Cannot generate comparison plots."
            )

    print("Analysis complete!")


if __name__ == "__main__":
    main()

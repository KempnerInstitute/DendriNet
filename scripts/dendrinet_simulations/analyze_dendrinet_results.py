#!/usr/bin/env python3
"""
analyze_dendrinet_results.py
---------------------------
Script to analyze and visualize results from dendrinet simulation array jobs.

This script:
1. Collects results from all completed jobs for a given experiment
2. Combines results into a unified dataset
3. Generates visualizations to understand parameter effects
4. Creates detailed reports on synergy and model performance

Usage:
    python scripts/analyze_dendrinet_results.py --timestamp 20240501123456
    python scripts/analyze_dendrinet_results.py --folder /path/to/results/dendrinet_20240501123456
    python scripts/analyze_dendrinet_results.py --all  # Analyze all dendrinet_ folders
"""

import argparse
import glob
import json
import os
import re
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze dendrinet simulation results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--timestamp",
        type=str,
        help="Timestamp ID of the experiment (e.g. 20240501123456)",
    )
    group.add_argument("--folder", type=str, help="Full path to the experiment folder")
    group.add_argument(
        "--all",
        action="store_true",
        help="Analyze all dendrinet_ folders in the output directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Directory to save output files (default: same as experiment folder)",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=os.path.join(os.getcwd(), "outputs"),
        help="Base output directory (default: ./outputs)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="dendrinet_",
        help="Prefix of experiment folders to look for (default: dendrinet_)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots in addition to saving them",
    )
    parser.add_argument(
        "--report-missing",
        action="store_true",
        help="Report detail of missing or unfinished runs",
    )
    return parser.parse_args()


def find_experiment_directory(timestamp, output_base, prefix="dendrinet_"):
    """Find the experiment directory based on timestamp"""
    folder_path = os.path.join(output_base, f"{prefix}{timestamp}")
    if not os.path.exists(folder_path):
        # Try searching for folders with matching timestamp
        folders = glob.glob(os.path.join(output_base, f"{prefix}*{timestamp}*"))
        if folders:
            folder_path = folders[0]
        else:
            raise ValueError(
                f"Could not find experiment folder with timestamp {timestamp}"
            )
    return folder_path


def collect_configs(exp_dir):
    """Collect configuration files from the config directory"""
    configs_dir = os.path.join(exp_dir, "configs")
    if not os.path.exists(configs_dir):
        print(f"Warning: Config directory not found at {configs_dir}")
        return {}

    # Find all config files
    config_files = glob.glob(os.path.join(configs_dir, "dendrinet_sim_config_*.yaml"))

    configs = {}
    for config_file in config_files:
        try:
            # Extract task id from filename
            task_match = re.search(
                r"dendrinet_sim_config_(\d+)\.yaml", os.path.basename(config_file)
            )
            if not task_match:
                continue

            task_id = int(task_match.group(1))

            # Load config
            with open(config_file) as f:
                config = yaml.safe_load(f)

            configs[task_id] = {"config_file": config_file, "parameters": config}
        except Exception as e:
            print(f"Error processing config file {config_file}: {e!s}")

    return configs


def collect_results(exp_dir, configs):
    """
    Collect results from all completed jobs.
    Returns a DataFrame with combined results and a list of missing jobs.
    """
    results_dir = os.path.join(exp_dir, "results")
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory not found at {results_dir}")
        return pd.DataFrame(), list(configs.keys())

    # Find all job directories
    job_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    all_results = []
    missing_jobs = []

    # Track which tasks have results
    found_task_ids = set()

    for job_dir in job_dirs:
        job_path = os.path.join(results_dir, job_dir)

        # Find all task directories
        task_dirs = glob.glob(os.path.join(job_path, "task_*"))

        for task_dir in task_dirs:
            # Extract task ID
            task_match = re.search(r"task_(\d+)$", task_dir)
            if not task_match:
                continue

            task_id = int(task_match.group(1))

            # Check for results file
            results_file = os.path.join(task_dir, "simulation_results.json")
            if not os.path.exists(results_file):
                missing_jobs.append(task_id)
                continue

            # Load results
            try:
                with open(results_file) as f:
                    job_results = json.load(f)

                # Add task ID and config parameters to each result
                for result in job_results:
                    result["task_id"] = task_id

                    # Add any missing parameters from config
                    if task_id in configs:
                        for key, value in configs[task_id]["parameters"].items():
                            if key not in result:
                                result[key] = value

                all_results.extend(job_results)
                found_task_ids.add(task_id)
            except Exception as e:
                print(f"Error processing results from task {task_id}: {e!s}")
                missing_jobs.append(task_id)

    # Check for configs without results
    for task_id in configs:
        if task_id not in found_task_ids:
            missing_jobs.append(task_id)

    # Convert to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
    else:
        results_df = pd.DataFrame()

    return results_df, list(set(missing_jobs))


def save_combined_results(results_df, exp_dir):
    """Save the combined results to CSV"""
    if results_df.empty:
        print("No results to save")
        return

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Save to CSV
    csv_path = os.path.join(plots_dir, "combined_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Combined results saved to {csv_path}")

    # Skip parquet save due to dependency issues
    print("Skipping parquet save due to dependency issues")


def create_parameter_effect_plots(results_df, output_dir):
    """Create plots showing the effect of each parameter on performance"""
    if results_df.empty:
        print("No results available for parameter effect plots")
        return

    # Create directory for parameter effect plots
    params_dir = os.path.join(output_dir, "parameter_effects")
    os.makedirs(params_dir, exist_ok=True)

    # Define metrics to plot
    metrics = ["accuracy", "final_mi", "syn_ee", "syn_ii", "syn_ei"]

    # Define parameters to analyze
    parameters = ["N_E", "N_I", "rho", "cEE", "cII", "cEI", "E_group", "I_group"]

    # Plot effect of each parameter on each metric
    for param in parameters:
        if param not in results_df.columns:
            continue

        # Create plot
        plt.figure(figsize=(12, 8))

        for i, metric in enumerate(metrics):
            if metric not in results_df.columns:
                continue

            plt.subplot(len(metrics), 1, i + 1)

            # Group by parameter and model_type, calculate mean and std
            grouped = (
                results_df.groupby([param, "model_type"])[metric]
                .agg(["mean", "std"])
                .reset_index()
            )

            # Plot for each model type
            for model_type in results_df["model_type"].unique():
                model_data = grouped[grouped["model_type"] == model_type]
                plt.errorbar(
                    model_data[param],
                    model_data["mean"],
                    yerr=model_data["std"],
                    marker="o",
                    label=f"{model_type}",
                )

            plt.ylabel(metric)
            if i == 0:
                plt.title(f"Effect of {param} on performance metrics")
            if i == len(metrics) - 1:
                plt.xlabel(param)
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(params_dir, f"effect_of_{param}.png"), dpi=300)
        plt.close()

    # Create E/I ratio plots if both N_E and N_I are present
    if "N_E" in results_df.columns and "N_I" in results_df.columns:
        # Calculate E/I ratio
        results_df["EI_ratio"] = results_df["N_E"] / results_df["N_I"]

        plt.figure(figsize=(12, 8))

        for i, metric in enumerate(metrics):
            if metric not in results_df.columns:
                continue

            plt.subplot(len(metrics), 1, i + 1)

            # Group by E/I ratio and model_type
            results_df["EI_ratio_bin"] = pd.cut(results_df["EI_ratio"], bins=10)
            grouped = (
                results_df.groupby(["EI_ratio_bin", "model_type"])[metric]
                .agg(["mean", "std"])
                .reset_index()
            )

            # Convert bin edges to midpoints for plotting
            grouped["EI_ratio_value"] = grouped["EI_ratio_bin"].apply(lambda x: x.mid)

            # Plot for each model type
            for model_type in results_df["model_type"].unique():
                model_data = grouped[grouped["model_type"] == model_type]
                plt.errorbar(
                    model_data["EI_ratio_value"],
                    model_data["mean"],
                    yerr=model_data["std"],
                    marker="o",
                    label=f"{model_type}",
                )

            plt.ylabel(metric)
            if i == 0:
                plt.title("Effect of E/I ratio on performance metrics")
            if i == len(metrics) - 1:
                plt.xlabel("E/I ratio")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(params_dir, "effect_of_ei_ratio.png"), dpi=300)
        plt.close()


def create_correlation_heatmaps(results_df, output_dir):
    """Create heatmaps showing the effect of correlation parameters"""
    if results_df.empty:
        print("No results available for correlation heatmaps")
        return

    # Check if required columns exist
    required_cols = ["cEE", "cII", "cEI", "accuracy", "final_mi", "model_type"]
    if not all(col in results_df.columns for col in required_cols):
        print("Missing required columns for correlation heatmaps")
        return

    # Create directory for correlation plots
    corr_dir = os.path.join(output_dir, "correlation_effects")
    os.makedirs(corr_dir, exist_ok=True)

    # Define metrics to plot
    metrics = ["accuracy", "final_mi", "syn_ee", "syn_ii", "syn_ei"]

    # Create heatmaps for different combinations of correlation parameters
    correlation_pairs = [("cEE", "cII"), ("cEE", "cEI"), ("cII", "cEI")]

    for metric in metrics:
        if metric not in results_df.columns:
            continue

        for model_type in results_df["model_type"].unique():
            model_data = results_df[results_df["model_type"] == model_type]

            for corr_x, corr_y in correlation_pairs:
                plt.figure(figsize=(10, 8))

                # Create pivot table
                pivot = model_data.pivot_table(
                    index=corr_x, columns=corr_y, values=metric, aggfunc="mean"
                )

                # Create heatmap
                sns.heatmap(
                    pivot,
                    annot=True,
                    fmt=".3f",
                    cmap="viridis",
                    cbar_kws={"label": f"Mean {metric}"},
                )

                plt.title(f"{metric} by {corr_x} and {corr_y} ({model_type} model)")
                plt.tight_layout()

                # Save figure
                plt.savefig(
                    os.path.join(
                        corr_dir, f"{metric}_{corr_x}_{corr_y}_{model_type}.png"
                    ),
                    dpi=300,
                )
                plt.close()


def create_group_info_plots(results_df, output_dir):
    """Create plots showing the effect of group information assignments"""
    if results_df.empty:
        print("No results available for group information plots")
        return

    # Check if required columns exist
    required_cols = ["E_group", "I_group", "accuracy", "final_mi", "model_type"]
    if not all(col in results_df.columns for col in required_cols):
        print("Missing required columns for group information plots")
        return

    # Create directory for group info plots
    group_dir = os.path.join(output_dir, "group_effects")
    os.makedirs(group_dir, exist_ok=True)

    # Define metrics to plot
    metrics = ["accuracy", "final_mi", "syn_ee", "syn_ii", "syn_ei"]

    # Create mapping for group labels
    group_labels = {0: "No info", 1: "Medium info", 2: "High info"}

    for metric in metrics:
        if metric not in results_df.columns:
            continue

        for model_type in results_df["model_type"].unique():
            model_data = results_df[results_df["model_type"] == model_type]

            plt.figure(figsize=(10, 8))

            # Create pivot table
            pivot = model_data.pivot_table(
                index="E_group", columns="I_group", values=metric, aggfunc="mean"
            )

            # Rename index and columns with more descriptive labels
            pivot.index = [group_labels.get(i, i) for i in pivot.index]
            pivot.columns = [group_labels.get(i, i) for i in pivot.columns]

            # Create heatmap
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                cbar_kws={"label": f"Mean {metric}"},
            )

            plt.title(f"{metric} by E and I Group Information ({model_type} model)")
            plt.xlabel("I Group Information")
            plt.ylabel("E Group Information")
            plt.tight_layout()

            # Save figure
            plt.savefig(
                os.path.join(group_dir, f"{metric}_by_group_{model_type}.png"), dpi=300
            )
            plt.close()


def create_synergy_plots(results_df, output_dir):
    """Create specialized plots to visualize synergy patterns"""
    if results_df.empty:
        print("No results available for synergy plots")
        return

    # Check if required columns exist
    required_cols = [
        "syn_ee",
        "syn_ii",
        "syn_ei",
        "red_ee",
        "red_ii",
        "red_ei",
        "model_type",
    ]
    if not all(col in results_df.columns for col in required_cols):
        print("Missing required columns for synergy plots")
        return

    # Create directory for synergy plots
    synergy_dir = os.path.join(output_dir, "synergy_analysis")
    os.makedirs(synergy_dir, exist_ok=True)

    # Melt the dataframe to get synergy and redundancy in long format
    synergy_cols = ["syn_ee", "syn_ii", "syn_ei"]
    redundancy_cols = ["red_ee", "red_ii", "red_ei"]

    syn_df = results_df.melt(
        id_vars=[
            "E_group",
            "I_group",
            "cEE",
            "cII",
            "cEI",
            "model_type",
            "accuracy",
            "final_mi",
        ],
        value_vars=synergy_cols,
        var_name="synergy_type",
        value_name="synergy_value",
    )

    red_df = results_df.melt(
        id_vars=[
            "E_group",
            "I_group",
            "cEE",
            "cII",
            "cEI",
            "model_type",
            "accuracy",
            "final_mi",
        ],
        value_vars=redundancy_cols,
        var_name="redundancy_type",
        value_name="redundancy_value",
    )

    # Plot 1: Synergy by type and model
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=syn_df, x="synergy_type", y="synergy_value", hue="model_type")
    plt.title("Synergy Distribution by Type and Model")
    plt.xlabel("Synergy Type")
    plt.ylabel("Synergy Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(synergy_dir, "synergy_by_type.png"), dpi=300)
    plt.close()

    # Plot 2: Redundancy by type and model
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=red_df, x="redundancy_type", y="redundancy_value", hue="model_type"
    )
    plt.title("Redundancy Distribution by Type and Model")
    plt.xlabel("Redundancy Type")
    plt.ylabel("Redundancy Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(synergy_dir, "redundancy_by_type.png"), dpi=300)
    plt.close()

    # Plot 3: Correlation between synergy and performance
    for model_type in results_df["model_type"].unique():
        plt.figure(figsize=(14, 8))

        # Filter data for this model type
        syn_model_df = syn_df[syn_df["model_type"] == model_type]

        plt.subplot(1, 2, 1)
        sns.scatterplot(
            data=syn_model_df,
            x="synergy_value",
            y="accuracy",
            hue="synergy_type",
            alpha=0.7,
        )
        plt.title(f"Synergy vs Accuracy ({model_type} model)")
        plt.xlabel("Synergy Value")
        plt.ylabel("Accuracy")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        sns.scatterplot(
            data=syn_model_df,
            x="synergy_value",
            y="final_mi",
            hue="synergy_type",
            alpha=0.7,
        )
        plt.title(f"Synergy vs Mutual Information ({model_type} model)")
        plt.xlabel("Synergy Value")
        plt.ylabel("Mutual Information")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(synergy_dir, f"synergy_vs_performance_{model_type}.png"),
            dpi=300,
        )
        plt.close()

    # Plot 4: Combined scatter plot with synergy types
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=syn_df,
        x="synergy_value",
        y="accuracy",
        hue="synergy_type",
        style="model_type",
        alpha=0.7,
    )
    plt.title("Synergy vs Accuracy (All Models)")
    plt.xlabel("Synergy Value")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(synergy_dir, "synergy_vs_accuracy.png"), dpi=300)
    plt.close()


def find_optimal_configurations(results_df, output_dir):
    """Find and report the best-performing configurations"""
    if results_df.empty:
        print("No results available for optimal configuration analysis")
        return

    # Create directory for optimal configurations
    optimal_dir = os.path.join(output_dir, "optimal_configs")
    os.makedirs(optimal_dir, exist_ok=True)

    # Define metrics for optimization
    optimization_metrics = {
        "accuracy": "max",
        "final_mi": "max",
        "syn_ee": "max",
        "syn_ii": "max",
        "syn_ei": "max",
    }

    optimal_configs = {}

    # Find optimal configurations for each metric and model type
    for model_type in results_df["model_type"].unique():
        model_data = results_df[results_df["model_type"] == model_type]

        for metric, direction in optimization_metrics.items():
            if metric not in model_data.columns:
                continue

            if direction == "max":
                optimal_idx = model_data[metric].idxmax()
            else:
                optimal_idx = model_data[metric].idxmin()

            optimal_config = model_data.loc[optimal_idx].to_dict()
            optimal_configs[f"{model_type}_{metric}"] = optimal_config

    # Save optimal configurations
    with open(os.path.join(optimal_dir, "optimal_configurations.json"), "w") as f:
        # Filter out non-serializable values (like numpy arrays)
        filtered_configs = {}
        for key, config in optimal_configs.items():
            filtered_config = {}
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                    filtered_config[k] = v
            filtered_configs[key] = filtered_config
        json.dump(filtered_configs, f, indent=2)

    # Create summary table
    summary_rows = []

    for key, config in optimal_configs.items():
        model_type, metric = key.split("_", 1)

        # Extract relevant parameters
        summary_row = {
            "Model Type": model_type,
            "Metric": metric,
            "Value": config.get(metric, "N/A"),
            "N_E": config.get("N_E", "N/A"),
            "N_I": config.get("N_I", "N/A"),
            "E/I Ratio": config.get("N_E", 0) / config.get("N_I", 1),
            "cEE": config.get("cEE", "N/A"),
            "cII": config.get("cII", "N/A"),
            "cEI": config.get("cEI", "N/A"),
            "E_group": config.get("E_group", "N/A"),
            "I_group": config.get("I_group", "N/A"),
        }

        summary_rows.append(summary_row)

    # Create and save summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(optimal_dir, "optimal_configs_summary.csv"), index=False
    )

    # Create visual comparison of optimal configurations
    plt.figure(figsize=(12, 10))

    # Map group numbers to information content for better readability
    group_info_map = {0: "No Info", 1: "Medium Info", 2: "High Info"}

    # Organize by model type
    for i, model_type in enumerate(results_df["model_type"].unique()):
        model_keys = [
            k for k in optimal_configs.keys() if k.startswith(f"{model_type}_")
        ]

        if not model_keys:
            continue

        plt.subplot(len(results_df["model_type"].unique()), 1, i + 1)

        # Prepare data for plotting
        plot_data = []

        for key in model_keys:
            config = optimal_configs[key]
            metric = key.split("_", 1)[1]

            # Get E and I group info
            e_group_info = group_info_map.get(config.get("E_group", 0), "Unknown")
            i_group_info = group_info_map.get(config.get("I_group", 0), "Unknown")

            plot_data.append(
                {
                    "Metric": metric,
                    "N_E": config.get("N_E", 0),
                    "N_I": config.get("N_I", 0),
                    "E/I Ratio": config.get("N_E", 0) / config.get("N_I", 1),
                    "E Group": e_group_info,
                    "I Group": i_group_info,
                    "Value": config.get(metric, 0),
                }
            )

        # Convert to DataFrame for plotting
        plot_df = pd.DataFrame(plot_data)

        # Create bar plot
        sns.barplot(data=plot_df, x="Metric", y="E/I Ratio", palette="viridis")

        # Add labels
        for j, row in enumerate(plot_data):
            plt.text(
                j,
                row["E/I Ratio"] + 0.1,
                f"E: {row['E Group']}\nI: {row['I Group']}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.title(f"Optimal E/I Ratio by Metric ({model_type} model)")
        plt.ylabel("E/I Ratio")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(optimal_dir, "optimal_ei_ratios.png"), dpi=300)
    plt.close()

    return optimal_configs


def create_summary_report(results_df, missing_jobs, optimal_configs, exp_dir):
    """Create a summary report of the analysis"""
    if results_df.empty and not missing_jobs:
        print("No data available for summary report")
        return

    report_path = os.path.join(exp_dir, "plots", "analysis_summary.txt")

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DENDRINET SIMULATION ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Experiment directory: {exp_dir}\n\n")

        # Results summary
        if not results_df.empty:
            f.write("-" * 80 + "\n")
            f.write("RESULTS SUMMARY\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Total configurations analyzed: {len(results_df)}\n")
            f.write(f"Model types: {', '.join(results_df['model_type'].unique())}\n\n")

            # Performance summary
            f.write("Performance metrics summary:\n")
            for metric in ["accuracy", "final_mi"]:
                if metric in results_df.columns:
                    f.write(f"{metric}:\n")
                    f.write(f"  Mean: {results_df[metric].mean():.4f}\n")
                    f.write(f"  Std: {results_df[metric].std():.4f}\n")
                    f.write(f"  Min: {results_df[metric].min():.4f}\n")
                    f.write(f"  Max: {results_df[metric].max():.4f}\n\n")

            # Synergy summary
            f.write("Synergy metrics summary:\n")
            for metric in ["syn_ee", "syn_ii", "syn_ei"]:
                if metric in results_df.columns:
                    f.write(f"{metric}:\n")
                    f.write(f"  Mean: {results_df[metric].mean():.4f}\n")
                    f.write(f"  Std: {results_df[metric].std():.4f}\n")
                    f.write(f"  Min: {results_df[metric].min():.4f}\n")
                    f.write(f"  Max: {results_df[metric].max():.4f}\n\n")

        # Missing jobs
        if missing_jobs:
            f.write("-" * 80 + "\n")
            f.write("MISSING JOBS\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Number of missing jobs: {len(missing_jobs)}\n")
            f.write("Missing task IDs: " + ", ".join(map(str, missing_jobs[:20])))
            if len(missing_jobs) > 20:
                f.write(f" ... and {len(missing_jobs) - 20} more")
            f.write("\n\n")

        # Optimal configurations
        if optimal_configs:
            f.write("-" * 80 + "\n")
            f.write("OPTIMAL CONFIGURATIONS\n")
            f.write("-" * 80 + "\n\n")

            for key, config in optimal_configs.items():
                model_type, metric = key.split("_", 1)

                f.write(f"Best configuration for {metric} ({model_type} model):\n")
                f.write(f"  {metric} value: {config.get(metric, 'N/A'):.4f}\n")
                f.write(f"  N_E: {config.get('N_E', 'N/A')}\n")
                f.write(f"  N_I: {config.get('N_I', 'N/A')}\n")
                f.write(
                    f"  E/I Ratio: {config.get('N_E', 0) / config.get('N_I', 1):.2f}\n"
                )
                f.write(
                    f"  Correlations: cEE={config.get('cEE', 'N/A'):.2f}, cII={config.get('cII', 'N/A'):.2f}, cEI={config.get('cEI', 'N/A'):.2f}\n"
                )
                f.write(
                    f"  Groups: E_group={config.get('E_group', 'N/A')}, I_group={config.get('I_group', 'N/A')}\n\n"
                )

        f.write("\nAnalysis completed successfully.\n")

    print(f"Summary report saved to {report_path}")


def process_experiment(exp_dir, show_plots=False, report_missing=False):
    """Process a single experiment directory"""
    print(f"Processing experiment directory: {exp_dir}")

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Collect config files
    configs = collect_configs(exp_dir)
    print(f"Found {len(configs)} configuration files")

    # Collect results
    results_df, missing_jobs = collect_results(exp_dir, configs)

    if results_df.empty:
        print("No results found in the experiment directory")
        return

    print(f"Collected {len(results_df)} results, {len(missing_jobs)} missing jobs")

    # Save combined results
    save_combined_results(results_df, exp_dir)

    # Generate visualizations
    print("Generating visualizations...")

    create_parameter_effect_plots(results_df, plots_dir)
    create_correlation_heatmaps(results_df, plots_dir)
    create_group_info_plots(results_df, plots_dir)
    create_synergy_plots(results_df, plots_dir)

    # Find optimal configurations
    optimal_configs = find_optimal_configurations(results_df, plots_dir)

    # Create summary report
    create_summary_report(results_df, missing_jobs, optimal_configs, exp_dir)

    print(f"Analysis completed. Results saved in {plots_dir}")

    # Show plots if requested
    if show_plots:
        print("Opening plots in viewer...")
        # Find the most recent plot and open it
        plot_files = glob.glob(os.path.join(plots_dir, "**/*.png"), recursive=True)
        if plot_files:
            most_recent = max(plot_files, key=os.path.getmtime)
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 10))
                plt.imshow(plt.imread(most_recent))
                plt.axis("off")
                plt.title(f"Sample plot: {os.path.basename(most_recent)}")
                plt.show()
            except Exception as e:
                print(f"Failed to display plot: {e!s}")


def main():
    args = parse_args()

    # Set matplotlib style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context(
        "paper", rc={"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11}
    )

    # Suppress warnings
    warnings.filterwarnings("ignore")

    try:
        if args.timestamp:
            # Find experiment directory by timestamp
            exp_dir = find_experiment_directory(
                args.timestamp, args.output_base, args.prefix
            )
            process_experiment(exp_dir, args.show_plots, args.report_missing)

        elif args.folder:
            # Use specified folder
            if not os.path.exists(args.folder):
                print(f"Error: Specified folder does not exist: {args.folder}")
                return 1
            process_experiment(args.folder, args.show_plots, args.report_missing)

        elif args.all:
            # Find all folders with the specified prefix
            folders = glob.glob(os.path.join(args.output_base, f"{args.prefix}*"))
            if not folders:
                print(
                    f"Error: No folders found with prefix '{args.prefix}' in {args.output_base}"
                )
                return 1

            for folder in folders:
                print(f"\nProcessing folder: {folder}")
                process_experiment(folder, False, args.report_missing)

            print(f"\nProcessed {len(folders)} experiment folders")

    except Exception as e:
        print(f"Error during analysis: {e!s}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

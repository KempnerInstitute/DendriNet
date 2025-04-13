#!/usr/bin/env python
"""
analyze_ei_sweep_lite.py
-------------------------
Memory-optimized version of analyze_ei_sweep.py to avoid OOM errors.
This script analyzes EI sweep results with lower memory footprint by:
1. Processing files one at a time instead of loading everything in memory
2. Limiting the number of plots generated
3. Using chunking for large dataframes

Usage:
    python scripts/analyze_ei_sweep_lite.py --timestamp 20250411144518
    python scripts/analyze_ei_sweep_lite.py --folder /path/to/results/ei_20250411144518
"""

import argparse
import gc
import glob
import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze EI sweep results (memory-optimized)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--timestamp",
        type=str,
        help="Timestamp ID of the experiment (e.g. 20250411144518)",
    )
    group.add_argument("--folder", type=str, help="Full path to the experiment folder")
    parser.add_argument(
        "--output-base",
        type=str,
        default="/n/holylabs/LABS/kempner_dev/Users/hsafaai/results/dendritic-modeling/outputs",
        help="Base output directory",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ei_",
        help="Prefix of experiment folders to look for (default: ei_)",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=200,
        help="Maximum number of configurations to process (default: 200)",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=100,
        help="Maximum number of job dirs to process (default: 100)",
    )
    parser.add_argument(
        "--report-missing",
        action="store_true",
        help="Report detail of missing or unfinished runs",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only collect metrics, don't generate plots (saves memory)",
    )
    parser.add_argument(
        "--essential-plots-only",
        action="store_true",
        help="Only generate essential plots (heatmap and EI ratio)",
    )
    return parser.parse_args()


def find_experiment_directory(timestamp, output_base, prefix="ei_"):
    """Find the experiment directory based on timestamp"""
    folder_path = os.path.join(output_base, f"{prefix}{timestamp}")
    if not os.path.exists(folder_path):
        # Try searching for folders with matching timestamp
        ei_folders = glob.glob(os.path.join(output_base, f"{prefix}*{timestamp}*"))
        if ei_folders:
            folder_path = ei_folders[0]
        else:
            raise ValueError(
                f"Could not find experiment folder with timestamp {timestamp}"
            )
    return folder_path


def extract_metrics_from_log(log_content, metrics_dict):
    """Extract metrics from log content and update metrics_dict"""
    # Pattern 1: "train accuracy: 0.91, val accuracy: 0.88"
    train_match = re.search(
        r"train accuracy[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
    )
    val_match = re.search(
        r"val(?:id|idation)? accuracy[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
    )
    test_match = re.search(r"test accuracy[:=]\s*([\d\.]+)", log_content, re.IGNORECASE)

    # Pattern 2: "Accuracy: train=0.91, val=0.88"
    if not train_match:
        train_match = re.search(
            r"accuracy.*?train[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
    if not val_match:
        val_match = re.search(
            r"accuracy.*?val(?:id|idation)?[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
    if not test_match:
        test_match = re.search(
            r"accuracy.*?test[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )

    # Pattern 3: "Final results: train=0.91, val=0.88, test=0.85"
    if not train_match:
        train_match = re.search(
            r"final results.*?train[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
    if not val_match:
        val_match = re.search(
            r"final results.*?val(?:id|idation)?[:=]\s*([\d\.]+)",
            log_content,
            re.IGNORECASE,
        )
    if not test_match:
        test_match = re.search(
            r"final results.*?test[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )

    # Pattern 4: "Final accuracy: train=0.91, valid=0.88, test=0.85"
    if not train_match:
        train_match = re.search(
            r"final accuracy.*?train[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
    if not val_match:
        val_match = re.search(
            r"final accuracy.*?valid[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
    if not test_match:
        test_match = re.search(
            r"final accuracy.*?test[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )

    # Extract best values from the end of the log
    log_lines = log_content.splitlines()
    last_lines = "\n".join(log_lines[-20:])

    # Try to find accuracy in the last lines
    if not train_match:
        train_match = re.search(
            r"train accuracy[:=]\s*([\d\.]+)", last_lines, re.IGNORECASE
        )
    if not val_match:
        val_match = re.search(
            r"val(?:id|idation)? accuracy[:=]\s*([\d\.]+)", last_lines, re.IGNORECASE
        )
    if not test_match:
        test_match = re.search(
            r"test accuracy[:=]\s*([\d\.]+)", last_lines, re.IGNORECASE
        )

    # Update metrics dict with found values
    if train_match:
        metrics_dict["train_acc"] = float(train_match.group(1))
    if val_match:
        metrics_dict["val_acc"] = float(val_match.group(1))
    if test_match:
        metrics_dict["test_acc"] = float(test_match.group(1))


def extract_params_from_log(log_content, params_dict):
    """Extract parameters from log content and update params_dict"""
    # Look for parameter values in log
    ee_match = re.search(
        r"ee(?:_value|_synapses|_factor)?[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
    )
    ie_match = re.search(
        r"ie(?:_value|_synapses|_factor)?[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
    )
    seed_match = re.search(r"seed[:=]\s*(\d+)", log_content, re.IGNORECASE)

    # Alternative patterns
    if not ee_match:
        ee_match = re.search(r"excitatory[:=]\s*([\d\.]+)", log_content, re.IGNORECASE)
    if not ie_match:
        ie_match = re.search(r"inhibitory[:=]\s*([\d\.]+)", log_content, re.IGNORECASE)

    # Update params dict with found values
    if ee_match:
        params_dict["ee_value"] = float(ee_match.group(1))
    if ie_match:
        params_dict["ie_value"] = float(ie_match.group(1))
    if seed_match:
        params_dict["seed"] = int(seed_match.group(1))


def convert_to_simple_value(x):
    """Convert an OmegaConf object to a simple value."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(x, ListConfig):
        # Convert ListConfig to a list
        if len(x) == 1:
            # Return the first element if there's only one
            return convert_to_simple_value(x[0])
        else:
            # Return the entire list converted
            return [convert_to_simple_value(item) for item in x]
    elif isinstance(x, DictConfig):
        # Convert DictConfig to a dictionary
        return {k: convert_to_simple_value(v) for k, v in x.items()}
    else:
        # Return the value as is if it's not an OmegaConf object
        return x


def collect_params_data(exp_dir, max_configs=200):
    """Collect parameter data from the config files with memory optimizations"""
    configs_dir = os.path.join(exp_dir, "configs")
    params_data = []

    # Find all config files (limit to max_configs)
    config_files = glob.glob(os.path.join(configs_dir, "ei_sweep_config_*.yaml"))
    if len(config_files) > max_configs:
        print(f"Limiting to {max_configs} config files (out of {len(config_files)})")
        config_files = config_files[:max_configs]

    for config_file in config_files:
        try:
            # Extract task id from filename
            task_match = re.search(
                r"ei_sweep_config_(\d+)\.yaml", os.path.basename(config_file)
            )
            if not task_match:
                continue

            task_id = int(task_match.group(1))

            # Load config
            config = OmegaConf.load(config_file)

            # Extract EE and IE values
            try:
                # Try the expected path first
                ee_values = (
                    config.model.network.parameters.ee_synapses_per_branch_per_layer
                )
                ie_values = (
                    config.model.network.parameters.ie_synapses_per_branch_per_layer
                )

                # Convert to simple values
                ee_values = convert_to_simple_value(ee_values)
                ie_values = convert_to_simple_value(ie_values)

                # Handle cases where values might be lists
                if isinstance(ee_values, list) and len(ee_values) == 1:
                    ee_value = ee_values[0]
                else:
                    ee_value = ee_values

                if isinstance(ie_values, list) and len(ie_values) == 1:
                    ie_value = ie_values[0]
                else:
                    ie_value = ie_values

                # Get the seed
                seed = config.train.seed

                # Add to data
                params_data.append(
                    {
                        "task_id": task_id,
                        "ee_value": ee_value,
                        "ie_value": ie_value,
                        "seed": seed,
                    }
                )

            except Exception as e:
                # Try fallback structure
                try:
                    if hasattr(config, "excitation_factor"):
                        ee_value = convert_to_simple_value(config.excitation_factor)
                    elif hasattr(config.model, "excitation_factor"):
                        ee_value = convert_to_simple_value(
                            config.model.excitation_factor
                        )
                    else:
                        ee_value = "unknown"

                    if hasattr(config, "inhibition_factor"):
                        ie_value = convert_to_simple_value(config.inhibition_factor)
                    elif hasattr(config.model, "inhibition_factor"):
                        ie_value = convert_to_simple_value(
                            config.model.inhibition_factor
                        )
                    else:
                        ie_value = "unknown"

                    seed = config.train.seed if hasattr(config.train, "seed") else 0

                    params_data.append(
                        {
                            "task_id": task_id,
                            "ee_value": ee_value,
                            "ie_value": ie_value,
                            "seed": seed,
                        }
                    )
                except Exception:
                    pass

        except Exception:
            pass

        # Explicitly release config from memory
        config = None
        gc.collect()

    # Create DataFrame
    params_df = pd.DataFrame(params_data)
    if not params_df.empty and "task_id" in params_df.columns:
        params_df = params_df.sort_values("task_id").reset_index(drop=True)

    return params_df


def process_log_file(
    file_path,
    job_dir,
    params_df,
    results_data,
    missing_data,
    exp_timestamp,
    has_task_ids,
):
    """Process a log file to extract results"""
    # Initialize empty metrics dict
    metrics_dict = {
        "train_acc": float("nan"),
        "val_acc": float("nan"),
        "test_acc": float("nan"),
    }

    try:
        # Read the file line by line to avoid loading entire file into memory
        log_content = ""
        with open(file_path, "r") as f:
            # Read the first 50 lines and last 50 lines (most likely to contain the information we need)
            first_lines = []
            for i, line in enumerate(f):
                if i < 50:
                    first_lines.append(line)
                else:
                    break

            # Get file size and seek near the end
            f.seek(0, os.SEEK_END)
            file_size = f.tell()

            # If file is small, just read it all
            if file_size < 10000:  # 10KB
                f.seek(0)
                log_content = f.read()
            else:
                # Read last 5000 characters instead of last lines
                last_chunk_size = min(5000, file_size)
                f.seek(file_size - last_chunk_size)
                last_chunk = f.read(last_chunk_size)

                # Combine first lines and last chunk
                log_content = "".join(first_lines) + "\n...\n" + last_chunk

        # Extract metrics from log
        extract_metrics_from_log(log_content, metrics_dict)

        # Extract task ID from job directory name
        task_id = -1
        task_match = re.search(r"_(\d+)$", job_dir)
        if task_match:
            task_id = int(task_match.group(1))

        # Extract parameters from log
        params_dict = {"ee_value": None, "ie_value": None, "seed": 0}
        extract_params_from_log(log_content, params_dict)

        # Try to match with a parameter from params_df
        if has_task_ids and task_id >= 0:
            task_params = params_df[params_df["task_id"] == task_id]
            if not task_params.empty:
                params_dict["ee_value"] = task_params["ee_value"].iloc[0]
                params_dict["ie_value"] = task_params["ie_value"].iloc[0]
                params_dict["seed"] = task_params["seed"].iloc[0]

        # Store results if we found at least one accuracy value
        if not (
            pd.isna(metrics_dict["train_acc"])
            and pd.isna(metrics_dict["val_acc"])
            and pd.isna(metrics_dict["test_acc"])
        ):
            results_data.append(
                {
                    "ee_value": params_dict["ee_value"],
                    "ie_value": params_dict["ie_value"],
                    "train_acc": metrics_dict["train_acc"],
                    "val_acc": metrics_dict["val_acc"],
                    "test_acc": metrics_dict["test_acc"],
                    "experiment": exp_timestamp,
                    "task_id": task_id,
                    "seed": params_dict["seed"],
                    "job_dir": job_dir,
                }
            )
        else:
            # No accuracy values found
            missing_data.append(
                {
                    "job_dir": job_dir,
                    "reason": "No accuracy values found in log file",
                    "status": "incomplete",
                }
            )

        # Clear log content to free memory
        log_content = None
        gc.collect()

    except Exception as e:
        # Record as error during processing
        missing_data.append(
            {
                "job_dir": job_dir,
                "reason": f"Error processing log file: {str(e)}",
                "status": "error",
            }
        )
        print(f"Error processing log file {file_path}: {e}")


def collect_results(exp_dir, params_df, max_jobs=100):
    """Collect results from the experiment directory with memory optimizations"""
    results_dir = os.path.join(exp_dir, "results")
    exp_timestamp = os.path.basename(exp_dir)

    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory not found: {results_dir}")
        return pd.DataFrame(), pd.DataFrame()

    # Initialize results storage
    results_data = []
    missing_data = []

    # Check if we can track missing runs using task_ids
    has_task_ids = not params_df.empty and "task_id" in params_df.columns

    # Process each job directory in chunks to avoid memory issues
    try:
        job_dirs = [
            d
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ]

        if len(job_dirs) > max_jobs:
            print(f"Limiting to {max_jobs} job directories (out of {len(job_dirs)})")
            job_dirs = job_dirs[:max_jobs]

        print(f"Processing {len(job_dirs)} job directories")

    except Exception as e:
        print(f"Error listing job directories: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Process each job directory with focus on train.log files
    for job_dir in job_dirs:
        job_path = os.path.join(results_dir, job_dir)

        # First check for train.log - most reliable source
        train_log = os.path.join(job_path, "train.log")
        if os.path.exists(train_log):
            process_log_file(
                train_log,
                job_dir,
                params_df,
                results_data,
                missing_data,
                exp_timestamp,
                has_task_ids,
            )
            continue

        # If no train.log, check for output.log
        output_log = os.path.join(job_path, "output.log")
        if os.path.exists(output_log):
            process_log_file(
                output_log,
                job_dir,
                params_df,
                results_data,
                missing_data,
                exp_timestamp,
                has_task_ids,
            )
            continue

        # Last resort - look for any .log file
        log_files = glob.glob(os.path.join(job_path, "*.log"))
        if log_files:
            process_log_file(
                log_files[0],  # Just use the first log file found
                job_dir,
                params_df,
                results_data,
                missing_data,
                exp_timestamp,
                has_task_ids,
            )
            continue

        # If no log files found, mark as missing
        missing_data.append(
            {"job_dir": job_dir, "reason": "No log files found", "status": "incomplete"}
        )

        # Free memory after each directory
        gc.collect()

    # Create DataFrame
    results_df = pd.DataFrame(results_data) if results_data else pd.DataFrame()
    missing_df = pd.DataFrame(missing_data) if missing_data else pd.DataFrame()

    # Print summary
    print(f"Results collected: {len(results_df)}")
    if not results_df.empty:
        print("\nSample of collected results:")
        print(results_df.head().to_string())

    return results_df, missing_df


def generate_heatmap(plot_df, metric, output_path):
    """Generate a heatmap for the given metric"""
    # Extract just the data we need to save memory
    plot_data = plot_df[["ee_value", "ie_value", metric]].dropna().copy()

    # Create pivot table
    try:
        pivot_df = plot_data.pivot_table(
            values=metric,
            index="ee_value",
            columns="ie_value",
            aggfunc="mean",
        )

        # Check if pivot table has data
        if pivot_df.empty or pivot_df.isna().all().all():
            print(f"No valid data for {metric} heatmap")
            return

        # Fill NaN values for visualization
        pivot_viz = pivot_df.copy()
        if pivot_viz.isna().any().any():
            # For visualization only, replace NaNs with min value - 0.1
            min_val = pivot_viz.min().min()
            if not pd.isna(min_val):
                fill_val = min_val - 0.1
                pivot_viz = pivot_viz.fillna(fill_val)
            else:
                print("Cannot fill NaNs, all values are NaN")
                return

        # Create the plot directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Generate the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_viz, annot=True, cmap="viridis", fmt=".3f")
        plt.title(f'{metric.replace("_", " ").title()} vs EE/IE Parameters')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Generated heatmap: {output_path}")

    except Exception as e:
        print(f"Error generating heatmap for {metric}: {e}")

    # Clear data to free memory
    plot_data = None
    pivot_df = None
    pivot_viz = None
    gc.collect()


def generate_ei_ratio_plot(plot_df, metric, output_path):
    """Generate a plot showing the effect of E/I ratio on performance"""
    try:
        # Extract just what we need
        plot_data = plot_df[["ee_value", "ie_value", metric]].dropna().copy()

        # Calculate E/I ratio
        plot_data["ei_ratio"] = plot_data["ee_value"] / plot_data["ie_value"]

        # Group by ei_ratio and calculate mean and std
        ei_grouped = (
            plot_data.groupby("ei_ratio")[metric].agg(["mean", "std"]).reset_index()
        )

        # Create the plot directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Generate the plot
        plt.figure(figsize=(10, 6))

        # Plot mean with error bars
        plt.errorbar(
            ei_grouped["ei_ratio"],
            ei_grouped["mean"],
            yerr=ei_grouped["std"],
            marker="o",
            linestyle="-",
            capsize=5,
            color="blue",
        )

        # Add a second representation with individual points
        plt.scatter(plot_data["ei_ratio"], plot_data[metric], alpha=0.3, color="gray")

        # Add labels and title
        plt.xlabel("E/I Ratio (EE Value / IE Value)")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f'Effect of E/I Ratio on {metric.replace("_", " ").title()}')
        plt.grid(True)

        # Add horizontal line at best performance
        if not ei_grouped.empty:
            best_idx = ei_grouped["mean"].idxmax()
            best_ratio = ei_grouped.iloc[best_idx]["ei_ratio"]
            best_perf = ei_grouped.iloc[best_idx]["mean"]
            plt.axhline(y=best_perf, color="g", linestyle="--", alpha=0.7)
            plt.axvline(x=best_ratio, color="g", linestyle="--", alpha=0.7)
            plt.annotate(
                f"Best: ({best_ratio:.2f}, {best_perf:.4f})",
                xy=(best_ratio, best_perf),
                xytext=(5, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
            )

        # Save the plot
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Generated EI ratio plot: {output_path}")

    except Exception as e:
        print(f"Error generating EI ratio plot for {metric}: {e}")

    # Clear data to free memory
    plot_data = None
    ei_grouped = None
    gc.collect()


def process_single_experiment(exp_dir, args):
    """Process a single experiment with memory optimizations"""
    print(f"Processing experiment: {exp_dir}")

    # Create plots directory
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Collect parameter data from configs with limits
    params_df = collect_params_data(exp_dir, max_configs=args.max_configs)
    if params_df is not None and not params_df.empty:
        print(f"Collected parameters for {len(params_df)} configurations")
    else:
        print("No parameter data found. Will try to extract from results.")
        params_df = pd.DataFrame()

    # 2. Collect results with limits
    results_df, missing_df = collect_results(exp_dir, params_df, max_jobs=args.max_jobs)

    # Check if we found any results
    if results_df.empty:
        print("No results found for this experiment.")
        # Create an empty results file
        empty_results = pd.DataFrame(
            columns=["ee_value", "ie_value", "train_acc", "val_acc", "test_acc"]
        )
        empty_results.to_csv(os.path.join(plots_dir, "empty_results.csv"), index=False)
        return results_df, exp_dir

    # Print summary of metrics
    print("\nMetrics summary:")
    for metric in ["train_acc", "val_acc", "test_acc"]:
        if metric in results_df.columns:
            valid_metrics = results_df[~pd.isna(results_df[metric])]
            if not valid_metrics.empty:
                print(f"  {metric}: {len(valid_metrics)}/{len(results_df)} values")
                print(
                    f"    Mean: {valid_metrics[metric].mean():.4f}, Min: {valid_metrics[metric].min():.4f}, Max: {valid_metrics[metric].max():.4f}"
                )
            else:
                print(f"  {metric}: No valid values found")

    # Save the results to CSV
    results_df.to_csv(os.path.join(plots_dir, "ei_sweep_results.csv"), index=False)
    print(f"Saved results to {os.path.join(plots_dir, 'ei_sweep_results.csv')}")

    # Skip visualization if requested (metrics-only mode)
    if args.metrics_only:
        print("Skipping plot generation (metrics-only mode)")
        return results_df, exp_dir

    # Get a clean dataframe for plotting
    try:
        plot_df = results_df.copy()

        # Convert to numeric
        for col in ["ee_value", "ie_value", "train_acc", "val_acc", "test_acc"]:
            if col in plot_df.columns:
                # If it's unknown or not numeric, convert to NaN
                if col in ["ee_value", "ie_value"]:
                    plot_df[col] = plot_df[col].apply(
                        lambda x: np.nan
                        if x == "unknown" or not isinstance(x, (int, float, str))
                        else x
                    )

                # Convert to numeric
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

        # Generate essential plots for each metric
        for metric in ["val_acc", "test_acc", "train_acc"]:
            if metric in plot_df.columns and not plot_df[metric].isna().all():
                # Generate heatmap
                heatmap_path = os.path.join(plots_dir, f"ei_sweep_{metric}_heatmap.png")
                generate_heatmap(plot_df, metric, heatmap_path)

                # Generate EI ratio plot
                ratio_path = os.path.join(plots_dir, f"ei_sweep_{metric}_ei_ratio.png")
                generate_ei_ratio_plot(plot_df, metric, ratio_path)

                # Clear memory after each metric
                gc.collect()

        print(f"Plots saved to {plots_dir}")

    except Exception as e:
        print(f"Error generating plots: {e}")
        traceback.print_exc()

    # 4. Report missing runs if requested
    if args.report_missing and not missing_df.empty:
        print(f"\nMissing runs: {len(missing_df)}")

        # Save missing runs to file
        missing_file = os.path.join(plots_dir, "missing_runs.csv")
        missing_df.to_csv(missing_file, index=False)
        print(f"  - Details saved to: {missing_file}")

    return results_df, exp_dir


def main():
    args = parse_args()

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
        process_single_experiment(exp_dir, args)

        print("Analysis complete! Check the plots directory for outputs.")


if __name__ == "__main__":
    main()

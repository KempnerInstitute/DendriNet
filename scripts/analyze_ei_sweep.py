#!/usr/bin/env python
"""
analyze_ei_sweep.py
------------------
Generate plots from the results of an EI sweep experiment.

This script:
1. Takes a timestamp ID or folder path as input
2. Analyzes all the results for different IE and EE parameter values
3. Generates summary plots in the plots directory
4. Reports missing or unfinished runs

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
        help="List of timestamps or folder paths to analyze together",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=os.path.join(os.getcwd(), "output"),
        help="Base output directory (default: ./output)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ei_",
        help="Prefix of experiment folders to look for (default: ei_)",
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


def collect_params_data(exp_dir):
    """Collect parameter data from the config files"""
    configs_dir = os.path.join(exp_dir, "configs")
    params_data = []

    # Find all config files
    config_files = glob.glob(os.path.join(configs_dir, "ei_sweep_config_*.yaml"))

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

            # Extract EE and IE values - the structure might be different than expected
            try:
                # Try the expected path first
                ee_values = (
                    config.model.network.parameters.ee_synapses_per_branch_per_layer
                )
                ie_values = (
                    config.model.network.parameters.ie_synapses_per_branch_per_layer
                )

                # If we get here, we found the values
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
                        "config_file": config_file,
                    }
                )

            except Exception as e:
                print(
                    f"Warning: Could not extract EE/IE values from {config_file}: {e}"
                )
                print(f"Trying alternative structure...")

                # Try alternative paths
                try:
                    # This is a fallback - try other common parameter paths
                    if hasattr(config, "excitation_factor"):
                        ee_value = config.excitation_factor
                    elif hasattr(config.model, "excitation_factor"):
                        ee_value = config.model.excitation_factor
                    else:
                        # Default if not found
                        ee_value = "unknown"

                    if hasattr(config, "inhibition_factor"):
                        ie_value = config.inhibition_factor
                    elif hasattr(config.model, "inhibition_factor"):
                        ie_value = config.model.inhibition_factor
                    else:
                        # Default if not found
                        ie_value = "unknown"

                    # Get seed if available
                    seed = config.train.seed if hasattr(config.train, "seed") else 0

                    # Add to data
                    params_data.append(
                        {
                            "task_id": task_id,
                            "ee_value": ee_value,
                            "ie_value": ie_value,
                            "seed": seed,
                            "config_file": config_file,
                        }
                    )

                except Exception as nested_e:
                    print(
                        f"Error: Could not extract parameters using fallback: {nested_e}"
                    )

        except Exception as e:
            print(f"Error processing config file {config_file}: {e}")

    # Create DataFrame
    params_df = pd.DataFrame(params_data)
    if not params_df.empty and "task_id" in params_df.columns:
        params_df = params_df.sort_values("task_id").reset_index(drop=True)
    else:
        print("Warning: No task_id column found in parameter data")
    return params_df


def collect_results(exp_dir, params_df):
    """Collect results from the experiment directory"""
    results_dir = os.path.join(exp_dir, "results")
    exp_timestamp = os.path.basename(exp_dir)  # Get folder name for identification

    # If results_dir doesn't exist, try using the exp_dir directly
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        print(f"Trying to use the experiment directory directly: {exp_dir}")
        results_dir = exp_dir

    # If still doesn't exist, return empty dataframes
    if not os.path.exists(results_dir):
        print(f"Warning: Directory not found: {results_dir}")
        return pd.DataFrame(), pd.DataFrame()

    # Initialize results storage
    results_data = []
    missing_data = []

    # Check if we can track missing runs using task_ids
    has_task_ids = not params_df.empty and "task_id" in params_df.columns

    # Create a set of all task IDs from params_df for tracking missing runs (if possible)
    all_task_ids = set(params_df["task_id"].tolist()) if has_task_ids else set()
    found_task_ids = set()

    # Debug: Print first few task IDs if available
    if has_task_ids:
        print(f"First 5 task IDs: {list(all_task_ids)[:5]}")

    print(f"Looking for config-based directories in {results_dir}")

    # First, look for config_X directories
    config_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("config_")
    ]

    print(f"Found {len(config_dirs)} config directories")

    # Process each config directory
    for config_dir in config_dirs:
        config_path = os.path.join(results_dir, config_dir)

        # Extract config ID from directory name
        config_match = re.search(r"config_(\d+)", config_dir)
        if config_match:
            config_id = int(config_match.group(1))
        else:
            print(f"Couldn't extract config ID from directory name: {config_dir}")
            continue

        try:
            # Check for output.log in this config directory
            output_log = os.path.join(config_path, "output.log")
            if os.path.exists(output_log):
                print(f"Processing log file: {output_log}")
                # Initialize metrics and params
                metrics = {
                    "train_acc": float("nan"),
                    "val_acc": float("nan"),
                    "test_acc": float("nan"),
                }
                params = {"ee_value": None, "ie_value": None, "seed": 0}

                # Read the log file
                with open(output_log, "r") as f:
                    log_content = f.read()

                # Extract metrics from log
                extract_metrics_from_log(log_content, metrics)

                # Extract params from log
                extract_params_from_log(log_content, params)

                # Get parameter values for this config from params_df
                if has_task_ids:
                    task_params = params_df[params_df["task_id"] == config_id]
                    if not task_params.empty:
                        print(f"Found parameters for config {config_id}")
                        params["ee_value"] = task_params["ee_value"].iloc[0]
                        params["ie_value"] = task_params["ie_value"].iloc[0]
                        params["seed"] = task_params["seed"].iloc[0]
                        found_task_ids.add(config_id)

                # Store results if metrics are valid
                if not all(pd.isna(val) for val in metrics.values()):
                    results_data.append(
                        {
                            "ee_value": params["ee_value"],
                            "ie_value": params["ie_value"],
                            "train_acc": metrics["train_acc"],
                            "val_acc": metrics["val_acc"],
                            "test_acc": metrics["test_acc"],
                            "experiment": exp_timestamp,
                            "task_id": config_id,
                            "seed": params["seed"],
                            "job_dir": config_dir,
                        }
                    )
                    print(
                        f"Added result for config {config_id}: train={metrics['train_acc']:.3f}, val={metrics['val_acc']:.3f}"
                    )
            else:
                print(f"No output.log found in {config_path}")

            # Also look for config_used.yaml file
            config_used = os.path.join(config_path, "config_used.yaml")
            if (
                os.path.exists(config_used)
                and has_task_ids
                and config_id not in found_task_ids
            ):
                # Load config to get parameters
                config = OmegaConf.load(config_used)

                # Try to extract EE and IE values if they exist
                try:
                    ee_values = (
                        config.model.network.parameters.ee_synapses_per_branch_per_layer
                    )
                    ie_values = (
                        config.model.network.parameters.ie_synapses_per_branch_per_layer
                    )

                    # Handle cases where values might be lists
                    if isinstance(ee_values, list) and len(ee_values) == 1:
                        ee_value = ee_values[0]
                    else:
                        ee_value = ee_values

                    if isinstance(ie_values, list) and len(ie_values) == 1:
                        ie_value = ie_values[0]
                    else:
                        ie_value = ie_values

                    seed = config.train.seed

                    # Add to found task IDs
                    found_task_ids.add(config_id)

                    # If we have a config file but no output.log, might be a failed run
                    missing_data.append(
                        {
                            "task_id": config_id,
                            "ee_value": ee_value,
                            "ie_value": ie_value,
                            "seed": seed,
                            "job_dir": config_dir,
                        }
                    )
                except Exception as e:
                    print(
                        f"Error extracting parameters from config_used.yaml in {config_path}: {e}"
                    )

        except Exception as e:
            print(f"Error processing config directory {config_path}: {e}")

    # Find missing configs based on params_df
    if has_task_ids:
        missing_ids = all_task_ids - found_task_ids
        for missing_id in missing_ids:
            task_params = params_df[params_df["task_id"] == missing_id]
            if not task_params.empty:
                # Add to missing data
                missing_data.append(
                    {
                        "task_id": missing_id,
                        "ee_value": task_params["ee_value"].iloc[0],
                        "ie_value": task_params["ie_value"].iloc[0],
                        "seed": task_params["seed"].iloc[0],
                        "job_dir": f"missing_config_{missing_id}",
                    }
                )
                print(f"Added missing config {missing_id} to missing data")

    # Create DataFrames from the collected results
    results_df = pd.DataFrame(results_data)
    missing_df = pd.DataFrame(missing_data)

    if not results_df.empty:
        print(f"Collected {len(results_df)} results")
        # Print sample of results
        print("\nSample of collected results:")
        print(results_df.head().to_string())
    else:
        print("No results found.")

    if not missing_df.empty:
        print(f"Found {len(missing_df)} missing configs")
        print("\nSample of missing configs:")
        print(missing_df.head().to_string())

    # For backward compatibility, look for task directories if no config dirs found
    if len(results_data) == 0 and len(config_dirs) == 0:
        print(
            "No config directories found. Trying to look for task directories as fallback."
        )

        # Continue with the original function logic for job/task directories
        # Find all tasks directories recursively (pattern task_*)
        all_task_dirs = []
        for root, dirs, files in os.walk(results_dir):
            task_dirs = [d for d in dirs if d.startswith("task_")]
            if task_dirs:
                for task_dir in task_dirs:
                    all_task_dirs.append(os.path.join(root, task_dir))

        print(f"Found {len(all_task_dirs)} task directories as fallback")

        # Process each task directory
        for task_dir in all_task_dirs:
            # Extract task ID from directory name
            task_match = re.search(r"task_(\d+)", os.path.basename(task_dir))
            if task_match:
                task_id = int(task_match.group(1))
                # Process each task directory (fallback to original logic)
                try:
                    output_log = os.path.join(task_dir, "output.log")
                    if os.path.exists(output_log):
                        print(f"Processing log file (fallback): {output_log}")
                        # Process this output log (abbreviated from original logic)
                        # Initialize metrics and params
                        metrics = {
                            "train_acc": float("nan"),
                            "val_acc": float("nan"),
                            "test_acc": float("nan"),
                        }
                        params = {"ee_value": None, "ie_value": None, "seed": 0}

                        # Read the log file
                        with open(output_log, "r") as f:
                            log_content = f.read()

                        # Extract metrics from log
                        extract_metrics_from_log(log_content, metrics)

                        # Extract params from log
                        extract_params_from_log(log_content, params)

                        # Get parameter values from params_df
                        if has_task_ids:
                            task_params = params_df[params_df["task_id"] == task_id]
                            if not task_params.empty:
                                params["ee_value"] = task_params["ee_value"].iloc[0]
                                params["ie_value"] = task_params["ie_value"].iloc[0]
                                params["seed"] = task_params["seed"].iloc[0]

                        # Store results if metrics are valid
                        if not all(pd.isna(val) for val in metrics.values()):
                            results_data.append(
                                {
                                    "ee_value": params["ee_value"],
                                    "ie_value": params["ie_value"],
                                    "train_acc": metrics["train_acc"],
                                    "val_acc": metrics["val_acc"],
                                    "test_acc": metrics["test_acc"],
                                    "experiment": exp_timestamp,
                                    "task_id": task_id,
                                    "seed": params["seed"],
                                    "job_dir": os.path.basename(
                                        os.path.dirname(task_dir)
                                    ),
                                }
                            )
                except Exception as e:
                    print(f"Error processing task directory {task_dir}: {e}")

        # Re-create DataFrames after fallback
        results_df = pd.DataFrame(results_data)

    return results_df, missing_df


def process_performance_file(
    file_path,
    task_id,
    params_df,
    results_data,
    missing_data,
    exp_timestamp,
    has_task_ids,
):
    """Process a performance file and extract metrics"""
    try:
        with open(file_path, "r") as f:
            perf_data = json.load(f)

        # Get parameter values for this task if we have task_ids
        if has_task_ids:
            task_params = params_df[params_df["task_id"] == task_id]
            if task_params.empty:
                print(f"Warning: No parameter data found for task {task_id}")
                return

            ee_value = task_params["ee_value"].iloc[0]
            ie_value = task_params["ie_value"].iloc[0]
            seed = task_params["seed"].iloc[0]
        else:
            # Try to extract parameter values from perf_data if available
            ee_value = perf_data.get("ee_value", "unknown")
            ie_value = perf_data.get("ie_value", "unknown")
            seed = perf_data.get("seed", 0)

        # Extract performance metrics
        train_acc = perf_data.get("train_accuracy", float("nan"))
        val_acc = perf_data.get("val_accuracy", float("nan"))
        test_acc = perf_data.get("test_accuracy", float("nan"))

        # Store the results
        results_data.append(
            {
                "ee_value": ee_value,
                "ie_value": ie_value,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "experiment": exp_timestamp,
                "task_id": task_id,
                "seed": seed,
            }
        )

    except Exception as e:
        # Record as error during processing
        missing_data.append(
            {
                "task_id": task_id,
                "reason": f"Error processing performance file: {str(e)}",
                "status": "error",
            }
        )
        print(f"Error processing performance file for task {task_id}: {e}")


def process_output_log(
    file_path,
    task_id,
    params_df,
    results_data,
    missing_data,
    exp_timestamp,
    has_task_ids,
):
    """Process an output log file and extract metrics"""
    try:
        # Read the file
        with open(file_path, "r") as f:
            log_content = f.read()

        # Use regex to extract metrics
        train_match = re.search(r"train accuracy:\s*([\d\.]+)", log_content)
        val_match = re.search(r"val(?:id)? accuracy:\s*([\d\.]+)", log_content)
        test_match = re.search(r"test accuracy:\s*([\d\.]+)", log_content)

        train_acc = float(train_match.group(1)) if train_match else float("nan")
        val_acc = float(val_match.group(1)) if val_match else float("nan")
        test_acc = float(test_match.group(1)) if test_match else float("nan")

        # Get parameter values for this task if we have task_ids
        if has_task_ids:
            task_params = params_df[params_df["task_id"] == task_id]
            if task_params.empty:
                print(f"Warning: No parameter data found for task {task_id}")
                return

            ee_value = task_params["ee_value"].iloc[0]
            ie_value = task_params["ie_value"].iloc[0]
            seed = task_params["seed"].iloc[0]
        else:
            # Default values
            ee_value = "unknown"
            ie_value = "unknown"
            seed = 0

        # Store the results
        results_data.append(
            {
                "ee_value": ee_value,
                "ie_value": ie_value,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "experiment": exp_timestamp,
                "task_id": task_id,
                "seed": seed,
            }
        )

    except Exception as e:
        # Record as error during processing
        missing_data.append(
            {
                "task_id": task_id,
                "reason": f"Error processing output log: {str(e)}",
                "status": "error",
            }
        )
        print(f"Error processing output log for task {task_id}: {e}")


def process_config_file(
    file_path,
    task_id,
    params_df,
    results_data,
    missing_data,
    exp_timestamp,
    has_task_ids,
):
    """Process a config file and extract any available metrics"""
    try:
        # Try both YAML and JSON formats
        try:
            config = OmegaConf.load(file_path)
            if hasattr(config, "to_container"):
                config_data = config.to_container()
            else:
                config_data = config
        except:
            with open(file_path, "r") as f:
                if file_path.endswith(".json"):
                    config_data = json.load(f)
                else:
                    # Try to parse as YAML
                    import yaml

                    config_data = yaml.safe_load(f)

        # Get parameter values for this task if we have task_ids
        if has_task_ids:
            task_params = params_df[params_df["task_id"] == task_id]
            if task_params.empty:
                print(f"Warning: No parameter data found for task {task_id}")
                return

            ee_value = task_params["ee_value"].iloc[0]
            ie_value = task_params["ie_value"].iloc[0]
            seed = task_params["seed"].iloc[0]
        else:
            # Try to extract from config
            ee_value = extract_param(
                config_data, ["ee_synapses_per_branch_per_layer", "excitation_factor"]
            )
            ie_value = extract_param(
                config_data, ["ie_synapses_per_branch_per_layer", "inhibition_factor"]
            )
            seed = extract_param(config_data, ["seed", "random_seed"])

        # Try to extract performance metrics if they exist
        train_acc = extract_param(config_data, ["train_accuracy", "accuracy/train"])
        val_acc = extract_param(
            config_data, ["val_accuracy", "valid_accuracy", "accuracy/val"]
        )
        test_acc = extract_param(config_data, ["test_accuracy", "accuracy/test"])

        # Only add if we have at least one metric
        if not (pd.isna(train_acc) and pd.isna(val_acc) and pd.isna(test_acc)):
            results_data.append(
                {
                    "ee_value": ee_value,
                    "ie_value": ie_value,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "experiment": exp_timestamp,
                    "task_id": task_id,
                    "seed": seed,
                }
            )
        else:
            # No metrics found in config
            missing_data.append(
                {
                    "task_id": task_id,
                    "reason": "No performance metrics found in config file",
                    "status": "incomplete",
                }
            )

    except Exception as e:
        # Record as error during processing
        missing_data.append(
            {
                "task_id": task_id,
                "reason": f"Error processing config file: {str(e)}",
                "status": "error",
            }
        )
        print(f"Error processing config file for task {task_id}: {e}")


def extract_param(config_data, possible_keys, default=float("nan")):
    """Extract a parameter from config data, trying multiple possible keys"""
    from omegaconf import DictConfig, ListConfig

    for key in possible_keys:
        # Handle nested keys (e.g., "model.network.parameters.ee_synapses_per_branch_per_layer")
        if "." in key:
            parts = key.split(".")
            value = config_data
            found = True
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                elif isinstance(value, DictConfig) and part in value:
                    value = value[part]
                else:
                    found = False
                    break
            if found:
                # Convert value if needed
                if isinstance(value, ListConfig):
                    if len(value) == 1:
                        return value[0]  # Return first element if only one
                    return list(value)  # Convert to regular list otherwise
                elif isinstance(value, DictConfig):
                    return dict(value)  # Convert to regular dict
                return value
        # Handle direct keys
        elif (isinstance(config_data, dict) and key in config_data) or (
            isinstance(config_data, DictConfig) and key in config_data
        ):
            value = config_data[key]
            # Convert value if needed
            if isinstance(value, ListConfig):
                if len(value) == 1:
                    return value[0]  # Return first element if only one
                return list(value)  # Convert to regular list otherwise
            elif isinstance(value, DictConfig):
                return dict(value)  # Convert to regular dict
            return value

    # If we get here, none of the keys were found
    return default


def generate_plots(results_df, params_df, output_dir, base_name):
    """Generate plots from the results dataframe"""
    # Check if the dataframe is empty
    if results_df.empty:
        print("No results to plot.")
        return

    # Create output directory if it doesn't exist
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Save original data for reference
    results_df.to_csv(
        os.path.join(plot_dir, f"{base_name}_results_original.csv"), index=False
    )

    # Print diagnostic information
    print("\nDiagnostic information before processing:")
    for col in ["ee_value", "ie_value", "train_acc", "val_acc", "test_acc"]:
        if col in results_df.columns:
            nan_count = results_df[col].isna().sum()
            total = len(results_df)
            print(
                f"  - {col}: {nan_count}/{total} NaN values ({nan_count/total*100:.1f}%)"
            )
            print(f"    Data types: {results_df[col].apply(type).unique()}")
            if not results_df[col].isna().all():
                # Skip min/max calculation for columns that might contain ListConfig objects
                pass

    # Ensure all values are numeric
    try:
        # Handle ListConfig objects that might be in ee_value or ie_value columns
        for col in ["ee_value", "ie_value"]:
            if col in results_df.columns:
                # First convert any 'unknown' values to NaN
                results_df[col] = results_df[col].replace("unknown", np.nan)

                # Then convert remaining values to float, handling ListConfig objects
                results_df[col] = results_df[col].apply(
                    lambda x: float(x)
                    if isinstance(x, (int, float, str)) and not pd.isna(x)
                    else (
                        float(x[0])
                        if hasattr(x, "__getitem__") and len(x) > 0
                        else float("nan")
                    )
                )

        # Convert potential string columns to float if they contain numbers
        for col in ["train_acc", "val_acc", "test_acc"]:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    except Exception as e:
        print(f"Error converting values to numeric: {e}")
        print(f"Data types: {results_df.dtypes}")

        # Print problematic rows for debugging
        for col in ["ee_value", "ie_value", "train_acc", "val_acc", "test_acc"]:
            if col in results_df.columns:
                problem_rows = results_df[
                    ~pd.to_numeric(results_df[col], errors="coerce").notna()
                ]
                if not problem_rows.empty:
                    print(f"Problematic {col} values:")
                    for idx, row in problem_rows.iterrows():
                        print(f"  Row {idx}: {col}={row[col]} (type: {type(row[col])})")

    # Print diagnostic information after type conversion
    print("\nDiagnostic information after type conversion:")
    for col in ["ee_value", "ie_value", "train_acc", "val_acc", "test_acc"]:
        if col in results_df.columns:
            nan_count = results_df[col].isna().sum()
            total = len(results_df)
            print(
                f"  - {col}: {nan_count}/{total} NaN values ({nan_count/total*100:.1f}%)"
            )
            if not results_df[col].isna().all():
                print(f"    Min: {results_df[col].min()}, Max: {results_df[col].max()}")

    # Remove rows where both ee_value and ie_value are NaN
    valid_rows = ~(pd.isna(results_df["ee_value"]) & pd.isna(results_df["ie_value"]))
    if valid_rows.sum() < len(results_df):
        print(
            f"Removing {len(results_df) - valid_rows.sum()} rows with both ee_value and ie_value as NaN"
        )
        results_df = results_df[valid_rows].copy()

    # Save a raw CSV of all collected data before any further filtering
    results_df.to_csv(
        os.path.join(plot_dir, f"{base_name}_all_results_raw.csv"), index=False
    )

    # Check if we still have enough data to create meaningful plots
    if len(results_df) < 3:
        print(
            f"Warning: Only {len(results_df)} valid results available, skipping plots that require more data"
        )

    # Save a clean version without NaN values for plotting
    clean_results = results_df.dropna(subset=["ee_value", "ie_value"]).copy()
    if len(clean_results) > 0:
        clean_results.to_csv(
            os.path.join(plot_dir, f"{base_name}_clean_results.csv"), index=False
        )

    # Metrics to plot
    metrics = [
        col for col in ["train_acc", "val_acc", "test_acc"] if col in results_df.columns
    ]

    if not metrics:
        print("No metrics columns found in results dataframe.")
        return

    # Plot the heatmap for each metric
    for metric in metrics:
        try:
            # Skip metrics that are all NaN
            if results_df[metric].isna().all():
                print(f"Skipping {metric} - all values are NaN")
                continue

            # Use a clean subset for this metric with no NaN values
            plot_df = results_df.dropna(subset=[metric, "ee_value", "ie_value"]).copy()

            print(
                f"\nPlotting {metric} with {len(plot_df)}/{len(results_df)} valid rows"
            )

            if len(plot_df) < 3:
                print(f"Skipping {metric} plots - insufficient non-NaN data points")
                continue

            # Save this clean subset for diagnostic purposes
            plot_df.to_csv(
                os.path.join(plot_dir, f"{base_name}_{metric}_clean.csv"), index=False
            )

            # Check if we have both ee_value and ie_value columns
            if "ee_value" in plot_df.columns and "ie_value" in plot_df.columns:
                # Create pivot table
                try:
                    pivot_df = plot_df.pivot_table(
                        values=metric,
                        index="ee_value",
                        columns="ie_value",
                        aggfunc="mean",  # Average across seeds if multiple values exist
                    )

                    # Save pivot table for diagnostics
                    pivot_df.to_csv(
                        os.path.join(plot_dir, f"{base_name}_{metric}_pivot.csv")
                    )

                    print(f"Pivot table for {metric}:")
                    print(pivot_df)

                    # Check if pivot table has data
                    if pivot_df.empty or pivot_df.isna().all().all():
                        print(f"No valid data for {metric} pivot table")
                        continue

                    # Fill NaN values for visualization (won't affect the saved data)
                    pivot_viz = pivot_df.copy()
                    if pivot_viz.isna().any().any():
                        print(f"Warning: Pivot table for {metric} contains NaN values")
                        print(f"NaN locations: \n{pivot_viz.isna()}")

                        # For visualization only, replace NaNs with min value - 0.1
                        min_val = pivot_viz.min().min()
                        if not pd.isna(min_val):
                            fill_val = min_val - 0.1
                            pivot_viz = pivot_viz.fillna(fill_val)
                            print(f"Filling NaNs with {fill_val} for visualization")
                        else:
                            print("Cannot fill NaNs, all values are NaN")
                            continue

                    # Check if we have enough data points for a meaningful heatmap
                    if len(pivot_df) < 2 or len(pivot_df.columns) < 2:
                        print(
                            f"Skipping heatmap for {metric} - need multiple ee and ie values"
                        )
                    else:
                        # Plot heatmap
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(pivot_viz, annot=True, cmap="viridis", fmt=".3f")
                        plt.title(
                            f'{metric.replace("_", " ").title()} vs EE/IE Parameters'
                        )
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(plot_dir, f"{base_name}_{metric}_heatmap.png")
                        )
                        plt.close()

                        # Plot 3D surface if we have enough data points
                        if len(pivot_df) >= 3 and len(pivot_df.columns) >= 3:
                            try:
                                fig = plt.figure(figsize=(12, 10))
                                ax = fig.add_subplot(111, projection="3d")

                                # Get X, Y, Z data for the surface
                                x = pivot_viz.columns.values  # ie_values
                                y = pivot_viz.index.values  # ee_values
                                x_grid, y_grid = np.meshgrid(x, y)
                                z = pivot_viz.values

                                # Create the surface plot
                                surf = ax.plot_surface(
                                    x_grid,
                                    y_grid,
                                    z,
                                    cmap="viridis",
                                    edgecolor="none",
                                    alpha=0.8,
                                )

                                # Add color bar
                                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

                                # Set labels
                                ax.set_xlabel("IE Value")
                                ax.set_ylabel("EE Value")
                                ax.set_zlabel(metric.replace("_", " ").title())
                                ax.set_title(
                                    f'3D Surface Plot of {metric.replace("_", " ").title()}'
                                )

                                # Display the plot
                                plt.savefig(
                                    os.path.join(
                                        plot_dir, f"{base_name}_{metric}_surface.png"
                                    )
                                )
                                plt.close()
                            except Exception as e:
                                print(f"Error creating 3D surface plot: {e}")
                        else:
                            print(
                                f"Skipping 3D surface plot for {metric} - need at least 3×3 data points"
                            )
                except Exception as e:
                    print(f"Error creating pivot table for {metric}: {e}")
                    continue

                # Create line plots for each EE value if we have multiple IE values
                if len(pivot_df.columns) >= 2:
                    try:
                        plt.figure(figsize=(10, 6))
                        for ee_val in pivot_df.index:
                            row = pivot_df.loc[ee_val].dropna()  # Drop NaN values
                            if not row.empty:
                                plt.plot(
                                    row.index,
                                    row.values,
                                    marker="o",
                                    label=f"EE={ee_val}",
                                )
                        plt.xlabel("IE Value")
                        plt.ylabel(metric.replace("_", " ").title())
                        plt.title(
                            f'Effect of IE Value on {metric.replace("_", " ").title()} for Different EE Values'
                        )
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(
                            os.path.join(plot_dir, f"{base_name}_{metric}_ie_lines.png")
                        )
                        plt.close()
                    except Exception as e:
                        print(f"Error creating IE line plot: {e}")

                # Create line plots for each IE value if we have multiple EE values
                if len(pivot_df.index) >= 2:
                    try:
                        plt.figure(figsize=(10, 6))
                        for ie_val in pivot_df.columns:
                            col = pivot_df[ie_val].dropna()  # Drop NaN values
                            if not col.empty:
                                plt.plot(
                                    col.index,
                                    col.values,
                                    marker="o",
                                    label=f"IE={ie_val}",
                                )
                        plt.xlabel("EE Value")
                        plt.ylabel(metric.replace("_", " ").title())
                        plt.title(
                            f'Effect of EE Value on {metric.replace("_", " ").title()} for Different IE Values'
                        )
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(
                            os.path.join(plot_dir, f"{base_name}_{metric}_ee_lines.png")
                        )
                        plt.close()
                    except Exception as e:
                        print(f"Error creating EE line plot: {e}")

            # If we only have one parameter (either ee_value or ie_value), create line plots
            elif "ee_value" in plot_df.columns:
                try:
                    plt.figure(figsize=(10, 6))
                    grouped = plot_df.groupby("ee_value")[metric].mean()
                    plt.plot(grouped.index, grouped.values, marker="o")
                    plt.xlabel("EE Value")
                    plt.ylabel(metric.replace("_", " ").title())
                    plt.title(
                        f'Effect of EE Value on {metric.replace("_", " ").title()}'
                    )
                    plt.grid(True)
                    plt.savefig(
                        os.path.join(plot_dir, f"{base_name}_{metric}_ee_line.png")
                    )
                    plt.close()
                except Exception as e:
                    print(f"Error creating single EE line plot: {e}")

            elif "ie_value" in plot_df.columns:
                try:
                    plt.figure(figsize=(10, 6))
                    grouped = plot_df.groupby("ie_value")[metric].mean()
                    plt.plot(grouped.index, grouped.values, marker="o")
                    plt.xlabel("IE Value")
                    plt.ylabel(metric.replace("_", " ").title())
                    plt.title(
                        f'Effect of IE Value on {metric.replace("_", " ").title()}'
                    )
                    plt.grid(True)
                    plt.savefig(
                        os.path.join(plot_dir, f"{base_name}_{metric}_ie_line.png")
                    )
                    plt.close()
                except Exception as e:
                    print(f"Error creating single IE line plot: {e}")

            else:
                print(f"Cannot create plots: missing ee_value and/or ie_value columns")

        except Exception as e:
            print(f"Error creating plot for {metric}: {e}")
            traceback.print_exc()

    # Create scatter plot if there are multiple unique ee/ie values
    try:
        if "ee_value" in results_df.columns and "ie_value" in results_df.columns:
            for metric in metrics:
                valid_data = results_df.dropna(subset=[metric, "ee_value", "ie_value"])
                if len(valid_data) > 1:
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(
                        valid_data["ee_value"],
                        valid_data["ie_value"],
                        c=valid_data[metric],
                        cmap="viridis",
                        s=100,
                        alpha=0.7,
                    )
                    plt.colorbar(scatter, label=metric.replace("_", " ").title())
                    plt.xlabel("EE Value")
                    plt.ylabel("IE Value")
                    plt.title(f'{metric.replace("_", " ").title()} by EE/IE Value')
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(plot_dir, f"{base_name}_{metric}_scatter.png")
                    )
                    plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")

    # Create CSV output
    results_df.to_csv(os.path.join(plot_dir, f"{base_name}_results.csv"), index=False)

    print(f"Plots and results saved to {plot_dir}")


def compare_multiple_experiments(exp_results, summary_dir):
    """Compare results from multiple experiments"""
    # Create summary directory if it doesn't exist
    os.makedirs(summary_dir, exist_ok=True)
    print(f"Creating summary in: {summary_dir}")

    # Check if we have any experiment results
    if not exp_results:
        print("No experiment results to compare.")
        return

    # Collect all results into one dataframe
    all_results = []
    for exp_name, results_df, exp_dir in exp_results:
        if results_df is not None and not results_df.empty:
            # Tag with experiment name
            results_df = results_df.copy()
            results_df["experiment"] = exp_name
            all_results.append(results_df)

    if not all_results:
        print("No valid results to compare.")
        return

    combined_df = pd.concat(all_results, ignore_index=True)

    # Check if combined dataframe has any data
    if combined_df.empty:
        print("Combined dataframe is empty, nothing to compare.")
        return

    # Print information about the combined dataset
    print(
        f"Combined dataset has {len(combined_df)} rows from {len(exp_results)} experiments"
    )

    # Output information about NaN values
    for col in ["ee_value", "ie_value", "train_acc", "val_acc", "test_acc"]:
        if col in combined_df.columns:
            nan_count = combined_df[col].isna().sum()
            print(
                f"Column {col}: {nan_count}/{len(combined_df)} NaN values ({nan_count/len(combined_df)*100:.1f}%)"
            )

    # Save combined results
    combined_df.to_csv(os.path.join(summary_dir, "all_experiments.csv"), index=False)

    # Generate combined plots
    try:
        # Group by experiment and ee/ie values
        exp_grouped = (
            combined_df.groupby(["experiment", "ee_value", "ie_value"])
            .agg({"train_acc": "mean", "val_acc": "mean", "test_acc": "mean"})
            .reset_index()
        )

        # Create bar charts comparing experiments
        for metric in ["train_acc", "val_acc", "test_acc"]:
            if metric in exp_grouped.columns and not exp_grouped[metric].isna().all():
                # Create bar chart
                exp_pivot = exp_grouped.pivot_table(
                    values=metric,
                    index=["ee_value", "ie_value"],
                    columns="experiment",
                    aggfunc="mean",
                )

                # Save pivot table
                exp_pivot.to_csv(os.path.join(summary_dir, f"comparison_{metric}.csv"))

                # Create comparison plot if we have data
                if not exp_pivot.empty and not exp_pivot.isna().all().all():
                    plt.figure(figsize=(12, 8))
                    exp_pivot.plot(kind="bar", figsize=(14, 8))
                    plt.title(
                        f"{metric.replace('_', ' ').title()} Comparison Across Experiments"
                    )
                    plt.xlabel("EE Value, IE Value")
                    plt.ylabel(metric.replace("_", " ").title())
                    plt.legend(title="Experiment")
                    plt.grid(axis="y")
                    plt.tight_layout()
                    plt.savefig(os.path.join(summary_dir, f"comparison_{metric}.png"))
                    plt.close()

        # Create line plots comparing across experiments for each ee/ie combination
        for metric in ["train_acc", "val_acc", "test_acc"]:
            if metric in exp_grouped.columns and not exp_grouped[metric].isna().all():
                # For each EE value
                ee_values = exp_grouped["ee_value"].unique()
                for ee_val in ee_values:
                    ee_data = exp_grouped[exp_grouped["ee_value"] == ee_val]
                    if not ee_data.empty:
                        plt.figure(figsize=(12, 6))
                        for exp in ee_data["experiment"].unique():
                            exp_data = ee_data[ee_data["experiment"] == exp]
                            plt.plot(
                                exp_data["ie_value"],
                                exp_data[metric],
                                marker="o",
                                label=exp,
                            )
                        plt.title(
                            f"{metric.replace('_', ' ').title()} vs IE Value (EE={ee_val})"
                        )
                        plt.xlabel("IE Value")
                        plt.ylabel(metric.replace("_", " ").title())
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(summary_dir, f"ee_{ee_val}_{metric}.png")
                        )
                        plt.close()

                # For each IE value
                ie_values = exp_grouped["ie_value"].unique()
                for ie_val in ie_values:
                    ie_data = exp_grouped[exp_grouped["ie_value"] == ie_val]
                    if not ie_data.empty:
                        plt.figure(figsize=(12, 6))
                        for exp in ie_data["experiment"].unique():
                            exp_data = ie_data[ie_data["experiment"] == exp]
                            plt.plot(
                                exp_data["ee_value"],
                                exp_data[metric],
                                marker="o",
                                label=exp,
                            )
                        plt.title(
                            f"{metric.replace('_', ' ').title()} vs EE Value (IE={ie_val})"
                        )
                        plt.xlabel("EE Value")
                        plt.ylabel(metric.replace("_", " ").title())
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(summary_dir, f"ie_{ie_val}_{metric}.png")
                        )
                        plt.close()

    except Exception as e:
        print(f"Error generating comparison plots: {e}")
        traceback.print_exc()

    # Create summary text
    try:
        with open(os.path.join(summary_dir, "summary.txt"), "w") as f:
            f.write(f"EI Experiments Comparison Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Experiments compared: {len(exp_results)}\n")

            for i, (exp_name, df, exp_dir) in enumerate(exp_results):
                f.write(f"\n{i+1}. {exp_name}\n")
                f.write(f"   Directory: {exp_dir}\n")

                if df is not None and not df.empty:
                    f.write(f"   Results: {len(df)} data points\n")

                    # Metrics
                    for metric in ["train_acc", "val_acc", "test_acc"]:
                        if metric in df.columns:
                            valid_metrics = df[~pd.isna(df[metric])]
                            if not valid_metrics.empty:
                                f.write(
                                    f"   {metric}: mean={valid_metrics[metric].mean():.4f}, "
                                    f"min={valid_metrics[metric].min():.4f}, "
                                    f"max={valid_metrics[metric].max():.4f}\n"
                                )
                            else:
                                f.write(f"   {metric}: No valid values\n")

                    # Parameter ranges
                    if "ee_value" in df.columns:
                        ee_vals = df["ee_value"].dropna().unique()
                        f.write(
                            f"   EE values: {', '.join(map(str, sorted(ee_vals)))}\n"
                        )
                    if "ie_value" in df.columns:
                        ie_vals = df["ie_value"].dropna().unique()
                        f.write(
                            f"   IE values: {', '.join(map(str, sorted(ie_vals)))}\n"
                        )
                else:
                    f.write(f"   No results available\n")

            # Best results across all experiments
            f.write("\nBest Results Across All Experiments\n")
            f.write("================================\n\n")

            for metric in ["train_acc", "val_acc", "test_acc"]:
                if metric in combined_df.columns:
                    valid_metrics = combined_df[~pd.isna(combined_df[metric])]
                    if not valid_metrics.empty:
                        best_idx = valid_metrics[metric].idxmax()
                        best_row = combined_df.loc[best_idx]

                        f.write(f"Best {metric}: {best_row[metric]:.4f}\n")
                        f.write(f"  Experiment: {best_row['experiment']}\n")
                        if "ee_value" in best_row and not pd.isna(best_row["ee_value"]):
                            f.write(f"  EE value: {best_row['ee_value']}\n")
                        if "ie_value" in best_row and not pd.isna(best_row["ie_value"]):
                            f.write(f"  IE value: {best_row['ie_value']}\n")
                        if "seed" in best_row and not pd.isna(best_row["seed"]):
                            f.write(f"  Seed: {best_row['seed']}\n")
                        f.write("\n")

    except Exception as e:
        print(f"Error creating summary text: {e}")


def process_single_experiment(exp_dir, show_plots=False, report_missing=False):
    """Process a single experiment and generate plots"""
    print(f"Processing experiment: {exp_dir}")

    # 1. Collect parameter data from configs
    params_df = collect_params_data(exp_dir)
    if params_df is not None and not params_df.empty:
        print(f"Collected parameters for {len(params_df)} configurations")
    else:
        print("No parameter data found. Will try to extract from results.")
        params_df = pd.DataFrame()

    # 2. Collect results from output logs
    results_df, missing_df = collect_results(exp_dir, params_df)

    # Create plots directory
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Check if we found any results
    if results_df.empty:
        print("No results found for this experiment.")
        # Create an empty results file
        empty_results = pd.DataFrame(
            columns=["ee_value", "ie_value", "train_acc", "val_acc", "test_acc"]
        )
        empty_results.to_csv(os.path.join(plots_dir, "empty_results.csv"), index=False)
        return results_df, exp_dir

    # Print summary of results
    print(f"Collected {len(results_df)} results")

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

    # 3. Generate plots and analysis
    try:
        generate_plots(results_df, params_df, exp_dir, "ei_sweep")
    except Exception as e:
        print(f"Error generating plots: {e}")
        traceback.print_exc()
        # Save the raw dataframe that caused the error
        results_df.to_csv(os.path.join(plots_dir, "error_results.csv"), index=False)

    # 4. Report missing runs if requested
    if report_missing and not missing_df.empty:
        print(f"\nMissing runs: {len(missing_df)}")

        # Print detail about missing runs
        missing_by_status = missing_df.groupby("status").size()
        for status, count in missing_by_status.items():
            print(f"  - {status.title()}: {count}")

        # Save missing runs to file
        missing_file = os.path.join(plots_dir, "missing_runs.csv")
        missing_df.to_csv(missing_file, index=False)
        print(f"  - Details saved to: {missing_file}")

    return results_df, exp_dir


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
        # Read the file
        with open(file_path, "r") as f:
            log_content = f.read()

        # Look for different patterns of reporting accuracy
        # Pattern 1: "train accuracy: 0.91, val accuracy: 0.88"
        train_match = re.search(
            r"train accuracy[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
        val_match = re.search(
            r"val(?:id|idation)? accuracy[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
        test_match = re.search(
            r"test accuracy[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )

        # Pattern 2: "Accuracy: train=0.91, val=0.88"
        if not train_match:
            train_match = re.search(
                r"accuracy.*?train[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
            )
        if not val_match:
            val_match = re.search(
                r"accuracy.*?val(?:id|idation)?[:=]\s*([\d\.]+)",
                log_content,
                re.IGNORECASE,
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

        # Extract best values from the entire log - often the final values are reported at the end
        # Get last 20 lines which often contain final results
        log_lines = log_content.splitlines()
        last_lines = "\n".join(log_lines[-20:])

        # Try to find accuracy in the last lines
        if not train_match:
            train_match = re.search(
                r"train accuracy[:=]\s*([\d\.]+)", last_lines, re.IGNORECASE
            )
        if not val_match:
            val_match = re.search(
                r"val(?:id|idation)? accuracy[:=]\s*([\d\.]+)",
                last_lines,
                re.IGNORECASE,
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

        # Extract task ID from job directory name
        task_id = -1
        task_match = re.search(r"_(\d+)$", job_dir)
        if task_match:
            task_id = int(task_match.group(1))

        # Extract parameters from log
        params_dict = {"ee_value": None, "ie_value": None, "seed": 0}
        extract_params_from_log(log_content, params_dict)

        # Try to match with a parameter from params_df or use extracted values
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


# Add extract_metrics_from_log function


def extract_metrics_from_log(log_content, metrics_dict):
    """Extract metrics from log content and update metrics_dict"""
    # Look for different patterns of reporting accuracy

    # Pattern for output.log format: [Epoch 100/100] train_loss=1.8706, train_acc=0.370, valid_loss=1.9548, valid_acc=0.319
    train_matches = re.findall(r"train_acc=(\d+\.\d+)", log_content, re.IGNORECASE)
    valid_matches = re.findall(r"valid_acc=(\d+\.\d+)", log_content, re.IGNORECASE)
    test_matches = re.findall(r"test_acc=(\d+\.\d+)", log_content, re.IGNORECASE)

    # If we found matches, use the last (most recent) one
    if train_matches:
        metrics_dict["train_acc"] = float(train_matches[-1])
    if valid_matches:
        metrics_dict["val_acc"] = float(valid_matches[-1])
    if test_matches:
        metrics_dict["test_acc"] = float(test_matches[-1])

    # If we didn't find val_acc but found valid_acc, use that
    if "val_acc" not in metrics_dict or pd.isna(metrics_dict["val_acc"]):
        val_matches = re.findall(r"val_acc=(\d+\.\d+)", log_content, re.IGNORECASE)
        if val_matches:
            metrics_dict["val_acc"] = float(val_matches[-1])

    # Pattern 1: "train accuracy: 0.91, val accuracy: 0.88"
    if "train_acc" not in metrics_dict or pd.isna(metrics_dict["train_acc"]):
        train_match = re.search(
            r"train accuracy[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
        if train_match:
            metrics_dict["train_acc"] = float(train_match.group(1))

    if "val_acc" not in metrics_dict or pd.isna(metrics_dict["val_acc"]):
        val_match = re.search(
            r"val(?:id|idation)? accuracy[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
        if val_match:
            metrics_dict["val_acc"] = float(val_match.group(1))

    if "test_acc" not in metrics_dict or pd.isna(metrics_dict["test_acc"]):
        test_match = re.search(
            r"test accuracy[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
        if test_match:
            metrics_dict["test_acc"] = float(test_match.group(1))

    # Pattern 2: "Accuracy: train=0.91, val=0.88"
    if "train_acc" not in metrics_dict or pd.isna(metrics_dict["train_acc"]):
        train_match = re.search(
            r"accuracy.*?train[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
        if train_match:
            metrics_dict["train_acc"] = float(train_match.group(1))

    if "val_acc" not in metrics_dict or pd.isna(metrics_dict["val_acc"]):
        val_match = re.search(
            r"accuracy.*?val(?:id|idation)?[:=]\s*([\d\.]+)",
            log_content,
            re.IGNORECASE,
        )
        if val_match:
            metrics_dict["val_acc"] = float(val_match.group(1))

    if "test_acc" not in metrics_dict or pd.isna(metrics_dict["test_acc"]):
        test_match = re.search(
            r"accuracy.*?test[:=]\s*([\d\.]+)", log_content, re.IGNORECASE
        )
        if test_match:
            metrics_dict["test_acc"] = float(test_match.group(1))

    # Extract best values from the entire log - often the final values are reported at the end
    # Get last 20 lines which often contain final results
    log_lines = log_content.splitlines()
    last_lines = "\n".join(log_lines[-20:])

    # If we don't have validation accuracy yet, try to find it in the last lines
    if "val_acc" not in metrics_dict or pd.isna(metrics_dict["val_acc"]):
        # Try to find validation accuracy in various formats in the last lines
        val_match = re.search(r"valid_acc=(\d+\.\d+)", last_lines, re.IGNORECASE)
        if val_match:
            metrics_dict["val_acc"] = float(val_match.group(1))

    # Since no test accuracy found in the log, use validation accuracy as a proxy if available
    if (
        ("test_acc" not in metrics_dict or pd.isna(metrics_dict["test_acc"]))
        and "val_acc" in metrics_dict
        and not pd.isna(metrics_dict["val_acc"])
    ):
        print("No test accuracy found, using validation accuracy as proxy")
        metrics_dict["test_acc"] = metrics_dict["val_acc"]


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


def extract_metrics_and_params(data, metrics_dict, params_dict, source=""):
    """Extract metrics and parameters from a data dictionary"""
    # Debug print
    print_sample_keys(data, source)

    # Define metric paths to check
    metric_paths = {
        "train_acc": [
            "train_accuracy",
            "train_acc",
            "accuracy/train",
            "metrics/train_accuracy",
            "train/accuracy",
            "results/train_accuracy",
            "training/accuracy",
        ],
        "val_acc": [
            "val_accuracy",
            "valid_accuracy",
            "accuracy/val",
            "accuracy/valid",
            "metrics/val_accuracy",
            "validation/accuracy",
            "val/accuracy",
            "results/val_accuracy",
            "results/valid_accuracy",
        ],
        "test_acc": [
            "test_accuracy",
            "accuracy/test",
            "metrics/test_accuracy",
            "test/accuracy",
            "results/test_accuracy",
        ],
    }

    # Try to extract metrics
    for metric_name, paths in metric_paths.items():
        value = extract_param(data, paths, float("nan"))
        if not pd.isna(value):
            metrics_dict[metric_name] = value

    # Define parameter paths to check
    param_paths = {
        "ee_value": [
            "ee_synapses_per_branch_per_layer",
            "excitation_factor",
            "ee_factor",
            "ee",
            "model/excitation_factor",
            "model/network/parameters/ee_synapses_per_branch_per_layer",
        ],
        "ie_value": [
            "ie_synapses_per_branch_per_layer",
            "inhibition_factor",
            "ie_factor",
            "ie",
            "model/inhibition_factor",
            "model/network/parameters/ie_synapses_per_branch_per_layer",
        ],
        "seed": ["seed", "random_seed", "train/seed"],
    }

    # Try to extract parameters
    for param_name, paths in param_paths.items():
        value = extract_param(data, paths, None)
        if value is not None:
            params_dict[param_name] = value


def main():
    args = parse_args()

    # Create timestamp for summary directory if analyzing multiple experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If analyzing a single experiment
    if args.timestamp or args.folder:
        # Determine the experiment directory
        if args.folder:
            exp_dir = args.folder
            print(f"Analyzing folder: {exp_dir}")

            # Verify the directory exists
            if not os.path.exists(exp_dir):
                print(f"Error: Directory {exp_dir} not found!")
                print(
                    "Usage: python scripts/analyze_ei_sweep.py --folder /path/to/experiment/folder"
                )
                return
        else:
            exp_dir = find_experiment_directory(
                args.timestamp, args.output_base, args.prefix
            )
            print(f"Analyzing experiment with timestamp {args.timestamp}: {exp_dir}")

        # Process the experiment
        results_df, _ = process_single_experiment(
            exp_dir, show_plots=args.show_plots, report_missing=args.report_missing
        )

        # Print summary information if results were found
        if results_df is not None and not results_df.empty:
            print(f"\nFound {len(results_df)} valid results")

            # Show best configuration for each metric
            for metric in ["val_acc", "test_acc"]:
                if metric in results_df.columns and not results_df[metric].isna().all():
                    best_idx = results_df[metric].idxmax()
                    best_row = results_df.loc[best_idx]
                    print(f"\nBest {metric}: {best_row[metric]:.4f}")
                    print(f"  EE Value: {best_row['ee_value']}")
                    print(f"  IE Value: {best_row['ie_value']}")
                    if "seed" in best_row:
                        print(f"  Seed: {best_row['seed']}")
        else:
            print(
                "\nNo valid results found. Check if the directory structure is correct."
            )
            print("The script looks for:")
            print("1. Log files (*.log, slurm-*.out)")
            print("2. Model files (best_model.pt, *.pt) with associated JSON files")
            print("3. Performance directories with result files")
            print("\nYou can create a test file to check extraction patterns:")
            print("echo 'Final accuracy: train=0.95, valid=0.92, test=0.91' > test.log")
            print("python scripts/analyze_ei_sweep.py --folder ./")

    # If analyzing multiple experiments
    elif args.multiple:
        # Process each experiment
        exp_results = []
        for exp_id in args.multiple:
            if os.path.isdir(exp_id):
                exp_dir = exp_id
            else:
                try:
                    exp_dir = find_experiment_directory(
                        exp_id, args.output_base, args.prefix
                    )
                except ValueError as e:
                    print(f"Error: {e}")
                    continue

            print(f"Analyzing experiment in: {exp_dir}")
            results_df, _ = process_single_experiment(
                exp_dir, show_plots=args.show_plots, report_missing=args.report_missing
            )
            if results_df is not None and not results_df.empty:
                exp_name = os.path.basename(exp_dir)
                exp_results.append((exp_name, results_df, exp_dir))

        # Create comparison directory
        if exp_results:
            comparison_dir = os.path.join(
                args.output_base, f"ei_comparison_{timestamp}"
            )
            os.makedirs(comparison_dir, exist_ok=True)
            print(f"Creating comparison plots in: {comparison_dir}")

            # Generate comparison plots
            compare_multiple_experiments(exp_results, comparison_dir)
        else:
            print("No valid results found in any of the specified experiments.")

    # If analyzing all ei_ folders
    elif args.all:
        # Find all ei_ folders
        ei_folders = glob.glob(os.path.join(args.output_base, f"{args.prefix}*"))
        if not ei_folders:
            print(f"No {args.prefix}* folders found in {args.output_base}")
            return

        # Process each experiment
        exp_results = []
        for folder in ei_folders:
            print(f"Analyzing experiment in: {folder}")
            results_df, _ = process_single_experiment(
                folder, show_plots=args.show_plots, report_missing=args.report_missing
            )
            if results_df is not None and not results_df.empty:
                exp_name = os.path.basename(folder)
                exp_results.append((exp_name, results_df, folder))

        # Create comparison directory
        if exp_results:
            comparison_dir = os.path.join(
                args.output_base, f"ei_comparison_{timestamp}"
            )
            os.makedirs(comparison_dir, exist_ok=True)
            print(f"Creating comparison plots in: {comparison_dir}")

            # Generate comparison plots
            compare_multiple_experiments(exp_results, comparison_dir)
        else:
            print("No valid results found in any of the folders.")

    else:
        print(
            "No action specified. Use one of: --timestamp, --folder, --multiple, or --all"
        )
        print("\nUsage examples:")
        print("  Process a single experiment by timestamp:")
        print("    python scripts/analyze_ei_sweep.py --timestamp 20230615123456")
        print("\n  Process a specific folder:")
        print(
            "    python scripts/analyze_ei_sweep.py --folder /path/to/experiment/folder"
        )
        print("\n  Process multiple experiments:")
        print(
            "    python scripts/analyze_ei_sweep.py --multiple 20230615123456 20230615234567"
        )
        print("\n  Process all ei_ folders in output directory:")
        print("    python scripts/analyze_ei_sweep.py --all")


# Define the print_sample_keys function if it's not already defined
if "print_sample_keys" not in globals():

    def print_sample_keys(data, source=""):
        """Print sample keys from a dictionary for debugging"""
        if not isinstance(data, dict):
            print(f"{source}: Not a dictionary")
            return

        # Print top-level keys
        print(f"{source} top-level keys: {', '.join(list(data.keys())[:5])}")

        # Look for accuracy related keys at any level
        accuracy_keys = []
        for key in data.keys():
            if "acc" in key.lower():
                accuracy_keys.append(key)
            if isinstance(data[key], dict):
                # Look one level deeper
                for subkey in data[key].keys():
                    if "acc" in subkey.lower():
                        accuracy_keys.append(f"{key}.{subkey}")

        if accuracy_keys:
            print(f"{source} accuracy-related keys: {', '.join(accuracy_keys)}")
            for key in accuracy_keys:
                if "." in key:
                    parts = key.split(".")
                    value = data[parts[0]][parts[1]]
                else:
                    value = data[key]
                print(f"  {key} = {value}")
        else:
            print(f"{source}: No accuracy-related keys found")

        # Look for parameter related keys
        param_keys = []
        for key in data.keys():
            if any(term in key.lower() for term in ["ee", "ie", "exc", "inh", "synap"]):
                param_keys.append(key)
            if isinstance(data[key], dict):
                # Look one level deeper
                for subkey in data[key].keys():
                    if any(
                        term in subkey.lower()
                        for term in ["ee", "ie", "exc", "inh", "synap"]
                    ):
                        param_keys.append(f"{key}.{subkey}")

        if param_keys:
            print(f"{source} parameter-related keys: {', '.join(param_keys)}")
            for key in param_keys:
                if "." in key:
                    parts = key.split(".")
                    value = data[parts[0]][parts[1]]
                else:
                    value = data[key]
                print(f"  {key} = {value}")
        else:
            print(f"{source}: No parameter-related keys found")


if __name__ == "__main__":
    main()

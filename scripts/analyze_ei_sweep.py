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

    # Initialize results storage
    results_data = []
    missing_data = []

    # Check if we can track missing runs using task_ids
    has_task_ids = not params_df.empty and "task_id" in params_df.columns

    # Create a set of all task IDs from params_df for tracking missing runs (if possible)
    all_task_ids = set(params_df["task_id"].tolist()) if has_task_ids else set()
    found_task_ids = set()

    # Find all subdirectories in the results directory
    job_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    print(f"Found {len(job_dirs)} job directories")

    # Keep track of different types of result directories found
    job_types = {"numeric": 0, "timestamp": 0, "other": 0}
    found_formats = set()

    # Process each job directory
    for job_dir in job_dirs:
        job_path = os.path.join(results_dir, job_dir)

        # Check if this is a numeric job ID directory with task_X subdirectories
        if re.match(r"^\d+$", job_dir):
            job_types["numeric"] += 1
            # This is a job ID directory, look for task_X subdirectories
            task_dirs = glob.glob(os.path.join(job_path, "task_*"))

            for task_dir in task_dirs:
                # Extract task id from directory name
                task_match = re.search(r"task_(\d+)", os.path.basename(task_dir))
                if not task_match:
                    continue

                task_id = int(task_match.group(1))

                # Record that we found this task directory
                if has_task_ids:
                    found_task_ids.add(task_id)

                # Look for results in various possible locations

                # Option 1: Check for performance directory
                performance_dir = os.path.join(task_dir, "performance")
                if os.path.exists(performance_dir):
                    found_formats.add("performance_dir")
                    final_perf_file = os.path.join(performance_dir, "final.json")
                    if os.path.exists(final_perf_file):
                        process_performance_file(
                            final_perf_file,
                            task_id,
                            params_df,
                            results_data,
                            missing_data,
                            exp_timestamp,
                            has_task_ids,
                        )
                        continue

                # Option 2: Check for output log file
                output_log = os.path.join(task_dir, "output.log")
                if os.path.exists(output_log):
                    found_formats.add("output_log")
                    process_output_log(
                        output_log,
                        task_id,
                        params_df,
                        results_data,
                        missing_data,
                        exp_timestamp,
                        has_task_ids,
                    )
                    continue

                # Option 3: Look for config_used.yaml which might have results
                config_used = os.path.join(task_dir, "config_used.yaml")
                if os.path.exists(config_used):
                    found_formats.add("config_used.yaml")
                    process_config_file(
                        config_used,
                        task_id,
                        params_df,
                        results_data,
                        missing_data,
                        exp_timestamp,
                        has_task_ids,
                    )
                    continue

                # If we get here, no results were found
                missing_data.append(
                    {
                        "task_id": task_id,
                        "reason": "No performance data found in any expected location",
                        "status": "incomplete",
                    }
                )

        elif "dendritic_ei_sweep" in job_dir or re.search(r"\d{8}_\d{6}$", job_dir):
            job_types["timestamp"] += 1
            # This is a named directory (likely with timestamp), treat it as a single result
            # Get task_id from job_dir if possible
            task_id = -1
            task_match = re.search(r"_(\d+)$", job_dir)
            if task_match:
                task_id = int(task_match.group(1))

            # Check for EE/IE values in job_dir name
            ee_match = re.search(r"ee(\d+)", job_dir.lower())
            ie_match = re.search(r"ie(\d+)", job_dir.lower())
            ee_value_from_dir = int(ee_match.group(1)) if ee_match else None
            ie_value_from_dir = int(ie_match.group(1)) if ie_match else None

            # Try extracting values from config, logs, and model files
            metrics = {
                "train_acc": float("nan"),
                "val_acc": float("nan"),
                "test_acc": float("nan"),
            }
            params = {"ee_value": None, "ie_value": None, "seed": 0}

            # 1. Try config.json first
            config_json = os.path.join(job_path, "config.json")
            if os.path.exists(config_json):
                found_formats.add("config.json")
                try:
                    with open(config_json, "r") as f:
                        config_data = json.load(f)

                    # Extract metrics and parameters
                    extract_metrics_and_params(
                        config_data, metrics, params, f"config.json in {job_dir}"
                    )
                except Exception as e:
                    print(f"Error processing config.json in {job_path}: {e}")

            # 2. Try train.log
            train_log = os.path.join(job_path, "train.log")
            if os.path.exists(train_log):
                found_formats.add("train.log")
                try:
                    with open(train_log, "r") as f:
                        log_content = f.read()

                    # Extract metrics
                    extract_metrics_from_log(log_content, metrics)

                    # Extract params
                    extract_params_from_log(log_content, params)
                except Exception as e:
                    print(f"Error processing train.log in {job_path}: {e}")

            # 3. Try model files
            model_files = glob.glob(os.path.join(job_path, "*.pt"))
            if model_files:
                found_formats.add("model_files")
                try:
                    # Look for matching JSON files with the same name
                    for model_file in model_files:
                        base_name = os.path.splitext(model_file)[0]
                        json_file = f"{base_name}.json"
                        if os.path.exists(json_file):
                            found_formats.add("model_json")
                            try:
                                with open(json_file, "r") as f:
                                    model_data = json.load(f)
                                extract_metrics_and_params(
                                    model_data,
                                    metrics,
                                    params,
                                    f"{os.path.basename(json_file)} in {job_dir}",
                                )
                            except Exception as e:
                                print(f"Error processing {json_file}: {e}")
                except Exception as e:
                    print(f"Error checking for model JSON files in {job_path}: {e}")

            # 4. Search for general JSON files if we still don't have metrics
            if all(pd.isna(val) for val in metrics.values()):
                json_files = [
                    f
                    for f in glob.glob(os.path.join(job_path, "*.json"))
                    if f != config_json and not f.endswith(".pt.json")
                ]
                if json_files:
                    found_formats.add("other_json")
                    for json_file in json_files:
                        try:
                            with open(json_file, "r") as f:
                                json_data = json.load(f)
                            extract_metrics_and_params(
                                json_data,
                                metrics,
                                params,
                                f"{os.path.basename(json_file)} in {job_dir}",
                            )
                        except Exception:
                            pass  # Silently ignore errors in other JSON files

            # Use parameters from directory name if not found elsewhere
            if params["ee_value"] is None and ee_value_from_dir is not None:
                params["ee_value"] = ee_value_from_dir
            if params["ie_value"] is None and ie_value_from_dir is not None:
                params["ie_value"] = ie_value_from_dir

            # Use parameters from params_df if task_id is valid
            if has_task_ids and task_id >= 0:
                task_params = params_df[params_df["task_id"] == task_id]
                if not task_params.empty:
                    params["ee_value"] = task_params["ee_value"].iloc[0]
                    params["ie_value"] = task_params["ie_value"].iloc[0]
                    params["seed"] = task_params["seed"].iloc[0]
                    found_task_ids.add(task_id)

            # Set default values if still not found
            if params["ee_value"] is None:
                params["ee_value"] = "unknown"
            if params["ie_value"] is None:
                params["ie_value"] = "unknown"

            # Check if we got valid metrics
            if not all(pd.isna(val) for val in metrics.values()):
                # Append results
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
                        "job_dir": job_dir,
                    }
                )
            else:
                # No metrics found
                missing_data.append(
                    {
                        "job_dir": job_dir,
                        "reason": "No performance metrics found in any files",
                        "status": "incomplete",
                    }
                )

        else:
            job_types["other"] += 1
            # Other type of directory, try to extract information
            # List files
            files = os.listdir(job_path)
            found_formats.add(f"other_dir_{','.join(files[:3])}")

    # Find completely missing task IDs (no directory found)
    if has_task_ids:
        missing_ids = all_task_ids - found_task_ids
        for task_id in missing_ids:
            task_params = params_df[params_df["task_id"] == task_id]
            if not task_params.empty:
                ee_value = task_params["ee_value"].iloc[0]
                ie_value = task_params["ie_value"].iloc[0]
                seed = task_params["seed"].iloc[0]

                # Record as completely missing
                missing_data.append(
                    {
                        "task_id": task_id,
                        "ee_value": ee_value,
                        "ie_value": ie_value,
                        "seed": seed,
                        "reason": "No task directory found",
                        "status": "missing",
                    }
                )

                # Add NaN entries to results for complete parameter grid
                results_data.append(
                    {
                        "ee_value": ee_value,
                        "ie_value": ie_value,
                        "train_acc": float("nan"),
                        "val_acc": float("nan"),
                        "test_acc": float("nan"),
                        "experiment": exp_timestamp,
                        "task_id": task_id,
                        "seed": seed,
                    }
                )

    # Print summary of what was found
    print(
        f"Directory types: numeric={job_types['numeric']}, timestamp={job_types['timestamp']}, other={job_types['other']}"
    )
    print(f"Formats found: {', '.join(found_formats)}")
    print(f"Results collected: {len(results_data)}")

    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    missing_df = pd.DataFrame(missing_data)

    # Drop duplicates if any (might happen with multiple result sources)
    if "task_id" in results_df.columns:
        results_df = results_df.drop_duplicates(
            subset=["task_id", "ee_value", "ie_value", "seed"]
        )

    # Print sample of results
    if not results_df.empty:
        print("\nSample of collected results:")
        # Check if certain important columns are all NaN
        nan_cols = []
        for col in ["train_acc", "val_acc", "test_acc"]:
            if col in results_df.columns and results_df[col].isna().all():
                nan_cols.append(col)

        if nan_cols:
            print(f"Warning: The following columns are all NaN: {', '.join(nan_cols)}")

        # Print first few rows
        print(results_df.head().to_string())

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

    # Ensure all values are numeric
    try:
        # Handle ListConfig objects that might be in ee_value or ie_value columns
        for col in ["ee_value", "ie_value"]:
            if col in results_df.columns:
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

            # Check if we have both ee_value and ie_value columns
            if "ee_value" in results_df.columns and "ie_value" in results_df.columns:
                # Create pivot table
                pivot_df = results_df.pivot_table(
                    values=metric,
                    index="ee_value",
                    columns="ie_value",
                    aggfunc="mean",  # Average across seeds if multiple values exist
                )

                # Check if pivot table has data
                if pivot_df.empty or pivot_df.isna().all().all():
                    print(f"No valid data for {metric} pivot table")
                    continue

                # Plot heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_df, annot=True, cmap="viridis", fmt=".3f")
                plt.title(f'{metric.replace("_", " ").title()} vs EE/IE Parameters')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"{base_name}_{metric}_heatmap.png"))
                plt.close()

                # Plot 3D surface
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection="3d")

                # Get X, Y, Z data for the surface
                x = pivot_df.columns.values  # ie_values
                y = pivot_df.index.values  # ee_values
                x_grid, y_grid = np.meshgrid(x, y)
                z = pivot_df.values

                # Create the surface plot
                surf = ax.plot_surface(
                    x_grid, y_grid, z, cmap="viridis", edgecolor="none", alpha=0.8
                )

                # Add color bar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

                # Set labels
                ax.set_xlabel("IE Value")
                ax.set_ylabel("EE Value")
                ax.set_zlabel(metric.replace("_", " ").title())
                ax.set_title(f'3D Surface Plot of {metric.replace("_", " ").title()}')

                # Display the plot
                plt.savefig(os.path.join(plot_dir, f"{base_name}_{metric}_surface.png"))
                plt.close()

                # Create line plots for each EE value
                plt.figure(figsize=(10, 6))
                for ee_val in pivot_df.index:
                    row = pivot_df.loc[ee_val]
                    plt.plot(row.index, row.values, marker="o", label=f"EE={ee_val}")
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

                # Create line plots for each IE value
                plt.figure(figsize=(10, 6))
                for ie_val in pivot_df.columns:
                    col = pivot_df[ie_val]
                    plt.plot(col.index, col.values, marker="o", label=f"IE={ie_val}")
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

            # If we only have one parameter (either ee_value or ie_value), create line plots
            elif "ee_value" in results_df.columns:
                plt.figure(figsize=(10, 6))
                grouped = results_df.groupby("ee_value")[metric].mean()
                plt.plot(grouped.index, grouped.values, marker="o")
                plt.xlabel("EE Value")
                plt.ylabel(metric.replace("_", " ").title())
                plt.title(f'Effect of EE Value on {metric.replace("_", " ").title()}')
                plt.grid(True)
                plt.savefig(os.path.join(plot_dir, f"{base_name}_{metric}_ee_line.png"))
                plt.close()

            elif "ie_value" in results_df.columns:
                plt.figure(figsize=(10, 6))
                grouped = results_df.groupby("ie_value")[metric].mean()
                plt.plot(grouped.index, grouped.values, marker="o")
                plt.xlabel("IE Value")
                plt.ylabel(metric.replace("_", " ").title())
                plt.title(f'Effect of IE Value on {metric.replace("_", " ").title()}')
                plt.grid(True)
                plt.savefig(os.path.join(plot_dir, f"{base_name}_{metric}_ie_line.png"))
                plt.close()

            else:
                print(f"Cannot create plots: missing ee_value and/or ie_value columns")

        except Exception as e:
            print(f"Error creating plot for {metric}: {e}")
            traceback.print_exc()

    # Create CSV output
    results_df.to_csv(os.path.join(plot_dir, f"{base_name}_results.csv"), index=False)

    print(f"Plots and results saved to {plot_dir}")


def compare_multiple_experiments(exp_results, summary_dir):
    """Compare results from multiple experiments"""
    if not exp_results:
        print("No experiment results to compare")
        return

    # Combine all results
    all_results_df = pd.concat([df for _, df, _ in exp_results])

    # Create dataframe with experiment names for easier reference
    exp_names = pd.DataFrame(
        [{"experiment": name, "path": path} for name, _, path in exp_results]
    )

    # Get the best parameters for each experiment
    best_by_exp = []
    for exp_name, df, _ in exp_results:
        if not df.empty:
            best_idx = df["test_acc"].idxmax()
            best_row = df.loc[best_idx].copy()
            best_row["experiment"] = exp_name
            best_by_exp.append(best_row)

    best_by_exp = pd.DataFrame(best_by_exp)

    # Create comparison plots
    os.makedirs(summary_dir, exist_ok=True)

    # 1. Bar plot comparing best results across experiments
    plt.figure(figsize=(14, 8))
    best_by_exp = best_by_exp.sort_values("test_acc", ascending=False)

    x = np.arange(len(best_by_exp))
    width = 0.35

    plt.bar(
        x - width / 2,
        best_by_exp["val_acc"],
        width,
        label="Validation Accuracy",
        color="skyblue",
    )
    plt.bar(
        x + width / 2,
        best_by_exp["test_acc"],
        width,
        label="Test Accuracy",
        color="lightcoral",
    )

    plt.xlabel("Experiment")
    plt.ylabel("Accuracy")
    plt.title("Best Performance Comparison Across Experiments")
    plt.xticks(x, best_by_exp["experiment"], rotation=45, ha="right")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "experiment_comparison.png"), dpi=300)
    plt.close()

    # 2. Get unique parameter combinations across all experiments
    param_combos = (
        all_results_df[["ee_value", "ie_value"]]
        .drop_duplicates()
        .sort_values(["ee_value", "ie_value"])
        .reset_index(drop=True)
    )

    # Create a summary text file
    summary_text = f"""
    Multi-Experiment Comparison Summary
    -----------------------------------

    Experiments analyzed ({len(exp_results)}):
    """

    for idx, (name, _, path) in enumerate(exp_results, 1):
        summary_text += f"\n    {idx}. {name}: {path}"

    summary_text += """

    Best Parameters by Experiment:
    -----------------------------
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


def process_single_experiment(exp_dir, show_plots=False, report_missing=False):
    """Process and analyze a single experiment directory"""
    # Check that the directory exists and has the expected structure
    configs_dir = os.path.join(exp_dir, "configs")
    results_dir = os.path.join(exp_dir, "results")

    # Check if we have the configs directory
    has_configs = os.path.exists(configs_dir)

    # Check if we have the results directory
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found in {exp_dir}")
        return None, None

    # Create plots directory
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Collect parameter data from config files if available
    params_df = pd.DataFrame()
    if has_configs:
        params_df = collect_params_data(exp_dir)
        if params_df.empty:
            print(
                f"Warning: No parameter data found in {exp_dir}, will try to collect results directly"
            )
    else:
        print(
            f"Warning: No configs directory found in {exp_dir}, will try to collect results directly"
        )

    # Collect results
    results_df, missing_df = collect_results(exp_dir, params_df)

    # Generate plots if we have results
    if not results_df.empty:
        # Convert any ListConfig objects to simple values
        for col in ["ee_value", "ie_value"]:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(
                    lambda x: convert_to_simple_value(x)
                )

        # Convert numeric columns to appropriate types
        for col in ["ee_value", "ie_value", "train_acc", "val_acc", "test_acc"]:
            if col in results_df.columns:
                # Try to convert to numeric, coerce errors to NaN
                results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

        generate_plots(results_df, params_df, plots_dir, os.path.basename(exp_dir))

        # Save summary information
        print(f"\nResults Summary for {os.path.basename(exp_dir)}:")
        print(f"  - Total runs analyzed: {len(results_df)}")

        # Safely calculate mean values, handling NaN
        train_mean = (
            results_df["train_acc"].mean()
            if "train_acc" in results_df
            else float("nan")
        )
        val_mean = (
            results_df["val_acc"].mean() if "val_acc" in results_df else float("nan")
        )
        test_mean = (
            results_df["test_acc"].mean() if "test_acc" in results_df else float("nan")
        )

        print(f"  - Average validation accuracy: {val_mean:.4f}")
        print(f"  - Average test accuracy: {test_mean:.4f}")

        # Find best performing parameter combination, handling NaN values
        valid_results = results_df.dropna(subset=["val_acc"])
        if not valid_results.empty:
            best_params = valid_results.loc[valid_results["val_acc"].idxmax()]
            print(f"\nBest Parameter Combination:")
            print(f"  - EE value: {best_params['ee_value']}")
            print(f"  - IE value: {best_params['ie_value']}")
            print(f"  - Validation accuracy: {best_params['val_acc']:.4f}")
            print(f"  - Test accuracy: {best_params.get('test_acc', float('nan')):.4f}")

            # Save summary to file
            summary_file = os.path.join(plots_dir, "summary.txt")
            with open(summary_file, "w") as f:
                f.write(f"Results Summary for {os.path.basename(exp_dir)}:\n")
                f.write(f"  - Total runs analyzed: {len(results_df)}\n")
                f.write(f"  - Average validation accuracy: {val_mean:.4f}\n")
                f.write(f"  - Average test accuracy: {test_mean:.4f}\n\n")

                f.write(f"Best Parameter Combination:\n")
                f.write(f"  - EE value: {best_params['ee_value']}\n")
                f.write(f"  - IE value: {best_params['ie_value']}\n")
                f.write(f"  - Validation accuracy: {best_params['val_acc']:.4f}\n")
                f.write(
                    f"  - Test accuracy: {best_params.get('test_acc', float('nan')):.4f}\n"
                )
        else:
            print(
                f"\nNo valid parameter combinations found (all validation accuracies are NaN)"
            )
    else:
        print(f"Warning: No results found for {exp_dir}")

    # Report missing or unfinished runs
    if report_missing and not missing_df.empty:
        print(f"\nMissing or Unfinished Runs Summary for {os.path.basename(exp_dir)}:")
        print(f"  - Missing/unfinished runs: {len(missing_df)}")

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

        # Convert matches to values or NaN
        train_acc = float(train_match.group(1)) if train_match else float("nan")
        val_acc = float(val_match.group(1)) if val_match else float("nan")
        test_acc = float(test_match.group(1)) if test_match else float("nan")

        # Look for parameter values in log
        ee_match = re.search(
            r"ee(?:_value|_synapses|_factor)?[:=]\s*([\d\.]+)",
            log_content,
            re.IGNORECASE,
        )
        ie_match = re.search(
            r"ie(?:_value|_synapses|_factor)?[:=]\s*([\d\.]+)",
            log_content,
            re.IGNORECASE,
        )

        # Try to extract task ID from job directory name
        task_id = -1
        task_match = re.search(r"_(\d+)$", job_dir)
        if task_match:
            task_id = int(task_match.group(1))

        # Try to match with a parameter from params_df or use extracted values
        if has_task_ids and task_id >= 0:
            task_params = params_df[params_df["task_id"] == task_id]
            if not task_params.empty:
                ee_value = task_params["ee_value"].iloc[0]
                ie_value = task_params["ie_value"].iloc[0]
                seed = task_params["seed"].iloc[0]
            else:
                ee_value = float(ee_match.group(1)) if ee_match else "unknown"
                ie_value = float(ie_match.group(1)) if ie_match else "unknown"
                seed = 0
        else:
            # Use values from log if found
            ee_value = float(ee_match.group(1)) if ee_match else "unknown"
            ie_value = float(ie_match.group(1)) if ie_match else "unknown"
            seed = 0

        # Store the results - only if we found at least one accuracy value
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
        else:
            exp_dir = find_experiment_directory(
                args.timestamp, args.output_base, args.prefix
            )

        print(f"Analyzing experiment in: {exp_dir}")
        process_single_experiment(
            exp_dir, show_plots=args.show_plots, report_missing=args.report_missing
        )

    # If analyzing multiple experiments
    elif args.multiple:
        # Process each experiment
        exp_results = []
        for exp_id in args.multiple:
            if os.path.isdir(exp_id):
                exp_dir = exp_id
            else:
                exp_dir = find_experiment_directory(
                    exp_id, args.output_base, args.prefix
                )

            print(f"Analyzing experiment in: {exp_dir}")
            results_df, _ = process_single_experiment(
                exp_dir, show_plots=args.show_plots, report_missing=args.report_missing
            )
            if results_df is not None and not results_df.empty:
                exp_name = os.path.basename(exp_dir)
                exp_results.append((exp_name, results_df, exp_dir))

        # Create comparison directory
        comparison_dir = os.path.join(args.output_base, f"ei_comparison_{timestamp}")
        os.makedirs(comparison_dir, exist_ok=True)

        # Generate comparison plots
        compare_multiple_experiments(exp_results, comparison_dir)

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
        comparison_dir = os.path.join(args.output_base, f"ei_comparison_{timestamp}")
        os.makedirs(comparison_dir, exist_ok=True)

        # Generate comparison plots
        compare_multiple_experiments(exp_results, comparison_dir)


if __name__ == "__main__":
    main()

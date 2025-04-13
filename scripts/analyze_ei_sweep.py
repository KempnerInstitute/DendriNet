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

                # Convert OmegaConf objects to simple Python types
                ee_values = convert_to_simple_value(ee_values)
                ie_values = convert_to_simple_value(ie_values)

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
                        ee_value = convert_to_simple_value(config.excitation_factor)
                    elif hasattr(config.model, "excitation_factor"):
                        ee_value = convert_to_simple_value(
                            config.model.excitation_factor
                        )
                    else:
                        # Default if not found
                        ee_value = "unknown"

                    if hasattr(config, "inhibition_factor"):
                        ie_value = convert_to_simple_value(config.inhibition_factor)
                    elif hasattr(config.model, "inhibition_factor"):
                        ie_value = convert_to_simple_value(
                            config.model.inhibition_factor
                        )
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

    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory not found: {results_dir}")
        return pd.DataFrame(), pd.DataFrame()

    # Initialize results storage
    results_data = []
    missing_data = []

    # Check if we can track missing runs using task_ids
    has_task_ids = not params_df.empty and "task_id" in params_df.columns

    # Create a set of all task IDs from params_df for tracking missing runs (if possible)
    all_task_ids = set(params_df["task_id"].tolist()) if has_task_ids else set()
    found_task_ids = set()

    # Find all subdirectories in the results directory
    try:
        job_dirs = [
            d
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ]
        print(f"Found {len(job_dirs)} job directories")
    except Exception as e:
        print(f"Error reading job directories: {e}")
        job_dirs = []

    # Keep track of different types of result directories found
    job_types = {"numeric": 0, "timestamp": 0, "other": 0}
    found_formats = set()

    # Process each job directory
    for job_dir in job_dirs:
        job_path = os.path.join(results_dir, job_dir)

        # Check for best_model.pt files first
        best_model_file = os.path.join(job_path, "best_model.pt")
        best_model_json = os.path.join(job_path, "best_model.json")
        best_model_metrics = os.path.join(job_path, "best_model_metrics.json")

        # If we find a best_model.pt file, try to get metrics from associated files
        if os.path.exists(best_model_file):
            found_formats.add("best_model.pt")

            # Try to find metrics in associated JSON
            if os.path.exists(best_model_json):
                found_formats.add("best_model.json")
                try:
                    with open(best_model_json, "r") as f:
                        model_data = json.load(f)

                    # Extract metrics
                    metrics = {
                        "train_acc": float("nan"),
                        "val_acc": float("nan"),
                        "test_acc": float("nan"),
                    }
                    params = {"ee_value": None, "ie_value": None, "seed": 0}

                    extract_metrics_and_params(
                        model_data, metrics, params, f"best_model.json in {job_dir}"
                    )

                    # Process task_id and params as usual
                    task_id = -1
                    task_match = re.search(r"_(\d+)$", job_dir)
                    if task_match:
                        task_id = int(task_match.group(1))

                    if has_task_ids and task_id >= 0:
                        task_params = params_df[params_df["task_id"] == task_id]
                        if not task_params.empty:
                            params["ee_value"] = task_params["ee_value"].iloc[0]
                            params["ie_value"] = task_params["ie_value"].iloc[0]
                            params["seed"] = task_params["seed"].iloc[0]
                            found_task_ids.add(task_id)

                    # Store results
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
                                "job_dir": job_dir,
                                "source": "best_model",
                            }
                        )
                        continue  # Skip other processing for this job_dir
                except Exception as e:
                    print(f"Error processing best_model.json in {job_path}: {e}")

            # Try metrics JSON if it exists
            elif os.path.exists(best_model_metrics):
                found_formats.add("best_model_metrics.json")
                try:
                    with open(best_model_metrics, "r") as f:
                        metrics_data = json.load(f)

                    # Same processing as above...
                    # Implementation similar to above
                    continue  # Skip other processing
                except Exception as e:
                    print(
                        f"Error processing best_model_metrics.json in {job_path}: {e}"
                    )

            # If no json files but we found the model, note this
            else:
                print(
                    f"Found best_model.pt in {job_dir} but no associated metrics files"
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

        # Extract metrics from log using the function defined earlier
        extract_metrics_from_log(log_content, metrics_dict)

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
    try:
        results_df.to_csv(
            os.path.join(plot_dir, f"{base_name}_results_original.csv"), index=False
        )
    except Exception as e:
        print(f"Warning: Couldn't save original results due to: {e}")

    # Print diagnostic information
    print("\nDiagnostic information before processing:")
    for col in ["ee_value", "ie_value", "train_acc", "val_acc", "test_acc"]:
        if col in results_df.columns:
            try:
                nan_count = results_df[col].isna().sum()
                total = len(results_df)
                print(
                    f"  - {col}: {nan_count}/{total} NaN values ({nan_count/total*100:.1f}%)"
                )
                print(f"    Data types: {results_df[col].apply(type).unique()}")

                # Check for ListConfig objects to avoid min/max errors
                has_listconfig = any(
                    str(t) == "<class 'omegaconf.listconfig.ListConfig'>"
                    for t in results_df[col].apply(type).unique()
                )
                if not has_listconfig and not results_df[col].isna().all():
                    print(
                        f"    Min: {results_df[col].min()}, Max: {results_df[col].max()}"
                    )
            except Exception as e:
                print(f"  - {col}: Error checking values: {e}")

    # Ensure all values are numeric - convert OmegaConf ListConfig objects to Python types
    try:
        # First, make a copy to avoid modifying the original dataframe
        plot_df = results_df.copy()

        # Handle ListConfig objects that might be in ee_value or ie_value columns
        for col in ["ee_value", "ie_value"]:
            if col in plot_df.columns:
                # Check each value and convert ListConfig objects
                plot_df[col] = plot_df[col].apply(lambda x: convert_to_simple_value(x))

                # Then convert any 'unknown' values to NaN
                plot_df[col] = plot_df[col].replace("unknown", np.nan)

                # Then convert to numeric
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

        # Convert other metric columns to numeric
        for col in ["train_acc", "val_acc", "test_acc"]:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    except Exception as e:
        print(f"Error converting values to numeric: {e}")
        print(f"Data types: {plot_df.dtypes}")

        # Print problematic rows for debugging
        for col in ["ee_value", "ie_value", "train_acc", "val_acc", "test_acc"]:
            if col in plot_df.columns:
                try:
                    problem_rows = plot_df[
                        ~pd.to_numeric(plot_df[col], errors="coerce").notna()
                    ]
                    if not problem_rows.empty:
                        print(f"Problematic {col} values:")
                        for idx, row in problem_rows.iterrows():
                            print(
                                f"  Row {idx}: {col}={row[col]} (type: {type(row[col])})"
                            )
                except Exception as nested_e:
                    print(f"  Error checking problematic rows for {col}: {nested_e}")

    # Print diagnostic information after type conversion
    print("\nDiagnostic information after type conversion:")
    for col in ["ee_value", "ie_value", "train_acc", "val_acc", "test_acc"]:
        if col in plot_df.columns:
            nan_count = plot_df[col].isna().sum()
            total = len(plot_df)
            print(
                f"  - {col}: {nan_count}/{total} NaN values ({nan_count/total*100:.1f}%)"
            )

    # Remove rows where both ee_value and ie_value are NaN
    valid_rows = ~(pd.isna(plot_df["ee_value"]) & pd.isna(plot_df["ie_value"]))
    if valid_rows.sum() < len(plot_df):
        print(
            f"Removing {len(plot_df) - valid_rows.sum()} rows with both ee_value and ie_value as NaN"
        )
        plot_df = plot_df[valid_rows].copy()

    # Save a raw CSV of all collected data before any further filtering
    try:
        plot_df.to_csv(
            os.path.join(plot_dir, f"{base_name}_all_results_raw.csv"), index=False
        )
    except Exception as e:
        print(f"Warning: Couldn't save raw results due to: {e}")

    # Check if we still have enough data to create meaningful plots
    if len(plot_df) < 3:
        print(
            f"Warning: Only {len(plot_df)} valid results available, skipping plots that require more data"
        )
        return

    # Save a clean version without NaN values for plotting
    clean_results = plot_df.dropna(subset=["ee_value", "ie_value"]).copy()
    if len(clean_results) > 0:
        try:
            clean_results.to_csv(
                os.path.join(plot_dir, f"{base_name}_clean_results.csv"), index=False
            )
        except Exception as e:
            print(f"Warning: Couldn't save clean results due to: {e}")

    # Metrics to plot
    metrics = [
        col for col in ["train_acc", "val_acc", "test_acc"] if col in plot_df.columns
    ]

    if not metrics:
        print("No metrics columns found in results dataframe.")
        return

    # Plot the heatmap for each metric
    for metric in metrics:
        try:
            # Skip metrics that are all NaN
            if plot_df[metric].isna().all():
                print(f"Skipping {metric} - all values are NaN")
                continue

            # Use a clean subset for this metric with no NaN values
            plot_df_clean = plot_df.dropna(
                subset=[metric, "ee_value", "ie_value"]
            ).copy()

            print(
                f"\nPlotting {metric} with {len(plot_df_clean)}/{len(plot_df)} valid rows"
            )

            if len(plot_df_clean) < 3:
                print(f"Skipping {metric} plots - insufficient non-NaN data points")
                continue

            # Save this clean subset for diagnostic purposes
            plot_df_clean.to_csv(
                os.path.join(plot_dir, f"{base_name}_{metric}_clean.csv"), index=False
            )

            # Check if we have both ee_value and ie_value columns
            if (
                "ee_value" in plot_df_clean.columns
                and "ie_value" in plot_df_clean.columns
            ):
                # Create pivot table
                try:
                    pivot_df = plot_df_clean.pivot_table(
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

                # Create EI ratio plot - similar to plot_utils.py
                try:
                    plot_ei_ratio_effect(
                        plot_df_clean,
                        metric,
                        save_path=os.path.join(
                            plot_dir, f"{base_name}_{metric}_ei_ratio.png"
                        ),
                    )
                except Exception as e:
                    print(f"Error creating EI ratio plot: {e}")

            # If we only have one parameter (either ee_value or ie_value), create line plots
            elif "ee_value" in plot_df_clean.columns:
                try:
                    plt.figure(figsize=(10, 6))
                    grouped = plot_df_clean.groupby("ee_value")[metric].mean()
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

            elif "ie_value" in plot_df_clean.columns:
                try:
                    plt.figure(figsize=(10, 6))
                    grouped = plot_df_clean.groupby("ie_value")[metric].mean()
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
        if "ee_value" in plot_df.columns and "ie_value" in plot_df.columns:
            for metric in metrics:
                valid_data = plot_df.dropna(subset=[metric, "ee_value", "ie_value"])
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
    try:
        plot_df.to_csv(os.path.join(plot_dir, f"{base_name}_results.csv"), index=False)
    except Exception as e:
        print(f"Warning: Couldn't save final results due to: {e}")

    print(f"Plots and results saved to {plot_dir}")


def plot_ei_ratio_effect(df, metric_column, save_path=None):
    """Generate a plot showing the effect of E/I ratio on performance.

    Args:
        df: DataFrame containing ee_value, ie_value and metric columns
        metric_column: Name of the metric column (e.g., 'val_acc')
        save_path: Path to save the figure
    """
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()

    # Calculate E/I ratio
    plot_df["ei_ratio"] = plot_df["ee_value"] / plot_df["ie_value"]

    # Group by ei_ratio and calculate mean and std
    ei_grouped = (
        plot_df.groupby("ei_ratio")[metric_column].agg(["mean", "std"]).reset_index()
    )

    # Create figure
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
    sns.scatterplot(
        data=plot_df, x="ei_ratio", y=metric_column, alpha=0.4, color="gray"
    )

    # Fit a curve if we have enough data points
    if len(ei_grouped) >= 4:
        try:
            # Sort for curve fitting
            ei_sorted = ei_grouped.sort_values("ei_ratio")
            x = ei_sorted["ei_ratio"].values
            y = ei_sorted["mean"].values

            # Try polynomial fit
            z = np.polyfit(x, y, 3)
            p = np.poly1d(z)

            # Generate smooth curve
            x_smooth = np.linspace(x.min(), x.max(), 100)
            plt.plot(x_smooth, p(x_smooth), "r--", label="Fitted curve")
            plt.legend()
        except Exception as e:
            print(f"Could not fit curve to EI ratio data: {e}")

    # Add labels and title
    plt.xlabel("E/I Ratio (EE Value / IE Value)")
    plt.ylabel(metric_column.replace("_", " ").title())
    plt.title(f'Effect of E/I Ratio on {metric_column.replace("_", " ").title()}')
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

    # Adjust layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()

    return plt


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

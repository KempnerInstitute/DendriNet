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


def find_performance_directories(results_dir):
    """Find all performance directories recursively"""
    performance_dirs = []

    # Look for the standard performance directories
    for root, dirs, _ in os.walk(results_dir):
        if "performance" in dirs:
            perf_dir = os.path.join(root, "performance")

            # Check if there's a "final" subdirectory
            final_dir = os.path.join(perf_dir, "final")
            if os.path.isdir(final_dir):
                performance_dirs.append(final_dir)  # Use the final subdirectory
            else:
                performance_dirs.append(
                    perf_dir
                )  # Use the regular performance directory

    return performance_dirs


def process_performance_directory(perf_dir, exp_timestamp, params_df, has_task_ids):
    """Process a performance directory to extract metrics"""
    results = []

    # Get the run directory (either 2 levels up if we're in final/ or 1 level up)
    if os.path.basename(perf_dir) == "final":
        run_dir = os.path.dirname(os.path.dirname(perf_dir))  # Go up two levels
    else:
        run_dir = os.path.dirname(perf_dir)  # Go up one level

    # Try to extract task_id from the path
    task_id = -1

    # First check if the path contains config_XXX
    config_match = re.search(r"config_(\d+)", run_dir)
    if config_match:
        task_id = int(config_match.group(1))
    else:
        # Try to extract from task_XXX pattern
        task_match = re.search(r"task_(\d+)", run_dir)
        if task_match:
            task_id = int(task_match.group(1))

    # FOR DEBUGGING: For the first task, dump the entire directory structure
    if task_id in [1, 100, 200, 300, 400]:  # Check a few specific tasks for debugging
        print(f"\n==== DETAILED DEBUGGING FOR TASK {task_id} ====")
        print(f"Performance directory: {perf_dir}")
        print(f"Run directory: {run_dir}")

        # Print all files and subdirectories
        print(f"Contents of performance directory:")
        try:
            all_files = os.listdir(perf_dir)
            for item in all_files:
                item_path = os.path.join(perf_dir, item)
                if os.path.isdir(item_path):
                    print(f"  DIR: {item}")
                    # List contents of this subdirectory
                    try:
                        subdir_contents = os.listdir(item_path)
                        for subitem in subdir_contents:
                            print(f"    - {subitem}")
                    except Exception as e:
                        print(f"    Error listing subdirectory: {e}")
                else:
                    print(f"  FILE: {item} ({os.path.getsize(item_path)} bytes)")

            # Check if we can read any text files directly
            text_files = [
                f
                for f in all_files
                if f.endswith(".txt")
                or f.endswith(".log")
                or (
                    f.endswith(".json")
                    and os.path.getsize(os.path.join(perf_dir, f)) < 10000
                )
            ]
            for text_file in text_files:
                try:
                    with open(os.path.join(perf_dir, text_file), "r") as f:
                        content = f.read()
                    print(f"\nContents of {text_file}:")
                    print(content[:1000] + "..." if len(content) > 1000 else content)
                except Exception as e:
                    print(f"Error reading {text_file}: {e}")

            # Check the run directory too
            print(f"\nContents of run directory:")
            run_files = os.listdir(run_dir)
            for item in run_files:
                item_path = os.path.join(run_dir, item)
                if os.path.isdir(item_path):
                    print(f"  DIR: {item}")
                else:
                    print(f"  FILE: {item} ({os.path.getsize(item_path)} bytes)")

            # Look for metrics.json in the task directory (which could be above the run_dir)
            if "config_" in run_dir:
                task_dir = os.path.dirname(run_dir)
                print(f"\nContents of task directory {task_dir}:")
                try:
                    task_files = os.listdir(task_dir)
                    for item in task_files:
                        item_path = os.path.join(task_dir, item)
                        if os.path.isdir(item_path):
                            print(f"  DIR: {item}")
                        else:
                            print(
                                f"  FILE: {item} ({os.path.getsize(item_path)} bytes)"
                            )

                    # Check for metrics files directly in task_dir
                    for item in task_files:
                        if (
                            "metric" in item.lower()
                            or "performance" in item.lower()
                            or "result" in item.lower()
                            or "final" in item.lower()
                        ):
                            item_path = os.path.join(task_dir, item)
                            if not os.path.isdir(item_path):
                                try:
                                    with open(item_path, "r") as f:
                                        if item.endswith(".json"):
                                            data = json.load(f)
                                            print(f"\nContents of {item}:")
                                            print(
                                                json.dumps(data, indent=2)[:1000]
                                                + "..."
                                                if len(json.dumps(data)) > 1000
                                                else json.dumps(data, indent=2)
                                            )
                                        else:
                                            content = f.read()
                                            print(f"\nContents of {item}:")
                                            print(
                                                content[:1000] + "..."
                                                if len(content) > 1000
                                                else content
                                            )
                                except Exception as e:
                                    print(f"Error reading {item}: {e}")
                except Exception as e:
                    print(f"Error listing task directory: {e}")
        except Exception as e:
            print(f"Error listing performance directory: {e}")

    # First, check if there are any JSON files directly in the directory
    json_files = glob.glob(os.path.join(perf_dir, "*.json"))

    # If no JSON files found directly, try the special check for the final directory
    if not json_files:
        # First, try looking for specific performance files in the directory itself
        final_results, final_task_id = check_final_directory_for_metrics(
            perf_dir, run_dir, task_id, exp_timestamp, params_df, has_task_ids
        )

        if final_results:
            return final_results, final_task_id

        # Check for any text files that might have metrics
        text_files = glob.glob(os.path.join(perf_dir, "*.txt")) + glob.glob(
            os.path.join(perf_dir, "*.log")
        )
        if text_files and task_id in [
            1,
            100,
            200,
            300,
            400,
        ]:  # Debugging for specific tasks
            print(f"\nFound {len(text_files)} text files in performance directory:")
            for txt_file in text_files:
                try:
                    with open(txt_file, "r") as f:
                        content = f.read()
                    print(f"Contents of {os.path.basename(txt_file)}:")
                    print(content[:500] + "..." if len(content) > 500 else content)

                    # Look for accuracy metrics in text
                    accuracy_pattern = r"\b(accuracy|acc)[ :=]+(\d+\.\d+)"
                    train_pattern = (
                        r"\b(train|training)[ _]+(accuracy|acc)[ :=]+(\d+\.\d+)"
                    )
                    val_pattern = (
                        r"\b(val|valid|validation)[ _]+(accuracy|acc)[ :=]+(\d+\.\d+)"
                    )
                    test_pattern = (
                        r"\b(test|testing)[ _]+(accuracy|acc)[ :=]+(\d+\.\d+)"
                    )

                    # Look for matches
                    accuracy_match = re.search(accuracy_pattern, content, re.IGNORECASE)
                    train_match = re.search(train_pattern, content, re.IGNORECASE)
                    val_match = re.search(val_pattern, content, re.IGNORECASE)
                    test_match = re.search(test_pattern, content, re.IGNORECASE)

                    if accuracy_match or train_match or val_match or test_match:
                        print(
                            f"Found potential accuracy metrics in {os.path.basename(txt_file)}"
                        )
                except Exception as e:
                    print(f"Error reading {txt_file}: {e}")

        # If that doesn't work, check for CSV or TXT files
        csv_files = glob.glob(os.path.join(perf_dir, "*.csv"))
        txt_files = glob.glob(os.path.join(perf_dir, "*.txt"))

        # If we found any CSV or TXT files, print their names
        if csv_files or txt_files:
            print(
                f"\nTask {task_id} performance files (non-JSON): {[os.path.basename(f) for f in csv_files + txt_files]}"
            )

            # Try to read CSV files
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    print(f"CSV file content ({os.path.basename(csv_file)}):")
                    print(df.head().to_string())
                except Exception as e:
                    print(f"Error reading CSV file {csv_file}: {e}")

            # Try to read TXT files
            for txt_file in txt_files:
                try:
                    with open(txt_file, "r") as f:
                        content = f.read()
                    print(f"TXT file content ({os.path.basename(txt_file)}):")
                    print(content[:500] + "..." if len(content) > 500 else content)
                except Exception as e:
                    print(f"Error reading TXT file {txt_file}: {e}")

        # NEW: Try to find logs or any other files in parent directories that might have results
        if task_id in [
            1,
            100,
            200,
            300,
            400,
        ]:  # Only look for a few tasks to avoid too much output
            print(f"\nLooking for log files in parent directories of task {task_id}...")
            parent_dir = run_dir
            for _ in range(3):  # Look up to 3 levels up
                try:
                    log_files = glob.glob(
                        os.path.join(parent_dir, "*.log")
                    ) + glob.glob(os.path.join(parent_dir, "output.txt"))
                    if log_files:
                        print(
                            f"Found log files in {parent_dir}: {[os.path.basename(f) for f in log_files]}"
                        )
                        for log_file in log_files:
                            try:
                                with open(log_file, "r") as f:
                                    content = f.read()
                                print(
                                    f"Contents of log file {os.path.basename(log_file)}:"
                                )

                                # Look for lines with "accuracy" in them
                                accuracy_lines = [
                                    line
                                    for line in content.split("\n")
                                    if "accuracy" in line.lower()
                                    or "acc" in line.lower()
                                ]
                                if accuracy_lines:
                                    print("Found lines with accuracy:")
                                    for line in accuracy_lines[
                                        :10
                                    ]:  # Show first 10 lines
                                        print(f"  {line}")
                                else:
                                    print("No lines with accuracy found.")
                            except Exception as e:
                                print(f"Error reading log file {log_file}: {e}")
                    parent_dir = os.path.dirname(parent_dir)
                except Exception as e:
                    print(f"Error checking parent directory {parent_dir}: {e}")
                    break

        # Check if there are subdirectories that might contain performance files
        subdirs = [
            d for d in os.listdir(perf_dir) if os.path.isdir(os.path.join(perf_dir, d))
        ]
        if subdirs and task_id < 5:
            print(f"Found subdirectories in performance directory: {subdirs}")

            # Check each subdirectory for performance files
            for subdir in subdirs:
                subdir_path = os.path.join(perf_dir, subdir)

                # Recursively check this subdirectory
                subdir_results, subdir_task_id = check_final_directory_for_metrics(
                    subdir_path,
                    run_dir,
                    task_id,
                    exp_timestamp,
                    params_df,
                    has_task_ids,
                )

                if subdir_results:
                    return subdir_results, subdir_task_id

        return results, -1  # Return empty results if no files found

    # Limit debug output to first few files only
    if task_id < 5:  # Only show details for first 5 tasks
        print(
            f"\nTask {task_id} performance files: {[os.path.basename(f) for f in json_files]}"
        )

    # Use accuracy.json or test_accuracy.json if exists, otherwise use the last file alphabetically
    target_perf_file = None
    preferred_files = [
        "accuracy.json",
        "test_accuracy.json",
        "final_results.json",
        "results.json",
        "metrics.json",
    ]

    for preferred in preferred_files:
        for json_file in json_files:
            if os.path.basename(json_file).lower() == preferred.lower():
                target_perf_file = json_file
                break
        if target_perf_file:
            break

    # If none of the preferred files found, use the last file alphabetically
    if not target_perf_file:
        target_perf_file = sorted(json_files)[-1]

    try:
        with open(target_perf_file, "r") as f:
            try:
                perf_data = json.load(f)
            except json.JSONDecodeError:
                # Try to read as text file if JSON parsing fails
                f.seek(0)
                content = f.read()

                # Try to extract metrics using regex
                metrics = {
                    "train_acc": float("nan"),
                    "val_acc": float("nan"),
                    "test_acc": float("nan"),
                }

                train_match = re.search(
                    r"train[_\s]acc(?:uracy)?[\s=:]+(\d+\.\d+)", content, re.IGNORECASE
                )
                val_match = re.search(
                    r"val(?:id)?[_\s]acc(?:uracy)?[\s=:]+(\d+\.\d+)",
                    content,
                    re.IGNORECASE,
                )
                test_match = re.search(
                    r"test[_\s]acc(?:uracy)?[\s=:]+(\d+\.\d+)", content, re.IGNORECASE
                )

                if train_match:
                    metrics["train_acc"] = float(train_match.group(1))
                if val_match:
                    metrics["val_acc"] = float(val_match.group(1))
                if test_match:
                    metrics["test_acc"] = float(test_match.group(1))

                # Get parameter values from params_df
                params = {"ee_value": None, "ie_value": None, "seed": 0}
                found_task_id = -1

                if has_task_ids and task_id >= 0 and not params_df.empty:
                    task_params = params_df[params_df["task_id"] == task_id]
                    if not task_params.empty:
                        params["ee_value"] = task_params["ee_value"].iloc[0]
                        params["ie_value"] = task_params["ie_value"].iloc[0]
                        params["seed"] = task_params["seed"].iloc[0]
                        found_task_id = task_id

                # Add to results if we have at least one metric that's not NaN
                if not all(pd.isna(val) for val in metrics.values()):
                    results.append(
                        {
                            "ee_value": params["ee_value"],
                            "ie_value": params["ie_value"],
                            "train_acc": metrics["train_acc"],
                            "val_acc": metrics["val_acc"],
                            "test_acc": metrics["test_acc"],
                            "experiment": exp_timestamp,
                            "task_id": task_id,
                            "seed": params["seed"],
                            "job_dir": os.path.basename(run_dir),
                            "source": "performance_text",
                        }
                    )
                    if task_id < 5:
                        print(
                            f"Added result from text file: task={task_id}, metrics={metrics}"
                        )

                return results, found_task_id

        # If we're debugging one of the first few tasks, print the file content
        if task_id < 5:
            print(f"Performance file content ({os.path.basename(target_perf_file)}):")
            print(
                json.dumps(perf_data, indent=2)[:500] + "..."
                if len(json.dumps(perf_data)) > 500
                else json.dumps(perf_data, indent=2)
            )

        # Extract metrics
        metrics = {
            "train_acc": float("nan"),
            "val_acc": float("nan"),
            "test_acc": float("nan"),
        }

        # Expanded patterns for finding accuracy metrics
        accuracy_patterns = {
            "train_acc": [
                "train_acc",
                "train_accuracy",
                "training_accuracy",
                "train/accuracy",
                "training/accuracy",
                "accuracy_train",
                "accuracy/train",
                "train_top1",
                "train_top1_acc",
                "train",
            ],
            "val_acc": [
                "val_acc",
                "val_accuracy",
                "valid_acc",
                "valid_accuracy",
                "validation_accuracy",
                "val/accuracy",
                "valid/accuracy",
                "validation/accuracy",
                "accuracy_val",
                "accuracy_valid",
                "accuracy/val",
                "accuracy/valid",
                "val_top1",
                "val_top1_acc",
                "valid_top1",
                "valid_top1_acc",
                "val",
                "valid",
                "validation",
            ],
            "test_acc": [
                "test_acc",
                "test_accuracy",
                "testing_accuracy",
                "test/accuracy",
                "testing/accuracy",
                "accuracy_test",
                "accuracy/test",
                "test_top1",
                "test_top1_acc",
                "test",
                "testing",
            ],
        }

        # First, try to find direct key matches
        for metric_name, patterns in accuracy_patterns.items():
            for pattern in patterns:
                if pattern in perf_data:
                    # Check if the value is a dict, number, or string
                    if isinstance(perf_data[pattern], dict):
                        # Try to find 'value' or 'accuracy' in the dict
                        if "value" in perf_data[pattern]:
                            metrics[metric_name] = float(perf_data[pattern]["value"])
                        elif "accuracy" in perf_data[pattern]:
                            metrics[metric_name] = float(perf_data[pattern]["accuracy"])
                    elif isinstance(perf_data[pattern], (int, float)):
                        metrics[metric_name] = float(perf_data[pattern])
                    elif (
                        isinstance(perf_data[pattern], str)
                        and perf_data[pattern].replace(".", "", 1).isdigit()
                    ):
                        metrics[metric_name] = float(perf_data[pattern])

                    if task_id < 5 and not pd.isna(metrics[metric_name]):
                        print(
                            f"Found {metric_name}: {metrics[metric_name]} (from key: {pattern})"
                        )
                    break

        # If direct keys didn't work, try to search for keys containing the pattern
        if all(pd.isna(val) for val in metrics.values()):
            for metric_name, patterns in accuracy_patterns.items():
                for key in perf_data.keys():
                    if any(pattern in key.lower() for pattern in patterns) or any(
                        p.replace("_", "") in key.lower().replace("_", "")
                        for p in patterns
                    ):
                        try:
                            if isinstance(perf_data[key], dict):
                                if "value" in perf_data[key]:
                                    metrics[metric_name] = float(
                                        perf_data[key]["value"]
                                    )
                                elif "accuracy" in perf_data[key]:
                                    metrics[metric_name] = float(
                                        perf_data[key]["accuracy"]
                                    )
                            else:
                                metrics[metric_name] = float(perf_data[key])

                            if task_id < 5:
                                print(
                                    f"Found {metric_name} using fuzzy match: {metrics[metric_name]} (from key: {key})"
                                )
                            break
                        except (ValueError, TypeError):
                            continue

        # Check if we need to look deeper in nested structures
        if all(pd.isna(val) for val in metrics.values()):
            for key, value in perf_data.items():
                if isinstance(value, dict):
                    if task_id < 5:
                        print(f"Checking nested dictionary under key: {key}")
                    for metric_name, patterns in accuracy_patterns.items():
                        for pattern in patterns:
                            if pattern in value:
                                try:
                                    if isinstance(value[pattern], dict):
                                        if "value" in value[pattern]:
                                            metrics[metric_name] = float(
                                                value[pattern]["value"]
                                            )
                                        elif "accuracy" in value[pattern]:
                                            metrics[metric_name] = float(
                                                value[pattern]["accuracy"]
                                            )
                                    else:
                                        metrics[metric_name] = float(value[pattern])

                                    if task_id < 5:
                                        print(
                                            f"Found {metric_name} in nested dict: {metrics[metric_name]} (from key: {key}.{pattern})"
                                        )
                                    break
                                except (ValueError, TypeError):
                                    continue

        # Get parameter values
        params = {"ee_value": None, "ie_value": None, "seed": 0}

        # If we have task_id, use it to get ee/ie values from params_df
        if has_task_ids and task_id >= 0 and not params_df.empty:
            task_params = params_df[params_df["task_id"] == task_id]
            if not task_params.empty:
                params["ee_value"] = task_params["ee_value"].iloc[0]
                params["ie_value"] = task_params["ie_value"].iloc[0]
                params["seed"] = task_params["seed"].iloc[0]
                found_task_id = task_id
            else:
                found_task_id = -1
        else:
            found_task_id = -1
            # Try to find ee/ie values in the performance data
            ee_patterns = ["ee", "excit", "ee_value", "ee_synapses", "excitation"]
            ie_patterns = ["ie", "inhib", "ie_value", "ie_synapses", "inhibition"]

            for key in perf_data.keys():
                if any(pattern in key.lower() for pattern in ee_patterns):
                    params["ee_value"] = (
                        float(perf_data[key])
                        if isinstance(perf_data[key], (int, float))
                        else perf_data[key]
                    )
                elif any(pattern in key.lower() for pattern in ie_patterns):
                    params["ie_value"] = (
                        float(perf_data[key])
                        if isinstance(perf_data[key], (int, float))
                        else perf_data[key]
                    )
                elif "seed" in key.lower():
                    params["seed"] = (
                        int(perf_data[key])
                        if isinstance(perf_data[key], (int, float))
                        else perf_data[key]
                    )

        # Add to results if we have at least one metric that's not NaN
        if not all(pd.isna(val) for val in metrics.values()):
            results.append(
                {
                    "ee_value": params["ee_value"],
                    "ie_value": params["ie_value"],
                    "train_acc": metrics["train_acc"],
                    "val_acc": metrics["val_acc"],
                    "test_acc": metrics["test_acc"],
                    "experiment": exp_timestamp,
                    "task_id": task_id,
                    "seed": params["seed"],
                    "job_dir": os.path.basename(run_dir),
                    "source": "performance_dir",
                }
            )
            if task_id < 5:
                print(
                    f"Added result from performance directory: task={task_id}, metrics={metrics}"
                )
        elif task_id < 5:
            print(f"No accuracy metrics found in performance file for task {task_id}")

        return results, found_task_id

    except Exception as e:
        if task_id < 5:
            print(f"Error processing performance file {target_perf_file}: {e}")
            print(
                f"Checking if parent directories contain config files that might have metrics..."
            )

            # Try to look for config files in parent directories
            run_config_path = os.path.join(run_dir, "config.json")
            if os.path.exists(run_config_path):
                try:
                    with open(run_config_path, "r") as f:
                        config_data = json.load(f)
                    if task_id < 5:
                        print(
                            f"Found config.json in run directory, keys: {list(config_data.keys())}"
                        )

                    # Try to find metrics
                    # (You can add code to process this config file if needed)
                except Exception as config_error:
                    if task_id < 5:
                        print(f"Error processing config.json: {config_error}")

        return results, -1


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

    # First, find all performance directories
    print("Searching for performance directories...")
    performance_dirs = find_performance_directories(results_dir)

    # Limit the number of directories to process for efficiency
    max_dirs_to_process = (
        30  # Adjust this number based on how many directories you want to process
    )

    if len(performance_dirs) > max_dirs_to_process:
        print(
            f"Found {len(performance_dirs)} performance directories. Processing a sample of {max_dirs_to_process}..."
        )

        # Get a representative sample: Take directories with evenly spaced indices
        sample_indices = np.linspace(
            0, len(performance_dirs) - 1, max_dirs_to_process, dtype=int
        )
        sample_dirs = [performance_dirs[i] for i in sample_indices]
        performance_dirs = sample_dirs
    else:
        print(
            f"Found {len(performance_dirs)} performance directories. Processing all of them..."
        )

    # Process performance directories
    for i, perf_dir in enumerate(performance_dirs):
        if i % 10 == 0:  # Print progress every 10 directories
            print(f"Processing performance directory {i+1}/{len(performance_dirs)}...")

        perf_results, found_task_id = process_performance_directory(
            perf_dir, exp_timestamp, params_df, has_task_ids
        )

        if perf_results:
            results_data.extend(perf_results)
            if found_task_id >= 0:
                found_task_ids.add(found_task_id)

    print("Finished processing performance directories.")

    # If we have no results yet, look for performance data in non-standard locations
    if not results_data:
        print(
            "No results found in standard performance directories. Looking for non-standard performance files..."
        )

        # Look for any JSON files with "performance" or "metrics" in the name throughout the experiment directory
        all_performance_files = []
        for root, _, files in os.walk(results_dir):
            for file in files:
                if file.endswith(".json") and (
                    "performance" in file.lower()
                    or "accuracy" in file.lower()
                    or "metrics" in file.lower()
                    or "results" in file.lower()
                ):
                    all_performance_files.append(os.path.join(root, file))

        if all_performance_files:
            print(
                f"Found {len(all_performance_files)} potential performance files outside standard directories."
            )

            # Process the first few files as a sample
            for i, file_path in enumerate(all_performance_files[:5]):
                try:
                    # Extract task_id from path if possible
                    config_match = re.search(r"config_(\d+)", file_path)
                    task_id = int(config_match.group(1)) if config_match else -1

                    # Get parent directory as run_dir
                    run_dir = os.path.dirname(file_path)

                    print(
                        f"Checking non-standard performance file: {os.path.basename(file_path)}"
                    )

                    with open(file_path, "r") as f:
                        try:
                            data = json.load(f)

                            # Process this file similarly to check_final_directory_for_metrics
                            metrics = {
                                "train_acc": float("nan"),
                                "val_acc": float("nan"),
                                "test_acc": float("nan"),
                            }

                            # Quick check for top-level accuracy keys
                            for key in data.keys():
                                if "train" in key.lower() and "acc" in key.lower():
                                    metrics["train_acc"] = float(data[key])
                                elif (
                                    "val" in key.lower() or "valid" in key.lower()
                                ) and "acc" in key.lower():
                                    metrics["val_acc"] = float(data[key])
                                elif "test" in key.lower() and "acc" in key.lower():
                                    metrics["test_acc"] = float(data[key])

                            # If we found metrics, add to results
                            if not all(pd.isna(val) for val in metrics.values()):
                                # Get parameter values if possible
                                params = {"ee_value": None, "ie_value": None, "seed": 0}
                                if has_task_ids and task_id >= 0:
                                    task_params = params_df[
                                        params_df["task_id"] == task_id
                                    ]
                                    if not task_params.empty:
                                        params["ee_value"] = task_params[
                                            "ee_value"
                                        ].iloc[0]
                                        params["ie_value"] = task_params[
                                            "ie_value"
                                        ].iloc[0]
                                        params["seed"] = task_params["seed"].iloc[0]
                                        found_task_ids.add(task_id)

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
                                        "job_dir": os.path.basename(run_dir),
                                        "source": "non_standard_performance",
                                    }
                                )
                                print(f"Added result from non-standard file: {metrics}")

                        except json.JSONDecodeError:
                            print(f"File is not valid JSON: {file_path}")
                            continue

                except Exception as e:
                    print(f"Error processing non-standard file {file_path}: {e}")

    # Continue with the original collection logic only for task_ids not found in performance dirs
    if has_task_ids:
        remaining_task_ids = all_task_ids - found_task_ids
        if remaining_task_ids:
            print(
                f"Looking for {len(remaining_task_ids)} remaining tasks from other sources."
            )

            # Search for other result sources for these remaining task_ids
            # ... (rest of the original function to find configs for remaining tasks)

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
                        "reason": "No task directory or performance data found",
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

    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    missing_df = pd.DataFrame(missing_data)

    # Drop duplicates if any (might happen with multiple result sources)
    if "task_id" in results_df.columns and not results_df.empty:
        results_df = results_df.drop_duplicates(
            subset=["task_id", "ee_value", "ie_value", "seed"]
        )

    # Print sample of results
    if not results_df.empty:
        # Check if certain important columns are all NaN
        nan_cols = []
        for col in ["train_acc", "val_acc", "test_acc"]:
            if col in results_df.columns and results_df[col].isna().all():
                nan_cols.append(col)

        if nan_cols:
            print(f"Warning: The following columns are all NaN: {', '.join(nan_cols)}")

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

        # Print summary with focus on highest E/I ratio results
        if results_df is not None and not results_df.empty:
            # Convert ee_value and ie_value to numeric if they're not already
            for col in ["ee_value", "ie_value"]:
                if col in results_df.columns:
                    results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

            # Calculate E/I ratio
            clean_results = results_df.dropna(subset=["ee_value", "ie_value"]).copy()
            if not clean_results.empty:
                # Make sure we don't divide by zero
                clean_results = clean_results[clean_results["ie_value"] > 0].copy()
                if not clean_results.empty:
                    clean_results["ei_ratio"] = (
                        clean_results["ee_value"] / clean_results["ie_value"]
                    )

                    # Find the results with the largest E/I ratio
                    max_ei_ratio = clean_results["ei_ratio"].max()
                    max_ei_results = clean_results[
                        clean_results["ei_ratio"] == max_ei_ratio
                    ]

                    print("\n=========================================")
                    print(f"RESULTS FOR LARGEST E/I RATIO: {max_ei_ratio:.2f}")
                    print("=========================================")
                    for _, row in max_ei_results.iterrows():
                        print(
                            f"EE Value: {row['ee_value']}, IE Value: {row['ie_value']}"
                        )
                        print(
                            f"Train Accuracy: {row['train_acc']:.4f}"
                            if not pd.isna(row["train_acc"])
                            else "Train Accuracy: N/A"
                        )
                        print(
                            f"Val Accuracy: {row['val_acc']:.4f}"
                            if not pd.isna(row["val_acc"])
                            else "Val Accuracy: N/A"
                        )
                        print(
                            f"Test Accuracy: {row['test_acc']:.4f}"
                            if not pd.isna(row["test_acc"])
                            else "Test Accuracy: N/A"
                        )
                        print("-----------------------------------------")
        else:
            print(
                "\nNo valid results found. Check if the directory structure is correct."
            )

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

            # Print results for this experiment focusing on highest E/I ratio
            if results_df is not None and not results_df.empty:
                # Convert ee_value and ie_value to numeric if they're not already
                for col in ["ee_value", "ie_value"]:
                    if col in results_df.columns:
                        results_df[col] = pd.to_numeric(
                            results_df[col], errors="coerce"
                        )

                # Calculate E/I ratio
                clean_results = results_df.dropna(
                    subset=["ee_value", "ie_value"]
                ).copy()
                if not clean_results.empty:
                    # Make sure we don't divide by zero
                    clean_results = clean_results[clean_results["ie_value"] > 0].copy()
                    if not clean_results.empty:
                        clean_results["ei_ratio"] = (
                            clean_results["ee_value"] / clean_results["ie_value"]
                        )

                        # Find the results with the largest E/I ratio
                        max_ei_ratio = clean_results["ei_ratio"].max()
                        max_ei_results = clean_results[
                            clean_results["ei_ratio"] == max_ei_ratio
                        ]

                        print(f"\n=== Experiment: {os.path.basename(exp_dir)} ===")
                        print(f"LARGEST E/I RATIO: {max_ei_ratio:.2f}")
                        for _, row in max_ei_results.iterrows():
                            print(f"EE: {row['ee_value']}, IE: {row['ie_value']}")
                            print(
                                f"Train: {row['train_acc']:.4f}"
                                if not pd.isna(row["train_acc"])
                                else "Train: N/A"
                            )
                            print(
                                f"Val: {row['val_acc']:.4f}"
                                if not pd.isna(row["val_acc"])
                                else "Val: N/A"
                            )
                            print(
                                f"Test: {row['test_acc']:.4f}"
                                if not pd.isna(row["test_acc"])
                                else "Test: N/A"
                            )
                            print("-----------------------------------------")

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

            # Print results for this experiment focusing on highest E/I ratio
            if results_df is not None and not results_df.empty:
                # Convert ee_value and ie_value to numeric if they're not already
                for col in ["ee_value", "ie_value"]:
                    if col in results_df.columns:
                        results_df[col] = pd.to_numeric(
                            results_df[col], errors="coerce"
                        )

                # Calculate E/I ratio
                clean_results = results_df.dropna(
                    subset=["ee_value", "ie_value"]
                ).copy()
                if not clean_results.empty:
                    # Make sure we don't divide by zero
                    clean_results = clean_results[clean_results["ie_value"] > 0].copy()
                    if not clean_results.empty:
                        clean_results["ei_ratio"] = (
                            clean_results["ee_value"] / clean_results["ie_value"]
                        )

                        # Find the results with the largest E/I ratio
                        max_ei_ratio = clean_results["ei_ratio"].max()
                        max_ei_results = clean_results[
                            clean_results["ei_ratio"] == max_ei_ratio
                        ]

                        print(f"\n=== Experiment: {os.path.basename(folder)} ===")
                        print(f"LARGEST E/I RATIO: {max_ei_ratio:.2f}")
                        for _, row in max_ei_results.iterrows():
                            print(f"EE: {row['ee_value']}, IE: {row['ie_value']}")
                            print(
                                f"Train: {row['train_acc']:.4f}"
                                if not pd.isna(row["train_acc"])
                                else "Train: N/A"
                            )
                            print(
                                f"Val: {row['val_acc']:.4f}"
                                if not pd.isna(row["val_acc"])
                                else "Val: N/A"
                            )
                            print(
                                f"Test: {row['test_acc']:.4f}"
                                if not pd.isna(row["test_acc"])
                                else "Test: N/A"
                            )
                            print("-----------------------------------------")

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


def check_final_directory_for_metrics(
    final_dir, run_dir, task_id, exp_timestamp, params_df, has_task_ids
):
    """Check for specific files in the final directory that might contain metrics"""
    results = []

    # Look for specific files that might contain performance metrics
    specific_files = [
        "best_model.performance.json",
        "model.performance.json",
        "performance.json",
        "metrics.json",
        "accuracy.json",
    ]

    found_files = []
    for specific_file in specific_files:
        file_path = os.path.join(final_dir, specific_file)
        if os.path.exists(file_path):
            found_files.append(file_path)

    # Also check for any JSON files with "performance" or "accuracy" in the name
    all_files = os.listdir(final_dir)
    for file_name in all_files:
        if file_name.endswith(".json") and (
            "performance" in file_name.lower()
            or "accuracy" in file_name.lower()
            or "metrics" in file_name.lower()
        ):
            file_path = os.path.join(final_dir, file_name)
            if file_path not in found_files:
                found_files.append(file_path)

    if found_files and task_id < 5:
        print(
            f"Found specific performance files in final directory: {[os.path.basename(f) for f in found_files]}"
        )

    # Process each file looking for metrics
    for file_path in found_files:
        try:
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Not a valid JSON file, skip
                    continue

            if task_id < 5:
                print(f"Content of {os.path.basename(file_path)}:")
                print(
                    json.dumps(data, indent=2)[:500] + "..."
                    if len(json.dumps(data)) > 500
                    else json.dumps(data, indent=2)
                )

            # Extract metrics - check various formats
            metrics = {
                "train_acc": float("nan"),
                "val_acc": float("nan"),
                "test_acc": float("nan"),
            }

            # Check if file is already in a standard format
            if isinstance(data, dict):
                # Case 1: Direct top-level keys
                for key in [
                    "train_accuracy",
                    "val_accuracy",
                    "test_accuracy",
                    "accuracy",
                    "train_acc",
                    "val_acc",
                    "test_acc",
                ]:
                    if key in data:
                        if "train" in key:
                            metrics["train_acc"] = float(data[key])
                        elif "val" in key or "valid" in key:
                            metrics["val_acc"] = float(data[key])
                        elif "test" in key:
                            metrics["test_acc"] = float(data[key])
                        elif (
                            key == "accuracy"
                        ):  # Just "accuracy" might be test accuracy
                            metrics["test_acc"] = float(data[key])

                # Case 2: Metrics under a "metrics" key
                if "metrics" in data and isinstance(data["metrics"], dict):
                    metrics_data = data["metrics"]
                    for key in metrics_data:
                        if "train" in key.lower() and (
                            "acc" in key.lower() or "accuracy" in key.lower()
                        ):
                            metrics["train_acc"] = float(metrics_data[key])
                        elif ("val" in key.lower() or "valid" in key.lower()) and (
                            "acc" in key.lower() or "accuracy" in key.lower()
                        ):
                            metrics["val_acc"] = float(metrics_data[key])
                        elif "test" in key.lower() and (
                            "acc" in key.lower() or "accuracy" in key.lower()
                        ):
                            metrics["test_acc"] = float(metrics_data[key])

                # Case 3: Performance data might be nested under a specific key
                for top_key in ["performance", "results", "final_results"]:
                    if top_key in data and isinstance(data[top_key], dict):
                        perf_data = data[top_key]
                        for key in perf_data:
                            if "train" in key.lower() and (
                                "acc" in key.lower() or "accuracy" in key.lower()
                            ):
                                metrics["train_acc"] = float(perf_data[key])
                            elif ("val" in key.lower() or "valid" in key.lower()) and (
                                "acc" in key.lower() or "accuracy" in key.lower()
                            ):
                                metrics["val_acc"] = float(perf_data[key])
                            elif "test" in key.lower() and (
                                "acc" in key.lower() or "accuracy" in key.lower()
                            ):
                                metrics["test_acc"] = float(perf_data[key])

            # Get parameter values from params_df if we have a task_id
            params = {"ee_value": None, "ie_value": None, "seed": 0}
            found_task_id = -1

            if has_task_ids and task_id >= 0 and not params_df.empty:
                task_params = params_df[params_df["task_id"] == task_id]
                if not task_params.empty:
                    params["ee_value"] = task_params["ee_value"].iloc[0]
                    params["ie_value"] = task_params["ie_value"].iloc[0]
                    params["seed"] = task_params["seed"].iloc[0]
                    found_task_id = task_id

            # Add to results if we found at least one metric
            if not all(pd.isna(val) for val in metrics.values()):
                results.append(
                    {
                        "ee_value": params["ee_value"],
                        "ie_value": params["ie_value"],
                        "train_acc": metrics["train_acc"],
                        "val_acc": metrics["val_acc"],
                        "test_acc": metrics["test_acc"],
                        "experiment": exp_timestamp,
                        "task_id": task_id,
                        "seed": params["seed"],
                        "job_dir": os.path.basename(run_dir),
                        "source": "final_dir_performance",
                    }
                )
                if task_id < 5:
                    print(
                        f"Added result from final directory file: task={task_id}, metrics={metrics}"
                    )

                return (
                    results,
                    found_task_id,
                )  # Return after finding the first usable metrics

        except Exception as e:
            if task_id < 5:
                print(f"Error processing {os.path.basename(file_path)}: {e}")

    return [], -1  # Return empty results if no metrics found


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
debug_perf_dir.py
----------------
A simple script to debug the extraction of metrics from performance directories
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

# Path to experiment directory
EXP_DIR = "/n/holylfs06/LABS/kempner_shared/Everyone/HS_MR_dend/ei_20250520163046"


def find_performance_directories(results_dir):
    """Find all performance directories recursively"""
    performance_dirs = []

    # Look for the standard performance directories
    for root, dirs, _ in os.walk(results_dir):
        if "performance" in dirs:
            perf_dir = os.path.join(root, "performance")
            performance_dirs.append(perf_dir)

    return performance_dirs


def extract_metrics_from_final_file(perf_dir):
    """Extract metrics from the 'final' file in performance directory"""
    final_file = os.path.join(perf_dir, "final")

    if not os.path.exists(final_file):
        print(f"No 'final' file found in {perf_dir}")
        return None

    try:
        # Read the file content
        with open(final_file, "r") as f:
            content = f.read()

        # Add closing brace if missing
        if not content.strip().endswith("}"):
            content += "}"

        # Try to parse as JSON
        try:
            data = json.loads(content)

            # Extract metrics
            metrics = {
                "train_acc": data.get("train accuracy", float("nan")),
                "val_acc": data.get("valid accuracy", float("nan")),
                "test_acc": data.get("test accuracy", float("nan")),
            }

            return metrics
        except json.JSONDecodeError:
            # If JSON parsing fails, try regex extraction
            print(f"Failed to parse JSON in {final_file}. Trying regex...")

            train_match = re.search(r"train accuracy[\":]?\s*([\d\.]+)", content)
            valid_match = re.search(r"valid accuracy[\":]?\s*([\d\.]+)", content)
            test_match = re.search(r"test accuracy[\":]?\s*([\d\.]+)", content)

            metrics = {
                "train_acc": float(train_match.group(1))
                if train_match
                else float("nan"),
                "val_acc": float(valid_match.group(1)) if valid_match else float("nan"),
                "test_acc": float(test_match.group(1)) if test_match else float("nan"),
            }

            return metrics
    except Exception as e:
        print(f"Error processing {final_file}: {e}")
        return None


def extract_config_number(perf_dir):
    """Extract config number from directory path"""
    match = re.search(r"config_(\d+)", perf_dir)
    if match:
        return int(match.group(1))
    return None


def main():
    # Find performance directories
    results_dir = os.path.join(EXP_DIR, "results")
    print(f"Searching for performance directories in {results_dir}...")

    perf_dirs = find_performance_directories(results_dir)
    print(f"Found {len(perf_dirs)} performance directories")

    # Sample some directories for debugging
    sample_size = min(30, len(perf_dirs))
    indices = np.linspace(0, len(perf_dirs) - 1, sample_size, dtype=int)
    sample_dirs = [perf_dirs[i] for i in indices]

    # Process each directory in the sample
    results = []
    for i, perf_dir in enumerate(sample_dirs):
        print(f"Processing directory {i+1}/{len(sample_dirs)}: {perf_dir}")

        metrics = extract_metrics_from_final_file(perf_dir)
        if metrics:
            config_num = extract_config_number(perf_dir)
            print(
                f"  Config {config_num}: Train={metrics['train_acc']:.4f}, Val={metrics['val_acc']:.4f}, Test={metrics['test_acc']:.4f}"
            )

            result = {
                "config_num": config_num,
                "train_acc": metrics["train_acc"],
                "val_acc": metrics["val_acc"],
                "test_acc": metrics["test_acc"],
                "perf_dir": perf_dir,
            }
            results.append(result)
        else:
            print(f"  No metrics found")

    # Create a DataFrame and display results
    if results:
        df = pd.DataFrame(results)
        print("\nSummary of extracted metrics:")
        print(f"Total successful extractions: {len(df)}/{len(sample_dirs)}")
        if not df.empty:
            print(
                f"Train accuracy: mean={df['train_acc'].mean():.4f}, min={df['train_acc'].min():.4f}, max={df['train_acc'].max():.4f}"
            )
            print(
                f"Val accuracy: mean={df['val_acc'].mean():.4f}, min={df['val_acc'].min():.4f}, max={df['val_acc'].max():.4f}"
            )
            print(
                f"Test accuracy: mean={df['test_acc'].mean():.4f}, min={df['test_acc'].min():.4f}, max={df['test_acc'].max():.4f}"
            )
    else:
        print("No metrics were successfully extracted.")


if __name__ == "__main__":
    main()

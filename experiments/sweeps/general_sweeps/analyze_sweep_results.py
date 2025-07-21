#!/usr/bin/env python3
"""
analyze_sweep_results.py
------------------------
Script that analyzes the results of parameter sweeps, loads the data from
output folders, and generates plots showing performance with error bars
across multiple random seeds.

Usage:
    python analyze_sweep_results.py --results_dir outputs/sweep_results --metadata_file configs/enhanced_sweep_config_metadata.yaml
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from omegaconf import OmegaConf


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze sweep results and generate plots with error bars."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing the sweep results.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Metadata YAML file containing sweep parameter details.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_acc",
        choices=["train_acc", "valid_acc", "test_acc"],
        help="Performance metric to visualize.",
    )
    return parser.parse_args()


def load_metadata(metadata_file):
    """
    Load sweep configuration metadata.

    Parameters
    ----------
    metadata_file : str
        Path to the metadata YAML file.

    Returns
    -------
    dict
        Dictionary containing the sweep metadata.
    """
    if not os.path.exists(metadata_file):
        print(f"[ERROR] Metadata file not found: {metadata_file}")
        return None

    try:
        if metadata_file.endswith(".yaml"):
            metadata = OmegaConf.load(metadata_file)
            return OmegaConf.to_container(metadata)
        else:
            with open(metadata_file) as f:
                return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load metadata file: {e}")
        return None


def find_performance_files(results_dir):
    """
    Find performance.csv files in the results directory.

    Parameters
    ----------
    results_dir : str
        Directory containing the sweep results.

    Returns
    -------
    list
        List of paths to performance.csv files.
    """
    return glob.glob(os.path.join(results_dir, "**", "performance.csv"), recursive=True)


def extract_config_id(performance_file, results_dir):
    """
    Extract the configuration ID from the performance file path.

    Parameters
    ----------
    performance_file : str
        Path to the performance.csv file.
    results_dir : str
        Base directory for all results.

    Returns
    -------
    str
        The configuration ID.
    """
    # Extract the job ID from the path
    # The pattern will be something like: {run_name}_{job_id} or {run_name}_{array_job_id}/{array_job_id}_{array_task_id}
    rel_path = os.path.relpath(performance_file, results_dir)
    parts = rel_path.split(os.sep)

    # Look for a part that ends with an ID number pattern
    for part in parts:
        if re.search(r"_\d+$", part):
            return part

        # Also check for array job pattern
        array_match = re.search(r"_(\d+)/\1_(\d+)", "/".join(parts))
        if array_match:
            return f"{array_match.group(1)}_{array_match.group(2)}"

    # If no match found, use a hash of the path
    return f"unknown_{hash(performance_file) % 10000}"


def load_performance_data(performance_files, results_dir, metadata=None):
    """
    Load performance data from the CSV files.

    Parameters
    ----------
    performance_files : list
        List of paths to performance.csv files.
    results_dir : str
        Base directory for all results.
    metadata : dict, optional
        Dictionary mapping config IDs to parameter settings.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the performance data and sweep parameters.
    """
    data = []

    for file_path in performance_files:
        config_id = extract_config_id(file_path, results_dir)
        try:
            # Load the performance.csv file
            perf_df = pd.read_csv(file_path)

            # Get the performance data (usually should be just one row)
            for _, row in perf_df.iterrows():
                result_data = {
                    "config_id": config_id,
                    "train_acc": row.get("train_acc", np.nan),
                    "valid_acc": row.get("valid_acc", np.nan),
                    "test_acc": row.get("test_acc", np.nan),
                    "file_path": file_path,
                }

                # Add metadata if available
                if metadata and config_id in metadata:
                    config_metadata = metadata[config_id]
                    for key, value in config_metadata.items():
                        result_data[key] = value

                data.append(result_data)
        except Exception as e:
            print(f"[WARNING] Error loading file {file_path}: {e}")
            continue

    return pd.DataFrame(data)


def map_configs_to_metadata(perf_df, metadata):
    """
    Map configuration IDs in performance data to metadata.

    Parameters
    ----------
    perf_df : pandas.DataFrame
        DataFrame containing performance data.
    metadata : dict
        Dictionary mapping config names to parameter settings.

    Returns
    -------
    pandas.DataFrame
        DataFrame with metadata added.
    """
    # Create a mapping from config names to numeric IDs
    config_name_to_id = {}
    for config_name, params in metadata.items():
        # Extract the numeric part of the config name (e.g., "enhanced_sweep_config_42.yaml" -> 42)
        match = re.search(r"_(\d+)\.yaml$", config_name)
        if match:
            config_id = match.group(1)
            config_name_to_id[config_id] = params

    # Now look for job IDs in the perf_df that might match the extracted IDs
    for i, row in perf_df.iterrows():
        config_id = row["config_id"]
        # Try to find this job ID in the slurm output patterns
        for output_dir in glob.glob(os.path.join(args.results_dir, "*")):
            dir_name = os.path.basename(output_dir)
            task_id_match = re.search(r"_(\d+)$", dir_name)
            if task_id_match:
                task_id = task_id_match.group(1)
                if task_id in config_name_to_id:
                    for key, value in config_name_to_id[task_id].items():
                        perf_df.at[i, key] = value
                    break

    return perf_df


def aggregate_results(df, group_by, metric="test_acc"):
    """
    Aggregate results by parameter combinations, computing means and standard errors.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing performance data and sweep parameters.
    group_by : list
        List of column names to group by.
    metric : str
        The performance metric to aggregate.

    Returns
    -------
    pandas.DataFrame
        DataFrame with aggregated results.
    """
    # Group by the specified parameters and compute mean and standard error
    grouped = df.groupby(group_by)[metric].agg(["mean", "std", "count"]).reset_index()

    # Calculate standard error
    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])

    return grouped


def create_comparison_plots(
    df,
    param_name,
    metric="test_acc",
    output_dir="figures",
    other_params=None,
    plot_type="bar",
):
    """
    Create comparison plots for different values of a parameter.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing aggregated performance data.
    param_name : str
        The parameter to plot on the x-axis.
    metric : str
        The performance metric to visualize.
    output_dir : str
        Directory to save the plots.
    other_params : list, optional
        Other parameters to create separate plots for.
    plot_type : str
        Type of plot to create ('bar', 'line', or 'box').
    """
    os.makedirs(output_dir, exist_ok=True)

    if param_name not in df.columns:
        print(f"[WARNING] Parameter {param_name} not found in data")
        return

    # Determine if we need to create multiple plots for different parameter combinations
    if other_params:
        # Create a plot for each combination of other parameters
        param_combinations = df[other_params].drop_duplicates()
        for _, combination in param_combinations.iterrows():
            # Filter data for this combination
            mask = np.ones(len(df), dtype=bool)
            for param, value in combination.items():
                mask = mask & (df[param] == value)

            subset = df[mask].copy()
            if len(subset) == 0:
                continue

            # Create the plot
            create_single_plot(
                subset,
                param_name,
                metric,
                output_dir,
                f"{param_name}_vs_{metric}_"
                + "_".join(f"{p}_{combination[p]}" for p in other_params),
                combination,
                plot_type,
            )
    else:
        # Just create one plot for the specified parameter
        create_single_plot(
            df,
            param_name,
            metric,
            output_dir,
            f"{param_name}_vs_{metric}",
            None,
            plot_type,
        )


def create_single_plot(
    df, param_name, metric, output_dir, filename_base, title_info=None, plot_type="bar"
):
    """
    Create a single comparison plot.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing aggregated performance data.
    param_name : str
        The parameter to plot on the x-axis.
    metric : str
        The performance metric to visualize.
    output_dir : str
        Directory to save the plot.
    filename_base : str
        Base filename for the plot.
    title_info : dict, optional
        Additional information to include in the plot title.
    plot_type : str
        Type of plot to create ('bar', 'line', or 'box').
    """
    plt.figure(figsize=(10, 6))

    # Format parameter values for categorical display
    if param_name.endswith("synapses_per_branch_per_layer"):
        # For synapse counts, convert lists to string representation
        df["param_value"] = df[param_name].apply(
            lambda x: str(x) if isinstance(x, list) else str([x])
        )
    else:
        df["param_value"] = df[param_name].astype(str)

    # Create the appropriate plot type
    if plot_type == "bar":
        ax = sns.barplot(x="param_value", y="mean", data=df, color="cornflowerblue")
        # Add error bars
        for i, row in df.iterrows():
            ax.errorbar(i, row["mean"], yerr=row["se"], color="black", capsize=5)
    elif plot_type == "line":
        # Sort by parameter value for line plots if the parameter is numeric
        try:
            if param_name.endswith("synapses_per_branch_per_layer"):
                # Extract the first value from the list for sorting
                df["numeric_value"] = df[param_name].apply(
                    lambda x: x[0] if isinstance(x, list) else x
                )
                df = df.sort_values("numeric_value")
            else:
                df["numeric_value"] = pd.to_numeric(df["param_value"], errors="coerce")
                df = df.dropna(subset=["numeric_value"]).sort_values("numeric_value")
        except:
            pass

        ax = sns.lineplot(
            x="param_value", y="mean", data=df, markers=True, color="cornflowerblue"
        )
        # Add error bands
        plt.fill_between(
            df["param_value"],
            df["mean"] - df["se"],
            df["mean"] + df["se"],
            alpha=0.3,
            color="cornflowerblue",
        )
    elif plot_type == "box":
        # For box plots, we need the original data points, not aggregated
        if "original_data" in df.columns:
            ax = sns.boxplot(
                x="param_value",
                y=metric,
                data=df["original_data"],
                color="cornflowerblue",
            )
            ax = sns.swarmplot(
                x="param_value",
                y=metric,
                data=df["original_data"],
                color="black",
                alpha=0.7,
            )
        else:
            print(
                "[WARNING] Original data not available for box plot, using bar plot instead"
            )
            ax = sns.barplot(x="param_value", y="mean", data=df, color="cornflowerblue")
            for i, row in df.iterrows():
                ax.errorbar(i, row["mean"], yerr=row["se"], color="black", capsize=5)

    # Set plot labels and title
    plt.xlabel(param_name)
    plt.ylabel(f"{metric} (mean ± SE)")

    title = f"Effect of {param_name} on {metric}"
    if title_info is not None:
        title += "\n" + ", ".join(f"{k}={v}" for k, v in title_info.items())
    plt.title(title)

    # Rotate x-axis labels if they're long
    if max(len(str(val)) for val in df["param_value"]) > 10:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{filename_base}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{filename_base}.pdf"))
    plt.close()


def analyze_parameter_importance(df, metric="test_acc"):
    """
    Analyze the relative importance of different parameters on performance.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing performance data and sweep parameters.
    metric : str
        The performance metric to analyze.

    Returns
    -------
    pandas.DataFrame
        DataFrame with parameter importance results.
    """
    # Get all sweep parameters (excluding metadata columns)
    meta_columns = [
        "config_id",
        "train_acc",
        "valid_acc",
        "test_acc",
        "file_path",
        "seed",
    ]
    param_columns = [col for col in df.columns if col not in meta_columns]

    results = []

    # For each parameter, compute the variance in performance across its values
    for param in param_columns:
        try:
            # Get unique parameter values
            unique_values = df[param].unique()

            if len(unique_values) < 2:
                continue

            # For each value, compute the mean performance
            means = []
            for value in unique_values:
                mean_perf = df[df[param] == value][metric].mean()
                means.append(mean_perf)

            # Compute the range of mean performance across parameter values
            performance_range = max(means) - min(means)

            # Store the result
            results.append(
                {
                    "parameter": param,
                    "unique_values": len(unique_values),
                    "performance_range": performance_range,
                    "best_value": unique_values[np.argmax(means)],
                    "worst_value": unique_values[np.argmin(means)],
                }
            )
        except Exception as e:
            print(f"[WARNING] Error analyzing parameter {param}: {e}")
            continue

    # Convert to DataFrame and sort by importance
    importance_df = pd.DataFrame(results)
    importance_df = importance_df.sort_values("performance_range", ascending=False)

    return importance_df


def main(args):
    """Main function to analyze sweep results and generate plots."""
    # Create output directory for plots
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    print(f"[INFO] Loading metadata from {args.metadata_file}")
    metadata = load_metadata(args.metadata_file)

    if metadata is None:
        print(
            "[WARNING] No metadata available, proceeding without parameter information"
        )

    # Find performance files
    print(f"[INFO] Searching for performance files in {args.results_dir}")
    performance_files = find_performance_files(args.results_dir)
    print(f"[INFO] Found {len(performance_files)} performance files")

    if len(performance_files) == 0:
        print("[ERROR] No performance files found")
        return

    # Load performance data
    print("[INFO] Loading performance data")
    perf_df = load_performance_data(performance_files, args.results_dir)

    # Try to map job IDs to config settings from metadata
    if metadata:
        print("[INFO] Mapping job IDs to config settings")
        perf_df = map_configs_to_metadata(perf_df, metadata)

    # Save the raw data
    print("[INFO] Saving raw performance data")
    perf_df.to_csv(os.path.join(args.output_dir, "performance_data.csv"), index=False)

    # Analyze which parameters affected performance most
    print("[INFO] Analyzing parameter importance")
    param_importance = analyze_parameter_importance(perf_df, args.metric)
    param_importance.to_csv(
        os.path.join(args.output_dir, "parameter_importance.csv"), index=False
    )

    # Print the top 5 most important parameters
    if len(param_importance) > 0:
        print("\nTop parameters affecting performance:")
        print(param_importance.head(5).to_string(index=False))

    # Get sweep parameters (excluding metadata columns)
    meta_columns = ["config_id", "train_acc", "valid_acc", "test_acc", "file_path"]
    sweep_params = [col for col in perf_df.columns if col not in meta_columns]

    # Group the most important parameters first
    if len(param_importance) > 0:
        top_params = param_importance["parameter"].tolist()
        # Only keep params that are actually in the DataFrame
        top_params = [p for p in top_params if p in perf_df.columns]

        # If there are multiple parameters, create combination plots for the top 3
        if len(top_params) >= 3:
            for i, param in enumerate(top_params[:3]):
                # For each parameter, use the next two as secondary variables
                secondary_params = [top_params[j] for j in range(3) if j != i]

                # Only include seed in group_by if it's available
                group_by = [param]
                if "seed" in perf_df.columns:
                    group_by.append("seed")

                # Aggregate results
                print(f"[INFO] Generating plots for {param}")
                agg_df = aggregate_results(perf_df, group_by, args.metric)
                create_comparison_plots(
                    agg_df,
                    param,
                    args.metric,
                    args.output_dir,
                    secondary_params[:1],
                    "bar",
                )

        # Generate individual plots for the top 5 parameters
        for param in top_params[:5]:
            if param == "seed":
                continue  # Skip seed parameter

            # Only include seed in group_by if it's available
            group_by = [param]
            if "seed" in perf_df.columns:
                group_by.append("seed")

            agg_df = aggregate_results(perf_df, group_by, args.metric)
            create_comparison_plots(agg_df, param, args.metric, args.output_dir)

    print(f"\n[INFO] Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

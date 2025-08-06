"""
General Parameter Sweep Implementation
=====================================

Migrated and consolidated from scripts/sweeps/general_sweeps/
Combines functionality from:
- generate_sweep_configs.py
- analyze_sweep_results.py
"""

import glob
import itertools
import json
import os
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except AttributeError:
    # Handle seaborn version compatibility issue
    import matplotlib

    if hasattr(matplotlib.cm, "register_cmap"):
        import seaborn as sns
    else:
        matplotlib.cm.register_cmap = matplotlib.colormaps.register
        import seaborn as sns

import sys
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Use absolute imports to avoid relative import issues
try:
    from dendritic_modeling.scripts.sweeps.unified.base_sweep import (
        BaseSweepAnalyzer,
        BaseSweepGenerator,
    )
except ImportError:
    # Fallback for when run as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from base_sweep import BaseSweepAnalyzer, BaseSweepGenerator


class GeneralSweepGenerator(BaseSweepGenerator):
    """
    Generator for general parameter sweeps.

    Supports sweeping over arbitrary parameters in the configuration.
    """

    def __init__(self):
        super().__init__("general")

    def validate_config(self, config: DictConfig) -> bool:
        """Validate general sweep configuration."""
        required_sections = ["sweep_config", "base_config"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")

        # Check that sweep_config has parameters to sweep
        sweep_config = config.sweep_config
        if not sweep_config or len(sweep_config) == 0:
            raise ValueError(
                "sweep_config must contain at least one parameter to sweep"
            )

        # Validate that all sweep parameters have lists of values
        for param_name, param_values in sweep_config.items():
            if not isinstance(param_values, (list, tuple, ListConfig)):
                raise ValueError(
                    f"Sweep parameter '{param_name}' must be a list of values"
                )
            if len(param_values) == 0:
                raise ValueError(
                    f"Sweep parameter '{param_name}' must have at least one value"
                )

        return True

    def generate_configs(self, base_config: DictConfig) -> list[DictConfig]:
        """Generate general sweep configurations."""
        sweep_params = base_config.sweep_config
        base_train_config = base_config.base_config
        experiment_settings = base_config.get("experiment_settings", {})

        # Generate seeds
        seeds_per_condition = experiment_settings.get("seeds_per_condition", 1)
        base_seed = experiment_settings.get("base_seed", 42)
        seeds = [base_seed + i for i in range(seeds_per_condition)]

        # Generate all parameter combinations
        param_names = list(sweep_params.keys())
        param_values = [sweep_params[name] for name in param_names]

        all_combinations = list(itertools.product(*param_values))

        # Generate configurations
        configs = []
        config_id = 0

        for combination in all_combinations:
            for seed in seeds:
                config = deepcopy(base_train_config)

                # Apply parameter combination
                param_dict = {}
                for param_name, param_value in zip(param_names, combination):
                    self._set_nested_parameter(config, param_name, param_value)
                    param_dict[param_name] = param_value

                # Set seed
                if "train" not in config:
                    config["train"] = {}
                config["train"]["seed"] = seed

                # Add metadata
                config["_sweep_metadata"] = {
                    "sweep_type": self.sweep_type,
                    "config_id": str(config_id),
                    "seed": seed,
                    "parameters": param_dict,
                }

                configs.append(OmegaConf.create(config))
                config_id += 1

        return configs

    def _set_nested_parameter(self, config: dict, param_path: str, value: Any):
        """
        Set a parameter in nested configuration using dot notation.

        Args:
            config: Configuration dictionary to modify
            param_path: Dot-separated path (e.g., "model.network.type")
            value: Value to set
        """
        keys = param_path.split(".")
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value


class GeneralSweepAnalyzer(BaseSweepAnalyzer):
    """
    Analyzer for general parameter sweep results.

    Provides flexible analysis for arbitrary parameter sweeps.
    """

    def __init__(self):
        super().__init__("general")

    def collect_results(self, results_dir: str) -> dict[str, Any]:
        """Collect and aggregate general sweep results."""
        # Find config metadata
        config_dir = os.path.join(results_dir, "configs")
        metadata_path = os.path.join(config_dir, "sweep_metadata.yaml")

        if os.path.exists(metadata_path):
            metadata = OmegaConf.load(metadata_path)
        else:
            # Fallback: collect from individual config files
            metadata = self._collect_metadata_from_configs(config_dir)

        # Collect parameter data from configs
        params_df = self._collect_params_data(config_dir)

        # Collect performance results
        results_df, missing_df = self._collect_performance_results(
            results_dir, params_df
        )

        if results_df.empty:
            print("Warning: No performance results found")
            return {"configs": [], "missing": missing_df}

        # Clean and process results
        clean_results = self._clean_results_data(results_df)

        # Identify sweep parameters
        sweep_params = self._identify_sweep_parameters(clean_results)

        return {
            "configs": params_df,
            "results": clean_results,
            "sweep_parameters": sweep_params,
            "missing": missing_df,
            "metadata": metadata,
        }

    def generate_plots(self, results: dict[str, Any], output_dir: str) -> list[str]:
        """Generate comprehensive general sweep analysis plots."""
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        plot_paths = []

        clean_results = results["results"]
        sweep_params = results["sweep_parameters"]

        if clean_results.empty:
            print("Warning: No results to plot")
            return plot_paths

        # 1. Parameter effect plots
        plot_paths.extend(
            self._generate_parameter_effect_plots(clean_results, sweep_params, plot_dir)
        )

        # 2. Heatmaps for parameter pairs
        plot_paths.extend(
            self._generate_parameter_heatmaps(clean_results, sweep_params, plot_dir)
        )

        # 3. Performance distribution plots
        plot_paths.extend(
            self._generate_performance_distributions(clean_results, plot_dir)
        )

        # 4. Correlation analysis
        plot_paths.extend(
            self._generate_correlation_plots(clean_results, sweep_params, plot_dir)
        )

        return plot_paths

    def _collect_params_data(self, config_dir: str) -> pd.DataFrame:
        """Collect parameter data from config files."""
        config_files = glob.glob(os.path.join(config_dir, "general_config_*.yaml"))

        params_data = []
        for config_file in config_files:
            try:
                config = OmegaConf.load(config_file)

                # Extract metadata
                metadata = config.get("_sweep_metadata", {})

                # Extract key parameters
                params = {
                    "config_file": os.path.basename(config_file),
                    "config_id": metadata.get("config_id", len(params_data)),
                    "seed": metadata.get("seed", config.get("train", {}).get("seed")),
                }

                # Add sweep parameters
                sweep_params = metadata.get("parameters", {})
                params.update(sweep_params)

                params_data.append(params)

            except Exception as e:
                print(f"Warning: Could not load config {config_file}: {e}")

        return pd.DataFrame(params_data)

    def _collect_performance_results(
        self, results_dir: str, params_df: pd.DataFrame
    ) -> tuple:
        """Collect performance results from experiment outputs."""
        results_data = []
        missing_data = []

        for _, row in params_df.iterrows():
            config_id = row["config_id"]
            result_path = os.path.join(results_dir, "results", f"config_{config_id}")

            if not os.path.exists(result_path):
                missing_data.append(row.to_dict())
                continue

            # Look for performance metrics
            performance_file = os.path.join(result_path, "performance", "metrics.json")
            if os.path.exists(performance_file):
                try:
                    with open(performance_file) as f:
                        metrics = json.load(f)

                    result_row = row.to_dict()
                    result_row.update(metrics)
                    results_data.append(result_row)

                except Exception as e:
                    print(
                        f"Warning: Could not load metrics for config {config_id}: {e}"
                    )
                    missing_data.append(row.to_dict())
            else:
                missing_data.append(row.to_dict())

        results_df = pd.DataFrame(results_data) if results_data else pd.DataFrame()
        missing_df = pd.DataFrame(missing_data) if missing_data else pd.DataFrame()

        return results_df, missing_df

    def _clean_results_data(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare results data for analysis."""
        if results_df.empty:
            return results_df

        # Convert numeric columns
        for col in results_df.columns:
            if col not in ["config_file", "config_id"]:
                results_df[col] = pd.to_numeric(results_df[col], errors="ignore")

        return results_df

    def _identify_sweep_parameters(self, results_df: pd.DataFrame) -> list[str]:
        """Identify which parameters were swept (have multiple unique values)."""
        if results_df.empty:
            return []

        sweep_params = []
        exclude_cols = [
            "config_file",
            "config_id",
            "seed",
            "train_acc",
            "valid_acc",
            "test_acc",
            "train_loss",
            "valid_loss",
            "test_loss",
        ]

        for col in results_df.columns:
            if col not in exclude_cols and results_df[col].nunique() > 1:
                sweep_params.append(col)

        return sweep_params

    def _generate_parameter_effect_plots(
        self, results_df: pd.DataFrame, sweep_params: list[str], plot_dir: str
    ) -> list[str]:
        """Generate plots showing the effect of each parameter on performance."""
        plot_paths = []

        if "test_acc" not in results_df.columns or not sweep_params:
            return plot_paths

        param_effects_dir = os.path.join(plot_dir, "parameter_effects")
        os.makedirs(param_effects_dir, exist_ok=True)

        for param in sweep_params:
            if param in results_df.columns:
                plt.figure(figsize=(10, 6))

                # Group by parameter value and compute statistics
                grouped = (
                    results_df.groupby(param)["test_acc"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )

                # Plot with error bars
                plt.errorbar(
                    grouped[param],
                    grouped["mean"],
                    yerr=grouped["std"],
                    marker="o",
                    capsize=5,
                    linewidth=2,
                    markersize=8,
                )

                plt.xlabel(param)
                plt.ylabel("Test Accuracy")
                plt.title(f"Effect of {param} on Performance")
                plt.grid(True, alpha=0.3)

                # Add count annotations
                for _, row in grouped.iterrows():
                    plt.annotate(
                        f'n={row["count"]}',
                        xy=(row[param], row["mean"]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )

                plot_path = os.path.join(param_effects_dir, f"effect_of_{param}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                plot_paths.append(plot_path)

        return plot_paths

    def _generate_parameter_heatmaps(
        self, results_df: pd.DataFrame, sweep_params: list[str], plot_dir: str
    ) -> list[str]:
        """Generate heatmaps for pairs of parameters."""
        plot_paths = []

        if "test_acc" not in results_df.columns or len(sweep_params) < 2:
            return plot_paths

        heatmap_dir = os.path.join(plot_dir, "parameter_heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)

        # Generate heatmaps for parameter pairs
        for i, param1 in enumerate(sweep_params):
            for param2 in sweep_params[i + 1 :]:
                if param1 in results_df.columns and param2 in results_df.columns:
                    try:
                        # Group by both parameters and average performance
                        grouped = (
                            results_df.groupby([param1, param2])["test_acc"]
                            .mean()
                            .reset_index()
                        )
                        heatmap_data = grouped.pivot(
                            index=param2, columns=param1, values="test_acc"
                        )

                        plt.figure(figsize=(10, 8))
                        sns.heatmap(
                            heatmap_data,
                            annot=True,
                            fmt=".3f",
                            cmap="viridis",
                            cbar_kws={"label": "Test Accuracy"},
                        )
                        plt.title(f"Performance Heatmap: {param1} vs {param2}")
                        plt.xlabel(param1)
                        plt.ylabel(param2)

                        plot_path = os.path.join(
                            heatmap_dir, f"heatmap_{param1}_vs_{param2}.png"
                        )
                        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                        plt.close()
                        plot_paths.append(plot_path)

                    except Exception as e:
                        print(
                            f"Warning: Could not generate heatmap for {param1} vs {param2}: {e}"
                        )

        return plot_paths

    def _generate_performance_distributions(
        self, results_df: pd.DataFrame, plot_dir: str
    ) -> list[str]:
        """Generate performance distribution plots."""
        plot_paths = []

        performance_cols = ["train_acc", "valid_acc", "test_acc"]
        available_cols = [col for col in performance_cols if col in results_df.columns]

        if not available_cols:
            return plot_paths

        # Overall performance distribution
        plt.figure(figsize=(12, 6))

        data_for_plot = []
        labels = []

        for col in available_cols:
            data_for_plot.append(results_df[col].dropna())
            labels.append(col.replace("_", " ").title())

        plt.boxplot(data_for_plot, labels=labels)
        plt.ylabel("Accuracy")
        plt.title("Performance Distribution Across All Configurations")
        plt.xticks(rotation=45)

        plot_path = os.path.join(plot_dir, "performance_distributions.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths.append(plot_path)

        return plot_paths

    def _generate_correlation_plots(
        self, results_df: pd.DataFrame, sweep_params: list[str], plot_dir: str
    ) -> list[str]:
        """Generate correlation analysis plots."""
        plot_paths = []

        # Select numeric columns for correlation
        numeric_cols = [*sweep_params, "train_acc", "valid_acc", "test_acc"]
        available_numeric_cols = [
            col for col in numeric_cols if col in results_df.columns
        ]

        if len(available_numeric_cols) < 2:
            return plot_paths

        correlation_data = results_df[available_numeric_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_data,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            square=True,
            cbar_kws={"label": "Correlation"},
        )
        plt.title("Parameter and Performance Correlation Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plot_path = os.path.join(plot_dir, "correlation_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths.append(plot_path)

        return plot_paths

    def _collect_metadata_from_configs(self, config_dir: str) -> dict:
        """Fallback method to collect metadata from individual config files."""
        config_files = glob.glob(os.path.join(config_dir, "general_config_*.yaml"))

        return {
            "sweep_type": "general",
            "num_configs": len(config_files),
            "config_files": [os.path.basename(f) for f in config_files],
        }

#!/usr/bin/env python
"""
Unified Result Analyzer for Dendritic Modeling Framework
=======================================================

This script provides comprehensive analysis capabilities for all sweep types.
It can automatically detect the sweep type and generate sophisticated plots
including heatmaps, line plots with error bars, noise analysis, ablation studies,
and information theory visualizations.

Features:
- Performance heatmaps (metrics vs E/I parameters)
- Metric vs E/I ratio line plots with error bars
- Noise perturbation analysis (heatmaps and curves)
- Ablation analysis (layer-wise and averaged)
- Information theory plots (MI, branch information)
- Parameter scaling analysis
- Network category comparisons (shunting vs non-shunting)

Usage:
    # Analyze specific sweep type
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --sweep-type ei --results-dir output/ei_20240101120000

    # Auto-detect sweep type
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --results-dir output/ei_20240101120000 --auto-detect

    # Analyze multiple directories
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --results-dirs output/ei_* --sweep-type ei
"""

import argparse
import glob
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d

# Try importing seaborn with compatibility handling
try:
    import seaborn as sns
except AttributeError:
    import matplotlib

    if hasattr(matplotlib.cm, "register_cmap"):
        import seaborn as sns
    else:
        matplotlib.cm.register_cmap = matplotlib.colormaps.register
        import seaborn as sns

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Define supported sweep types directly to avoid circular imports
SUPPORTED_SWEEP_TYPES = ["unified", "ei", "branch", "noise", "general"]


class UnifiedResultAnalyzer:
    """
    Comprehensive analyzer that can handle results from any sweep type.
    Generates sophisticated plots including heatmaps, line plots, noise analysis, etc.
    """

    def __init__(self):
        self.supported_types = SUPPORTED_SWEEP_TYPES

        # Define metric mappings for consistent labeling
        self.metric_mapping = {
            "train accuracy": "Training Accuracy",
            "valid accuracy": "Validation Accuracy",
            "test accuracy": "Test Accuracy",
            "train categorical_loglikelihood": "Training Log-Likelihood",
            "valid categorical_loglikelihood": "Validation Log-Likelihood",
            "test categorical_loglikelihood": "Test Log-Likelihood",
            # Original MI metrics
            "mi_E_I": "MI(E:I)",
            "mi_EI_Vout": "MI((E,I):Vout)",
            "si_EI_Vout": "Syn((E,I):Vout)",
            "ri_EI_Vout": "Red((E,I):Vout)",
            "mi_E_Vout": "MI(E:Vout)",
            "mi_I_Vout": "MI(I:Vout)",
            "mi_EI_label": "MI((E,I):Class)",
            # New format MI metrics (converted from I(X;Y) format)
            "mi_E_C": "MI(E:Class)",
            "mi_I_C": "MI(I:Class)",
            "mi_Vout_C": "MI(Vout:Class)",
            "mi_Vout_I": "MI(Vout:I)",
            # Pairwise MI
            "pairwise_mi_E_I": "Pairwise MI(E:I)",
            "pairwise_mi_E_C": "Pairwise MI(E:Class)",
            "pairwise_mi_I_C": "Pairwise MI(I:Class)",
            # Conditional MI
            "conditional_mi_E_C_given_I": "MI(E:Class|I)",
            "conditional_mi_I_C_given_E": "MI(I:Class|E)",
        }

        # Define network category colors and markers
        self.category_colors = {
            "shunting": "#1f77b4",  # Blue
            "no_shunting": "#ff7f0e",  # Orange
            "unknown": "gray",
            "default": "green",
        }
        self.category_markers = {
            "shunting": "o",  # Circle
            "no_shunting": "s",  # Square
            "unknown": "x",
            "default": "^",  # Triangle
        }

        # Information theory color scheme (matching old analysis)
        self.info_colors = {
            "excitatory": "#FF6B6B",  # Red for excitatory
            "inhibitory": "#4ECDC4",  # Teal for inhibitory
            "combined": "#45B7D1",  # Blue for combined/output
            "conditional": "#8B5A96",  # Purple for conditional
            "cross": "#FFA726",  # Orange for cross-information
        }

    def detect_sweep_type(self, results_dir: str) -> Optional[str]:
        """
        Auto-detect the sweep type from the results directory.

        Args:
            results_dir: Directory containing sweep results

        Returns:
            Detected sweep type or None if unable to detect
        """
        # Check for metadata file
        metadata_path = os.path.join(results_dir, "configs", "sweep_metadata.yaml")
        if os.path.exists(metadata_path):
            try:
                metadata = OmegaConf.load(metadata_path)
                return metadata.get("sweep_type")
            except Exception:
                pass

        # Check directory name patterns
        dir_name = os.path.basename(results_dir.rstrip("/"))
        for sweep_type in self.supported_types:
            if dir_name.startswith(f"{sweep_type}_"):
                return sweep_type

        # Check config file patterns - handle both old and new naming
        config_dir = os.path.join(results_dir, "configs")
        if os.path.exists(config_dir):
            # Check for unified sweep configs
            unified_configs = glob.glob(
                os.path.join(config_dir, "unified_config_*.yaml")
            )
            if unified_configs:
                return "unified"

            # Check for other sweep types
            for sweep_type in self.supported_types:
                config_files = glob.glob(
                    os.path.join(config_dir, f"{sweep_type}_*config_*.yaml")
                )
                if config_files:
                    return sweep_type

        return None

    def collect_params_data(self, exp_dir: str) -> pd.DataFrame:
        """Collect parameter data from the config files"""
        configs_dir = os.path.join(exp_dir, "configs")
        params_data = []

        if not os.path.exists(configs_dir):
            print(f"Configs directory not found: {configs_dir}")
            return pd.DataFrame()

        # Find all config files - handle both naming conventions
        config_patterns = [
            "unified_config_*.yaml",
            "ei_*config_*.yaml",
            "branch_*config_*.yaml",
            "noise_*config_*.yaml",
            "general_*config_*.yaml",
            "*_config_*.yaml",
        ]

        config_files = []
        for pattern in config_patterns:
            config_files.extend(glob.glob(os.path.join(configs_dir, pattern)))

        # Remove duplicates
        config_files = list(set(config_files))

        print(f"Found {len(config_files)} config files to process.")

        for config_file in config_files:
            try:
                # Extract task id from filename
                task_match = re.search(
                    r"config_(\d+)\.yaml", os.path.basename(config_file)
                )
                if not task_match:
                    continue

                task_id = int(task_match.group(1))

                # Load config
                config = OmegaConf.load(config_file)

                # Extract parameters with robust handling
                data = self._extract_config_params(config, task_id, config_file)
                if data:
                    params_data.append(data)

            except Exception as e:
                print(f"Error processing config file {config_file}: {e}")

        # Create DataFrame
        params_df = pd.DataFrame(params_data)
        if not params_df.empty and "task_id" in params_df.columns:
            params_df = params_df.sort_values("task_id").reset_index(drop=True)
        else:
            print("Warning: No task_id column found in parameter data")
        return params_df

    def _extract_config_params(
        self, config: Any, task_id: int, config_file: str
    ) -> Optional[dict]:
        """Extract parameters from a config object with robust fallback strategies"""
        try:
            data = {
                "task_id": task_id,
                "ee_value": None,
                "ie_value": None,
                "branch_factors": None,
                "use_shunting": None,
                "seed": 42,  # Default seed
                "config_file": config_file,
            }

            # Extract EE and IE values with multiple fallback strategies
            try:
                # Try unified sweep structure first
                if hasattr(config.model, "core_network") and hasattr(
                    config.model.core_network, "parameters"
                ):
                    net_params = config.model.core_network.parameters

                    # Look for EI-specific parameters
                    if hasattr(net_params, "ee_synapses_per_branch_per_layer"):
                        ee_values = net_params.ee_synapses_per_branch_per_layer
                        data["ee_value"] = (
                            ee_values[0]
                            if isinstance(ee_values, list) and len(ee_values) == 1
                            else ee_values
                        )

                    if hasattr(net_params, "ie_synapses_per_branch_per_layer"):
                        ie_values = net_params.ie_synapses_per_branch_per_layer
                        data["ie_value"] = (
                            ie_values[0]
                            if isinstance(ie_values, list) and len(ie_values) == 1
                            else ie_values
                        )

                    if hasattr(net_params, "excitatory_branch_factors"):
                        branch_factors = net_params.excitatory_branch_factors
                        if (
                            isinstance(branch_factors, list)
                            and len(branch_factors) == 1
                        ):
                            data["branch_factors"] = branch_factors[0]
                        else:
                            data["branch_factors"] = "-".join(
                                str(x) for x in branch_factors
                            )

                    if hasattr(net_params, "use_shunting"):
                        data["use_shunting"] = net_params.use_shunting

                # Try old structure
                elif hasattr(config.model, "network") and hasattr(
                    config.model.network, "parameters"
                ):
                    net_params = config.model.network.parameters

                    if hasattr(net_params, "ee_synapses_per_branch_per_layer"):
                        ee_values = net_params.ee_synapses_per_branch_per_layer
                        data["ee_value"] = (
                            ee_values[0]
                            if isinstance(ee_values, list) and len(ee_values) == 1
                            else ee_values
                        )

                    if hasattr(net_params, "ie_synapses_per_branch_per_layer"):
                        ie_values = net_params.ie_synapses_per_branch_per_layer
                        data["ie_value"] = (
                            ie_values[0]
                            if isinstance(ie_values, list) and len(ie_values) == 1
                            else ie_values
                        )

                    if hasattr(net_params, "excitatory_branch_factors"):
                        branch_factors = net_params.excitatory_branch_factors
                        if (
                            isinstance(branch_factors, list)
                            and len(branch_factors) == 1
                        ):
                            data["branch_factors"] = branch_factors[0]
                        else:
                            data["branch_factors"] = "-".join(
                                str(x) for x in branch_factors
                            )

                    if hasattr(net_params, "use_shunting"):
                        data["use_shunting"] = net_params.use_shunting

                # Extract seed
                if hasattr(config.train, "seed"):
                    data["seed"] = config.train.seed

            except Exception as e:
                print(
                    f"Warning: Could not extract EE/IE values from {config_file}: {e}"
                )
                # Set default unknown values
                data["ee_value"] = "unknown"
                data["ie_value"] = "unknown"

            return data

        except Exception as e:
            print(f"Error: Could not extract parameters from {config_file}: {e}")
            return None

    def collect_results(self, exp_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Collect results from the experiment directory using the new flattened structure"""
        results_dir = os.path.join(exp_dir, "results")
        configs_dir = os.path.join(exp_dir, "configs")
        exp_timestamp = os.path.basename(exp_dir)

        # Initialize results storage
        results_data = []
        missing_data = []

        # Check if the directories exist
        if not os.path.exists(results_dir):
            print(f"Results directory not found: {results_dir}")
            return pd.DataFrame(), pd.DataFrame()

        # Get all config folders in results directory (new flattened structure)
        config_folders = [d for d in os.listdir(results_dir) if d.startswith("config_")]
        total_configs = len(config_folders)
        print(f"Found {total_configs} config folders to process.")

        # Process each config folder
        for i, config_folder in enumerate(config_folders):
            if i % 50 == 0:  # Print progress every 50 folders
                print(f"Processing config folder {i+1}/{total_configs}...")

            # Extract task_id from the folder name
            task_id_match = re.search(r"config_(\d+)", config_folder)
            if not task_id_match:
                continue

            task_id = int(task_id_match.group(1))
            config_path = os.path.join(results_dir, config_folder)

            try:
                # Collect data from this config folder (flattened structure)
                data = self._collect_config_results(
                    config_path, task_id, configs_dir, exp_timestamp
                )
                if data:
                    results_data.append(data)

            except Exception as e:
                print(f"Error processing config folder {config_folder}: {e}")
                missing_data.append(
                    {"task_id": task_id, "reason": f"Error: {e!s}", "status": "error"}
                )

        print("Finished processing all config folders.")
        print(f"Collected results for {len(results_data)} tasks.")

        # Create DataFrames
        results_df = pd.DataFrame(results_data)
        missing_df = pd.DataFrame(missing_data)

        return results_df, missing_df

    def _collect_config_results(
        self, config_path: str, task_id: int, configs_dir: str, exp_timestamp: str
    ) -> Optional[dict]:
        """Collect results from a single config directory (flattened structure)"""

        # Load config parameters
        config_yaml_patterns = [
            f"unified_config_{task_id}.yaml",
            f"ei_sweep_config_{task_id}.yaml",
            f"branch_config_{task_id}.yaml",
            f"noise_config_{task_id}.yaml",
            f"general_config_{task_id}.yaml",
        ]

        config_yaml = None
        for pattern in config_yaml_patterns:
            potential_path = os.path.join(configs_dir, pattern)
            if os.path.exists(potential_path):
                config_yaml = potential_path
                break

        if not config_yaml:
            print(f"Config YAML not found for task {task_id}")
            return None

        try:
            config = OmegaConf.load(config_yaml)
            # Extract parameters from config
            data = {
                "experiment": exp_timestamp,
                "task_id": task_id,
                "job_dir": os.path.basename(config_path),
                "ee_value": None,
                "ie_value": None,
                "branch_factors": None,
                "use_shunting": None,
                "seed": 42,  # Default seed
            }

            # Extract parameters (reuse extraction logic)
            extracted_params = self._extract_config_params(config, task_id, config_yaml)
            if extracted_params:
                data.update(extracted_params)

            # Now collect all analysis results from the flattened config directory
            # The config directory now contains files directly (not in experiment_* subdirs)
            self._collect_analysis_files(config_path, data)

            return data

        except Exception as e:
            print(f"Error processing config {task_id}: {e}")
            return None

    def _collect_analysis_files(self, config_path: str, data: dict):
        """Collect analysis results from various files in the config directory"""

        # Handle both flattened structure (new) and nested structure (old)
        # First check for flattened structure, then fall back to nested structure
        search_paths = [config_path]  # Flattened structure

        # Check for nested structure (experiment_* subdirectories)
        experiment_dirs = [
            d
            for d in os.listdir(config_path)
            if os.path.isdir(os.path.join(config_path, d))
            and d.startswith("experiment_")
        ]
        if experiment_dirs:
            # Use the first experiment directory found (usually only one)
            experiment_path = os.path.join(config_path, experiment_dirs[0])
            search_paths = [experiment_path]  # Nested structure
            # Using nested structure
        else:
            pass  # Using flattened structure

        for search_path in search_paths:
            # Collect parameter count
            nparams_file = os.path.join(search_path, "nparams")
            if os.path.exists(nparams_file):
                try:
                    with open(nparams_file) as f:
                        content = f.read()
                    content_dict = json.loads(content)
                    for key, value in content_dict.items():
                        data[key] = value
                except Exception as e:
                    print(f"Error reading nparams: {e}")

            # Collect performance results
            perf_file = os.path.join(search_path, "performance", "final")
            if os.path.exists(perf_file):
                try:
                    with open(perf_file) as f:
                        content = f.read()
                    content_dict = json.loads(content)
                    for key, value in content_dict.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                data[f"{k} {key}"] = v
                        else:
                            data[key] = value
                except Exception as e:
                    print(f"Error reading performance: {e}")

            # Collect information analysis results (new format)
            info_file = os.path.join(search_path, "information_analysis", "final")
            if os.path.exists(info_file):
                try:
                    with open(info_file) as f:
                        content = f.read()
                    content_dict = json.loads(content)

                    # Extract basic mutual information metrics
                    if "basic_mi" in content_dict and isinstance(
                        content_dict["basic_mi"], dict
                    ):
                        for mi_key, mi_value in content_dict["basic_mi"].items():
                            if isinstance(mi_value, (int, float)):
                                # Convert I(E;C) format to mi_E_C format for consistency
                                # Handle Vout -> V conversion
                                normalized_key = (
                                    mi_key.replace("I(", "mi_")
                                    .replace(";", "_")
                                    .replace(")", "")
                                    .replace("Vout", "V")
                                    .replace("_std", "_std")
                                )
                                data[normalized_key] = mi_value

                    # Extract pairwise MI if available (remove pairwise_ prefix)
                    if "pairwise_mi" in content_dict and isinstance(
                        content_dict["pairwise_mi"], dict
                    ):
                        for mi_key, mi_value in content_dict["pairwise_mi"].items():
                            if isinstance(mi_value, (int, float)):
                                # Don't add pairwise_ prefix, just normalize the key
                                normalized_key = (
                                    mi_key.replace("I(", "mi_")
                                    .replace(";", "_")
                                    .replace(")", "")
                                    .replace("Vout", "V")
                                    .replace("_std", "_std")
                                )
                                data[normalized_key] = mi_value

                    # Extract conditional MI if available (remove conditional_ prefix)
                    if "conditional_mi" in content_dict and isinstance(
                        content_dict["conditional_mi"], dict
                    ):
                        for mi_key, mi_value in content_dict["conditional_mi"].items():
                            if isinstance(mi_value, (int, float)):
                                # Don't add conditional_ prefix, just normalize the key
                                normalized_key = (
                                    mi_key.replace("I(", "mi_")
                                    .replace(";", "_")
                                    .replace(")", "")
                                    .replace("|", "_given_")
                                    .replace("Vout", "V")
                                    .replace("_std", "_std")
                                )
                                data[normalized_key] = mi_value

                    # Extract layer-specific MI data
                    if "layer_statistics" in content_dict and isinstance(
                        content_dict["layer_statistics"], dict
                    ):
                        for layer_name, layer_data in content_dict[
                            "layer_statistics"
                        ].items():
                            if isinstance(layer_data, dict):
                                # Extract layer depth from layer name (e.g., branch_layers.0 -> depth 0)
                                try:
                                    if "branch_layers." in layer_name:
                                        depth = int(
                                            layer_name.split("branch_layers.")[-1]
                                        )
                                    else:
                                        depth = 0  # Default depth
                                except (ValueError, IndexError):
                                    depth = 0

                                # Extract layer-specific MI metrics
                                for mi_key, mi_value in layer_data.items():
                                    if isinstance(mi_value, (int, float)):
                                        # Convert I(E;C) format to layer_mi_E_C format with layer info
                                        # Handle both mean and std values, and Vout -> V conversion
                                        # Remove existing _mean or _std suffixes before adding depth
                                        base_key = (
                                            mi_key.replace("I(", "layer_mi_")
                                            .replace(";", "_")
                                            .replace(")", "")
                                            .replace(",", "_")
                                            .replace("|", "_given_")
                                            .replace("Vout", "V")
                                        )
                                        # Remove _mean or _std suffix if present
                                        if base_key.endswith("_mean"):
                                            base_key = base_key[:-5]  # Remove "_mean"
                                            suffix = "_mean"
                                        elif base_key.endswith("_std"):
                                            base_key = base_key[:-4]  # Remove "_std"
                                            suffix = "_std"
                                        else:
                                            suffix = ""
                                        data[f"{base_key}_depth{depth}{suffix}"] = (
                                            mi_value
                                        )

                    # Store method and other metadata
                    for key in ["method", "n_samples"]:
                        if key in content_dict:
                            data[f"info_{key}"] = content_dict[key]

                except Exception as e:
                    print(f"Error reading information analysis: {e}")

            # Collect noise perturbation results
            noise_file = os.path.join(
                search_path, "noise_perturbation_analysis", "final"
            )
            if os.path.exists(noise_file):
                try:
                    with open(noise_file) as f:
                        content = f.read()
                    content_dict = json.loads(content)
                    for noise_type, noise_dict in content_dict.items():
                        if isinstance(noise_dict, dict):
                            for noise_level, noise_data in noise_dict.items():
                                if isinstance(noise_data, dict):
                                    for metric, value in noise_data.items():
                                        metric = metric.replace("_", "")
                                        if isinstance(value, list):
                                            for i, v in enumerate(value):
                                                data[
                                                    f"noise_{noise_type}_{metric}_{noise_level}_sample{i}"
                                                ] = v
                                        else:
                                            data[
                                                f"noise_{noise_type}_{metric}_{noise_level}_sample0"
                                            ] = value
                except Exception as e:
                    print(f"Error reading noise analysis: {e}")

            # Collect ablation results (new format)
            ablation_file = os.path.join(search_path, "ablation_analysis", "final")
            if os.path.exists(ablation_file):
                try:
                    with open(ablation_file) as f:
                        content = f.read()
                    content_dict = json.loads(content)

                    # Handle new ablation format: {"layer": {"ablation_type": {"layer_name": {"depth": X, "metrics": ...}}}}
                    if "layer" in content_dict and isinstance(
                        content_dict["layer"], dict
                    ):
                        for ablation_type, layer_data in content_dict["layer"].items():
                            if isinstance(layer_data, dict):
                                for _layer_name, layer_metrics in layer_data.items():
                                    if isinstance(layer_metrics, dict):
                                        # Extract depth information
                                        depth = layer_metrics.get("depth", 0)

                                        # Extract metrics (skip non-numeric metadata)
                                        for (
                                            metric_name,
                                            metric_value,
                                        ) in layer_metrics.items():
                                            if metric_name not in [
                                                "depth",
                                                "layer_name",
                                            ] and isinstance(
                                                metric_value, (int, float)
                                            ):
                                                # Create standardized ablation column name
                                                safe_ablation_type = (
                                                    ablation_type.replace(
                                                        " ", "_"
                                                    ).replace("-", "_")
                                                )
                                                data[
                                                    f"ablation_{safe_ablation_type}_{metric_name}_depth{depth}"
                                                ] = metric_value

                except Exception as e:
                    print(f"Error reading ablation analysis: {e}")

            # Break after processing the first valid path (flattened or first experiment dir)
            break

    def analyze_single_sweep(
        self,
        results_dir: str,
        sweep_type: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Analyze a single sweep directory with comprehensive plotting.

        Args:
            results_dir: Directory containing sweep results
            sweep_type: Type of sweep (auto-detected if None)
            output_dir: Output directory for analysis (defaults to results_dir)

        Returns:
            Analysis results dictionary
        """
        # Auto-detect sweep type if not provided
        if sweep_type is None:
            sweep_type = self.detect_sweep_type(results_dir)
            if sweep_type is None:
                raise ValueError(f"Could not detect sweep type for {results_dir}")
            print(f"Auto-detected sweep type: {sweep_type}")

        print(f"Analyzing {sweep_type} sweep results...")
        print(f"Results directory: {results_dir}")

        # Set output directory
        if output_dir is None:
            output_dir = results_dir

        # Collect results from the flattened directory structure
        print("Collecting results from config directories...")
        results_df, missing_df = self.collect_results(results_dir)

        if results_df.empty:
            print("No results found. Check if the experiment directory is correct.")
            return {
                "sweep_type": sweep_type,
                "results_dir": results_dir,
                "output_dir": output_dir,
                "plot_paths": [],
                "report_path": None,
                "results": {},
                "message": "No results found in directory.",
            }

        # Clean and process the results data
        print("Processing results data...")
        clean_results = self.clean_results_data(results_df)

        if clean_results.empty:
            print("No valid results after cleaning.")
            return {
                "sweep_type": sweep_type,
                "results_dir": results_dir,
                "output_dir": output_dir,
                "plot_paths": [],
                "report_path": None,
                "results": {},
                "message": "No valid results after data cleaning.",
            }

        # Aggregate results (mean/std over seeds)
        print("Aggregating results...")
        agg_results = self.aggregate_results(clean_results)

        # Generate comprehensive plots
        print("Generating plots...")
        plot_paths = self.generate_plots(agg_results, output_dir, f"{sweep_type}_sweep")

        # Save processed data
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Save cleaned results
        clean_results.to_csv(
            os.path.join(plots_dir, f"{sweep_type}_sweep_clean_results.csv"),
            index=False,
        )

        # Save aggregated results
        agg_results.to_csv(
            os.path.join(plots_dir, f"{sweep_type}_sweep_aggregated_results.csv"),
            index=False,
        )

        print(f"Analysis complete! Generated {len(plot_paths)} plots.")
        print(f"Plot files saved to {plots_dir}")

        return {
            "sweep_type": sweep_type,
            "results_dir": results_dir,
            "output_dir": output_dir,
            "plot_paths": plot_paths,
            "report_path": None,
            "results": {
                "total_configs": len(clean_results),
                "aggregated_configs": len(agg_results),
                "missing_configs": len(missing_df),
            },
            "message": f"Comprehensive analysis complete. Generated {len(plot_paths)} plots.",
        }

    def clean_results_data(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process the results data for plotting"""
        if results_df.empty:
            return results_df

        # Make a copy to avoid modifying the original
        clean_df = results_df.copy()

        print(f"\nInitial data: {len(clean_df)} rows")

        # Convert ee_value and ie_value from complex objects to numeric values
        for col in ["ee_value", "ie_value"]:
            if col in clean_df.columns:

                def extract_first(x):
                    try:
                        # If it has __getitem__ method (like lists, arrays, etc.)
                        if (
                            hasattr(x, "__getitem__")
                            and hasattr(x, "__len__")
                            and len(x) > 0
                        ):
                            return float(x[0])
                        # For scalar values
                        elif isinstance(x, (int, float)):
                            return float(x)
                        # For string representation of a number
                        elif (
                            isinstance(x, str)
                            and x.strip().replace(".", "", 1).isdigit()
                        ):
                            return float(x)
                        # For any other case that might be convertible to float
                        else:
                            return float(x)
                    except (TypeError, ValueError, IndexError):
                        return None

                # Apply the extraction function
                clean_df[col] = clean_df[col].apply(extract_first)

        # Calculate E/I ratio
        clean_df = clean_df.dropna(subset=["ee_value", "ie_value"]).copy()
        print(f"After dropping NaNs: {len(clean_df)} rows")

        if not clean_df.empty:
            # Make sure we don't divide by zero
            clean_df = clean_df[clean_df["ie_value"] > 0].copy()
            print(f"After removing zero/negative IE values: {len(clean_df)} rows")

            if not clean_df.empty:
                clean_df["ei_ratio"] = clean_df["ee_value"] / clean_df["ie_value"]
                print(
                    f"E/I ratio range: {clean_df['ei_ratio'].min():.2f} to {clean_df['ei_ratio'].max():.2f}"
                )

        # Create a network category based on use_shunting
        clean_df["network_category"] = "unknown"

        # Update network category based on use_shunting
        if "use_shunting" in clean_df.columns:
            clean_df.loc[clean_df["use_shunting"], "network_category"] = "shunting"
            clean_df.loc[not clean_df["use_shunting"], "network_category"] = (
                "no_shunting"
            )

        # Print diagnostics about the cleaned data
        print(f"\nCleaned data summary: {len(clean_df)} rows")
        if "network_category" in clean_df.columns and len(clean_df) > 0:
            for category in clean_df["network_category"].unique():
                count = len(clean_df[clean_df["network_category"] == category])
                print(f"  {category}: {count} rows")

        return clean_df

    def aggregate_results(self, cleaned_results: pd.DataFrame) -> pd.DataFrame:
        """Produce mean and std of metrics over seeds and samples"""
        if cleaned_results.empty:
            return cleaned_results

        # Define architecture columns for grouping
        arch_cols = [
            "ee_value",
            "ie_value",
            "ei_ratio",
            "branch_factors",
            "use_shunting",
            "network_category",
        ]
        if "nparams" in cleaned_results.columns:
            arch_cols.append("nparams")

        # Find noise columns
        noise_cols = [
            col for col in cleaned_results.columns if col.startswith("noise_")
        ]

        # Define columns to exclude from aggregation
        exclude_cols = (
            arch_cols
            + noise_cols
            + ["experiment", "task_id", "job_dir", "seed", "config_file"]
        )
        metric_cols = [
            col for col in cleaned_results.columns if col not in exclude_cols
        ]

        # Process noise data
        noise_patterns = []
        for col in noise_cols:
            parts = col.split("_")
            if len(parts) >= 4:
                noise_type = parts[1]
                metric = parts[2]
                noise_level = parts[-2]
                pattern = (noise_type, metric, noise_level)
                noise_patterns.append(pattern)

        noise_patterns = list(set(noise_patterns))

        # Group by architecture columns and aggregate
        results = []
        groupby_cols = [col for col in arch_cols if col in cleaned_results.columns]

        for _, group in cleaned_results.groupby(groupby_cols):
            row = dict(zip(groupby_cols, group[groupby_cols].iloc[0]))

            # Aggregate noise patterns
            for noise_type, metric, noise_level in noise_patterns:
                pattern_cols = [
                    col
                    for col in noise_cols
                    if f"noise_{noise_type}_{metric}_{noise_level}" in col
                ]

                if pattern_cols:
                    values = group[pattern_cols].values.flatten()
                    values = values[~pd.isna(values)]  # Remove NaN values
                    if len(values) > 0:
                        row[f"noise_{noise_type}_{metric}_{noise_level}_mean"] = (
                            np.mean(values)
                        )
                        row[f"noise_{noise_type}_{metric}_{noise_level}_std"] = np.std(
                            values
                        )

            # Aggregate regular metrics
            for col in metric_cols:
                if col in group.columns:
                    values = group[col].dropna().values
                    if len(values) > 0:
                        try:
                            # Ensure values are numeric
                            numeric_values = [
                                float(v) for v in values if not pd.isna(v)
                            ]
                            if len(numeric_values) > 0:
                                row[f"{col}_mean"] = np.mean(numeric_values)
                                row[f"{col}_std"] = np.std(numeric_values)
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Could not aggregate {col}: {e}")
                            continue

            results.append(row)

        agg_results = pd.DataFrame(results)

        # Reorder columns
        new_metric_cols = []
        for col in metric_cols:
            new_metric_cols.extend([f"{col}_mean", f"{col}_std"])

        # Determine final column order
        final_cols = groupby_cols.copy()
        for col in new_metric_cols:
            if col in agg_results.columns:
                final_cols.append(col)

        # Add remaining columns
        for col in agg_results.columns:
            if col not in final_cols:
                final_cols.append(col)

        agg_results = agg_results.reindex(columns=final_cols)

        print(f"Aggregated to {len(agg_results)} unique parameter combinations")
        return agg_results

    def generate_plots(
        self, results_df: pd.DataFrame, output_dir: str, base_name: str
    ) -> list[str]:
        """Generate comprehensive plots from the results dataframe"""
        if results_df.empty:
            print("No results to plot.")
            return []

        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plot_paths = []

        # Save original data for reference
        results_df.to_csv(
            os.path.join(plots_dir, f"{base_name}_results_aggregated.csv"), index=False
        )

        # Filter for valid data
        valid_data = (
            results_df.dropna(subset=["ei_ratio"])
            if "ei_ratio" in results_df.columns
            else results_df
        )

        if valid_data.empty:
            print("No valid data for plotting.")
            return plot_paths

        # Generate different types of plots
        try:
            # 1. Metric vs E/I ratio plots
            ei_plots = self._generate_ei_ratio_plots(valid_data, plots_dir, base_name)
            plot_paths.extend(ei_plots)

            # 2. Parameter count plots (if nparams available)
            if "nparams" in valid_data.columns:
                param_plots = self._generate_parameter_plots(
                    valid_data, plots_dir, base_name
                )
                plot_paths.extend(param_plots)

            # 3. Noise analysis plots (if noise data available)
            noise_cols = [col for col in valid_data.columns if col.startswith("noise_")]
            if noise_cols:
                noise_plots = self._generate_noise_plots(
                    valid_data, plots_dir, base_name
                )
                plot_paths.extend(noise_plots)

            # 4. Heatmap plots
            heatmap_plots = self._generate_heatmaps(valid_data, plots_dir, base_name)
            plot_paths.extend(heatmap_plots)

            # 5. Ablation analysis plots (if ablation data available)
            ablation_cols = [
                col for col in valid_data.columns if col.startswith("ablation_")
            ]
            if ablation_cols:
                ablation_plots = self._generate_ablation_plots(
                    valid_data, plots_dir, base_name
                )
                plot_paths.extend(ablation_plots)

            # 6. Information theory plots (if MI data available)
            mi_cols = [col for col in valid_data.columns if "mi_" in col.lower()]
            if mi_cols:
                info_plots = self._generate_information_plots(
                    valid_data, plots_dir, base_name
                )
                plot_paths.extend(info_plots)

        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback

            traceback.print_exc()

        print(f"Generated {len(plot_paths)} plots")
        return plot_paths

    def _generate_ei_ratio_plots(
        self, valid_data: pd.DataFrame, plots_dir: str, base_name: str
    ) -> list[str]:
        """Generate metric vs E/I ratio line plots with error bars"""
        plot_paths = []

        if "ei_ratio" not in valid_data.columns:
            return plot_paths

        metric_ei_plot_dir = os.path.join(plots_dir, "metric_vs_ei_ratio_plots")
        os.makedirs(metric_ei_plot_dir, exist_ok=True)

        # List of metrics to plot
        metrics = list(self.metric_mapping.keys())

        for metric in metrics:
            # Check if we have this metric in the data
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col not in valid_data.columns:
                continue

            try:
                # Generate plot for this metric
                fig, ax = plt.subplots(figsize=(10, 6))

                # Make sure we have a network_category column
                if "network_category" not in valid_data.columns:
                    valid_data["network_category"] = "default"

                # Get unique categories
                categories = valid_data["network_category"].unique()

                for category in categories:
                    category_data = valid_data[
                        valid_data["network_category"] == category
                    ]

                    if category_data.empty:
                        continue

                    # Sort by ei_ratio for proper line plotting
                    category_data = category_data.sort_values("ei_ratio")

                    # Plot with error bars
                    color = self.category_colors.get(category, "black")
                    marker = self.category_markers.get(category, "o")

                    y_values = category_data[mean_col]
                    yerr_values = (
                        category_data[std_col]
                        if std_col in category_data.columns
                        else None
                    )

                    ax.errorbar(
                        category_data["ei_ratio"],
                        y_values,
                        yerr=yerr_values,
                        label=f"{category}",
                        marker=marker,
                        color=color,
                        linestyle="-",
                        alpha=0.8,
                        capsize=3,
                    )

                ax.set_xlabel("E/I Ratio", fontsize=14)
                ax.set_ylabel(self.metric_mapping[metric], fontsize=14)
                ax.set_title(
                    f"{self.metric_mapping[metric]} vs E/I Ratio by Network Type",
                    fontsize=14,
                )
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", labelsize=12)

                # Set log scale for x-axis
                ax.set_xscale("log")
                formatter = ScalarFormatter()
                formatter.set_scientific(False)
                ax.xaxis.set_major_formatter(formatter)

                # Set reasonable tick positions
                ei_ratios = valid_data["ei_ratio"].dropna()
                if not ei_ratios.empty:
                    min_ratio = ei_ratios.min()
                    max_ratio = ei_ratios.max()
                    tick_positions = np.logspace(
                        np.log10(min_ratio), np.log10(max_ratio), 8
                    )
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels([f"{x:.1f}" for x in tick_positions])

                # Save the plot
                fig.tight_layout()
                plot_path = os.path.join(
                    metric_ei_plot_dir, f"{base_name}_{metric}_vs_ei_ratio.png"
                )
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating {metric} plot: {e}")

        print(f"Created {len(plot_paths)} metric vs E/I ratio plots")
        return plot_paths

    def _generate_parameter_plots(
        self, valid_data: pd.DataFrame, plots_dir: str, base_name: str
    ) -> list[str]:
        """Generate metric vs parameter count plots"""
        plot_paths = []

        if "nparams" not in valid_data.columns:
            return plot_paths

        metric_bf_plot_dir = os.path.join(plots_dir, "metric_vs_parameter_plots")
        os.makedirs(metric_bf_plot_dir, exist_ok=True)

        # List of metrics to plot
        metrics = list(self.metric_mapping.keys())

        for metric in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col not in valid_data.columns:
                continue

            try:
                fig, ax = plt.subplots(figsize=(10, 6))

                if "network_category" not in valid_data.columns:
                    valid_data["network_category"] = "default"

                categories = valid_data["network_category"].unique()

                for category in categories:
                    category_data = valid_data[
                        valid_data["network_category"] == category
                    ]

                    if category_data.empty:
                        continue

                    category_data = category_data.sort_values("nparams")

                    color = self.category_colors.get(category, "black")
                    marker = self.category_markers.get(category, "o")

                    y_values = category_data[mean_col]
                    yerr_values = (
                        category_data[std_col]
                        if std_col in category_data.columns
                        else None
                    )

                    ax.plot(
                        category_data["nparams"],
                        y_values,
                        marker=marker,
                        color=color,
                        linestyle="-",
                        alpha=0.8,
                        label=f"{category}",
                    )

                    if yerr_values is not None:
                        ax.fill_between(
                            category_data["nparams"],
                            y_values - yerr_values,
                            y_values + yerr_values,
                            color=color,
                            alpha=0.2,
                        )

                ax.set_xlabel("Parameter Count", fontsize=14)
                ax.set_ylabel(self.metric_mapping[metric], fontsize=14)
                ax.set_title(
                    f"{self.metric_mapping[metric]} vs Parameter Count by Network Type",
                    fontsize=14,
                )
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", labelsize=12)

                ax.set_xscale("log")
                formatter = ScalarFormatter()
                formatter.set_scientific(True)
                ax.xaxis.set_major_formatter(formatter)

                fig.tight_layout()
                plot_path = os.path.join(
                    metric_bf_plot_dir, f"{base_name}_{metric}_vs_nparams.png"
                )
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating parameter plot for {metric}: {e}")

        print(f"Created {len(plot_paths)} parameter plots")
        return plot_paths

    def _generate_noise_plots(
        self, valid_data: pd.DataFrame, plots_dir: str, base_name: str
    ) -> list[str]:
        """Generate noise perturbation analysis plots"""
        plot_paths = []

        noise_ei_plot_dir = os.path.join(plots_dir, "noise_vs_ei_ratio_plots")
        os.makedirs(noise_ei_plot_dir, exist_ok=True)

        noise_mapping = {"uniform": "Uniform", "gaussian": "Gaussian"}
        metric_mapping = {
            "accuracy": "Test Accuracy",
            "categoricalll": "Test Log-Likelihood",
        }

        noise_cols = [col for col in valid_data.columns if col.startswith("noise_")]
        if not noise_cols:
            return plot_paths

        # Extract noise types and metrics
        noise_types = []
        metrics = []
        for col in noise_cols:
            parts = col.split("_")
            if len(parts) >= 4:
                noise_type = parts[1]
                metric = parts[2]
                if noise_type not in noise_types:
                    noise_types.append(noise_type)
                if metric not in metrics:
                    metrics.append(metric)

        if not valid_data["network_category"].isin(["shunting", "no_shunting"]).any():
            print("No shunting/no_shunting data for noise analysis")
            return plot_paths

        # Generate noise plots for each combination
        for noise_type, metric in itertools.product(noise_types, metrics):
            try:
                mean_cols = [
                    col
                    for col in noise_cols
                    if noise_type in col and metric in col and "mean" in col
                ]

                if not mean_cols:
                    continue

                # Create heatmap data
                shunting_df = valid_data[
                    valid_data["network_category"] == "shunting"
                ].sort_values("ei_ratio")
                no_shunting_df = valid_data[
                    valid_data["network_category"] == "no_shunting"
                ].sort_values("ei_ratio")

                if shunting_df.empty or no_shunting_df.empty:
                    continue

                heatmap_data = []
                for mean_col in mean_cols:
                    parts = mean_col.split("_")
                    try:
                        noise_level = float(parts[-2])
                    except (ValueError, IndexError):
                        continue

                    for ei_ratio in shunting_df["ei_ratio"].unique():
                        shunt_vals = shunting_df[shunting_df["ei_ratio"] == ei_ratio][
                            mean_col
                        ]
                        no_shunt_vals = no_shunting_df[
                            no_shunting_df["ei_ratio"] == ei_ratio
                        ][mean_col]

                        if not shunt_vals.empty and not no_shunt_vals.empty:
                            diff = shunt_vals.iloc[0] - no_shunt_vals.iloc[0]
                            heatmap_data.append(
                                {
                                    "ei_ratio": ei_ratio,
                                    "noise_level": noise_level,
                                    "diff": diff,
                                }
                            )

                if heatmap_data:
                    heatmap_df = pd.DataFrame(heatmap_data)
                    heatmap_pivot = heatmap_df.pivot(
                        index="noise_level", columns="ei_ratio", values="diff"
                    )

                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        heatmap_pivot,
                        cmap="RdBu_r",
                        center=0,
                        annot=False,
                        ax=ax,
                        cbar_kws={
                            "label": f"{metric_mapping.get(metric, metric)} Difference (Shunting - Non-Shunting)"
                        },
                    )
                    ax.invert_yaxis()
                    ax.set_xticklabels(
                        [f"{float(x.get_text()):.1f}" for x in ax.get_xticklabels()]
                    )
                    ax.set_xlabel("E/I Ratio", fontsize=14)
                    ax.set_ylabel("Noise Level", fontsize=14)
                    ax.set_title(
                        f"{noise_mapping.get(noise_type, noise_type)} Noise Perturbations vs E/I Ratio\n"
                        f"{metric_mapping.get(metric, metric)} Difference",
                        fontsize=14,
                    )
                    ax.tick_params(axis="both", labelsize=12)

                    fig.tight_layout()
                    plot_path = os.path.join(
                        noise_ei_plot_dir, f"{noise_type}_{metric}_noise_heatmap.png"
                    )
                    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating noise plot for {noise_type}_{metric}: {e}")

        print(f"Created {len(plot_paths)} noise analysis plots")
        return plot_paths

    def _generate_heatmaps(
        self, valid_data: pd.DataFrame, plots_dir: str, base_name: str
    ) -> list[str]:
        """Generate performance heatmaps"""
        plot_paths = []

        if "ee_value" not in valid_data.columns or "ie_value" not in valid_data.columns:
            return plot_paths

        heatmap_plot_dir = os.path.join(plots_dir, "performance_heatmaps")
        os.makedirs(heatmap_plot_dir, exist_ok=True)

        # Generate heatmaps for key metrics
        key_metrics = ["test accuracy", "valid accuracy", "train accuracy"]

        for metric in key_metrics:
            mean_col = f"{metric}_mean"

            if mean_col not in valid_data.columns:
                continue

            try:
                # Create pivot table for heatmap
                pivot_data = valid_data.pivot_table(
                    values=mean_col,
                    index="ie_value",
                    columns="ee_value",
                    aggfunc="mean",
                )

                if not pivot_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        pivot_data,
                        cmap="viridis",
                        annot=True,
                        fmt=".3f",
                        cbar_kws={"label": self.metric_mapping.get(metric, metric)},
                        ax=ax,
                    )
                    ax.set_xlabel("EE Synapses per Branch", fontsize=14)
                    ax.set_ylabel("IE Synapses per Branch", fontsize=14)
                    ax.set_title(
                        f"{self.metric_mapping.get(metric, metric)} Heatmap",
                        fontsize=16,
                    )

                    fig.tight_layout()
                    plot_path = os.path.join(
                        heatmap_plot_dir,
                        f"{base_name}_{metric.replace(' ', '_')}_heatmap.png",
                    )
                    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating heatmap for {metric}: {e}")

        print(f"Created {len(plot_paths)} heatmap plots")
        return plot_paths

    def _generate_ablation_plots(
        self, valid_data: pd.DataFrame, plots_dir: str, base_name: str
    ) -> list[str]:
        """Generate ablation analysis plots matching old analysis format"""
        plot_paths = []

        ablation_plot_dir = os.path.join(plots_dir, "ablation_analysis_plots")
        os.makedirs(ablation_plot_dir, exist_ok=True)

        # Find ablation columns (look for both column name patterns)
        ablation_cols = [
            col
            for col in valid_data.columns
            if col.startswith("ablation_")
            or "ablation" in col.lower()
            or "contribution" in col.lower()
        ]

        if not ablation_cols:
            print("No ablation columns found")
            return plot_paths

        # Extract ablation types and metrics from column structure
        ablation_types = []
        metrics = []
        depths = []
        layer_cols = []

        for col in ablation_cols:
            if "_depth" in col and "_mean" in col:
                # Pattern: ablation_type_metric_depthX_mean
                parts = col.replace("_mean", "").split("_")
                if len(parts) >= 4:
                    ablation_type = parts[1]
                    metric = parts[2]
                    depth_part = parts[3]
                    if depth_part.startswith("depth"):
                        try:
                            depth = int(depth_part.replace("depth", ""))
                            if ablation_type not in ablation_types:
                                ablation_types.append(ablation_type)
                            if metric not in metrics:
                                metrics.append(metric)
                            if depth not in depths:
                                depths.append(depth)
                        except ValueError:
                            continue
            elif "_mean" in col:
                # Check if this could be a layer-agnostic ablation column
                layer_cols.append(col)

        depths.sort()

        if not valid_data["network_category"].isin(["shunting", "no_shunting"]).any():
            print("No shunting/no_shunting data for ablation analysis")
            return plot_paths

        # Generate E/I ratio plots by depth (current approach)
        for ablation_type, metric in itertools.product(ablation_types, metrics):
            try:
                plot_paths.extend(
                    self._create_ablation_ei_plots(
                        valid_data,
                        ablation_plot_dir,
                        base_name,
                        ablation_type,
                        metric,
                        depths,
                    )
                )
            except Exception as e:
                print(
                    f"Error generating E/I ablation plot for {ablation_type}_{metric}: {e}"
                )

        # Generate smoothed ablation per layer plots (matching old analysis)
        if layer_cols:
            try:
                plot_paths.extend(
                    self._create_smoothed_ablation_layer_plots(
                        valid_data, ablation_plot_dir, base_name, layer_cols
                    )
                )
            except Exception as e:
                print(f"Error generating smoothed ablation layer plots: {e}")

        print(f"Created {len(plot_paths)} ablation analysis plots")
        return plot_paths

    def _create_ablation_ei_plots(
        self,
        valid_data: pd.DataFrame,
        output_dir: str,
        base_name: str,
        ablation_type: str,
        metric: str,
        depths: list[int],
    ) -> list[str]:
        """Create ablation vs E/I ratio plots by depth"""
        plot_paths = []

        # Find columns for this ablation type and metric
        ablation_cols_filtered = [
            col
            for col in valid_data.columns
            if f"ablation_{ablation_type}_{metric}_" in col and "_mean" in col
        ]

        if not ablation_cols_filtered:
            return plot_paths

        # Create separate plots for each network category
        categories = ["shunting", "no_shunting"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            f"{ablation_type.title()} Ablation: {metric.title()} vs E/I Ratio by Depth",
            fontsize=16,
        )

        for idx, category in enumerate(categories):
            ax = axes[idx]
            category_data = valid_data[valid_data["network_category"] == category]

            if category_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No {category} data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{category.title()}")
                continue

            # Plot lines for each depth
            for depth in depths:
                depth_col = f"ablation_{ablation_type}_{metric}_depth{depth}_mean"
                if depth_col not in category_data.columns:
                    continue

                # Sort by E/I ratio for proper line plotting
                sorted_data = category_data.sort_values("ei_ratio")

                # Filter out non-numeric or missing values
                plot_data = sorted_data[[depth_col, "ei_ratio"]].dropna()
                if plot_data.empty:
                    continue

                try:
                    # Convert to numeric if needed
                    y_vals = pd.to_numeric(
                        plot_data[depth_col], errors="coerce"
                    ).dropna()
                    x_vals = plot_data.loc[y_vals.index, "ei_ratio"]

                    if len(y_vals) > 0:
                        ax.plot(
                            x_vals,
                            y_vals,
                            marker="o",
                            label=f"Depth {depth}",
                            alpha=0.8,
                            linewidth=2,
                        )
                except Exception as e:
                    print(f"Warning: Could not plot depth {depth} for {category}: {e}")
                    continue

            ax.set_xlabel("E/I Ratio", fontsize=12)
            ax.set_ylabel(f"{metric.title()}", fontsize=12)
            ax.set_title(f"{category.title()}", fontsize=14)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Format x-axis
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter)

        fig.tight_layout()
        plot_path = os.path.join(
            output_dir, f"{base_name}_{ablation_type}_{metric}_ablation.png"
        )
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(plot_path)

        return plot_paths

    def _create_smoothed_ablation_layer_plots(
        self,
        valid_data: pd.DataFrame,
        output_dir: str,
        base_name: str,
        ablation_cols: list[str],
    ) -> list[str]:
        """Create smoothed ablation per layer plots (matching old analysis exactly)"""
        plot_paths = []

        # Check if we have layer information
        if "layer" not in valid_data.columns:
            print("No layer information available for smoothed ablation plots")
            return plot_paths

        # Look for depth-specific ablation columns with layer information
        layer_ablation_cols = []
        for col in ablation_cols:
            if "_depth" in col and "_mean" in col:
                layer_ablation_cols.append(col)

        if not layer_ablation_cols:
            print(
                "No layer-specific ablation columns found for smoothed ablation plots"
            )
            return plot_paths

        # Create data structure with layer information
        layer_data = []
        for _, row in valid_data.iterrows():
            for col in layer_ablation_cols:
                if not pd.isna(row[col]):
                    # Extract depth from column name
                    try:
                        depth_match = re.search(r"depth(\d+)", col)
                        if depth_match:
                            layer = int(depth_match.group(1))
                            layer_data.append(
                                {
                                    "layer": layer,
                                    "use_shunting": row["network_category"]
                                    == "shunting",
                                    "ablation_value": row[col],
                                    "ablation_type": col,
                                    "ei_ratio": row.get("ei_ratio", 0),
                                }
                            )
                    except (KeyError, ValueError, TypeError):
                        continue

        if not layer_data:
            print("No valid layer ablation data found")
            return plot_paths

        layer_df = pd.DataFrame(layer_data)

        # Group by ablation type
        ablation_types = layer_df["ablation_type"].unique()

        for ablation_type in ablation_types:
            type_data = layer_df[layer_df["ablation_type"] == ablation_type]
            layers = sorted(type_data["layer"].unique())

            if len(layers) < 3:  # Need at least 3 points for smoothing
                continue

            try:
                fig, ax = plt.subplots(figsize=(12, 6))

                for use_shunting in [True, False]:
                    shunt_data = type_data[type_data["use_shunting"] == use_shunting]

                    if shunt_data.empty:
                        continue

                    # Calculate mean values per layer
                    layer_means = []
                    for layer in layers:
                        layer_subset = shunt_data[shunt_data["layer"] == layer]
                        if not layer_subset.empty:
                            layer_means.append(layer_subset["ablation_value"].mean())
                        else:
                            layer_means.append(np.nan)

                    # Remove NaN values for smoothing
                    valid_indices = ~np.isnan(layer_means)
                    if (
                        np.sum(valid_indices) < 3
                    ):  # Need at least 3 points for smoothing
                        continue

                    valid_layers = np.array(layers)[valid_indices]
                    valid_means = np.array(layer_means)[valid_indices]

                    # Apply Gaussian smoothing (matching old analysis)
                    smoothed_values = gaussian_filter1d(valid_means, sigma=1.0)

                    # Plot style (matching old analysis colors)
                    style = (
                        {"color": "#1f77b4", "marker": "o", "label": "Shunting"}
                        if use_shunting
                        else {
                            "color": "#ff7f0e",
                            "marker": "s",
                            "label": "Non-Shunting",
                        }
                    )

                    # Plot raw data (faded)
                    ax.plot(
                        valid_layers,
                        valid_means,
                        alpha=0.3,
                        color=style["color"],
                        marker=style["marker"],
                        linestyle=":",
                        markersize=4,
                        label=f"{style['label']} (Raw)",
                    )

                    # Plot smoothed data (prominent)
                    ax.plot(
                        valid_layers,
                        smoothed_values,
                        color=style["color"],
                        marker=style["marker"],
                        linewidth=2,
                        markersize=6,
                        label=f"{style['label']} (Smoothed)",
                    )

                # Clean up ablation type name for display
                display_name = (
                    ablation_type.replace("ablation_", "")
                    .replace("_mean", "")
                    .replace("_", " ")
                    .title()
                )

                ax.set_xlabel("Layer Index", fontsize=12)
                ax.set_ylabel(f"{display_name}", fontsize=12)
                ax.set_title(f"Smoothed {display_name} per Layer", fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                safe_name = ablation_type.replace("ablation_", "").replace("_mean", "")
                plot_path = os.path.join(
                    output_dir, f"{base_name}_smoothed_{safe_name}_per_layer.png"
                )
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                plot_paths.append(plot_path)

                print(f"Created smoothed ablation per layer plot for {display_name}")

            except Exception as e:
                print(f"Error creating smoothed ablation plot for {ablation_type}: {e}")

        return plot_paths

    def _generate_information_plots(
        self, valid_data: pd.DataFrame, plots_dir: str, base_name: str
    ) -> list[str]:
        """Generate information theory plots matching old analysis format"""
        plot_paths = []

        if "ei_ratio" not in valid_data.columns:
            return plot_paths

        info_plot_dir = os.path.join(plots_dir, "information_theory_plots")
        os.makedirs(info_plot_dir, exist_ok=True)

        # Collect global and layer-specific MI metrics
        global_mi_metrics = []
        layer_mi_metrics = []

        for col in valid_data.columns:
            if any(
                mi_pattern in col.lower()
                for mi_pattern in ["mi_", "mutual_information"]
            ):
                # Extract the metric name
                if "_mean" in col:
                    metric_name = col.replace("_mean", "")
                    if "layer_mi_" in metric_name:
                        # This is a layer-specific metric
                        base_metric = metric_name.replace("layer_mi_", "").split(
                            "_depth"
                        )[0]
                        if base_metric not in layer_mi_metrics:
                            layer_mi_metrics.append(base_metric)
                    else:
                        # This is a global metric
                        if metric_name not in global_mi_metrics:
                            global_mi_metrics.append(metric_name)

        if not global_mi_metrics and not layer_mi_metrics:
            print("No mutual information metrics found")
            return plot_paths

        if not valid_data["network_category"].isin(["shunting", "no_shunting"]).any():
            print("No shunting/no_shunting data for information analysis")
            return plot_paths

        # Define metric groups matching old analysis exactly
        metric_groups = {
            "class_information": {
                "metrics": ["mi_E_C", "mi_I_C", "mi_V_C"],
                "labels": ["I(E;C)", "I(I;C)", "I(V;C)"],
                "colors": [
                    self.info_colors["excitatory"],
                    self.info_colors["inhibitory"],
                    self.info_colors["combined"],
                ],
                "title": "Information with Class Labels",
                "ylabel": "Mutual Information (bits)",
            },
            "component_information": {
                "metrics": ["mi_E_I", "mi_E_V", "mi_I_V"],
                "labels": ["I(E;I)", "I(E;V)", "I(I;V)"],
                "colors": ["purple", "orange", "brown"],
                "title": "Component Information",
                "ylabel": "Mutual Information (bits)",
            },
            "conditional_information": {
                "metrics": ["mi_E_I_given_C", "mi_E_V_given_C", "mi_I_V_given_C"],
                "labels": ["I(E;I|C)", "I(E;V|C)", "I(I;V|C)"],
                "colors": ["darkgreen", "forestgreen", "olive"],
                "title": "Conditional Information",
                "ylabel": "Mutual Information (bits)",
            },
        }

        # Generate global information plots grouped by metric type (matching old format)
        for group_name, group_info in metric_groups.items():
            metrics = group_info["metrics"]
            labels = group_info["labels"]
            metric_colors = group_info["colors"]
            group_title = group_info["title"]
            ylabel = group_info["ylabel"]

            # Check which metrics are available
            available_metrics = []
            available_labels = []
            available_colors = []

            for metric, label, color in zip(metrics, labels, metric_colors):
                mean_col = f"{metric}_mean"
                if mean_col in valid_data.columns:
                    available_metrics.append(metric)
                    available_labels.append(label)
                    available_colors.append(color)

            if not available_metrics:
                print(f"No {group_name} metrics found")
                continue

            try:
                # Create separate plots for shunting vs no shunting (matching old analysis)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # First pass: collect all y-values to determine global y-axis range
                all_y_values = []
                plot_data = {}

                for use_shunting in [True, False]:
                    category = "shunting" if use_shunting else "no_shunting"
                    shunt_data = valid_data[valid_data["network_category"] == category]
                    plot_data[use_shunting] = {}

                    if not shunt_data.empty:
                        for metric in available_metrics:
                            mean_col = f"{metric}_mean"
                            # Group by E/I ratio and calculate statistics
                            grouped = shunt_data.groupby("ei_ratio")[mean_col].mean()
                            plot_data[use_shunting][metric] = grouped
                            all_y_values.extend(grouped.values)

                # Calculate global y-axis range
                if all_y_values:
                    global_y_min = min(all_y_values)
                    global_y_max = max(all_y_values)
                    y_margin = (global_y_max - global_y_min) * 0.1
                    final_y_min = global_y_min - y_margin
                    final_y_max = global_y_max + y_margin
                else:
                    final_y_min, final_y_max = 0, 1

                # Second pass: create the actual plots
                for use_shunting, ax, title_suffix in [
                    (True, ax1, "Shunting"),
                    (False, ax2, "No Shunting"),
                ]:
                    if use_shunting not in plot_data or not plot_data[use_shunting]:
                        ax.text(
                            0.5,
                            0.5,
                            f"No data for\\n{title_suffix}",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                        )
                        ax.set_title(f"{group_title}\\n{title_suffix} Networks")
                        ax.set_ylim(final_y_min, final_y_max)
                        continue

                    # Plot each metric
                    for metric, label, color in zip(
                        available_metrics, available_labels, available_colors
                    ):
                        if metric in plot_data[use_shunting]:
                            grouped = plot_data[use_shunting][metric]
                            ax.plot(
                                grouped.index,
                                grouped.values,
                                marker="o",
                                color=color,
                                label=label,
                                linewidth=2,
                                markersize=6,
                            )

                    ax.set_xlabel("E/I Ratio", fontsize=12)
                    ax.set_ylabel(ylabel, fontsize=12)
                    ax.set_title(
                        f"{group_title}\\n{title_suffix} Networks", fontsize=14
                    )
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.set_xscale("log")
                    ax.set_ylim(final_y_min, final_y_max)

                plt.suptitle(f"Branch Information Analysis: {group_title}", fontsize=16)
                plt.tight_layout()
                plot_path = os.path.join(
                    info_plot_dir, f"{base_name}_{group_name}_branch_info.png"
                )
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating {group_name} plot: {e}")

        # Generate layer-specific information plots and heatmaps
        layer_plot_dir = os.path.join(plots_dir, "layer_information_plots")
        os.makedirs(layer_plot_dir, exist_ok=True)

        # Create I(V;C) heatmap by layer (matching old analysis)
        plot_paths.extend(
            self._create_ivc_heatmap_by_layer(valid_data, layer_plot_dir, base_name)
        )

        # Create first vs last layer comparison plots
        plot_paths.extend(
            self._create_first_last_layer_plots(valid_data, layer_plot_dir, base_name)
        )

        print(
            f"Created {len(plot_paths)} information theory plots (global + layer-specific + heatmaps)"
        )
        return plot_paths

    def _create_ivc_heatmap_by_layer(
        self, valid_data: pd.DataFrame, output_dir: str, base_name: str
    ) -> list[str]:
        """Create I(V;C) heatmap by layer and E/I ratio (matching old analysis)"""
        plot_paths = []

        try:
            # Look for I(V;C) layer data (handle double _mean suffix from aggregation)
            ivc_cols = [
                col
                for col in valid_data.columns
                if "layer_mi_V_C_depth" in col and col.endswith("_mean_mean")
            ]

            if not ivc_cols:
                print("No layer I(V;C) data found for heatmap")
                return plot_paths

            # Extract layer information
            layer_data = []
            for col in ivc_cols:
                try:
                    depth = int(
                        col.split("_depth")[1]
                        .replace("_mean_mean", "")
                        .replace("_mean", "")
                    )
                    for _, row in valid_data.iterrows():
                        if not pd.isna(row[col]):
                            layer_data.append(
                                {
                                    "layer": depth,
                                    "ei_ratio": row["ei_ratio"],
                                    "use_shunting": row["network_category"]
                                    == "shunting",
                                    "ivc_value": row[col],
                                }
                            )
                except (KeyError, ValueError, TypeError):
                    continue

            if not layer_data:
                return plot_paths

            layer_df = pd.DataFrame(layer_data)

            # Create separate heatmaps for shunting and non-shunting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Calculate global min/max for consistent color scaling
            all_values = layer_df["ivc_value"].values
            vmin = np.nanmin(all_values) if len(all_values) > 0 else 0
            vmax = np.nanmax(all_values) if len(all_values) > 0 else 1

            for use_shunting, ax, title_suffix in [
                (True, ax1, "Shunting"),
                (False, ax2, "Non-Shunting"),
            ]:
                shunt_data = layer_df[layer_df["use_shunting"] == use_shunting]

                if shunt_data.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for\\n{title_suffix}",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    ax.set_title(f"I(V;C) Heatmap - {title_suffix}")
                    continue

                try:
                    # Create pivot table for heatmap
                    pivot = shunt_data.pivot_table(
                        values="ivc_value",
                        index="layer",
                        columns="ei_ratio",
                        aggfunc="mean",
                    )

                    if not pivot.empty:
                        # Create heatmap with consistent color scale
                        sns.heatmap(
                            pivot,
                            cmap="viridis",
                            annot=True,
                            fmt=".3f",
                            cbar_kws={"label": "I(V;C) (bits)"},
                            ax=ax,
                            vmin=vmin,
                            vmax=vmax,
                        )

                        ax.set_xlabel("E/I Ratio", fontsize=12)
                        ax.set_ylabel("Layer Index", fontsize=12)
                        ax.set_title(f"I(V;C) Heatmap - {title_suffix}", fontsize=14)

                        # Improve x-axis labels
                        x_labels = [f"{float(x):.1f}" for x in pivot.columns]
                        ax.set_xticklabels(x_labels, rotation=45)

                except Exception as e:
                    print(f"Error creating I(V;C) heatmap for {title_suffix}: {e}")
                    ax.text(
                        0.5,
                        0.5,
                        f"Error creating\\n{title_suffix} heatmap",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    ax.set_title(f"I(V;C) Heatmap - {title_suffix}")

            plt.suptitle("I(V;C) Information by Layer and E/I Ratio", fontsize=16)
            plt.tight_layout()
            plot_path = os.path.join(
                output_dir, f"{base_name}_ivc_heatmap_by_layer.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            plot_paths.append(plot_path)

            print("Created I(V;C) heatmap by layer")

        except Exception as e:
            print(f"Error creating I(V;C) heatmap: {e}")

        return plot_paths

    def _create_first_last_layer_plots(
        self, valid_data: pd.DataFrame, output_dir: str, base_name: str
    ) -> list[str]:
        """Create first vs last layer comparison plots (matching old analysis)"""
        plot_paths = []

        try:
            # Find all layer-specific MI columns (handle double _mean suffix from aggregation)
            layer_cols = [
                col
                for col in valid_data.columns
                if "layer_mi_" in col and "_depth" in col and col.endswith("_mean_mean")
            ]

            if not layer_cols:
                print("No layer-specific MI data found for first/last layer analysis")
                return plot_paths

            # Extract available depths
            depths = set()
            for col in layer_cols:
                try:
                    depth = int(
                        col.split("_depth")[1]
                        .replace("_mean_mean", "")
                        .replace("_mean", "")
                    )
                    depths.add(depth)
                except (ValueError, IndexError):
                    continue

            depths = sorted(depths)
            if len(depths) < 2:
                print(f"Not enough layers ({len(depths)}) for first/last comparison")
                return plot_paths

            first_layer = depths[0]
            last_layer = depths[-1]

            print(
                f"Creating first ({first_layer}) vs last ({last_layer}) layer comparison plots"
            )

            # Define metric groups for first/last layer analysis
            metric_groups = {
                "class_information": {
                    "metrics": ["E_C", "I_C", "V_C"],
                    "labels": ["I(E;C)", "I(I;C)", "I(V;C)"],
                    "colors": [
                        self.info_colors["excitatory"],
                        self.info_colors["inhibitory"],
                        self.info_colors["combined"],
                    ],
                    "title": "Information with Class Labels",
                    "ylabel": "Mutual Information (bits)",
                },
                "component_information": {
                    "metrics": ["E_I", "E_V", "I_V"],
                    "labels": ["I(E;I)", "I(E;V)", "I(I;V)"],
                    "colors": [
                        self.info_colors["conditional"],
                        self.info_colors["cross"],
                        "purple",
                    ],
                    "title": "Component Information",
                    "ylabel": "Mutual Information (bits)",
                },
                "conditional_information": {
                    "metrics": ["E_I_given_C", "E_V_given_C", "I_V_given_C"],
                    "labels": ["I(E;I|C)", "I(E;V|C)", "I(I;V|C)"],
                    "colors": ["darkgreen", "forestgreen", "olive"],
                    "title": "Conditional Information",
                    "ylabel": "Mutual Information (bits)",
                },
            }

            # Create plots for each metric group
            for group_name, group_info in metric_groups.items():
                metrics = group_info["metrics"]
                labels = group_info["labels"]
                metric_colors = group_info["colors"]
                group_title = group_info["title"]
                ylabel = group_info["ylabel"]

                # Check which metrics are available
                available_metrics = []
                available_labels = []
                available_colors = []

                for metric, label, color in zip(metrics, labels, metric_colors):
                    first_col = f"layer_mi_{metric}_depth{first_layer}_mean_mean"
                    last_col = f"layer_mi_{metric}_depth{last_layer}_mean_mean"

                    if (
                        first_col in valid_data.columns
                        and last_col in valid_data.columns
                    ):
                        available_metrics.append(metric)
                        available_labels.append(label)
                        available_colors.append(color)

                if not available_metrics:
                    print(
                        f"No {group_name} metrics found for first/last layer comparison"
                    )
                    continue

                # Create the comparison plot: first layer vs last layer
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                for layer_idx, ax, title_suffix in [
                    (first_layer, ax1, f"First Layer (Layer {first_layer})"),
                    (last_layer, ax2, f"Last Layer (Layer {last_layer})"),
                ]:

                    # Track y-axis range for this subplot
                    y_values = []

                    # Plot each metric for both shunting conditions
                    for metric, label, color in zip(
                        available_metrics, available_labels, available_colors
                    ):
                        layer_col = f"layer_mi_{metric}_depth{layer_idx}_mean_mean"

                        for use_shunting in [True, False]:
                            category = "shunting" if use_shunting else "no_shunting"
                            shunt_data = valid_data[
                                valid_data["network_category"] == category
                            ]

                            if shunt_data.empty:
                                continue

                            # Group by E/I ratio and calculate mean
                            grouped = shunt_data.groupby("ei_ratio")[layer_col].mean()

                            if grouped.empty:
                                continue

                            # Create line style based on shunting
                            linestyle = "-" if use_shunting else "--"
                            alpha = 0.8
                            linewidth = 2 if use_shunting else 1.5

                            # Create label that includes both metric and shunting info
                            plot_label = (
                                f"{label} ({'Shunt' if use_shunting else 'No Shunt'})"
                            )

                            ax.plot(
                                grouped.index,
                                grouped.values,
                                color=color,
                                linestyle=linestyle,
                                label=plot_label,
                                linewidth=linewidth,
                                marker="o" if use_shunting else "s",
                                markersize=6,
                                alpha=alpha,
                            )

                            y_values.extend(grouped.values)

                    ax.set_xlabel("E/I Ratio", fontsize=12)
                    ax.set_ylabel(ylabel, fontsize=12)
                    ax.set_title(title_suffix, fontsize=14)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.set_xscale("log")

                    # Set y-axis range
                    if y_values:
                        y_min, y_max = min(y_values), max(y_values)
                        y_margin = (y_max - y_min) * 0.1
                        ax.set_ylim(y_min - y_margin, y_max + y_margin)

                plt.suptitle(
                    f"{group_title} - First vs Last Layer Comparison", fontsize=16
                )
                plt.tight_layout()
                plot_path = os.path.join(
                    output_dir, f"{base_name}_{group_name}_first_last_layer.png"
                )
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                plot_paths.append(plot_path)

                print(f"Created {group_name} first/last layer comparison plot")

        except Exception as e:
            print(f"Error creating first/last layer plots: {e}")

        return plot_paths

    def analyze_multiple_sweeps(
        self, results_dirs: list[str], sweep_type: Optional[str] = None
    ) -> list[dict]:
        """
        Analyze multiple sweep directories.

        Args:
            results_dirs: List of directories containing sweep results
            sweep_type: Type of sweep (auto-detected if None)

        Returns:
            List of analysis results dictionaries
        """
        analysis_results = []

        for results_dir in results_dirs:
            if not os.path.exists(results_dir):
                print(f"Warning: Directory does not exist: {results_dir}")
                continue

            try:
                print(f"\nAnalyzing: {results_dir}")
                analysis_info = self.analyze_single_sweep(results_dir, sweep_type)
                analysis_results.append(analysis_info)
                print(f"Analysis complete for {results_dir}")

            except Exception as e:
                print(f"Error analyzing {results_dir}: {e}")
                continue

        return analysis_results

    def generate_comparison_report(
        self, analysis_results: list[dict], output_file: str
    ):
        """
        Generate a comparison report across multiple sweep analyses.

        Args:
            analysis_results: List of analysis results from multiple sweeps
            output_file: Path to save the comparison report
        """
        with open(output_file, "w") as f:
            f.write("Unified Sweep Analysis Comparison Report\n")
            f.write("=" * 50 + "\n\n")

            for i, analysis in enumerate(analysis_results):
                f.write(f"Analysis {i+1}: {analysis['sweep_type']} Sweep\n")
                f.write("-" * 30 + "\n")
                f.write(f"Results Directory: {analysis['results_dir']}\n")
                f.write(f"Output Directory: {analysis['output_dir']}\n")
                f.write(f"Number of Plots: {len(analysis['plot_paths'])}\n")
                f.write(f"Report Path: {analysis['report_path']}\n\n")

                # Add best configurations if available
                if "results" in analysis and "configs" in analysis["results"]:
                    f.write(
                        f"Number of Configurations: {len(analysis['results']['configs'])}\n"
                    )

                f.write("\n")

        print(f"Comparison report saved to: {output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Result Analyzer for Dendritic Modeling Sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific sweep type
  python result_analyzer.py --sweep-type ei --results-dir output/ei_20240101120000

  # Auto-detect sweep type
  python result_analyzer.py --results-dir output/ei_20240101120000 --auto-detect

  # Analyze multiple directories
  python result_analyzer.py --results-dirs output/ei_* --sweep-type ei

  # Generate comparison report
  python result_analyzer.py --results-dirs output/*_20240101* --auto-detect --comparison-report comparison.txt
""",
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results-dir", help="Single directory containing sweep results to analyze"
    )
    input_group.add_argument(
        "--results-dirs",
        nargs="+",
        help="Multiple directories containing sweep results to analyze",
    )

    # Sweep type options
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument(
        "--sweep-type",
        choices=SUPPORTED_SWEEP_TYPES,
        help="Type of sweep (required if not using --auto-detect)",
    )
    type_group.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect sweep type from directory contents",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        help="Output directory for analysis (defaults to results directory)",
    )
    parser.add_argument(
        "--comparison-report",
        help="Generate comparison report and save to this file (for multiple directories)",
    )

    return parser.parse_args()


def expand_glob_patterns(patterns: list[str]) -> list[str]:
    """Expand glob patterns in directory list."""
    expanded = []
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            matches = glob.glob(pattern)
            expanded.extend(matches)
        else:
            expanded.append(pattern)

    # Remove duplicates and sort
    return sorted(set(expanded))


def main():
    """Main entry point."""
    args = parse_args()

    try:
        analyzer = UnifiedResultAnalyzer()

        if args.results_dir:
            # Single directory analysis
            if not args.sweep_type and not args.auto_detect:
                print(
                    "Error: Must specify --sweep-type or --auto-detect for single directory"
                )
                sys.exit(1)

            sweep_type = args.sweep_type if not args.auto_detect else None
            analysis_results = [
                analyzer.analyze_single_sweep(
                    args.results_dir, sweep_type, args.output_dir
                )
            ]

        else:
            # Multiple directories analysis
            if not args.sweep_type and not args.auto_detect:
                print(
                    "Error: Must specify --sweep-type or --auto-detect for multiple directories"
                )
                sys.exit(1)

            # Expand glob patterns
            results_dirs = expand_glob_patterns(args.results_dirs)

            if not results_dirs:
                print("Error: No matching directories found")
                sys.exit(1)

            print(f"Found {len(results_dirs)} directories to analyze:")
            for dir_path in results_dirs:
                print(f"   {dir_path}")

            sweep_type = args.sweep_type if not args.auto_detect else None
            analysis_results = analyzer.analyze_multiple_sweeps(
                results_dirs, sweep_type
            )

        # Generate comparison report if requested
        if args.comparison_report and len(analysis_results) > 1:
            analyzer.generate_comparison_report(
                analysis_results, args.comparison_report
            )

        print(f"\nAnalysis complete! Processed {len(analysis_results)} sweep(s)")

        # Summary
        for analysis in analysis_results:
            print(
                f"{analysis['sweep_type']} sweep: {len(analysis['plot_paths'])} plots generated"
            )

    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

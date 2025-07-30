"""
EI (Excitation-Inhibition) Sweep Implementation
==============================================

Migrated and consolidated from scripts/sweeps/ei_sweeps/
Combines functionality from:
- generate_ei_sweep.py
- analyze_ei_sweep.py
- rerun_missing_configs.py
"""

import itertools
import math
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except AttributeError:
    # Handle seaborn version compatibility issue
    import matplotlib
    if hasattr(matplotlib.cm, "register_cmap"):
        import seaborn as sns
    else:
        # For newer matplotlib versions, patch the issue
        matplotlib.cm.register_cmap = matplotlib.colormaps.register
        import seaborn as sns

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Use absolute imports to avoid relative import issues
try:
    from dendritic_modeling.scripts.sweeps.unified.base_sweep import BaseSweepGenerator, BaseSweepAnalyzer
except ImportError:
    # Fallback for when run as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from base_sweep import BaseSweepGenerator, BaseSweepAnalyzer


class EISweepGenerator(BaseSweepGenerator):
    """
    Generator for EI parameter sweeps.
    
    Supports filtering modes:
    - all: Generate all E-I combinations (default)
    - ei-sum: Only combinations where E+I equals constant
    - e-const: Only combinations with constant E value
    - i-const: Only combinations with constant I value
    - ei-ratio: Only combinations where E/I approximates constant ratio
    """
    
    FILTER_MODES = ["all", "ei-sum", "e-const", "i-const", "ei-ratio"]
    
    def __init__(self):
        super().__init__("ei")
    
    def validate_config(self, config: DictConfig) -> bool:
        """Validate EI sweep configuration."""
        required_sections = ["sweep_config", "base_config", "slurm_config"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Check for EI parameters
        sweep_config = config.sweep_config
        ei_params = [
            "model.core_network.parameters.ee_synapses_per_branch_per_layer",
            "model.core_network.parameters.ie_synapses_per_branch_per_layer"
        ]
        
        for param in ei_params:
            if param not in sweep_config:
                raise ValueError(f"Missing EI parameter: {param}")
        
        # Validate filter mode if specified
        filter_config = config.get("filter_config", {})
        filter_mode = filter_config.get("mode", "all")
        if filter_mode not in self.FILTER_MODES:
            raise ValueError(f"Invalid filter mode: {filter_mode}. "
                           f"Available modes: {self.FILTER_MODES}")
        
        return True
    
    def generate_configs(self, base_config: DictConfig) -> List[DictConfig]:
        """Generate EI sweep configurations."""
        sweep_params = base_config.sweep_config
        base_train_config = base_config.base_config
        experiment_settings = base_config.get("experiment_settings", {})
        filter_config = base_config.get("filter_config", {})
        
        # Extract EI parameters
        ee_values = sweep_params["model.core_network.parameters.ee_synapses_per_branch_per_layer"]
        ie_values = sweep_params["model.core_network.parameters.ie_synapses_per_branch_per_layer"]
        
        # Handle use_shunting parameter (can include "MLP" for network comparison)
        use_shunting_values = sweep_params.get(
            "model.core_network.parameters.use_shunting", [True]
        )
        
        # Generate all parameter combinations
        all_combinations = list(itertools.product(ee_values, ie_values, use_shunting_values))
        
        # Apply filtering
        filtered_combinations = self._apply_filter(all_combinations, filter_config)
        
        # Generate seeds
        seeds_per_condition = experiment_settings.get("seeds_per_condition", 1)
        base_seed = experiment_settings.get("base_seed", 42)
        seeds = [base_seed + i for i in range(seeds_per_condition)]
        
        # Generate configurations
        configs = []
        config_id = 0
        
        for ee, ie, use_shunting in filtered_combinations:
            for seed in seeds:
                config = deepcopy(base_train_config)
                
                # Set EI parameters
                if "model" not in config:
                    config["model"] = {}
                if "core_network" not in config["model"]:
                    config["model"]["core_network"] = {}
                if "parameters" not in config["model"]["core_network"]:
                    config["model"]["core_network"]["parameters"] = {}
                
                config["model"]["core_network"]["parameters"]["ee_synapses_per_branch_per_layer"] = [ee]
                config["model"]["core_network"]["parameters"]["ie_synapses_per_branch_per_layer"] = [ie]
                
                # Handle network type
                if use_shunting == "MLP":
                    # Use MLP network for comparison
                    config["model"]["core_network"]["type"] = "MLP"
                    # Remove EI-specific parameters for MLP
                    if "use_shunting" in config["model"]["core_network"]["parameters"]:
                        del config["model"]["core_network"]["parameters"]["use_shunting"]
                else:
                    config["model"]["core_network"]["parameters"]["use_shunting"] = use_shunting
                
                # Set seed
                if "train" not in config:
                    config["train"] = {}
                config["train"]["seed"] = seed
                
                # Ensure task configuration exists with reasonable defaults
                if "task" not in config:
                    config["task"] = {
                        "dataset": "cifar10",  # Default to well-supported dataset
                        "data_path": "",
                        "parameters": {
                            "flatten": True,
                            "normalize": False
                        }
                    }
                
                # Add metadata
                config["_sweep_metadata"] = {
                    "sweep_type": self.sweep_type,
                    "config_id": str(config_id),
                    "ee_synapses": ee,
                    "ie_synapses": ie,
                    "use_shunting": use_shunting,
                    "seed": seed,
                    "ei_ratio": ee / ie if ie > 0 else float('inf'),
                    "ei_sum": ee + ie
                }
                
                configs.append(OmegaConf.create(config))
                config_id += 1
        
        return configs
    
    def _apply_filter(self, combinations: List[Tuple], filter_config: Dict) -> List[Tuple]:
        """Apply filtering to EI parameter combinations."""
        filter_mode = filter_config.get("mode", "all")
        filter_value = filter_config.get("value")
        filter_tolerance = filter_config.get("tolerance", 0.1)
        
        if filter_mode == "all":
            return combinations
        
        filtered = []
        
        for ee, ie, use_shunting in combinations:
            if filter_mode == "ei-sum":
                if filter_value is not None and (ee + ie) == filter_value:
                    filtered.append((ee, ie, use_shunting))
            
            elif filter_mode == "e-const":
                if filter_value is not None and ee == filter_value:
                    filtered.append((ee, ie, use_shunting))
            
            elif filter_mode == "i-const":
                if filter_value is not None and ie == filter_value:
                    filtered.append((ee, ie, use_shunting))
            
            elif filter_mode == "ei-ratio":
                if filter_value is not None and ie > 0:
                    actual_ratio = ee / ie
                    if abs(actual_ratio - filter_value) <= filter_tolerance:
                        filtered.append((ee, ie, use_shunting))
        
        return filtered


class EISweepAnalyzer(BaseSweepAnalyzer):
    """
    Analyzer for EI parameter sweep results.
    
    Migrated from analyze_ei_sweep.py with enhanced functionality.
    """
    
    def __init__(self):
        super().__init__("ei")
    
    def collect_results(self, results_dir: str) -> Dict[str, Any]:
        """Collect and aggregate EI sweep results."""
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
        results_df, missing_df = self._collect_performance_results(results_dir, params_df)
        
        if results_df.empty:
            print("Warning: No performance results found")
            return {"configs": [], "missing": missing_df}
        
        # Clean and process results
        clean_results = self._clean_results_data(results_df)
        
        # Group by network type for analysis
        network_groups = self._group_by_network_type(clean_results)
        
        return {
            "configs": params_df,
            "results": clean_results,
            "network_groups": network_groups,
            "missing": missing_df,
            "metadata": metadata
        }
    
    def generate_plots(self, results: Dict[str, Any], output_dir: str) -> List[str]:
        """Generate comprehensive EI sweep analysis plots."""
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        plot_paths = []
        
        clean_results = results["results"]
        network_groups = results["network_groups"]
        
        if clean_results.empty:
            print("Warning: No results to plot")
            return plot_paths
        
        # 1. Overall heatmaps
        plot_paths.extend(self._generate_heatmaps(clean_results, plot_dir))
        
        # 2. Network comparison heatmaps
        plot_paths.extend(self._generate_network_comparison_heatmaps(network_groups, plot_dir))
        
        # 3. Line plots for parameter exploration
        plot_paths.extend(self._generate_line_plots(network_groups, plot_dir))
        
        # 4. EI ratio analysis
        plot_paths.extend(self._generate_ei_ratio_plots(network_groups, plot_dir))
        
        # 5. Performance distribution plots
        plot_paths.extend(self._generate_distribution_plots(network_groups, plot_dir))
        
        return plot_paths
    
    def _collect_params_data(self, config_dir: str) -> pd.DataFrame:
        """Collect parameter data from config files."""
        config_files = glob.glob(os.path.join(config_dir, "ei_config_*.yaml"))
        
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
                    "ee_synapses": metadata.get("ee_synapses", 
                        config.get("model", {}).get("network", {}).get("parameters", {}).get("ee_synapses_per_branch_per_layer")),
                    "ie_synapses": metadata.get("ie_synapses",
                        config.get("model", {}).get("network", {}).get("parameters", {}).get("ie_synapses_per_branch_per_layer")),
                    "use_shunting": metadata.get("use_shunting",
                        config.get("model", {}).get("network", {}).get("parameters", {}).get("use_shunting")),
                    "seed": metadata.get("seed", config.get("train", {}).get("seed")),
                    "ei_ratio": metadata.get("ei_ratio"),
                    "ei_sum": metadata.get("ei_sum"),
                    "network_type": config.get("model", {}).get("network", {}).get("type", "EINet")
                }
                
                params_data.append(params)
                
            except Exception as e:
                print(f"Warning: Could not load config {config_file}: {e}")
        
        return pd.DataFrame(params_data)
    
    def _collect_performance_results(self, results_dir: str, params_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                    with open(performance_file, 'r') as f:
                        metrics = json.load(f)
                    
                    result_row = row.to_dict()
                    result_row.update(metrics)
                    results_data.append(result_row)
                    
                except Exception as e:
                    print(f"Warning: Could not load metrics for config {config_id}: {e}")
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
        numeric_cols = ["ee_synapses", "ie_synapses", "seed", "ei_ratio", "ei_sum"]
        for col in numeric_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        
        # Handle performance metrics
        performance_cols = ["train_acc", "valid_acc", "test_acc", "train_loss", "valid_loss", "test_loss"]
        for col in performance_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        
        # Add derived columns
        if "ei_ratio" not in results_df.columns and "ee_synapses" in results_df.columns and "ie_synapses" in results_df.columns:
            results_df["ei_ratio"] = results_df["ee_synapses"] / results_df["ie_synapses"].replace(0, np.nan)
        
        if "ei_sum" not in results_df.columns and "ee_synapses" in results_df.columns and "ie_synapses" in results_df.columns:
            results_df["ei_sum"] = results_df["ee_synapses"] + results_df["ie_synapses"]
        
        return results_df
    
    def _group_by_network_type(self, results_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group results by network type for comparison analysis."""
        if results_df.empty:
            return {}
        
        groups = {}
        
        # Determine network categories
        if "network_type" in results_df.columns:
            # Use explicit network type
            for net_type in results_df["network_type"].unique():
                groups[net_type] = results_df[results_df["network_type"] == net_type]
        else:
            # Infer from use_shunting parameter
            if "use_shunting" in results_df.columns:
                shunting_groups = results_df.groupby("use_shunting")
                for shunting_val, group_df in shunting_groups:
                    if shunting_val == "MLP":
                        groups["MLP"] = group_df
                    elif shunting_val is True:
                        groups["EINet_Shunting"] = group_df
                    elif shunting_val is False:
                        groups["EINet_NoShunting"] = group_df
                    else:
                        groups[f"Unknown_{shunting_val}"] = group_df
            else:
                # Default: treat all as EINet
                groups["EINet"] = results_df
        
        return groups
    
    def _generate_heatmaps(self, results_df: pd.DataFrame, plot_dir: str) -> List[str]:
        """Generate overall performance heatmaps."""
        plot_paths = []
        
        if "test_acc" not in results_df.columns:
            return plot_paths
        
        # Group by EI parameters and average across seeds
        grouped = results_df.groupby(["ee_synapses", "ie_synapses"]).agg({
            "test_acc": ["mean", "std"],
            "train_acc": "mean",
            "valid_acc": "mean"
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
        
        # Create pivot table for heatmap
        heatmap_data = grouped.pivot(index="ie_synapses", columns="ee_synapses", values="test_acc_mean")
        
        # Generate heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', 
                   cbar_kws={'label': 'Test Accuracy'})
        plt.title('EI Sweep: Test Accuracy Heatmap')
        plt.xlabel('EE Synapses per Branch per Layer')
        plt.ylabel('IE Synapses per Branch per Layer')
        
        heatmap_path = os.path.join(plot_dir, "ei_sweep_test_accuracy_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(heatmap_path)
        
        return plot_paths
    
    def _generate_network_comparison_heatmaps(self, network_groups: Dict[str, pd.DataFrame], plot_dir: str) -> List[str]:
        """Generate heatmaps comparing different network types."""
        plot_paths = []
        
        if len(network_groups) <= 1:
            return plot_paths
        
        # Create comparison directory
        comparison_dir = os.path.join(plot_dir, "network_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        for net_type, group_df in network_groups.items():
            if group_df.empty or "test_acc" not in group_df.columns:
                continue
            
            # Group by EI parameters and average across seeds
            grouped = group_df.groupby(["ee_synapses", "ie_synapses"])["test_acc"].mean().reset_index()
            heatmap_data = grouped.pivot(index="ie_synapses", columns="ee_synapses", values="test_acc")
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                       cbar_kws={'label': 'Test Accuracy'})
            plt.title(f'{net_type}: Test Accuracy Heatmap')
            plt.xlabel('EE Synapses per Branch per Layer')
            plt.ylabel('IE Synapses per Branch per Layer')
            
            heatmap_path = os.path.join(comparison_dir, f"{net_type}_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(heatmap_path)
        
        return plot_paths
    
    def _generate_line_plots(self, network_groups: Dict[str, pd.DataFrame], plot_dir: str) -> List[str]:
        """Generate line plots for parameter exploration."""
        plot_paths = []
        
        line_plot_dir = os.path.join(plot_dir, "line_plots")
        os.makedirs(line_plot_dir, exist_ok=True)
        
        for net_type, group_df in network_groups.items():
            if group_df.empty or "test_acc" not in group_df.columns:
                continue
            
            # Plot test accuracy vs IE synapses (for different EE values)
            plot_paths.append(self._generate_line_plot_vs_ie(group_df, net_type, line_plot_dir))
            
            # Plot test accuracy vs EE synapses (for different IE values)
            plot_paths.append(self._generate_line_plot_vs_ee(group_df, net_type, line_plot_dir))
        
        return plot_paths
    
    def _generate_line_plot_vs_ie(self, group_df: pd.DataFrame, net_type: str, plot_dir: str) -> str:
        """Generate line plot of performance vs IE synapses."""
        plt.figure(figsize=(10, 6))
        
        # Get representative EE values (up to 5)
        ee_values = sorted(group_df["ee_synapses"].unique())
        max_lines = 5
        step = max(1, len(ee_values) // max_lines)
        selected_ee = ee_values[::step][:max_lines]
        
        for ee in selected_ee:
            ee_data = group_df[group_df["ee_synapses"] == ee]
            if not ee_data.empty:
                # Group by IE and average across seeds
                line_data = ee_data.groupby("ie_synapses")["test_acc"].agg(['mean', 'std']).reset_index()
                
                plt.errorbar(line_data["ie_synapses"], line_data["mean"], yerr=line_data["std"],
                           label=f'EE={ee}', marker='o', capsize=3)
        
        plt.xlabel('IE Synapses per Branch per Layer')
        plt.ylabel('Test Accuracy')
        plt.title(f'{net_type}: Test Accuracy vs IE Synapses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(plot_dir, f"{net_type}_test_acc_vs_ie.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _generate_line_plot_vs_ee(self, group_df: pd.DataFrame, net_type: str, plot_dir: str) -> str:
        """Generate line plot of performance vs EE synapses."""
        plt.figure(figsize=(10, 6))
        
        # Get representative IE values (up to 5)
        ie_values = sorted(group_df["ie_synapses"].unique())
        max_lines = 5
        step = max(1, len(ie_values) // max_lines)
        selected_ie = ie_values[::step][:max_lines]
        
        for ie in selected_ie:
            ie_data = group_df[group_df["ie_synapses"] == ie]
            if not ie_data.empty:
                # Group by EE and average across seeds
                line_data = ie_data.groupby("ee_synapses")["test_acc"].agg(['mean', 'std']).reset_index()
                
                plt.errorbar(line_data["ee_synapses"], line_data["mean"], yerr=line_data["std"],
                           label=f'IE={ie}', marker='o', capsize=3)
        
        plt.xlabel('EE Synapses per Branch per Layer')
        plt.ylabel('Test Accuracy')
        plt.title(f'{net_type}: Test Accuracy vs EE Synapses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(plot_dir, f"{net_type}_test_acc_vs_ee.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _generate_ei_ratio_plots(self, network_groups: Dict[str, pd.DataFrame], plot_dir: str) -> List[str]:
        """Generate EI ratio analysis plots."""
        plot_paths = []
        
        ratio_plot_dir = os.path.join(plot_dir, "ei_ratio_analysis")
        os.makedirs(ratio_plot_dir, exist_ok=True)
        
        for net_type, group_df in network_groups.items():
            if group_df.empty or "test_acc" not in group_df.columns or "ei_ratio" not in group_df.columns:
                continue
            
            # Filter out infinite ratios
            finite_ratios = group_df[np.isfinite(group_df["ei_ratio"])]
            
            if finite_ratios.empty:
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Group by EI ratio and average across seeds
            ratio_data = finite_ratios.groupby("ei_ratio")["test_acc"].agg(['mean', 'std']).reset_index()
            
            plt.errorbar(ratio_data["ei_ratio"], ratio_data["mean"], yerr=ratio_data["std"],
                        marker='o', capsize=3, linewidth=2)
            
            plt.xlabel('E/I Ratio')
            plt.ylabel('Test Accuracy')
            plt.title(f'{net_type}: Test Accuracy vs E/I Ratio')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(ratio_plot_dir, f"{net_type}_test_acc_vs_ei_ratio.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def _generate_distribution_plots(self, network_groups: Dict[str, pd.DataFrame], plot_dir: str) -> List[str]:
        """Generate performance distribution plots."""
        plot_paths = []
        
        if len(network_groups) <= 1:
            return plot_paths
        
        dist_plot_dir = os.path.join(plot_dir, "distributions")
        os.makedirs(dist_plot_dir, exist_ok=True)
        
        # Combined distribution plot
        plt.figure(figsize=(12, 6))
        
        data_for_plot = []
        labels = []
        
        for net_type, group_df in network_groups.items():
            if not group_df.empty and "test_acc" in group_df.columns:
                data_for_plot.append(group_df["test_acc"].dropna())
                labels.append(net_type)
        
        if data_for_plot:
            plt.boxplot(data_for_plot, labels=labels)
            plt.ylabel('Test Accuracy')
            plt.title('Test Accuracy Distribution by Network Type')
            plt.xticks(rotation=45)
            
            plot_path = os.path.join(dist_plot_dir, "test_accuracy_distributions.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def _collect_metadata_from_configs(self, config_dir: str) -> Dict:
        """Fallback method to collect metadata from individual config files."""
        config_files = glob.glob(os.path.join(config_dir, "ei_config_*.yaml"))
        
        return {
            "sweep_type": "ei",
            "num_configs": len(config_files),
            "config_files": [os.path.basename(f) for f in config_files]
        } 
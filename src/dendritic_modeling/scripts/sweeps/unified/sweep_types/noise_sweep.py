"""
Noise Robustness Sweep Implementation
===================================

Migrated and consolidated from scripts/sweeps/experiment_sweeps/
Combines functionality from sweep_noise_experiment.py
"""

import itertools
import os
from typing import Dict, List, Any, Optional
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


class NoiseSweepGenerator(BaseSweepGenerator):
    """
    Generator for noise robustness parameter sweeps.
    
    Specializes in sweeping noise-related parameters like:
    - noise types (gaussian, uniform, poisson, frequency)
    - noise levels/intensities
    - noise application modes (branch-wise, network-wide)
    """
    
    NOISE_TYPES = ["gaussian", "uniform", "poisson", "frequency"]
    
    def __init__(self):
        super().__init__("noise")
    
    def validate_config(self, config: DictConfig) -> bool:
        """Validate noise sweep configuration."""
        required_sections = ["sweep_config", "base_config"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Check for noise-related parameters
        sweep_config = config.sweep_config
        noise_params = [
            "noise.type",
            "noise.level",
            "noise.intensity",
            "noise.application_mode"
        ]
        
        # At least one noise parameter should be present
        has_noise_param = any(param in sweep_config for param in noise_params)
        if not has_noise_param:
            print("Warning: No recognized noise parameters found in sweep_config")
        
        # Validate noise types if specified
        if "noise.type" in sweep_config:
            noise_types = sweep_config["noise.type"]
            if isinstance(noise_types, (list, tuple)):
                for noise_type in noise_types:
                    if noise_type not in self.NOISE_TYPES:
                        raise ValueError(f"Invalid noise type: {noise_type}. "
                                       f"Available types: {self.NOISE_TYPES}")
        
        return True
    
    def generate_configs(self, base_config: DictConfig) -> List[DictConfig]:
        """Generate noise sweep configurations."""
        sweep_params = base_config.sweep_config
        base_train_config = base_config.base_config
        experiment_settings = base_config.get("experiment_settings", {})
        
        # Generate seeds
        seeds_per_condition = experiment_settings.get("seeds_per_condition", 1)
        base_seed = experiment_settings.get("base_seed", 42)
        seeds = [base_seed + i for i in range(seeds_per_condition)]
        
        # Extract noise-specific parameters
        noise_configs = self._extract_noise_configurations(sweep_params)
        
        # Generate configurations
        configs = []
        config_id = 0
        
        for noise_config in noise_configs:
            for seed in seeds:
                config = deepcopy(base_train_config)
                
                # Apply noise configuration
                self._apply_noise_config(config, noise_config)
                
                # Set seed
                if "train" not in config:
                    config["train"] = {}
                config["train"]["seed"] = seed
                
                # Add metadata
                config["_sweep_metadata"] = {
                    "sweep_type": self.sweep_type,
                    "config_id": str(config_id),
                    "seed": seed,
                    "noise_config": noise_config
                }
                
                configs.append(OmegaConf.create(config))
                config_id += 1
        
        return configs
    
    def _extract_noise_configurations(self, sweep_params: DictConfig) -> List[Dict]:
        """Extract and generate noise parameter combinations."""
        noise_configs = []
        
        # Handle different parameter combinations
        param_combinations = {}
        
        # Extract noise types
        if "noise.type" in sweep_params:
            param_combinations["noise_type"] = sweep_params["noise.type"]
        else:
            param_combinations["noise_type"] = ["gaussian"]  # Default
        
        # Extract noise levels
        if "noise.level" in sweep_params:
            param_combinations["noise_level"] = sweep_params["noise.level"]
        elif "noise.intensity" in sweep_params:
            param_combinations["noise_level"] = sweep_params["noise.intensity"]
        else:
            param_combinations["noise_level"] = [0.0, 0.1, 0.2, 0.3]  # Default
        
        # Extract application mode
        if "noise.application_mode" in sweep_params:
            param_combinations["application_mode"] = sweep_params["noise.application_mode"]
        else:
            param_combinations["application_mode"] = ["network_wide"]  # Default
        
        # Generate all combinations
        param_names = list(param_combinations.keys())
        param_values = [param_combinations[name] for name in param_names]
        
        for combination in itertools.product(*param_values):
            noise_config = {}
            for param_name, param_value in zip(param_names, combination):
                noise_config[param_name] = param_value
            noise_configs.append(noise_config)
        
        return noise_configs
    
    def _apply_noise_config(self, config: dict, noise_config: Dict):
        """Apply noise configuration to the training config."""
        if "noise" not in config:
            config["noise"] = {}
        
        # Apply noise configuration
        config["noise"]["type"] = noise_config.get("noise_type", "gaussian")
        config["noise"]["level"] = noise_config.get("noise_level", 0.0)
        config["noise"]["application_mode"] = noise_config.get("application_mode", "network_wide")
        
        # Enable noise in training if level > 0
        if noise_config.get("noise_level", 0.0) > 0:
            config["noise"]["enabled"] = True
        else:
            config["noise"]["enabled"] = False


class NoiseSweepAnalyzer(BaseSweepAnalyzer):
    """
    Analyzer for noise robustness sweep results.
    
    Focuses on analyzing noise robustness and degradation patterns.
    """
    
    def __init__(self):
        super().__init__("noise")
    
    def collect_results(self, results_dir: str) -> Dict[str, Any]:
        """Collect and aggregate noise sweep results."""
        # Find config metadata
        config_dir = os.path.join(results_dir, "configs")
        metadata_path = os.path.join(config_dir, "sweep_metadata.yaml")
        
        if os.path.exists(metadata_path):
            metadata = OmegaConf.load(metadata_path)
        else:
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
        
        # Group by noise type for analysis
        noise_groups = self._group_by_noise_type(clean_results)
        
        return {
            "configs": params_df,
            "results": clean_results,
            "noise_groups": noise_groups,
            "missing": missing_df,
            "metadata": metadata
        }
    
    def generate_plots(self, results: Dict[str, Any], output_dir: str) -> List[str]:
        """Generate comprehensive noise sweep analysis plots."""
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        plot_paths = []
        
        clean_results = results["results"]
        noise_groups = results["noise_groups"]
        
        if clean_results.empty:
            print("Warning: No results to plot")
            return plot_paths
        
        # 1. Noise robustness curves
        plot_paths.extend(self._generate_robustness_curves(noise_groups, plot_dir))
        
        # 2. Noise type comparison
        plot_paths.extend(self._generate_noise_type_comparison(noise_groups, plot_dir))
        
        # 3. Performance degradation analysis
        plot_paths.extend(self._generate_degradation_analysis(clean_results, plot_dir))
        
        # 4. Robustness metrics
        plot_paths.extend(self._generate_robustness_metrics(clean_results, plot_dir))
        
        return plot_paths
    
    def _collect_params_data(self, config_dir: str) -> pd.DataFrame:
        """Collect parameter data from config files."""
        config_files = glob.glob(os.path.join(config_dir, "noise_config_*.yaml"))
        
        params_data = []
        for config_file in config_files:
            try:
                config = OmegaConf.load(config_file)
                
                # Extract metadata
                metadata = config.get("_sweep_metadata", {})
                noise_config = metadata.get("noise_config", {})
                
                # Extract key parameters
                params = {
                    "config_file": os.path.basename(config_file),
                    "config_id": metadata.get("config_id", len(params_data)),
                    "seed": metadata.get("seed", config.get("train", {}).get("seed")),
                    "noise_type": noise_config.get("noise_type"),
                    "noise_level": noise_config.get("noise_level"),
                    "application_mode": noise_config.get("application_mode")
                }
                
                params_data.append(params)
                
            except Exception as e:
                print(f"Warning: Could not load config {config_file}: {e}")
        
        return pd.DataFrame(params_data)
    
    def _collect_performance_results(self, results_dir: str, params_df: pd.DataFrame) -> tuple:
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
        numeric_cols = ["noise_level", "seed"]
        for col in numeric_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        
        # Handle performance metrics
        performance_cols = ["train_acc", "valid_acc", "test_acc", "train_loss", "valid_loss", "test_loss"]
        for col in performance_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        
        return results_df
    
    def _group_by_noise_type(self, results_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group results by noise type for comparison analysis."""
        if results_df.empty or "noise_type" not in results_df.columns:
            return {}
        
        groups = {}
        for noise_type in results_df["noise_type"].unique():
            if pd.notna(noise_type):
                groups[noise_type] = results_df[results_df["noise_type"] == noise_type]
        
        return groups
    
    def _generate_robustness_curves(self, noise_groups: Dict[str, pd.DataFrame], plot_dir: str) -> List[str]:
        """Generate robustness curves for different noise types."""
        plot_paths = []
        
        robustness_dir = os.path.join(plot_dir, "robustness_curves")
        os.makedirs(robustness_dir, exist_ok=True)
        
        for noise_type, group_df in noise_groups.items():
            if group_df.empty or "test_acc" not in group_df.columns or "noise_level" not in group_df.columns:
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Group by noise level and compute statistics
            grouped = group_df.groupby("noise_level")["test_acc"].agg(['mean', 'std']).reset_index()
            
            # Plot robustness curve
            plt.errorbar(grouped["noise_level"], grouped["mean"], yerr=grouped["std"],
                        marker='o', capsize=5, linewidth=2, markersize=8, label=f'{noise_type} noise')
            
            plt.xlabel('Noise Level')
            plt.ylabel('Test Accuracy')
            plt.title(f'Robustness to {noise_type.capitalize()} Noise')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add horizontal line for clean performance (noise_level = 0)
            if 0.0 in grouped["noise_level"].values:
                clean_acc = grouped[grouped["noise_level"] == 0.0]["mean"].iloc[0]
                plt.axhline(y=clean_acc, color='red', linestyle='--', alpha=0.7, label='Clean Performance')
            
            plot_path = os.path.join(robustness_dir, f"{noise_type}_robustness_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def _generate_noise_type_comparison(self, noise_groups: Dict[str, pd.DataFrame], plot_dir: str) -> List[str]:
        """Generate comparison plots between different noise types."""
        plot_paths = []
        
        if len(noise_groups) <= 1:
            return plot_paths
        
        comparison_dir = os.path.join(plot_dir, "noise_type_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Combined robustness curves
        plt.figure(figsize=(12, 8))
        
        for noise_type, group_df in noise_groups.items():
            if not group_df.empty and "test_acc" in group_df.columns and "noise_level" in group_df.columns:
                grouped = group_df.groupby("noise_level")["test_acc"].agg(['mean', 'std']).reset_index()
                
                plt.errorbar(grouped["noise_level"], grouped["mean"], yerr=grouped["std"],
                           marker='o', capsize=3, linewidth=2, label=f'{noise_type.capitalize()} Noise')
        
        plt.xlabel('Noise Level')
        plt.ylabel('Test Accuracy')
        plt.title('Robustness Comparison Across Noise Types')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plot_path = os.path.join(comparison_dir, "noise_type_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        # Performance distribution at high noise
        high_noise_threshold = 0.2  # Define what constitutes "high noise"
        
        data_for_plot = []
        labels = []
        
        for noise_type, group_df in noise_groups.items():
            if not group_df.empty and "test_acc" in group_df.columns and "noise_level" in group_df.columns:
                high_noise_data = group_df[group_df["noise_level"] >= high_noise_threshold]
                if not high_noise_data.empty:
                    data_for_plot.append(high_noise_data["test_acc"].dropna())
                    labels.append(f'{noise_type.capitalize()}')
        
        if len(data_for_plot) > 1:
            plt.figure(figsize=(10, 6))
            plt.boxplot(data_for_plot, labels=labels)
            plt.ylabel('Test Accuracy')
            plt.title(f'Performance Distribution at High Noise (≥{high_noise_threshold})')
            plt.xticks(rotation=45)
            
            plot_path = os.path.join(comparison_dir, "high_noise_performance_distribution.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def _generate_degradation_analysis(self, results_df: pd.DataFrame, plot_dir: str) -> List[str]:
        """Generate performance degradation analysis plots."""
        plot_paths = []
        
        if "test_acc" not in results_df.columns or "noise_level" not in results_df.columns:
            return plot_paths
        
        degradation_dir = os.path.join(plot_dir, "degradation_analysis")
        os.makedirs(degradation_dir, exist_ok=True)
        
        # Calculate performance degradation relative to clean performance
        clean_performance = results_df[results_df["noise_level"] == 0.0]["test_acc"].mean() if 0.0 in results_df["noise_level"].values else None
        
        if clean_performance is not None:
            results_df["performance_degradation"] = (clean_performance - results_df["test_acc"]) / clean_performance * 100
            
            plt.figure(figsize=(10, 6))
            
            # Group by noise level and compute degradation statistics
            grouped = results_df.groupby("noise_level")["performance_degradation"].agg(['mean', 'std']).reset_index()
            
            plt.errorbar(grouped["noise_level"], grouped["mean"], yerr=grouped["std"],
                        marker='s', capsize=5, linewidth=2, markersize=8, color='red')
            
            plt.xlabel('Noise Level')
            plt.ylabel('Performance Degradation (%)')
            plt.title('Performance Degradation vs Noise Level')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(degradation_dir, "performance_degradation.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def _generate_robustness_metrics(self, results_df: pd.DataFrame, plot_dir: str) -> List[str]:
        """Generate robustness metrics and analysis."""
        plot_paths = []
        
        if "test_acc" not in results_df.columns or "noise_level" not in results_df.columns:
            return plot_paths
        
        metrics_dir = os.path.join(plot_dir, "robustness_metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Calculate Area Under the Robustness Curve (AURC)
        if "noise_type" in results_df.columns:
            noise_types = results_df["noise_type"].unique()
            
            aurc_scores = []
            noise_type_labels = []
            
            for noise_type in noise_types:
                if pd.notna(noise_type):
                    noise_data = results_df[results_df["noise_type"] == noise_type]
                    grouped = noise_data.groupby("noise_level")["test_acc"].mean().reset_index()
                    grouped = grouped.sort_values("noise_level")
                    
                    if len(grouped) > 1:
                        # Calculate AURC using trapezoidal rule
                        aurc = np.trapz(grouped["test_acc"], grouped["noise_level"])
                        aurc_scores.append(aurc)
                        noise_type_labels.append(noise_type.capitalize())
            
            if len(aurc_scores) > 1:
                plt.figure(figsize=(10, 6))
                plt.bar(noise_type_labels, aurc_scores, alpha=0.7)
                plt.ylabel('Area Under Robustness Curve')
                plt.title('Robustness Comparison (Higher is Better)')
                plt.xticks(rotation=45)
                
                plot_path = os.path.join(metrics_dir, "aurc_comparison.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths.append(plot_path)
        
        return plot_paths
    
    def _collect_metadata_from_configs(self, config_dir: str) -> Dict:
        """Fallback method to collect metadata from individual config files."""
        config_files = glob.glob(os.path.join(config_dir, "noise_config_*.yaml"))
        
        return {
            "sweep_type": "noise",
            "num_configs": len(config_files),
            "config_files": [os.path.basename(f) for f in config_files]
        } 
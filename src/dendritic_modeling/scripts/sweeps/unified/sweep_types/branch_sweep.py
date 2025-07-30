"""
Branch Parameter Sweep Implementation
====================================

Migrated and consolidated from scripts/sweeps/experiment_sweeps/
Combines functionality from sweep_branch_experiment.py
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


class BranchSweepGenerator(BaseSweepGenerator):
    """
    Generator for branch architecture parameter sweeps.
    
    Specializes in sweeping branch-related parameters like:
    - num_branches_per_layer
    - branch connectivity patterns
    - synaptic distribution schemes
    """
    
    def __init__(self):
        super().__init__("branch")
    
    def validate_config(self, config: DictConfig) -> bool:
        """Validate branch sweep configuration."""
        required_sections = ["sweep_config", "base_config"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Check for branch-related parameters
        sweep_config = config.sweep_config
        branch_params = [
            "model.core_network.parameters.excitatory_layer_sizes",
            "model.core_network.parameters.excitatory_branch_factors",
            "model.core_network.parameters.use_shunting"
        ]
        
        # At least one branch parameter should be present
        has_branch_param = any(param in sweep_config for param in branch_params)
        if not has_branch_param:
            print("Warning: No recognized branch parameters found in sweep_config")
        
        return True
    
    def generate_configs(self, base_config: DictConfig) -> List[DictConfig]:
        """Generate branch sweep configurations."""
        sweep_params = base_config.sweep_config
        base_train_config = base_config.base_config
        experiment_settings = base_config.get("experiment_settings", {})
        
        # Generate seeds
        seeds_per_condition = experiment_settings.get("seeds_per_condition", 1)
        base_seed = experiment_settings.get("base_seed", 42)
        seeds = [base_seed + i for i in range(seeds_per_condition)]
        
        # Extract branch-specific parameters
        branch_configs = self._extract_branch_configurations(sweep_params)
        
        # Generate configurations
        configs = []
        config_id = 0
        
        for branch_config in branch_configs:
            for seed in seeds:
                config = deepcopy(base_train_config)
                
                # Apply branch configuration
                self._apply_branch_config(config, branch_config)
                
                # Set seed
                if "train" not in config:
                    config["train"] = {}
                config["train"]["seed"] = seed
                
                # Add metadata
                config["_sweep_metadata"] = {
                    "sweep_type": self.sweep_type,
                    "config_id": str(config_id),
                    "seed": seed,
                    "branch_config": branch_config
                }
                
                configs.append(OmegaConf.create(config))
                config_id += 1
        
        return configs
    
    def _extract_branch_configurations(self, sweep_params: DictConfig) -> List[Dict]:
        """Extract and generate branch parameter combinations."""
        branch_configs = []
        
        # Handle different parameter combinations
        param_combinations = {}
        
        # Extract branch factors
        if "model.network.parameters.excitatory_branch_factors" in sweep_params:
            param_combinations["branch_factors"] = sweep_params["model.network.parameters.excitatory_branch_factors"]
        
        # Extract number of branches per layer
        if "model.network.parameters.num_branches_per_layer" in sweep_params:
            param_combinations["num_branches"] = sweep_params["model.network.parameters.num_branches_per_layer"]
        
        # Extract number of layers
        if "model.network.parameters.num_layers" in sweep_params:
            param_combinations["num_layers"] = sweep_params["model.network.parameters.num_layers"]
        
        # Extract use_shunting
        if "model.network.parameters.use_shunting" in sweep_params:
            param_combinations["use_shunting"] = sweep_params["model.network.parameters.use_shunting"]
        
        # Generate all combinations
        if param_combinations:
            param_names = list(param_combinations.keys())
            param_values = [param_combinations[name] for name in param_names]
            
            for combination in itertools.product(*param_values):
                branch_config = {}
                for param_name, param_value in zip(param_names, combination):
                    branch_config[param_name] = param_value
                branch_configs.append(branch_config)
        else:
            # Fallback: create single default config
            branch_configs.append({})
        
        return branch_configs
    
    def _apply_branch_config(self, config: dict, branch_config: Dict):
        """Apply branch configuration to the training config."""
        if "model" not in config:
            config["model"] = {}
        if "network" not in config["model"]:
            config["model"]["network"] = {}
        if "parameters" not in config["model"]["network"]:
            config["model"]["network"]["parameters"] = {}
        
        params = config["model"]["network"]["parameters"]
        
        # Apply branch configuration
        if "branch_factors" in branch_config:
            params["excitatory_branch_factors"] = branch_config["branch_factors"]
        
        if "num_branches" in branch_config:
            params["num_branches_per_layer"] = branch_config["num_branches"]
        
        if "num_layers" in branch_config:
            params["num_layers"] = branch_config["num_layers"]
        
        if "use_shunting" in branch_config:
            params["use_shunting"] = branch_config["use_shunting"]


class BranchSweepAnalyzer(BaseSweepAnalyzer):
    """
    Analyzer for branch parameter sweep results.
    
    Focuses on analyzing the effects of branch architecture parameters.
    """
    
    def __init__(self):
        super().__init__("branch")
    
    def collect_results(self, results_dir: str) -> Dict[str, Any]:
        """Collect and aggregate branch sweep results."""
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
        
        return {
            "configs": params_df,
            "results": clean_results,
            "missing": missing_df,
            "metadata": metadata
        }
    
    def generate_plots(self, results: Dict[str, Any], output_dir: str) -> List[str]:
        """Generate comprehensive branch sweep analysis plots."""
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        plot_paths = []
        
        clean_results = results["results"]
        
        if clean_results.empty:
            print("Warning: No results to plot")
            return plot_paths
        
        # 1. Branch parameter effect plots
        plot_paths.extend(self._generate_branch_effect_plots(clean_results, plot_dir))
        
        # 2. Branch architecture comparison
        plot_paths.extend(self._generate_architecture_comparison_plots(clean_results, plot_dir))
        
        # 3. Performance vs complexity analysis
        plot_paths.extend(self._generate_complexity_analysis_plots(clean_results, plot_dir))
        
        return plot_paths
    
    def _collect_params_data(self, config_dir: str) -> pd.DataFrame:
        """Collect parameter data from config files."""
        config_files = glob.glob(os.path.join(config_dir, "branch_config_*.yaml"))
        
        params_data = []
        for config_file in config_files:
            try:
                config = OmegaConf.load(config_file)
                
                # Extract metadata
                metadata = config.get("_sweep_metadata", {})
                branch_config = metadata.get("branch_config", {})
                
                # Extract key parameters
                params = {
                    "config_file": os.path.basename(config_file),
                    "config_id": metadata.get("config_id", len(params_data)),
                    "seed": metadata.get("seed", config.get("train", {}).get("seed")),
                    "branch_factors": branch_config.get("branch_factors"),
                    "num_branches": branch_config.get("num_branches"),
                    "num_layers": branch_config.get("num_layers"),
                    "use_shunting": branch_config.get("use_shunting")
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
        numeric_cols = ["num_branches", "num_layers", "seed"]
        for col in numeric_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        
        # Handle performance metrics
        performance_cols = ["train_acc", "valid_acc", "test_acc", "train_loss", "valid_loss", "test_loss"]
        for col in performance_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        
        return results_df
    
    def _generate_branch_effect_plots(self, results_df: pd.DataFrame, plot_dir: str) -> List[str]:
        """Generate plots showing the effect of branch parameters."""
        plot_paths = []
        
        if "test_acc" not in results_df.columns:
            return plot_paths
        
        branch_effects_dir = os.path.join(plot_dir, "branch_effects")
        os.makedirs(branch_effects_dir, exist_ok=True)
        
        # Effect of number of branches
        if "num_branches" in results_df.columns and results_df["num_branches"].nunique() > 1:
            plt.figure(figsize=(10, 6))
            
            grouped = results_df.groupby("num_branches")["test_acc"].agg(['mean', 'std']).reset_index()
            
            plt.errorbar(grouped["num_branches"], grouped["mean"], yerr=grouped["std"],
                        marker='o', capsize=5, linewidth=2, markersize=8)
            
            plt.xlabel('Number of Branches per Layer')
            plt.ylabel('Test Accuracy')
            plt.title('Effect of Branch Count on Performance')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(branch_effects_dir, "effect_of_branch_count.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # Effect of number of layers
        if "num_layers" in results_df.columns and results_df["num_layers"].nunique() > 1:
            plt.figure(figsize=(10, 6))
            
            grouped = results_df.groupby("num_layers")["test_acc"].agg(['mean', 'std']).reset_index()
            
            plt.errorbar(grouped["num_layers"], grouped["mean"], yerr=grouped["std"],
                        marker='s', capsize=5, linewidth=2, markersize=8)
            
            plt.xlabel('Number of Layers')
            plt.ylabel('Test Accuracy')
            plt.title('Effect of Network Depth on Performance')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(branch_effects_dir, "effect_of_layer_count.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def _generate_architecture_comparison_plots(self, results_df: pd.DataFrame, plot_dir: str) -> List[str]:
        """Generate architecture comparison plots."""
        plot_paths = []
        
        if "use_shunting" not in results_df.columns or "test_acc" not in results_df.columns:
            return plot_paths
        
        comparison_dir = os.path.join(plot_dir, "architecture_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Compare shunting vs non-shunting
        shunting_groups = results_df.groupby("use_shunting")
        
        data_for_plot = []
        labels = []
        
        for shunting_val, group_df in shunting_groups:
            if not group_df.empty and "test_acc" in group_df.columns:
                data_for_plot.append(group_df["test_acc"].dropna())
                label = "With Shunting" if shunting_val else "Without Shunting"
                labels.append(label)
        
        if len(data_for_plot) > 1:
            plt.figure(figsize=(10, 6))
            plt.boxplot(data_for_plot, labels=labels)
            plt.ylabel('Test Accuracy')
            plt.title('Performance Comparison: Shunting vs Non-Shunting')
            
            plot_path = os.path.join(comparison_dir, "shunting_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def _generate_complexity_analysis_plots(self, results_df: pd.DataFrame, plot_dir: str) -> List[str]:
        """Generate complexity analysis plots."""
        plot_paths = []
        
        if "test_acc" not in results_df.columns:
            return plot_paths
        
        complexity_dir = os.path.join(plot_dir, "complexity_analysis")
        os.makedirs(complexity_dir, exist_ok=True)
        
        # Performance vs architectural complexity (branches * layers)
        if "num_branches" in results_df.columns and "num_layers" in results_df.columns:
            results_df["complexity"] = results_df["num_branches"] * results_df["num_layers"]
            
            plt.figure(figsize=(10, 6))
            
            # Scatter plot with trend line
            plt.scatter(results_df["complexity"], results_df["test_acc"], alpha=0.6)
            
            # Add trend line
            z = np.polyfit(results_df["complexity"].dropna(), 
                          results_df["test_acc"].dropna(), 1)
            p = np.poly1d(z)
            plt.plot(results_df["complexity"], p(results_df["complexity"]), "r--", alpha=0.8)
            
            plt.xlabel('Architectural Complexity (Branches × Layers)')
            plt.ylabel('Test Accuracy')
            plt.title('Performance vs Architectural Complexity')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(complexity_dir, "performance_vs_complexity.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def _collect_metadata_from_configs(self, config_dir: str) -> Dict:
        """Fallback method to collect metadata from individual config files."""
        config_files = glob.glob(os.path.join(config_dir, "branch_config_*.yaml"))
        
        return {
            "sweep_type": "branch",
            "num_configs": len(config_files),
            "config_files": [os.path.basename(f) for f in config_files]
        } 
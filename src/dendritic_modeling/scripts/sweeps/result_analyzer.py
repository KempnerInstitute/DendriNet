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
    python result_analyzer.py --sweep-type ei --results-dir output/ei_20240101120000
    
    # Auto-detect sweep type
    python result_analyzer.py --results-dir output/ei_20240101120000 --auto-detect
    
    # Analyze multiple directories
    python result_analyzer.py --results-dirs output/ei_* --sweep-type ei
"""

import argparse
import sys
import os
import glob
import json
import re
import itertools
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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

from omegaconf import OmegaConf

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
            "mi_E_I": "MI(E:I)",
            "mi_Vout_I": "MI(Vout:I)",
            "mi_E_Vout": "MI(E:Vout)",
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
            "shunting": "blue",
            "no_shunting": "red", 
            "unknown": "gray",
            "default": "green",
        }
        self.category_markers = {
            "shunting": "o",
            "no_shunting": "s",
            "unknown": "x", 
            "default": "^",
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
        dir_name = os.path.basename(results_dir.rstrip('/'))
        for sweep_type in self.supported_types:
            if dir_name.startswith(f"{sweep_type}_"):
                return sweep_type
        
        # Check config file patterns - handle both old and new naming
        config_dir = os.path.join(results_dir, "configs")
        if os.path.exists(config_dir):
            # Check for unified sweep configs
            unified_configs = glob.glob(os.path.join(config_dir, "unified_config_*.yaml"))
            if unified_configs:
                return "unified"
            
            # Check for other sweep types
            for sweep_type in self.supported_types:
                config_files = glob.glob(os.path.join(config_dir, f"{sweep_type}_*config_*.yaml"))
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
            "*_config_*.yaml"
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
                task_match = re.search(r"config_(\d+)\.yaml", os.path.basename(config_file))
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
    
    def _extract_config_params(self, config: Any, task_id: int, config_file: str) -> Optional[Dict]:
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
                if hasattr(config.model, 'core_network') and hasattr(config.model.core_network, 'parameters'):
                    net_params = config.model.core_network.parameters
                    
                    # Look for EI-specific parameters
                    if hasattr(net_params, 'ee_synapses_per_branch_per_layer'):
                        ee_values = net_params.ee_synapses_per_branch_per_layer
                        data["ee_value"] = ee_values[0] if isinstance(ee_values, list) and len(ee_values) == 1 else ee_values
                        
                    if hasattr(net_params, 'ie_synapses_per_branch_per_layer'):
                        ie_values = net_params.ie_synapses_per_branch_per_layer  
                        data["ie_value"] = ie_values[0] if isinstance(ie_values, list) and len(ie_values) == 1 else ie_values
                        
                    if hasattr(net_params, 'excitatory_branch_factors'):
                        branch_factors = net_params.excitatory_branch_factors
                        if isinstance(branch_factors, list) and len(branch_factors) == 1:
                            data["branch_factors"] = branch_factors[0]
                        else:
                            data["branch_factors"] = "-".join(str(x) for x in branch_factors)
                            
                    if hasattr(net_params, 'use_shunting'):
                        data["use_shunting"] = net_params.use_shunting
                        
                # Try old structure
                elif hasattr(config.model, 'network') and hasattr(config.model.network, 'parameters'):
                    net_params = config.model.network.parameters
                    
                    if hasattr(net_params, 'ee_synapses_per_branch_per_layer'):
                        ee_values = net_params.ee_synapses_per_branch_per_layer
                        data["ee_value"] = ee_values[0] if isinstance(ee_values, list) and len(ee_values) == 1 else ee_values
                        
                    if hasattr(net_params, 'ie_synapses_per_branch_per_layer'):
                        ie_values = net_params.ie_synapses_per_branch_per_layer
                        data["ie_value"] = ie_values[0] if isinstance(ie_values, list) and len(ie_values) == 1 else ie_values
                        
                    if hasattr(net_params, 'excitatory_branch_factors'):
                        branch_factors = net_params.excitatory_branch_factors  
                        if isinstance(branch_factors, list) and len(branch_factors) == 1:
                            data["branch_factors"] = branch_factors[0]
                        else:
                            data["branch_factors"] = "-".join(str(x) for x in branch_factors)
                            
                    if hasattr(net_params, 'use_shunting'):
                        data["use_shunting"] = net_params.use_shunting

                # Extract seed
                if hasattr(config.train, 'seed'):
                    data["seed"] = config.train.seed

            except Exception as e:
                print(f"Warning: Could not extract EE/IE values from {config_file}: {e}")
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
    
    def _collect_config_results(self, config_path: str, task_id: int, 
                               configs_dir: str, exp_timestamp: str) -> Optional[Dict]:
        """Collect results from a single config directory (flattened structure)"""
        
        # Load config parameters
        config_yaml_patterns = [
            f"unified_config_{task_id}.yaml",
            f"ei_sweep_config_{task_id}.yaml", 
            f"branch_config_{task_id}.yaml",
            f"noise_config_{task_id}.yaml",
            f"general_config_{task_id}.yaml"
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
    
    def _collect_analysis_files(self, config_path: str, data: Dict):
        """Collect analysis results from various files in the config directory"""
        
        # Handle both flattened structure (new) and nested structure (old)
        # First check for flattened structure, then fall back to nested structure
        search_paths = [config_path]  # Flattened structure
        
        # Check for nested structure (experiment_* subdirectories)
        experiment_dirs = [d for d in os.listdir(config_path) 
                          if os.path.isdir(os.path.join(config_path, d)) and d.startswith("experiment_")]
        if experiment_dirs:
            # Use the first experiment directory found (usually only one)
            experiment_path = os.path.join(config_path, experiment_dirs[0])
            search_paths = [experiment_path]  # Nested structure
            pass  # Using nested structure
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
                    if "basic_mi" in content_dict and isinstance(content_dict["basic_mi"], dict):
                        for mi_key, mi_value in content_dict["basic_mi"].items():
                            if isinstance(mi_value, (int, float)):
                                # Convert I(E;C) format to mi_E_C format for consistency
                                normalized_key = mi_key.replace("I(", "mi_").replace(";", "_").replace(")", "").replace("_std", "_std")
                                data[normalized_key] = mi_value
                    
                    # Extract pairwise MI if available
                    if "pairwise_mi" in content_dict and isinstance(content_dict["pairwise_mi"], dict):
                        for mi_key, mi_value in content_dict["pairwise_mi"].items():
                            if isinstance(mi_value, (int, float)):
                                normalized_key = f"pairwise_{mi_key}".replace("I(", "mi_").replace(";", "_").replace(")", "").replace("_std", "_std")
                                data[normalized_key] = mi_value
                    
                    # Extract conditional MI if available  
                    if "conditional_mi" in content_dict and isinstance(content_dict["conditional_mi"], dict):
                        for mi_key, mi_value in content_dict["conditional_mi"].items():
                            if isinstance(mi_value, (int, float)):
                                normalized_key = f"conditional_{mi_key}".replace("I(", "mi_").replace(";", "_").replace(")", "").replace("|", "_given_").replace("_std", "_std")
                                data[normalized_key] = mi_value
                    
                    # Extract layer-specific MI data
                    if "layer_statistics" in content_dict and isinstance(content_dict["layer_statistics"], dict):
                        for layer_name, layer_data in content_dict["layer_statistics"].items():
                            if isinstance(layer_data, dict):
                                # Extract layer depth from layer name (e.g., branch_layers.0 -> depth 0)
                                try:
                                    if "branch_layers." in layer_name:
                                        depth = int(layer_name.split("branch_layers.")[-1])
                                    else:
                                        depth = 0  # Default depth
                                except:
                                    depth = 0
                                
                                # Extract layer-specific MI metrics
                                for mi_key, mi_value in layer_data.items():
                                    if isinstance(mi_value, (int, float)):
                                        # Convert I(E;C) format to mi_E_C format with layer info
                                        normalized_key = mi_key.replace("I(", "layer_mi_").replace(";", "_").replace(")", "").replace(",", "_").replace("|", "_given_").replace("_std", "_std")
                                        data[f"{normalized_key}_depth{depth}"] = mi_value
                    
                    # Store method and other metadata
                    for key in ["method", "n_samples"]:
                        if key in content_dict:
                            data[f"info_{key}"] = content_dict[key]
                            
                except Exception as e:
                    print(f"Error reading information analysis: {e}")

            # Collect noise perturbation results
            noise_file = os.path.join(search_path, "noise_perturbation_analysis", "final")
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
                                                data[f"noise_{noise_type}_{metric}_{noise_level}_sample{i}"] = v
                                        else:
                                            data[f"noise_{noise_type}_{metric}_{noise_level}_sample0"] = value
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
                    if "layer" in content_dict and isinstance(content_dict["layer"], dict):
                        for ablation_type, layer_data in content_dict["layer"].items():
                            if isinstance(layer_data, dict):
                                for layer_name, layer_metrics in layer_data.items():
                                    if isinstance(layer_metrics, dict):
                                        # Extract depth information
                                        depth = layer_metrics.get("depth", 0)
                                        
                                        # Extract metrics (skip non-numeric metadata)
                                        for metric_name, metric_value in layer_metrics.items():
                                            if metric_name not in ["depth", "layer_name"] and isinstance(metric_value, (int, float)):
                                                # Create standardized ablation column name
                                                safe_ablation_type = ablation_type.replace(" ", "_").replace("-", "_")
                                                data[f"ablation_{safe_ablation_type}_{metric_name}_depth{depth}"] = metric_value
                                
                except Exception as e:
                    print(f"Error reading ablation analysis: {e}")
            
            # Break after processing the first valid path (flattened or first experiment dir)
            break
    
    def analyze_single_sweep(self, results_dir: str, sweep_type: Optional[str] = None,
                           output_dir: Optional[str] = None) -> dict:
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
        
        # Step 1: Collect results from the flattened directory structure
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
                "message": "No results found in directory."
            }
        
        # Step 2: Clean and process the results data
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
                "message": "No valid results after data cleaning."
            }

        # Step 3: Aggregate results (mean/std over seeds)
        print("Aggregating results...")
        agg_results = self.aggregate_results(clean_results)

        # Step 4: Generate comprehensive plots
        print("Generating plots...")
        plot_paths = self.generate_plots(agg_results, output_dir, f"{sweep_type}_sweep")

        # Step 5: Save processed data
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save cleaned results
        clean_results.to_csv(
            os.path.join(plots_dir, f"{sweep_type}_sweep_clean_results.csv"), index=False
        )
        
        # Save aggregated results
        agg_results.to_csv(
            os.path.join(plots_dir, f"{sweep_type}_sweep_aggregated_results.csv"), index=False
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
                "missing_configs": len(missing_df)
            },
            "message": f"Comprehensive analysis complete. Generated {len(plot_paths)} plots."
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
                        if (hasattr(x, "__getitem__") and hasattr(x, "__len__") and len(x) > 0):
                            return float(x[0])
                        # For scalar values
                        elif isinstance(x, (int, float)):
                            return float(x)
                        # For string representation of a number
                        elif isinstance(x, str) and x.strip().replace(".", "", 1).isdigit():
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
                print(f"E/I ratio range: {clean_df['ei_ratio'].min():.2f} to {clean_df['ei_ratio'].max():.2f}")

        # Create a network category based on use_shunting
        clean_df["network_category"] = "unknown"

        # Update network category based on use_shunting
        if "use_shunting" in clean_df.columns:
            clean_df.loc[clean_df["use_shunting"] == True, "network_category"] = "shunting"
            clean_df.loc[clean_df["use_shunting"] == False, "network_category"] = "no_shunting"

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
        arch_cols = ["ee_value", "ie_value", "ei_ratio", "branch_factors", "use_shunting", "network_category"]
        if "nparams" in cleaned_results.columns:
            arch_cols.append("nparams")

        # Find noise columns
        noise_cols = [col for col in cleaned_results.columns if col.startswith("noise_")]

        # Define columns to exclude from aggregation
        exclude_cols = arch_cols + noise_cols + ["experiment", "task_id", "job_dir", "seed", "config_file"]
        metric_cols = [col for col in cleaned_results.columns if col not in exclude_cols]

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
                pattern_cols = [col for col in noise_cols if 
                               f"noise_{noise_type}_{metric}_{noise_level}" in col]
                
                if pattern_cols:
                    values = group[pattern_cols].values.flatten()
                    values = values[~pd.isna(values)]  # Remove NaN values
                    if len(values) > 0:
                        row[f"noise_{noise_type}_{metric}_{noise_level}_mean"] = np.mean(values)
                        row[f"noise_{noise_type}_{metric}_{noise_level}_std"] = np.std(values)

            # Aggregate regular metrics
            for col in metric_cols:
                if col in group.columns:
                    values = group[col].dropna().values
                    if len(values) > 0:
                        try:
                            # Ensure values are numeric
                            numeric_values = [float(v) for v in values if not pd.isna(v)]
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
    
    def generate_plots(self, results_df: pd.DataFrame, output_dir: str, base_name: str) -> List[str]:
        """Generate comprehensive plots from the results dataframe"""
        if results_df.empty:
            print("No results to plot.")
            return []

        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plot_paths = []

        # Save original data for reference
        results_df.to_csv(os.path.join(plots_dir, f"{base_name}_results_aggregated.csv"), index=False)

        # Filter for valid data
        valid_data = results_df.dropna(subset=["ei_ratio"]) if "ei_ratio" in results_df.columns else results_df

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
                param_plots = self._generate_parameter_plots(valid_data, plots_dir, base_name)
                plot_paths.extend(param_plots)

            # 3. Noise analysis plots (if noise data available)
            noise_cols = [col for col in valid_data.columns if col.startswith("noise_")]
            if noise_cols:
                noise_plots = self._generate_noise_plots(valid_data, plots_dir, base_name)
                plot_paths.extend(noise_plots)

            # 4. Heatmap plots
            heatmap_plots = self._generate_heatmaps(valid_data, plots_dir, base_name)
            plot_paths.extend(heatmap_plots)

            # 5. Ablation analysis plots (if ablation data available)
            ablation_cols = [col for col in valid_data.columns if col.startswith("ablation_")]
            if ablation_cols:
                ablation_plots = self._generate_ablation_plots(valid_data, plots_dir, base_name)
                plot_paths.extend(ablation_plots)

            # 6. Information theory plots (if MI data available) 
            mi_cols = [col for col in valid_data.columns if "mi_" in col.lower()]
            if mi_cols:
                info_plots = self._generate_information_plots(valid_data, plots_dir, base_name)
                plot_paths.extend(info_plots)

        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()

        print(f"Generated {len(plot_paths)} plots")
        return plot_paths
    
    def _generate_ei_ratio_plots(self, valid_data: pd.DataFrame, plots_dir: str, base_name: str) -> List[str]:
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
                    category_data = valid_data[valid_data["network_category"] == category]
                    
                    if category_data.empty:
                        continue

                    # Sort by ei_ratio for proper line plotting
                    category_data = category_data.sort_values("ei_ratio")

                    # Plot with error bars
                    color = self.category_colors.get(category, "black")
                    marker = self.category_markers.get(category, "o")

                    y_values = category_data[mean_col]
                    yerr_values = category_data[std_col] if std_col in category_data.columns else None

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
                ax.set_title(f"{self.metric_mapping[metric]} vs E/I Ratio by Network Type", fontsize=14)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', labelsize=12)

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
                    tick_positions = np.logspace(np.log10(min_ratio), np.log10(max_ratio), 8)
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels([f"{x:.1f}" for x in tick_positions])

                # Save the plot
                fig.tight_layout()
                plot_path = os.path.join(metric_ei_plot_dir, f"{base_name}_{metric}_vs_ei_ratio.png")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating {metric} plot: {e}")

        print(f"Created {len(plot_paths)} metric vs E/I ratio plots")
        return plot_paths
    
    def _generate_parameter_plots(self, valid_data: pd.DataFrame, plots_dir: str, base_name: str) -> List[str]:
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
                    category_data = valid_data[valid_data["network_category"] == category]
                    
                    if category_data.empty:
                        continue

                    category_data = category_data.sort_values("nparams")

                    color = self.category_colors.get(category, "black")
                    marker = self.category_markers.get(category, "o")

                    y_values = category_data[mean_col]
                    yerr_values = category_data[std_col] if std_col in category_data.columns else None

                    ax.plot(
                        category_data["nparams"],
                        y_values,
                        marker=marker,
                        color=color,
                        linestyle="-",
                        alpha=0.8,
                        label=f"{category}"
                    )
                    
                    if yerr_values is not None:
                        ax.fill_between(
                            category_data["nparams"],
                            y_values - yerr_values,
                            y_values + yerr_values,
                            color=color,
                            alpha=0.2
                        )

                ax.set_xlabel("Parameter Count", fontsize=14)
                ax.set_ylabel(self.metric_mapping[metric], fontsize=14)
                ax.set_title(f"{self.metric_mapping[metric]} vs Parameter Count by Network Type", fontsize=14)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', labelsize=12)

                ax.set_xscale("log")
                formatter = ScalarFormatter()
                formatter.set_scientific(True)
                ax.xaxis.set_major_formatter(formatter)

                fig.tight_layout()
                plot_path = os.path.join(metric_bf_plot_dir, f"{base_name}_{metric}_vs_nparams.png")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating parameter plot for {metric}: {e}")

        print(f"Created {len(plot_paths)} parameter plots")
        return plot_paths
    
    def _generate_noise_plots(self, valid_data: pd.DataFrame, plots_dir: str, base_name: str) -> List[str]:
        """Generate noise perturbation analysis plots"""
        plot_paths = []
        
        noise_ei_plot_dir = os.path.join(plots_dir, "noise_vs_ei_ratio_plots")
        os.makedirs(noise_ei_plot_dir, exist_ok=True)
        
        noise_mapping = {'uniform': 'Uniform', 'gaussian': 'Gaussian'}
        metric_mapping = {"accuracy": "Test Accuracy", "categoricalll": "Test Log-Likelihood"}

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
                mean_cols = [col for col in noise_cols 
                            if noise_type in col and metric in col and "mean" in col]
                
                if not mean_cols:
                    continue

                # Create heatmap data
                shunting_df = valid_data[valid_data["network_category"] == "shunting"].sort_values("ei_ratio")
                no_shunting_df = valid_data[valid_data["network_category"] == "no_shunting"].sort_values("ei_ratio")

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
                        shunt_vals = shunting_df[shunting_df["ei_ratio"] == ei_ratio][mean_col]
                        no_shunt_vals = no_shunting_df[no_shunting_df["ei_ratio"] == ei_ratio][mean_col]
                        
                        if not shunt_vals.empty and not no_shunt_vals.empty:
                            diff = shunt_vals.iloc[0] - no_shunt_vals.iloc[0]
                            heatmap_data.append({
                                "ei_ratio": ei_ratio,
                                "noise_level": noise_level,
                                "diff": diff
                            })

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
                        cbar_kws={"label": f"{metric_mapping.get(metric, metric)} Difference (Shunting - Non-Shunting)"}
                    )
                    ax.invert_yaxis()
                    ax.set_xticklabels([f"{float(x.get_text()):.1f}" for x in ax.get_xticklabels()])
                    ax.set_xlabel("E/I Ratio", fontsize=14)
                    ax.set_ylabel("Noise Level", fontsize=14)
                    ax.set_title(
                        f"{noise_mapping.get(noise_type, noise_type)} Noise Perturbations vs E/I Ratio\n"
                        f"{metric_mapping.get(metric, metric)} Difference",
                        fontsize=14
                    )
                    ax.tick_params(axis='both', labelsize=12)
                    
                    fig.tight_layout()
                    plot_path = os.path.join(noise_ei_plot_dir, f"{noise_type}_{metric}_noise_heatmap.png")
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating noise plot for {noise_type}_{metric}: {e}")

        print(f"Created {len(plot_paths)} noise analysis plots")
        return plot_paths
    
    def _generate_heatmaps(self, valid_data: pd.DataFrame, plots_dir: str, base_name: str) -> List[str]:
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
                    aggfunc='mean'
                )

                if not pivot_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        pivot_data,
                        cmap="viridis",
                        annot=True,
                        fmt=".3f",
                        cbar_kws={"label": self.metric_mapping.get(metric, metric)},
                        ax=ax
                    )
                    ax.set_xlabel("EE Synapses per Branch", fontsize=14)
                    ax.set_ylabel("IE Synapses per Branch", fontsize=14)
                    ax.set_title(f"{self.metric_mapping.get(metric, metric)} Heatmap", fontsize=16)
                    
                    fig.tight_layout()
                    plot_path = os.path.join(heatmap_plot_dir, f"{base_name}_{metric.replace(' ', '_')}_heatmap.png")
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating heatmap for {metric}: {e}")

        print(f"Created {len(plot_paths)} heatmap plots")
        return plot_paths
    
    def _generate_ablation_plots(self, valid_data: pd.DataFrame, plots_dir: str, base_name: str) -> List[str]:
        """Generate ablation analysis plots"""
        plot_paths = []
        
        ablation_plot_dir = os.path.join(plots_dir, "ablation_analysis_plots")
        os.makedirs(ablation_plot_dir, exist_ok=True)

        # Find ablation columns
        ablation_cols = [col for col in valid_data.columns if col.startswith("ablation_")]
        if not ablation_cols:
            return plot_paths

        # Extract ablation types and metrics
        ablation_types = []
        metrics = []
        depths = []
        for col in ablation_cols:
            parts = col.split("_")
            if len(parts) >= 4:  # ablation_type_metric_depthX
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

        depths.sort()

        if not valid_data["network_category"].isin(["shunting", "no_shunting"]).any():
            print("No shunting/no_shunting data for ablation analysis")
            return plot_paths

        # Generate ablation plots for each ablation type and metric
        for ablation_type, metric in itertools.product(ablation_types, metrics):
            try:
                # Find columns for this ablation type and metric
                ablation_cols_filtered = [col for col in ablation_cols 
                                        if f"ablation_{ablation_type}_{metric}_" in col]
                
                if not ablation_cols_filtered:
                    continue

                # Create separate plots for each network category
                categories = ["shunting", "no_shunting"]
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(f"{ablation_type.title()} Ablation: {metric.title()} vs E/I Ratio by Depth", fontsize=16)

                for idx, category in enumerate(categories):
                    ax = axes[idx]
                    category_data = valid_data[valid_data["network_category"] == category]
                    
                    if category_data.empty:
                        ax.text(0.5, 0.5, f'No {category} data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f"{category.title()}")
                        continue

                    # Plot lines for each depth
                    for depth in depths:
                        depth_cols = [col for col in ablation_cols_filtered if f"depth{depth}" in col]
                        if not depth_cols:
                            continue
                            
                        depth_col = depth_cols[0]  # Take first match
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
                            y_vals = pd.to_numeric(plot_data[depth_col], errors='coerce').dropna()
                            x_vals = plot_data.loc[y_vals.index, "ei_ratio"]
                            
                            if len(y_vals) > 0:
                                ax.plot(x_vals, y_vals, 
                                       marker='o', label=f'Depth {depth}', 
                                       alpha=0.8, linewidth=2)
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
                plot_path = os.path.join(ablation_plot_dir, f"{base_name}_{ablation_type}_{metric}_ablation.png")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating ablation plot for {ablation_type}_{metric}: {e}")

        print(f"Created {len(plot_paths)} ablation analysis plots")
        return plot_paths
    
    def _generate_information_plots(self, valid_data: pd.DataFrame, plots_dir: str, base_name: str) -> List[str]:
        """Generate information theory analysis plots"""
        plot_paths = []
        
        info_plot_dir = os.path.join(plots_dir, "information_theory_plots")
        os.makedirs(info_plot_dir, exist_ok=True)

        # Find MI columns - separate global and layer-specific metrics
        global_mi_metrics = []
        layer_mi_metrics = []
        
        for col in valid_data.columns:
            if any(mi_pattern in col.lower() for mi_pattern in ["mi_", "mutual_information"]):
                # Extract the metric name
                if "_mean" in col:
                    metric_name = col.replace("_mean", "")
                    if "layer_mi_" in metric_name:
                        # This is a layer-specific metric
                        base_metric = metric_name.replace("layer_mi_", "").split("_depth")[0]
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

        # Generate global information plots for each MI metric
        for mi_metric in global_mi_metrics:
            mean_col = f"{mi_metric}_mean"
            std_col = f"{mi_metric}_std"
            
            if mean_col not in valid_data.columns:
                continue

            try:
                fig, ax = plt.subplots(figsize=(10, 6))

                categories = valid_data["network_category"].unique()
                for category in categories:
                    if category not in ["shunting", "no_shunting"]:
                        continue
                        
                    category_data = valid_data[valid_data["network_category"] == category]
                    
                    if category_data.empty:
                        continue

                    # Sort by ei_ratio for proper line plotting
                    category_data = category_data.sort_values("ei_ratio")

                    color = self.category_colors.get(category, "black")
                    marker = self.category_markers.get(category, "o")

                    y_values = category_data[mean_col]
                    yerr_values = category_data[std_col] if std_col in category_data.columns else None

                    ax.errorbar(
                        category_data["ei_ratio"],
                        y_values,
                        yerr=yerr_values,
                        label=f"{category.title()}",
                        marker=marker,
                        color=color,
                        linestyle="-",
                        alpha=0.8,
                        capsize=3,
                    )

                # Format the metric name for display
                display_name = self.metric_mapping.get(mi_metric, mi_metric.replace("_", " ").title())
                
                ax.set_xlabel("E/I Ratio", fontsize=14)
                ax.set_ylabel(display_name, fontsize=14)
                ax.set_title(f"{display_name} vs E/I Ratio by Network Type", fontsize=14)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', labelsize=12)

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
                    tick_positions = np.logspace(np.log10(min_ratio), np.log10(max_ratio), 8)
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels([f"{x:.1f}" for x in tick_positions])

                fig.tight_layout()
                safe_metric_name = mi_metric.replace("/", "_").replace(":", "_")
                plot_path = os.path.join(info_plot_dir, f"{base_name}_{safe_metric_name}_vs_ei_ratio.png")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating information plot for {mi_metric}: {e}")

        # Generate layer-specific information plots (MI vs E/I ratio by layer)
        layer_plot_dir = os.path.join(plots_dir, "layer_information_plots")
        os.makedirs(layer_plot_dir, exist_ok=True)
        
        # Define proper mathematical notation mapping
        mi_notation = {
            "E_C": "I(E;C)",
            "I_C": "I(I;C)", 
            "Vout_C": "I(V_{out};C)",
            "E_I": "I(E;I)",
            "E_Vout": "I(E;V_{out})",
            "I_Vout": "I(I;V_{out})",
            "E_I_C": "I(E,I;C)",
            "E_I_given_C": "I(E;I|C)",
            "E_Vout_given_C": "I(E;V_{out}|C)",
            "I_Vout_given_C": "I(I;V_{out}|C)",
        }
        
        # Define distinct colors and styles for layers
        layer_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        layer_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        layer_markers = ['o', 's', '^', 'v', 'D', '<', '>']
        
        for base_metric in layer_mi_metrics:
            try:
                # Find all depth columns for this metric
                depth_cols = [col for col in valid_data.columns 
                             if f"layer_mi_{base_metric}_depth" in col and "_mean" in col]
                
                if not depth_cols:
                    continue

                # Extract depths and sort them
                depths = []
                for col in depth_cols:
                    try:
                        depth = int(col.split("_depth")[1].replace("_mean", ""))
                        depths.append(depth)
                    except:
                        continue
                depths = sorted(list(set(depths)))

                if not depths:
                    continue

                # Create MI vs E/I ratio plot for each network category
                categories = ["shunting", "no_shunting"]
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Get proper mathematical notation
                display_metric = mi_notation.get(base_metric, f"I({base_metric.replace('_', ';')})")
                
                for idx, category in enumerate(categories):
                    ax = axes[idx]
                    category_data = valid_data[valid_data["network_category"] == category]
                    
                    if category_data.empty:
                        ax.text(0.5, 0.5, f'No {category} data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f"{category.replace('_', ' ').title()}")
                        continue

                    # Plot line for each layer depth
                    for depth_idx, depth in enumerate(depths):
                        mean_col = f"layer_mi_{base_metric}_depth{depth}_mean"
                        std_col = f"layer_mi_{base_metric}_depth{depth}_std"
                        
                        if mean_col not in category_data.columns:
                            continue

                        # Sort by E/I ratio for proper line plotting
                        sorted_data = category_data.sort_values("ei_ratio")
                        
                        # Filter out NaN values
                        plot_data = sorted_data[[mean_col, std_col, "ei_ratio"]].dropna()
                        if plot_data.empty:
                            continue

                        # Use distinct colors/styles for each layer
                        color = layer_colors[depth_idx % len(layer_colors)]
                        style = layer_styles[depth_idx % len(layer_styles)]
                        marker = layer_markers[depth_idx % len(layer_markers)]
                        
                        ax.errorbar(
                            plot_data["ei_ratio"],
                            plot_data[mean_col],
                            yerr=plot_data[std_col] if std_col in plot_data.columns else None,
                            label=f'Layer {depth}',
                            color=color,
                            linestyle=style,
                            marker=marker,
                            markersize=6,
                            linewidth=2,
                            alpha=0.8,
                            capsize=4
                        )

                    ax.set_xlabel("E/I Ratio", fontsize=14)
                    ax.set_ylabel(f"{display_metric} (nats)", fontsize=14)
                    ax.set_title(f"{category.replace('_', ' ').title()} Networks", fontsize=14)
                    ax.set_xscale("log")
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='both', labelsize=12)
                    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
                    
                    # Format x-axis
                    formatter = ScalarFormatter()
                    formatter.set_scientific(False)
                    ax.xaxis.set_major_formatter(formatter)
                    
                    # Set reasonable tick positions
                    ei_ratios = valid_data["ei_ratio"].dropna()
                    if not ei_ratios.empty:
                        min_ratio = ei_ratios.min()
                        max_ratio = ei_ratios.max()
                        tick_positions = np.logspace(np.log10(min_ratio), np.log10(max_ratio), 6)
                        ax.set_xticks(tick_positions)
                        ax.set_xticklabels([f"{x:.1f}" for x in tick_positions])

                # Overall figure title
                fig.suptitle(f"{display_metric} vs E/I Ratio by Dendritic Layer", fontsize=16, y=1.02)
                fig.tight_layout()
                
                safe_metric_name = base_metric.replace("/", "_").replace(":", "_")
                plot_path = os.path.join(layer_plot_dir, f"{base_name}_layer_{safe_metric_name}_vs_ei_ratio.png")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plot_paths.append(plot_path)

            except Exception as e:
                print(f"Error generating layer information plot for {base_metric}: {e}")

        print(f"Created {len(plot_paths)} information theory plots (global + layer-specific)")
        return plot_paths
    
    def analyze_multiple_sweeps(self, results_dirs: List[str], sweep_type: Optional[str] = None) -> List[dict]:
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
                print(f"❌ Error analyzing {results_dir}: {e}")
                continue
        
        return analysis_results
    
    def generate_comparison_report(self, analysis_results: List[dict], output_file: str):
        """
        Generate a comparison report across multiple sweep analyses.
        
        Args:
            analysis_results: List of analysis results from multiple sweeps
            output_file: Path to save the comparison report
        """
        with open(output_file, 'w') as f:
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
                if 'results' in analysis and 'configs' in analysis['results']:
                    f.write(f"Number of Configurations: {len(analysis['results']['configs'])}\n")
                
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
"""
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results-dir",
        help="Single directory containing sweep results to analyze"
    )
    input_group.add_argument(
        "--results-dirs",
        nargs="+",
        help="Multiple directories containing sweep results to analyze"
    )
    
    # Sweep type options
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument(
        "--sweep-type",
        choices=SUPPORTED_SWEEP_TYPES,
        help="Type of sweep (required if not using --auto-detect)"
    )
    type_group.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect sweep type from directory contents"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        help="Output directory for analysis (defaults to results directory)"
    )
    parser.add_argument(
        "--comparison-report",
        help="Generate comparison report and save to this file (for multiple directories)"
    )
    
    return parser.parse_args()


def expand_glob_patterns(patterns: List[str]) -> List[str]:
    """Expand glob patterns in directory list."""
    expanded = []
    for pattern in patterns:
        if '*' in pattern or '?' in pattern:
            matches = glob.glob(pattern)
            expanded.extend(matches)
        else:
            expanded.append(pattern)
    
    # Remove duplicates and sort
    return sorted(list(set(expanded)))


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        analyzer = UnifiedResultAnalyzer()
        
        if args.results_dir:
            # Single directory analysis
            if not args.sweep_type and not args.auto_detect:
                print("❌ Error: Must specify --sweep-type or --auto-detect for single directory")
                sys.exit(1)
            
            sweep_type = args.sweep_type if not args.auto_detect else None
            analysis_results = [analyzer.analyze_single_sweep(
                args.results_dir, sweep_type, args.output_dir
            )]
        
        else:
            # Multiple directories analysis
            if not args.sweep_type and not args.auto_detect:
                print("❌ Error: Must specify --sweep-type or --auto-detect for multiple directories")
                sys.exit(1)
            
            # Expand glob patterns
            results_dirs = expand_glob_patterns(args.results_dirs)
            
            if not results_dirs:
                print("❌ Error: No matching directories found")
                sys.exit(1)
            
            print(f"Found {len(results_dirs)} directories to analyze:")
            for dir_path in results_dirs:
                print(f"   {dir_path}")
            
            sweep_type = args.sweep_type if not args.auto_detect else None
            analysis_results = analyzer.analyze_multiple_sweeps(results_dirs, sweep_type)
        
        # Generate comparison report if requested
        if args.comparison_report and len(analysis_results) > 1:
            analyzer.generate_comparison_report(analysis_results, args.comparison_report)
        
        print(f"\nAnalysis complete! Processed {len(analysis_results)} sweep(s)")
        
        # Summary
        for analysis in analysis_results:
            print(f"{analysis['sweep_type']} sweep: {len(analysis['plot_paths'])} plots generated")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
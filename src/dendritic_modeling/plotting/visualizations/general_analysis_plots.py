"""General analysis plotting functions for dendritic modeling."""

import os
from typing import Optional, Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from dendritic_modeling import logger
from .plotting_utils import save_plot, create_figure, setup_basic_plot, handle_plot_error, convert_layer_names


def plot_layer_contributions(results: dict[str, Any], **kwargs):
    """Plot layer contributions showing how each layer contributes to model performance."""
    logger.info("Plotting layer contributions")
    try:
        fig, ax = create_figure(figsize=(10, 6))
        
        if results and isinstance(results, dict):
            # Extract layer names and their contribution values
            layers = []
            contributions = []
            
            for layer_name, metrics in results.items():
                layers.append(layer_name)
                # Try to extract a meaningful contribution metric
                if isinstance(metrics, dict):
                    # Look for common contribution metrics
                    if 'contribution' in metrics:
                        contributions.append(metrics['contribution'])
                    elif 'accuracy_drop' in metrics:
                        contributions.append(metrics['accuracy_drop'])
                    elif 'performance_drop' in metrics:
                        contributions.append(metrics['performance_drop'])
                    else:
                        # Use first numeric value found
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                contributions.append(value)
                                break
                        else:
                            contributions.append(0.0)
                elif isinstance(metrics, (int, float)):
                    contributions.append(metrics)
                else:
                    contributions.append(0.0)
            
            if layers and contributions:
                # Create bar plot
                bars = ax.bar(range(len(layers)), contributions)
                ax.set_xticks(range(len(layers)))
                ax.set_xticklabels(layers, rotation=45, ha='right')
                ax.set_ylabel('Contribution')
                ax.set_title('Layer Contributions to Model Performance')
                
                # Add value labels on bars
                for bar, value in zip(bars, contributions):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
            else:
                ax.text(0.5, 0.5, 'No layer contribution data available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Layer Contributions')
        else:
            ax.text(0.5, 0.5, 'No data provided',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Layer Contributions')
        
        save_path = kwargs.get('save_path')
        filename = kwargs.get('filename', 'layer_contributions')
        if save_path:
            save_plot(fig, save_path, filename)
        
        return fig, ax
    except Exception as e:
        return handle_plot_error("layer contribution plotting", e)


def plot_ablation_results(results: dict, **kwargs):
    """Plot ablation study results showing performance impact of removing components."""
    logger.info("Plotting ablation analysis results")
    try:
        fig, ax = create_figure(figsize=(10, 6))
        if results:
            # Handle nested structure from ablation analysis
            if isinstance(list(results.values())[0], dict) and 'depth' not in list(results.values())[0]:
                # This is the nested structure: ablation_type -> layer -> metrics
                # Flatten to show average performance drop per ablation type
                keys = []
                values = []
                for ablation_type, layer_data in results.items():
                    keys.append(ablation_type.replace('_', ' ').title())
                    # Calculate average accuracy drop across layers
                    acc_drops = []
                    for layer_name, metrics in layer_data.items():
                        if 'accuracy drop' in metrics:
                            drop_val = metrics['accuracy drop']
                            # Ensure we have a scalar value
                            if isinstance(drop_val, (list, np.ndarray)):
                                drop_val = np.mean(drop_val) if len(drop_val) > 0 else 0
                            elif hasattr(drop_val, 'item'):  # torch tensor
                                drop_val = drop_val.item()
                            acc_drops.append(float(drop_val))
                    avg_drop = np.mean(acc_drops) if acc_drops else 0
                    values.append(avg_drop)
            else:
                # Simple structure: key -> value
                keys = list(results.keys())
                values = [list(result.values())[0] if isinstance(result, dict) and result else 
                         result if isinstance(result, (int, float)) else 0 
                         for result in results.values()]
            
            ax.bar(keys, values)
            setup_basic_plot(ax, 'Ablation Study Results', 'Ablation Type', 'Average Performance Drop')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'Ablation Results\n(No data)', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ablation Results')
        
        save_path = kwargs.get('save_path')
        filename = kwargs.get('filename', 'ablation_results')
        if save_path:
            save_plot(fig, save_path, filename)
        
        return fig, ax
    except Exception as e:
        return handle_plot_error("ablation plotting", e)


def plot_per_layer_ablation(layer_results: dict[str, dict[str, dict]], **kwargs):
    """Plot comprehensive per-layer ablation results similar to information analysis."""
    logger.info("Per-layer ablation plotting - creating comprehensive visualizations")
    try:
        save_path = kwargs.get('save_path')
        filename_prefix = kwargs.get('filename_prefix', 'ablation_per_layer')
        
        if not layer_results:
            logger.warning("No layer ablation results provided")
            return None
        
        # Convert layer names and organize by depth
        layer_data_by_depth = {}
        
        # Use convert_layer_names for consistency with information analysis
        # First restructure the data to work with convert_layer_names
        sample_ablation = next(iter(layer_results.values()))
        temp_layer_results = {}
        for layer_name, layer_data in sample_ablation.items():
            temp_layer_results[layer_name] = layer_data
        
        # Convert layer names using the standard function
        converted_temp = convert_layer_names(temp_layer_results)
        layer_name_mapping = {}
        
        # Create mapping from original technical names to display names
        all_depths = set()
        for ablation_type, ablation_data in layer_results.items():
            if ablation_data:
                for layer_name, layer_metrics in ablation_data.items():
                    if 'depth' in layer_metrics:
                        all_depths.add(layer_metrics['depth'])
        
        all_depths = sorted(all_depths, reverse=True)  # Soma to distal order
        
        # Create consistent naming
        for ablation_type, ablation_data in layer_results.items():
            if not ablation_data:
                continue
                
            # Create a consistent structure for all depths
            layers_info = []
            for i, depth in enumerate(all_depths):
                # Use same naming as information analysis: Soma, Layer 1, Layer 2, etc.
                if i == 0:
                    display_name = "Soma"
                else:
                    display_name = f"Layer {i}"
                
                # Find the layer data for this depth
                layer_metrics = None
                for layer_name, metrics in ablation_data.items():
                    if metrics.get('depth') == depth:
                        layer_metrics = metrics
                        break
                
                # If no data for this depth, create empty metrics
                if layer_metrics is None:
                    layer_metrics = {'depth': depth}
                
                layers_info.append({
                    'depth': depth,
                    'name': display_name,
                    'metrics': layer_metrics
                })
            
            layer_data_by_depth[ablation_type] = layers_info
        
        plots = []
        
        # Create separate plots for each metric type
        metric_types = ['accuracy drop', 'auc drop', 'categorical_loglikelihood drop']
        
        for metric_type in metric_types:
            if any(any(metric_type in layer['metrics'] for layer in layers) 
                   for layers in layer_data_by_depth.values()):
                
                fig, ax = create_figure(figsize=(14, 8))
                
                # Plot data for each ablation type
                ablation_types = list(layer_data_by_depth.keys())
                n_layers = len(layer_data_by_depth[ablation_types[0]])
                x_pos = np.arange(n_layers) * 1.5  # Add spacing between layer groups
                width = 0.3  # Slightly wider bars for better visibility
                
                colors = ['red', 'blue', 'green', 'orange', 'purple']
                
                for i, ablation_type in enumerate(ablation_types):
                    layers = layer_data_by_depth[ablation_type]
                    layer_names = [layer['name'] for layer in layers]
                    values = [layer['metrics'].get(metric_type, 0) for layer in layers]
                    
                    bars = ax.bar(x_pos + i*width, values, width, 
                                 label=ablation_type.replace('_', ' ').title(),
                                 color=colors[i % len(colors)], alpha=0.8)
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, values):
                        if val > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                
                # Customize plot
                ax.set_xlabel('Layer (Soma → Distal)', fontsize=12)
                ax.set_ylabel(f'{metric_type.replace("_", " ").title()}', fontsize=12)
                ax.set_title(f'Ablation Effects: {metric_type.replace("_", " ").title()} by Layer', fontsize=14)
                ax.set_xticks(x_pos + width * (len(ablation_types) - 1) / 2)
                ax.set_xticklabels(layer_names)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plots.append(fig)
                
                # Save individual plot
                if save_path:
                    clean_metric = metric_type.replace(' ', '_').replace('_drop', '')
                    save_plot(fig, save_path, f"{filename_prefix}_{clean_metric}")
        
        # Create summary plot with all metrics
        if layer_data_by_depth:
            fig, axes = plt.subplots(1, len(metric_types), figsize=(18, 6))
            if len(metric_types) == 1:
                axes = [axes]
            
            for idx, metric_type in enumerate(metric_types):
                ax = axes[idx]
                
                if any(any(metric_type in layer['metrics'] for layer in layers) 
                       for layers in layer_data_by_depth.values()):
                    
                    ablation_types = list(layer_data_by_depth.keys())
                    n_layers = len(layer_data_by_depth[ablation_types[0]])
                    x_pos = np.arange(n_layers) * 1.5  # Add spacing between layer groups
                    width = 0.3  # Slightly wider bars for better visibility
                    
                    for i, ablation_type in enumerate(ablation_types):
                        layers = layer_data_by_depth[ablation_type]
                        layer_names = [layer['name'] for layer in layers]
                        values = [layer['metrics'].get(metric_type, 0) for layer in layers]
                        
                        bars = ax.bar(x_pos + i*width, values, width, 
                                     label=ablation_type.replace('_', ' ').title(),
                                     color=colors[i % len(colors)], alpha=0.8)
                    
                    ax.set_xlabel('Layer (Soma → Distal)')
                    ax.set_ylabel(metric_type.replace('_', ' ').title())
                    ax.set_title(f'{metric_type.replace("_", " ").title()}')
                    ax.set_xticks(x_pos + width * (len(ablation_types) - 1) / 2)
                    ax.set_xticklabels(layer_names)
                    if idx == 0:  # Only show legend on first subplot
                        ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'No {metric_type} data', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{metric_type.replace("_", " ").title()}')
            
            plt.suptitle('Ablation Analysis: Performance Drops by Layer and Synapse Type', fontsize=16)
            plt.tight_layout()
            plots.append(fig)
            
            if save_path:
                save_plot(fig, save_path, f"{filename_prefix}_summary")
        
        logger.info(f"Generated {len(plots)} ablation plots")
        return plots
        
    except Exception as e:
        logger.error(f"Error in per-layer ablation plotting: {e}")
        return handle_plot_error("per-layer ablation plotting", e)


def plot_noise_robustness(results: dict[str, list[float]], noise_levels: list[float], **kwargs):
    """Plot noise robustness analysis results showing performance vs noise levels."""
    logger.info("Plotting noise robustness analysis")
    try:
        fig, ax = create_figure(figsize=(10, 6))
        
        if results and noise_levels:
            for name, values in results.items():
                ax.plot(noise_levels[:len(values)], values, marker='o', label=name)
            setup_basic_plot(ax, 'Noise Robustness', 'Noise Level', 'Performance')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Noise Robustness\n(No data)', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Noise Robustness')
        
        save_path = kwargs.get('save_path')
        filename = kwargs.get('filename', 'noise_robustness')
        if save_path:
            save_plot(fig, save_path, filename)
        
        return fig, ax
    except Exception as e:
        return handle_plot_error("noise robustness plotting", e)


def plot_weight_distributions(weight_data: dict[str, list[np.ndarray]], **kwargs):
    """Plot weight distributions for different layer types."""
    logger.info("Plotting weight distributions")
    try:
        fig, ax = create_figure(figsize=(10, 6))
        
        if weight_data:
            for name, weights in weight_data.items():
                if weights:
                    # Flatten all weight arrays and plot histogram
                    flat_weights = np.concatenate([w.flatten() for w in weights])
                    ax.hist(flat_weights, bins=50, alpha=0.7, label=name)
            setup_basic_plot(ax, 'Weight Distributions', 'Weight Value', 'Frequency')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Weight Distributions\n(No data)', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Distributions')
        
        save_path = kwargs.get('save_path')
        filename = kwargs.get('filename', 'weight_distributions')
        if save_path:
            save_plot(fig, save_path, filename)
        
        return fig, ax
    except Exception as e:
        return handle_plot_error("weight distribution plotting", e)


def plot_dendrinet_activations(dendrinet, exc_input_list, inh_input_list, **kwargs):
    """Plot DendriNet activation patterns for excitatory and inhibitory branches."""
    logger.info("Plotting DendriNet activations")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('DendriNet Activation Patterns')
        
        # Convert inputs to numpy arrays if they're tensors
        if hasattr(exc_input_list[0], 'detach'):
            exc_activations = [x.detach().cpu().numpy() for x in exc_input_list]
        else:
            exc_activations = exc_input_list
            
        if hasattr(inh_input_list[0], 'detach'):
            inh_activations = [x.detach().cpu().numpy() for x in inh_input_list]
        else:
            inh_activations = inh_input_list
        
        # Plot excitatory activations heatmap
        if exc_activations:
            exc_data = np.vstack(exc_activations)
            im1 = axes[0, 0].imshow(exc_data.T, aspect='auto', cmap='viridis')
            axes[0, 0].set_title('Excitatory Branch Activations')
            axes[0, 0].set_xlabel('Branch Index')
            axes[0, 0].set_ylabel('Neuron Index')
            plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot inhibitory activations heatmap
        if inh_activations:
            inh_data = np.vstack(inh_activations)
            im2 = axes[0, 1].imshow(inh_data.T, aspect='auto', cmap='plasma')
            axes[0, 1].set_title('Inhibitory Branch Activations')
            axes[0, 1].set_xlabel('Branch Index')
            axes[0, 1].set_ylabel('Neuron Index')
            plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot activation distributions
        if exc_activations:
            exc_flat = np.concatenate([act.flatten() for act in exc_activations])
            axes[1, 0].hist(exc_flat, bins=50, alpha=0.7, color='blue', label='Excitatory')
            axes[1, 0].set_title('Excitatory Activation Distribution')
            axes[1, 0].set_xlabel('Activation Value')
            axes[1, 0].set_ylabel('Frequency')
        
        if inh_activations:
            inh_flat = np.concatenate([act.flatten() for act in inh_activations])
            axes[1, 1].hist(inh_flat, bins=50, alpha=0.7, color='red', label='Inhibitory')
            axes[1, 1].set_title('Inhibitory Activation Distribution')
            axes[1, 1].set_xlabel('Activation Value')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        save_path = kwargs.get('save_path')
        filename = kwargs.get('filename', 'dendrinet_activations')
        if save_path:
            save_plot(fig, save_path, filename)
        
        return fig, axes
    except Exception as e:
        return handle_plot_error("DendriNet activation plotting", e)


def plot_einet_activations(einet, input_list, **kwargs):
    """Plot EINet activation patterns for excitatory and inhibitory networks."""
    logger.info("Plotting EINet activations")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('EINet Activation Patterns')
        
        # Convert inputs to numpy arrays if they're tensors
        if hasattr(input_list[0], 'detach'):
            activations = [x.detach().cpu().numpy() for x in input_list]
        else:
            activations = input_list
        
        if activations:
            # Combine all activations for analysis
            combined_data = np.vstack(activations)
            
            # Plot activation heatmap
            im1 = axes[0, 0].imshow(combined_data.T, aspect='auto', cmap='RdBu_r')
            axes[0, 0].set_title('Network Activation Heatmap')
            axes[0, 0].set_xlabel('Time/Sample')
            axes[0, 0].set_ylabel('Neuron Index')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Plot activation distribution
            flat_activations = combined_data.flatten()
            axes[0, 1].hist(flat_activations, bins=50, alpha=0.7, color='green')
            axes[0, 1].set_title('Activation Value Distribution')
            axes[0, 1].set_xlabel('Activation Value')
            axes[0, 1].set_ylabel('Frequency')
            
            # Plot mean activation per neuron
            mean_activations = np.mean(combined_data, axis=0)
            axes[1, 0].bar(range(len(mean_activations)), mean_activations)
            axes[1, 0].set_title('Mean Activation per Neuron')
            axes[1, 0].set_xlabel('Neuron Index')
            axes[1, 0].set_ylabel('Mean Activation')
            
            # Plot activation variance per neuron
            var_activations = np.var(combined_data, axis=0)
            axes[1, 1].bar(range(len(var_activations)), var_activations)
            axes[1, 1].set_title('Activation Variance per Neuron')
            axes[1, 1].set_xlabel('Neuron Index')
            axes[1, 1].set_ylabel('Activation Variance')
        else:
            # No data case
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No activation data available',
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        save_path = kwargs.get('save_path')
        filename = kwargs.get('filename', 'einet_activations')
        if save_path:
            save_plot(fig, save_path, filename)
        
        return fig, axes
    except Exception as e:
        return handle_plot_error("EINet activation plotting", e)


def plot_synapse_turnover(turnover_results: dict, save_path: str = None, filename: str = "synapse_turnover"):
    """Generate plots for synaptic turnover analysis results."""
    try:
        # Validate input data
        if not turnover_results:
            logger.warning("Empty turnover results provided")
            return None
            
        logger.info(f"Plotting turnover results with keys: {list(turnover_results.keys())}")
        
        # Check if we have the new format with incremental and cumulative data
        if "incremental" in turnover_results and "cumulative" in turnover_results:
            # New format with both incremental and cumulative data
            figures = []
            
            # Log data availability
            inc_data = turnover_results["incremental"]
            cum_data = turnover_results["cumulative"]
            n_snapshots = turnover_results.get("n_snapshots", 0)
            
            logger.info(f"Found {n_snapshots} snapshots for turnover analysis")
            logger.info(f"Incremental data keys: {list(inc_data.keys()) if inc_data else 'None'}")
            logger.info(f"Cumulative data keys: {list(cum_data.keys()) if cum_data else 'None'}")
            
            # Create incremental turnover plot (t vs t-1)
            fig1 = _plot_turnover_data(
                turnover_results["incremental"],
                "Incremental Synaptic Turnover (t vs t-1)",
                "incremental"
            )
            figures.append(fig1)
            
            # Create cumulative turnover plot (t vs t-0)
            fig2 = _plot_turnover_data(
                turnover_results["cumulative"],
                "Cumulative Synaptic Turnover (t vs t-0)",
                "cumulative"
            )
            figures.append(fig2)
            
            # Save both plots
            if save_path:
                if fig1:
                    save_plot(fig1, save_path, f"{filename}_incremental")
                if fig2:
                    save_plot(fig2, save_path, f"{filename}_cumulative")
            
            return figures
        
        else:
            # Original format - assume it's the old structure
            logger.warning("Using original turnover format - may not display correctly")
            return _plot_turnover_original_format(turnover_results, save_path, filename)
            
    except Exception as e:
        logger.error(f"Error in synapse turnover plotting: {e}")
        return handle_plot_error("synapse turnover plotting", e, create_placeholder=False)


def _plot_turnover_data(turnover_data: dict, title: str, plot_type: str) -> plt.Figure:
    """Helper function to plot turnover data."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot for excitatory connections
        if "exc" in turnover_data:
            _plot_single_turnover(axes[0, 0], turnover_data["exc"], f"Excitatory {title}")
        
        # Plot for inhibitory connections  
        if "inh" in turnover_data:
            _plot_single_turnover(axes[0, 1], turnover_data["inh"], f"Inhibitory {title}")
        
        # Plot for both types combined
        if "both" in turnover_data:
            _plot_single_turnover(axes[1, 0], turnover_data["both"], f"Combined {title}")
        
        # Summary statistics plot
        _plot_turnover_summary(axes[1, 1], turnover_data, plot_type)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.warning(f"Error plotting turnover data: {e}")
        return None


def _plot_single_turnover(ax: plt.Axes, data: dict, title: str):
    """Plot turnover data for a single type."""
    try:
        # Check if data has required keys and non-empty values
        required_keys = ["active", "stable", "transient"]
        if not all(key in data for key in required_keys):
            ax.text(0.5, 0.5, f'Missing turnover data\nRequired: {required_keys}\nFound: {list(data.keys())}', 
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        # Check if data arrays are not empty
        if not any(len(data[key]) > 0 for key in required_keys):
            ax.text(0.5, 0.5, f'Empty turnover data\nNo snapshots to compare', 
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        # Plot if we have data
        max_len = max(len(data[key]) for key in required_keys if len(data[key]) > 0)
        if max_len > 0:
            snapshots = list(range(max_len))
            
            # Define colors and styles for better visibility
            plot_styles = [
                ("active", '#2E8B57', 'o', 'Active', '-'),      # Sea green, solid line
                ("stable", '#4169E1', 's', 'Stable', '--'),     # Royal blue, dashed line
                ("transient", '#DC143C', '^', 'Transient', ':') # Crimson, dotted line
            ]
            
            plotted_any = False
            for key, color, marker, label, linestyle in plot_styles:
                if key in data and len(data[key]) > 0:
                    values = data[key][:max_len]  # Ensure same length
                    line = ax.plot(snapshots[:len(values)], values, 
                                  color=color, marker=marker, label=label, 
                                  linestyle=linestyle, linewidth=2, markersize=6, alpha=0.8)
                    plotted_any = True
                    
                    # Add text annotation for flat lines
                    if len(set(values)) == 1:  # All values are the same (flat line)
                        mid_x = len(values) // 2
                        ax.annotate(f'{label}: {values[0]}', 
                                   xy=(mid_x, values[0]), 
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=9, alpha=0.7,
                                   bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor=color, alpha=0.2))
            
            if plotted_any:
                ax.set_xlabel('Snapshot Number')
                ax.set_ylabel('Synapse Count')
                ax.set_title(title, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # Improve axis formatting
                ax.tick_params(axis='both', which='major', labelsize=9)
                
                # Set y-axis to start from 0 if all values are positive
                y_min = min(min(data[key]) for key in required_keys if len(data[key]) > 0)
                if y_min >= 0:
                    ax.set_ylim(bottom=0)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Turnover data\n(Error: {str(e)[:50]})', 
                ha='center', va='center', transform=ax.transAxes)


def _plot_turnover_summary(ax: plt.Axes, turnover_data: dict, plot_type: str):
    """Plot summary statistics for turnover."""
    try:
        summary_metrics = []
        labels = []
        colors = []
        
        color_map = {"exc": "green", "inh": "red", "both": "blue"}
        
        for conn_type in ["exc", "inh", "both"]:
            if conn_type in turnover_data:
                data = turnover_data[conn_type]
                
                # Use the actual data keys: active, stable, transient
                for metric_name in ["active", "stable", "transient"]:
                    if metric_name in data and len(data[metric_name]) > 0:
                        # Calculate final value and change from start
                        values = data[metric_name]
                        if len(values) > 1:
                            final_value = values[-1]
                            initial_value = values[0]
                            change = final_value - initial_value
                            
                            summary_metrics.append(change)
                            labels.append(f"{conn_type.upper()}\n{metric_name.capitalize()}\nChange")
                            colors.append(color_map[conn_type])
        
        if summary_metrics:
            bars = ax.bar(range(len(labels)), summary_metrics, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, summary_metrics):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., 
                       height + (abs(height) * 0.02 if height >= 0 else -abs(height) * 0.02),
                       f'{value:+.0f}', ha='center', 
                       va='bottom' if height >= 0 else 'top', fontsize=9)
            
            ax.set_title(f'{plot_type.capitalize()} Net Change Summary')
            ax.set_ylabel('Net Change (Final - Initial)')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'No summary data available\n(No turnover data found)', 
                    ha='center', va='center', transform=ax.transAxes)
            
    except Exception as e:
        ax.text(0.5, 0.5, f'Summary Error:\n{str(e)[:50]}', 
                ha='center', va='center', transform=ax.transAxes)


def _plot_turnover_original_format(turnover_results: dict, save_path: str, filename: str):
    """Handle original turnover format."""
    try:
        figures = []
        
        for connection_type in ["exc", "inh", "both"]:
            if connection_type in turnover_results:
                fig = _plot_turnover_data(
                    {connection_type: turnover_results[connection_type]},
                    f"Synaptic Turnover - {connection_type.capitalize()}",
                    "original"
                )
                if fig:
                    figures.append(fig)
                    if save_path:
                        save_plot(fig, save_path, f"{filename}_{connection_type}")
        
        return figures
        
    except Exception as e:
        logger.warning(f"Error plotting original turnover format: {e}")
        return []


def plot_performance_metrics(performance_data: dict, **kwargs):
    """Plot performance metrics as a bar chart."""
    try:
        fig, ax = create_figure(figsize=(10, 6))
        
        if performance_data:
            # Handle nested dictionary structure from PerformanceAnalyzer
            flattened_metrics = {}
            for metric_name, metric_data in performance_data.items():
                if isinstance(metric_data, dict):
                    # Handle nested structure (metric -> dataset -> value)
                    for dataset, value in metric_data.items():
                        if isinstance(value, (int, float)):
                            flattened_metrics[f"{metric_name}_{dataset}"] = value
                elif isinstance(metric_data, (int, float)):
                    # Handle direct values
                    flattened_metrics[metric_name] = metric_data
            
            if flattened_metrics:
                metrics = list(flattened_metrics.keys())
                values = list(flattened_metrics.values())
                ax.bar(metrics, values)
                setup_basic_plot(ax, 'Performance Metrics', 'Metric', 'Value')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No valid performance data found', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Performance Metrics')
        else:
            ax.text(0.5, 0.5, 'No performance data available', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Metrics')
        
        save_path = kwargs.get('save_path')
        filename = kwargs.get('filename', 'performance_metrics')
        if save_path:
            save_plot(fig, save_path, filename)
        
        return fig, ax
        
    except Exception as e:
        return handle_plot_error("performance metrics plotting", e)


def plot_performance_evolution(performance_dir: str, save_path: Optional[str] = None, 
                              filename: str = "performance_evolution") -> dict[str, Tuple[plt.Figure, plt.Axes]]:
    """Generate performance evolution plots from per-epoch performance files."""
    try:
        plots = {}
        
        # Look for performance files in the directory
        if not os.path.exists(performance_dir):
            logger.warning(f"Performance directory does not exist: {performance_dir}")
            return {}
        
        import glob
        import json
        
        # Find all performance files (assuming JSON format with epoch data)
        perf_files = glob.glob(os.path.join(performance_dir, "*performance*.json"))
        
        if not perf_files:
            logger.warning(f"No performance files found in {performance_dir}")
            # Create placeholder plot
            fig, ax = create_figure(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No performance data found in directory',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Evolution Over Time')
            plots['evolution'] = (fig, ax)
            return plots
        
        # Load and parse performance data
        all_data = {}
        epochs = []
        
        for file_path in sorted(perf_files):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract epoch number from filename or data
                if 'epoch' in data:
                    epoch = data['epoch']
                else:
                    # Try to extract from filename
                    import re
                    match = re.search(r'epoch_(\d+)', file_path)
                    if match:
                        epoch = int(match.group(1))
                    else:
                        continue
                
                epochs.append(epoch)
                
                # Store metrics
                for metric, value in data.items():
                    if metric != 'epoch' and isinstance(value, (int, float)):
                        if metric not in all_data:
                            all_data[metric] = []
                        all_data[metric].append(value)
                        
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Could not parse performance file {file_path}: {e}")
                continue
        
        if not all_data or not epochs:
            logger.warning("No valid performance data found")
            fig, ax = create_figure(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No valid performance data found',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Evolution Over Time')
            plots['evolution'] = (fig, ax)
            return plots
        
        # Sort by epoch
        sorted_indices = sorted(range(len(epochs)), key=lambda i: epochs[i])
        epochs = [epochs[i] for i in sorted_indices]
        for metric in all_data:
            all_data[metric] = [all_data[metric][i] for i in sorted_indices]
        
        # Create evolution plot
        fig, ax = create_figure(figsize=(12, 8))
        
        # Plot each metric
        for metric, values in all_data.items():
            if len(values) == len(epochs):
                ax.plot(epochs, values, marker='o', label=metric, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Performance Metric Value')
        ax.set_title('Performance Evolution Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plots['evolution'] = (fig, ax)
        
        if save_path:
            save_plot(fig, save_path, filename)
        
        return plots
        
    except Exception as e:
        logger.warning(f"Error in performance evolution plotting: {e}")
        return {}


# Training and Sweep Plotting Functions
def plot_loss_curves(train_losses: list[float], valid_losses: list[float], 
                     loss_name: str, save_dir: Optional[str] = None, filename_prefix: str = ""):
    """Plot training and validation loss curves."""
    try:
        fig, ax = create_figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss')
        
        if valid_losses:
            epochs_val = range(1, len(valid_losses) + 1)
            ax.plot(epochs_val, valid_losses, 'r-', label='Validation Loss')
        
        setup_basic_plot(ax, f'{loss_name} Loss Curves', 'Epoch', 'Loss')
        ax.legend()
        
        if save_dir:
            save_plot(fig, save_dir, f"{filename_prefix}loss_curves")
        
        return fig, ax
        
    except Exception as e:
        return handle_plot_error("loss curves plotting", e)


def plot_branch_sweep_results(results_csv_path, split="test"):
    """Plot branch sweep results."""
    import pandas as pd
    
    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific split
    split_df = results_df[results_df["split"] == split]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Branch Sweep Results ({split} split)", fontsize=16)

    # Plot accuracy vs number of branches
    axes[0, 0].plot(split_df["n_branches"], split_df["accuracy"], "o-", color="blue")
    setup_basic_plot(axes[0, 0], "Accuracy vs Number of Branches", "Number of Branches", "Accuracy")

    # Plot loss vs number of branches
    axes[0, 1].plot(split_df["n_branches"], split_df["loss"], "o-", color="red")
    setup_basic_plot(axes[0, 1], "Loss vs Number of Branches", "Number of Branches", "Loss")

    # Plot precision vs number of branches
    axes[1, 0].plot(split_df["n_branches"], split_df["precision"], "o-", color="green")
    setup_basic_plot(axes[1, 0], "Precision vs Number of Branches", "Number of Branches", "Precision")

    # Plot recall vs number of branches
    axes[1, 1].plot(split_df["n_branches"], split_df["recall"], "o-", color="orange")
    setup_basic_plot(axes[1, 1], "Recall vs Number of Branches", "Number of Branches", "Recall")

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(
        os.path.dirname(results_csv_path), f"branch_sweep_results_{split}.png"
    )
    save_plot(fig, os.path.dirname(results_csv_path), f"branch_sweep_results_{split}")
    
    print(f"Plot saved to: {save_path_final}")
    return fig, axes


def plot_ei_sweep_results(results_csv_path, ei_ratio_range=None, split="test"):
    """Plot E/I ratio sweep results."""
    import pandas as pd
    
    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific split
    split_df = results_df[results_df["split"] == split]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"E/I Ratio Sweep Results ({split} split)", fontsize=16)

    # Plot accuracy vs E/I ratio
    axes[0, 0].plot(split_df["ei_ratio"], split_df["accuracy"], "o-", color="blue")
    setup_basic_plot(axes[0, 0], "Accuracy vs E/I Ratio", "E/I Ratio", "Accuracy")

    # Plot loss vs E/I ratio
    axes[0, 1].plot(split_df["ei_ratio"], split_df["loss"], "o-", color="red")
    setup_basic_plot(axes[0, 1], "Loss vs E/I Ratio", "E/I Ratio", "Loss")

    # Plot precision vs E/I ratio
    axes[1, 0].plot(split_df["ei_ratio"], split_df["precision"], "o-", color="green")
    setup_basic_plot(axes[1, 0], "Precision vs E/I Ratio", "E/I Ratio", "Precision")

    # Plot recall vs E/I ratio
    axes[1, 1].plot(split_df["ei_ratio"], split_df["recall"], "o-", color="orange")
    setup_basic_plot(axes[1, 1], "Recall vs E/I Ratio", "E/I Ratio", "Recall")

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(
        os.path.dirname(results_csv_path), f"ei_sweep_results_{split}.png"
    )
    save_plot(fig, os.path.dirname(results_csv_path), f"ei_sweep_results_{split}")
    
    print(f"Plot saved to: {save_path_final}")
    return fig, axes


def plot_noise_sweep_branch_results(results_csv_path, noise_type="gaussian"):
    """Plot noise sweep results for branch networks."""
    import pandas as pd
    
    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific noise type
    noise_df = results_df[results_df["noise_type"] == noise_type]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Noise Sweep Results - {noise_type.title()} Noise (Branch Networks)", fontsize=16)

    # Plot accuracy vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[0, 0].plot(branch_data["noise_level"], branch_data["accuracy"], "o-", label=f"Branches={n_branches}")
    setup_basic_plot(axes[0, 0], "Accuracy vs Noise Level", "Noise Level", "Accuracy")
    axes[0, 0].legend()

    # Plot loss vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[0, 1].plot(branch_data["noise_level"], branch_data["loss"], "o-", label=f"Branches={n_branches}")
    setup_basic_plot(axes[0, 1], "Loss vs Noise Level", "Noise Level", "Loss")
    axes[0, 1].legend()

    # Plot precision vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[1, 0].plot(branch_data["noise_level"], branch_data["precision"], "o-", label=f"Branches={n_branches}")
    setup_basic_plot(axes[1, 0], "Precision vs Noise Level", "Noise Level", "Precision")
    axes[1, 0].legend()

    # Plot recall vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[1, 1].plot(branch_data["noise_level"], branch_data["recall"], "o-", label=f"Branches={n_branches}")
    setup_basic_plot(axes[1, 1], "Recall vs Noise Level", "Noise Level", "Recall")
    axes[1, 1].legend()

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(os.path.dirname(results_csv_path), f"noise_sweep_branch_results_{noise_type}.png")
    save_plot(fig, os.path.dirname(results_csv_path), f"noise_sweep_branch_results_{noise_type}")
    
    print(f"Plot saved to: {save_path_final}")
    return fig, axes


def plot_noise_sweep_ei_results(results_csv_path, noise_type="gaussian"):
    """Plot noise sweep results for E/I networks."""
    import pandas as pd
    
    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific noise type
    noise_df = results_df[results_df["noise_type"] == noise_type]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Noise Sweep Results - {noise_type.title()} Noise (E/I Networks)", fontsize=16)

    # Plot accuracy vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[0, 0].plot(ei_data["noise_level"], ei_data["accuracy"], "o-", label=f"E/I_var = {ei_ratio}")
    setup_basic_plot(axes[0, 0], "Accuracy vs Noise Level", "Noise Level", "Accuracy")
    axes[0, 0].legend()

    # Plot loss vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[0, 1].plot(ei_data["noise_level"], ei_data["loss"], "o-", label=f"E/I_var = {ei_ratio}")
    setup_basic_plot(axes[0, 1], "Loss vs Noise Level", "Noise Level", "Loss")
    axes[0, 1].legend()

    # Plot precision vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[1, 0].plot(ei_data["noise_level"], ei_data["precision"], "o-", label=f"E/I_var = {ei_ratio}")
    setup_basic_plot(axes[1, 0], "Precision vs Noise Level", "Noise Level", "Precision")
    axes[1, 0].legend()

    # Plot recall vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[1, 1].plot(ei_data["noise_level"], ei_data["recall"], "o-", label=f"E/I_var = {ei_ratio}")
    setup_basic_plot(axes[1, 1], "Recall vs Noise Level", "Noise Level", "Recall")
    axes[1, 1].legend()

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(os.path.dirname(results_csv_path), f"noise_sweep_ei_results_{noise_type}.png")
    save_plot(fig, os.path.dirname(results_csv_path), f"noise_sweep_ei_results_{noise_type}")
    
    print(f"Plot saved to: {save_path_final}")
    return fig, axes 
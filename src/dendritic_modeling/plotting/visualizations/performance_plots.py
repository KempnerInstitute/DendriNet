"""
Performance and training visualization plotting.

This module provides plotting for training curves, performance metrics,
and evaluation results.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from .base import TimeSeriesPlotter, PlotConfig


class PerformancePlotter(TimeSeriesPlotter):
    """Plotter for performance metrics and training curves."""

    def plot_loss_curves(
        self,
        train_losses: List[float],
        valid_losses: List[float],
        loss_name: str,
        save_dir: Optional[str] = None,
        filename_prefix: str = "",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot training and validation loss curves."""
        if not self._validate_data(train_losses, "training losses"):
            return self._create_error_plot("Loss Curves", "No training data")
        
        fig, ax = self._create_figure()
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot raw data
        ax.plot(epochs, train_losses, 'b-', alpha=0.6, label='Train (raw)')
        if valid_losses:
            ax.plot(epochs[:len(valid_losses)], valid_losses, 'r-', 
                   alpha=0.6, label='Valid (raw)')
        
        # Add smoothed curves
        if len(train_losses) > 10:
            smooth_train = self._smooth_data(np.array(train_losses), window_size=5)
            ax.plot(epochs[2:-2], smooth_train, 'b-', linewidth=2, 
                   label='Train (smooth)')
            
            if valid_losses and len(valid_losses) > 10:
                smooth_valid = self._smooth_data(np.array(valid_losses), window_size=5)
                ax.plot(epochs[2:len(smooth_valid)+2], smooth_valid, 'r-', 
                       linewidth=2, label='Valid (smooth)')
        
        # Apply styling
        self._apply_styling(
            ax,
            title=f'{filename_prefix} {loss_name} Loss' if filename_prefix else f'{loss_name} Loss',
            xlabel='Epoch',
            ylabel='Loss',
            show_legend=True
        )
        
        # Add final values annotation
        final_train = train_losses[-1]
        ax.annotate(f'Final: {final_train:.4f}', 
                   xy=(len(train_losses), final_train),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        if save_dir:
            filename = f"{filename_prefix}_loss_curves" if filename_prefix else "loss_curves"
            self._save_plot(fig, save_dir, filename)
        
        return fig, ax

    def plot_performance_metrics(
        self,
        results_dict: Dict[str, Any],
        save_path: Optional[str] = None,
        filename: str = "performance_summary",
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """Generate comprehensive performance summary plots."""
        plots = {}
        
        # Extract metrics
        metrics = {}
        for key, value in results_dict.items():
            if isinstance(value, (int, float)) and not key.startswith('_'):
                metrics[key] = value
        
        if not metrics:
            fig, ax = self._create_error_plot("Performance Metrics", "No metrics found")
            plots["error"] = (fig, ax)
            return plots
        
        # Create bar plot of metrics
        fig, ax = self._create_figure(figsize=(12, 6))
        
        keys = list(metrics.keys())
        values = list(metrics.values())
        
        # Color based on metric type
        colors = []
        for key in keys:
            if 'accuracy' in key.lower():
                colors.append(self.config.colors["success"])
            elif 'loss' in key.lower():
                colors.append(self.config.colors["danger"])
            elif 'time' in key.lower():
                colors.append(self.config.colors["info"])
            else:
                colors.append(self.config.colors["primary"])
        
        bars = ax.bar(keys, values, color=colors, alpha=self.config.alpha["main"])
        self._add_value_labels(ax, bars, values, format_str="{:.4f}")
        
        self._apply_styling(
            ax,
            title="Performance Metrics Summary",
            ylabel="Value",
            rotate_xticks=True
        )
        
        plots["summary"] = (fig, ax)
        
        if save_path:
            self._save_plot(fig, save_path, filename)
        
        return plots

    def plot_performance_evolution(
        self,
        performance_dir: str,
        save_path: Optional[str] = None,
        filename: str = "performance_evolution",
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """Plot performance metrics evolution over epochs from saved files."""
        plots = {}
        
        # Collect all performance files
        perf_files = []
        if os.path.exists(performance_dir):
            for f in os.listdir(performance_dir):
                if f.startswith("performance_epoch_") and f.endswith(".json"):
                    epoch = int(f.split("_")[2].split(".")[0])
                    perf_files.append((epoch, os.path.join(performance_dir, f)))
        
        if not perf_files:
            fig, ax = self._create_error_plot(
                "Performance Evolution",
                f"No performance files found in {performance_dir}"
            )
            plots["error"] = (fig, ax)
            return plots
        
        # Sort by epoch
        perf_files.sort(key=lambda x: x[0])
        
        # Load data
        epochs = []
        metrics_data = {}
        
        for epoch, filepath in perf_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    epochs.append(epoch)
                    
                    for key, value in data.items():
                        if isinstance(value, (int, float)):
                            if key not in metrics_data:
                                metrics_data[key] = []
                            metrics_data[key].append(value)
            except Exception as e:
                self.logger.warning(f"Error loading {filepath}: {e}")
        
        if not metrics_data:
            fig, ax = self._create_error_plot(
                "Performance Evolution",
                "No valid metrics found in files"
            )
            plots["error"] = (fig, ax)
            return plots
        
        # Create subplots for different metric types
        n_metrics = len(metrics_data)
        n_rows = (n_metrics + 1) // 2
        fig, axes = self._create_figure(n_rows, 2, figsize=(15, 5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[idx]
            
            # Plot with smoothing
            ax.plot(epochs[:len(values)], values, 'o-', alpha=0.6, 
                   label=f'{metric_name} (raw)')
            
            if len(values) > 5:
                smooth_values = self._smooth_data(np.array(values), window_size=3)
                ax.plot(epochs[1:len(smooth_values)+1], smooth_values, '-', 
                       linewidth=2, label=f'{metric_name} (smooth)')
            
            # Add trend line
            if len(values) > 3:
                self._add_trend_line(ax, np.array(epochs[:len(values)]), 
                                   np.array(values), order=1)
            
            self._apply_styling(
                ax,
                title=f'{metric_name} Evolution',
                xlabel='Epoch',
                ylabel=metric_name,
                show_legend=True
            )
        
        # Hide unused subplots
        for idx in range(len(metrics_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Performance Evolution Over Training', fontsize=16)
        plt.tight_layout()
        
        plots["evolution"] = (fig, axes)
        
        if save_path:
            self._save_plot(fig, save_path, filename)
        
        return plots


class SweepResultsPlotter(TimeSeriesPlotter):
    """Plotter for hyperparameter sweep results."""

    def plot_sweep_results(
        self,
        results: List[Dict[str, Any]],
        x_param: str,
        y_metric: str,
        groupby: Optional[str] = None,
        save_path: Optional[str] = None,
        filename: str = "sweep_results",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot sweep results with optional grouping."""
        if not self._validate_data(results, "sweep results"):
            return self._create_error_plot("Sweep Results", "No data")
        
        fig, ax = self._create_figure(figsize=(12, 8))
        
        # Extract data
        if groupby:
            # Group results
            groups = {}
            for result in results:
                group_val = result.get(groupby, "unknown")
                if group_val not in groups:
                    groups[group_val] = {"x": [], "y": []}
                groups[group_val]["x"].append(result.get(x_param, 0))
                groups[group_val]["y"].append(result.get(y_metric, 0))
            
            # Plot each group
            for group_name, data in groups.items():
                ax.scatter(data["x"], data["y"], label=f"{groupby}={group_name}", 
                          s=100, alpha=self.config.alpha["main"])
                
                # Add trend line
                if len(data["x"]) > 2:
                    x_arr = np.array(data["x"])
                    y_arr = np.array(data["y"])
                    z = np.polyfit(x_arr, y_arr, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(x_arr.min(), x_arr.max(), 100)
                    ax.plot(x_trend, p(x_trend), '--', alpha=0.5)
        else:
            # Single series
            x_values = [r.get(x_param, 0) for r in results]
            y_values = [r.get(y_metric, 0) for r in results]
            
            ax.scatter(x_values, y_values, s=100, alpha=self.config.alpha["main"],
                      color=self.config.colors["primary"])
            
            # Add trend line
            if len(x_values) > 2:
                self._add_trend_line(ax, np.array(x_values), np.array(y_values))
        
        self._apply_styling(
            ax,
            title=f"{y_metric} vs {x_param}",
            xlabel=x_param,
            ylabel=y_metric,
            show_legend=bool(groupby)
        )
        
        if save_path:
            self._save_plot(fig, save_path, filename)
        
        return fig, ax

    def plot_branch_sweep_results(
        self,
        branch_factors: List[int],
        accuracies: List[float],
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot branch factor sweep results."""
        if not self._validate_data(branch_factors, "branch factors"):
            return self._create_error_plot("Branch Sweep", "No data")
        
        fig, ax = self._create_figure()
        
        # Main plot
        ax.plot(branch_factors, accuracies, 'o-', markersize=8, linewidth=2,
               color=self.config.colors["primary"])
        
        # Add value labels
        for bf, acc in zip(branch_factors, accuracies):
            ax.annotate(f'{acc:.3f}', (bf, acc), textcoords="offset points",
                       xytext=(0,10), ha='center')
        
        # Find and mark maximum
        max_idx = np.argmax(accuracies)
        ax.scatter(branch_factors[max_idx], accuracies[max_idx], 
                  color=self.config.colors["danger"], s=200, marker='*',
                  label=f'Best: {branch_factors[max_idx]}')
        
        self._apply_styling(
            ax,
            title="Branch Factor Sweep Results",
            xlabel="Branch Factor",
            ylabel="Test Accuracy",
            show_legend=True
        )
        
        if save_path:
            self._save_plot(fig, save_path, "branch_sweep_results")
        
        return fig, ax

    def plot_ei_sweep_results(
        self,
        ei_ratios: List[float],
        accuracies: List[float],
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot E/I ratio sweep results."""
        if not self._validate_data(ei_ratios, "E/I ratios"):
            return self._create_error_plot("E/I Sweep", "No data")
        
        fig, ax = self._create_figure()
        
        # Main plot
        ax.plot(ei_ratios, accuracies, 'o-', markersize=8, linewidth=2,
               color=self.config.colors["combined"])
        
        # Add value labels
        for ratio, acc in zip(ei_ratios, accuracies):
            ax.annotate(f'{acc:.3f}', (ratio, acc), textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=8)
        
        # Find and mark maximum
        max_idx = np.argmax(accuracies)
        ax.scatter(ei_ratios[max_idx], accuracies[max_idx], 
                  color=self.config.colors["danger"], s=200, marker='*',
                  label=f'Best: {ei_ratios[max_idx]:.2f}')
        
        self._apply_styling(
            ax,
            title="E/I Ratio Sweep Results",
            xlabel="E/I Ratio",
            ylabel="Test Accuracy",
            show_legend=True
        )
        ax.set_xscale('log')
        
        if save_path:
            self._save_plot(fig, save_path, "ei_sweep_results")
        
        return fig, ax

    def plot_noise_sweep_results(
        self,
        results: Dict[str, List[float]],
        param_values: List[float],
        param_name: str,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot noise sweep results for branch or E/I parameters."""
        if not self._validate_data(results, "noise sweep results"):
            return self._create_error_plot("Noise Sweep", "No data")
        
        fig, ax = self._create_figure(figsize=(12, 8))
        
        # Plot each noise level
        noise_levels = sorted(results.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(noise_levels)))
        
        for noise_level, color in zip(noise_levels, colors):
            if noise_level in results and len(results[noise_level]) == len(param_values):
                ax.plot(param_values, results[noise_level], 'o-', 
                       label=f'Noise={noise_level}', color=color,
                       markersize=6, linewidth=2)
        
        self._apply_styling(
            ax,
            title=f"{param_name} Sweep with Noise",
            xlabel=param_name,
            ylabel="Test Accuracy",
            show_legend=True
        )
        
        # Use log scale for E/I ratio
        if "E/I" in param_name:
            ax.set_xscale('log')
        
        if save_path:
            filename = f"noise_sweep_{param_name.lower().replace('/', '_').replace(' ', '_')}"
            self._save_plot(fig, save_path, filename)
        
        return fig, ax 
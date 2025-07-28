"""
Information-theoretic analysis plotting.

This module provides specialized plotting for information analysis results,
including mutual information, PID, and layer-wise analysis.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns

from .base import BasePlotter, PlotConfig


class InformationAnalysisPlotter(BasePlotter):
    """Comprehensive plotter for information-theoretic analysis results."""

    def plot_information_metrics(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        filename_prefix: str = "information",
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """
        Create comprehensive plots for information analysis results.

        Args:
            results: Dictionary containing information analysis results
            save_path: Directory to save plots
            filename_prefix: Prefix for saved files

        Returns:
            Dictionary mapping plot names to (figure, axes) tuples
        """
        plots = {}

        try:
            # Check format and route to appropriate method
            is_new_format = "basic_mi" in results or "pairwise_mi" in results
            
            if is_new_format:
                plots = self._plot_new_format(results, save_path, filename_prefix)
            else:
                plots = self._plot_old_format(results, save_path, filename_prefix)
                
        except Exception as e:
            self.logger.warning(f"Error in information analysis plotting: {e}")
            fig, ax = self._create_error_plot(
                "Information Analysis Error",
                str(e)[:50] + "..."
            )
            plots["error"] = (fig, ax)
            
            if save_path and filename_prefix:
                self._save_plot(fig, save_path, f"{filename_prefix}_error")
            
        return plots

    def _plot_new_format(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        filename_prefix: str = "information",
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """Plot information metrics in the new nested format."""
        plots = {}

        # Use GridSpec for comprehensive layout
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

        # 1. Basic MI plot
        if "basic_mi" in results:
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_basic_mi(ax1, results["basic_mi"])
            plots["basic_mi"] = (fig, ax1)

        # 2. Pairwise MI heatmap
        if "pairwise_mi" in results:
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_pairwise_mi_heatmap(ax2, results["pairwise_mi"])
            plots["pairwise_mi"] = (fig, ax2)

        # 3. Conditional MI
        if "conditional_mi" in results:
            ax3 = fig.add_subplot(gs[1, :2])
            self._plot_conditional_mi(ax3, results["conditional_mi"])
            plots["conditional_mi"] = (fig, ax3)

        # 4. PID results
        if "pid" in results:
            ax4 = fig.add_subplot(gs[1, 2:])
            self._plot_pid_decomposition(ax4, results["pid"])
            plots["pid"] = (fig, ax4)

        # 5. Layer statistics
        if "layer_statistics" in results:
            ax5 = fig.add_subplot(gs[2, :2])
            self._plot_layer_summary(ax5, results["layer_statistics"])
            plots["layer_stats"] = (fig, ax5)

        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_summary_text(ax6, results)
        plots["summary"] = (fig, ax6)

        plt.suptitle("Comprehensive Information Analysis", fontsize=18, y=0.98)
        
        if save_path:
            self._save_plot(fig, save_path, f"{filename_prefix}_comprehensive")
        
        return plots

    def _plot_basic_mi(self, ax: plt.Axes, basic_mi: Dict[str, float]):
        """Plot basic MI metrics."""
        # Extract non-normalized metrics
        metrics = {k: v for k, v in basic_mi.items() 
                  if not k.endswith("_normalized") and not k.endswith("_std")}
        
        if not metrics:
            ax.text(0.5, 0.5, 'No basic MI metrics found', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Basic Mutual Information')
            return
            
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        # Color mapping for different types
        colors = []
        for label in labels:
            if 'E' in label and 'I' not in label:
                colors.append(self.config.colors["excitatory"])
            elif 'I' in label and 'E' not in label:
                colors.append(self.config.colors["inhibitory"])
            else:
                colors.append(self.config.colors["combined"])
        
        bars = ax.bar(labels, values, color=colors, alpha=self.config.alpha["main"])
        
        # Add value labels
        self._add_value_labels(ax, bars, values)
        
        # Apply styling
        self._apply_styling(
            ax,
            title="Basic Mutual Information with Class Labels",
            ylabel="Mutual Information (bits)",
            rotate_xticks=True
        )

    def _plot_pairwise_mi_heatmap(self, ax: plt.Axes, pairwise_mi: Dict[str, float]):
        """Plot pairwise MI as a heatmap."""
        # Determine variables present
        variables = ["E", "I", "Vout"]
        if any("Vinf" in key for key in pairwise_mi.keys()):
            variables.append("Vinf")

        n_vars = len(variables)
        mi_matrix = np.zeros((n_vars, n_vars))

        # Fill matrix
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    # Try different key formats
                    for key_format in [f"I({var1};{var2})", f"I({var2};{var1})"]:
                        if key_format in pairwise_mi:
                            mi_matrix[i, j] = pairwise_mi[key_format]
                            break

        # Create heatmap
        sns.heatmap(
            mi_matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=variables,
            yticklabels=variables,
            ax=ax,
            cbar_kws={"label": "MI (bits)"},
            square=True
        )
        ax.set_title("Pairwise Mutual Information Matrix", 
                    fontsize=self.config.fontsize["title"])

    def _plot_conditional_mi(self, ax: plt.Axes, conditional_mi: Dict[str, float]):
        """Plot conditional MI metrics."""
        metrics = {k: v for k, v in conditional_mi.items() if not k.endswith("_std")}
        
        if not metrics:
            ax.text(0.5, 0.5, 'No conditional MI metrics found', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Conditional Mutual Information')
            return
            
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(labels, values, color=self.config.colors["combined"], 
                      alpha=self.config.alpha["main"])
        
        self._add_value_labels(ax, bars, values)
        self._apply_styling(
            ax,
            title="Conditional Mutual Information",
            ylabel="Conditional MI (bits)",
            rotate_xticks=True
        )

    def _plot_pid_decomposition(self, ax: plt.Axes, pid_results: Dict[str, Any]):
        """Plot PID (Partial Information Decomposition) results."""
        # Find PID data
        pid_data = None
        for key, value in pid_results.items():
            if isinstance(value, dict) and any(
                comp in value for comp in ["synergy", "redundancy", "unique"]
            ):
                pid_data = value
                break
        
        if not pid_data:
            ax.text(0.5, 0.5, 'No PID data found', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PID: Information Decomposition')
            return
        
        # Extract components
        component_map = {
            "synergy": ("Synergy", self.config.colors["danger"]),
            "redundancy": ("Redundancy", self.config.colors["info"]),
            "unique_E": ("Unique E", self.config.colors["excitatory"]),
            "unique_I": ("Unique I", self.config.colors["inhibitory"]),
            "unique_1": ("Unique X1", self.config.colors["warning"]),
            "unique_2": ("Unique X2", self.config.colors["success"]),
        }
        
        components = []
        values = []
        colors = []
        
        for key, (label, color) in component_map.items():
            if key in pid_data:
                components.append(label)
                values.append(pid_data[key])
                colors.append(color)
        
        if components:
            bars = ax.bar(components, values, color=colors, alpha=self.config.alpha["main"])
            self._add_value_labels(ax, bars, values)
            self._apply_styling(
                ax,
                title="PID: Information Decomposition",
                ylabel="Information (bits)",
                rotate_xticks=True
            )

    def _plot_layer_summary(self, ax: plt.Axes, layer_stats: Dict[str, Dict[str, float]]):
        """Plot layer statistics summary."""
        if not layer_stats:
            ax.text(0.5, 0.5, 'No layer statistics available', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Layer Information Summary')
            return
        
        # Convert layer names
        layer_names = list(layer_stats.keys())
        converted_names = self._convert_layer_names(layer_names)
        
        # Extract I(Vout;C) for each layer
        vout_values = []
        for name in layer_names:
            vout_values.append(layer_stats[name].get("I(Vout;C)_mean", 0))
        
        if any(v > 0 for v in vout_values):
            ax.plot(converted_names, vout_values, 'o-', linewidth=3, markersize=8,
                   color=self.config.colors["combined"])
            self._apply_styling(
                ax,
                title="Information Flow Across Layers",
                xlabel="Layer",
                ylabel="I(Vout;C) (bits)",
                rotate_xticks=True
            )

    def _plot_summary_text(self, ax: plt.Axes, results: Dict[str, Any]):
        """Add summary statistics text."""
        summary_lines = []
        
        # Basic MI summary
        if "basic_mi" in results:
            summary_lines.append("Basic MI:")
            for key in ["I(E;C)", "I(I;C)", "I(Vout;C)", "I(E,I;C)"]:
                if key in results["basic_mi"]:
                    summary_lines.append(f"  {key}: {results['basic_mi'][key]:.4f} bits")
        
        # Layer count
        if "layer_statistics" in results:
            n_layers = len(results["layer_statistics"])
            summary_lines.append(f"\nLayers analyzed: {n_layers}")
        
        # Accuracy if available
        if "accuracy" in results:
            summary_lines.append(f"\nNetwork Accuracy: {results['accuracy']:.3f}")
        
        if summary_lines:
            ax.text(0.05, 0.95, "\n".join(summary_lines), 
                   transform=ax.transAxes, 
                   verticalalignment="top", 
                   fontfamily="monospace", 
                   fontsize=self.config.fontsize["annotation"])
        
        ax.axis("off")
        ax.set_title("Summary Statistics", fontsize=self.config.fontsize["title"])

    def _convert_layer_names(self, layer_names: List[str]) -> List[str]:
        """Convert technical layer names to intuitive format."""
        # Extract and sort layer indices
        indexed_layers = []
        for name in layer_names:
            try:
                if "layers." in name:
                    parts = name.split("layers.")
                    if len(parts) > 1:
                        layer_idx = int(parts[1].split(".")[0])
                        indexed_layers.append((layer_idx, name))
                    else:
                        indexed_layers.append((999, name))
                else:
                    indexed_layers.append((999, name))
            except:
                indexed_layers.append((999, name))
        
        # Sort by index (reverse for soma → distal ordering)
        indexed_layers.sort(key=lambda x: x[0], reverse=True)
        
        # Create intuitive names
        converted = []
        for i, (_, _) in enumerate(indexed_layers):
            if i == 0:
                converted.append("Soma")
            elif i == len(indexed_layers) - 1:
                converted.append("Distal")
            else:
                converted.append(f"Layer {i}")
        
        return converted

    def plot_per_layer_information(
        self, 
        layer_results: Dict[str, Dict[str, Any]], 
        **kwargs
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """Plot detailed per-layer information with mean ± variance."""
        save_path = kwargs.get('save_path')
        filename_prefix = kwargs.get('filename_prefix', 'per_layer_info')
        
        if not self._validate_data(layer_results, "layer results"):
            return None
        
        # Convert layer names
        layer_names = list(layer_results.keys())
        converted_names = self._convert_layer_names(layer_names)
        
        # Update results with converted names
        converted_results = {}
        for conv_name, orig_name in zip(converted_names, 
                                      [x[1] for x in sorted(
                                          [(self._get_layer_index(n), n) for n in layer_names],
                                          key=lambda x: x[0], reverse=True
                                      )]):
            converted_results[conv_name] = layer_results[orig_name]
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Basic MI per layer
        self._plot_layer_basic_mi(axes[0, 0], converted_results, converted_names)
        
        # Plot 2: Pairwise MI per layer
        self._plot_layer_pairwise_mi(axes[0, 1], converted_results, converted_names)
        
        # Plot 3: Information flow
        self._plot_layer_info_flow(axes[1, 0], converted_results, converted_names)
        
        # Plot 4: Summary heatmap
        self._plot_layer_heatmap(axes[1, 1], converted_results, converted_names)
        
        plt.suptitle('Per-Layer Information Analysis (Soma → Distal)', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path, f"{filename_prefix}_detailed")
        
        return fig, axes

    def _get_layer_index(self, layer_name: str) -> int:
        """Extract layer index from name."""
        try:
            if "layers." in layer_name:
                return int(layer_name.split("layers.")[1].split(".")[0])
        except:
            pass
        return 999

    def _plot_layer_basic_mi(
        self, 
        ax: plt.Axes, 
        layer_results: Dict[str, Dict[str, float]], 
        layer_names: List[str]
    ):
        """Plot basic MI metrics across layers with error bars."""
        metrics = ["I(E;C)_mean", "I(I;C)_mean", "I(Vout;C)_mean"]
        labels = ["I(E;C)", "I(I;C)", "I(Vout;C)"]
        colors = [
            self.config.colors["excitatory"],
            self.config.colors["inhibitory"],
            self.config.colors["combined"]
        ]
        
        x = np.arange(len(layer_names))
        width = 0.25
        
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            values = []
            errors = []
            
            for layer_name in layer_names:
                values.append(layer_results[layer_name].get(metric, 0))
                std_key = metric.replace("_mean", "_std")
                errors.append(layer_results[layer_name].get(std_key, 0))
            
            bars = ax.bar(x + i * width, values, width, yerr=errors,
                          label=label, color=color, alpha=self.config.alpha["main"],
                          capsize=4, elinewidth=1.5)
            
            # Add value labels
            for bar, val, err in zip(bars, values, errors):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           bar.get_height() + err + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', 
                           fontsize=8, rotation=0)
        
        self._apply_styling(
            ax,
            title="Basic MI Across Layers (Mean ± Std)",
            xlabel="Layer (Soma → Distal)",
            ylabel="MI (bits)",
            show_legend=True,
            rotate_xticks=True
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(layer_names)

    def _plot_layer_pairwise_mi(
        self, 
        ax: plt.Axes, 
        layer_results: Dict[str, Dict[str, float]], 
        layer_names: List[str]
    ):
        """Plot pairwise MI across layers."""
        pairwise_metrics = ["I(E;I)_mean", "I(E;Vout)_mean", "I(I;Vout)_mean"]
        pairwise_labels = ["I(E;I)", "I(E;Vout)", "I(I;Vout)"]
        
        x_pos = np.arange(len(pairwise_metrics))
        
        for layer_idx, layer_name in enumerate(layer_names):
            values = []
            errors = []
            
            for metric in pairwise_metrics:
                values.append(layer_results[layer_name].get(metric, 0))
                std_key = metric.replace("_mean", "_std")
                errors.append(layer_results[layer_name].get(std_key, 0))
            
            x_offset = x_pos + (layer_idx - len(layer_names)/2) * 0.1
            ax.errorbar(x_offset, values, yerr=errors, marker='o', 
                       label=layer_name, linewidth=2, markersize=6, 
                       capsize=4, capthick=1.5)
        
        self._apply_styling(
            ax,
            title="Pairwise MI Across Layers (Mean ± Std)",
            xlabel="Pairwise MI Type",
            ylabel="MI (bits)",
            show_legend=True
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pairwise_labels, rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_layer_info_flow(
        self, 
        ax: plt.Axes, 
        layer_results: Dict[str, Dict[str, float]], 
        layer_names: List[str]
    ):
        """Plot information flow across layers."""
        layer_indices = np.arange(len(layer_names))
        vout_values = []
        vout_errors = []
        
        for layer_name in layer_names:
            vout_values.append(layer_results[layer_name].get("I(Vout;C)_mean", 0))
            vout_errors.append(layer_results[layer_name].get("I(Vout;C)_std", 0))
        
        vout_values = np.array(vout_values)
        vout_errors = np.array(vout_errors)
        
        # Main line with error region
        ax.errorbar(layer_indices, vout_values, yerr=vout_errors, 
                   marker='o', linewidth=3, markersize=8, 
                   color=self.config.colors["combined"],
                   capsize=6, capthick=2, label='I(Vout;C)')
        
        ax.fill_between(layer_indices, 
                       vout_values - vout_errors,
                       vout_values + vout_errors,
                       alpha=self.config.alpha["fill"], 
                       color=self.config.colors["combined"])
        
        # Annotations
        for i, (val, err) in enumerate(zip(vout_values, vout_errors)):
            if val > 0:
                ax.annotate(f'{val:.3f}±{err:.3f}', 
                           (layer_indices[i], val + err + 0.02),
                           ha='center', va='bottom', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor="white", 
                                   alpha=self.config.alpha["annotation"]))
        
        self._apply_styling(
            ax,
            title="Information Flow Across Layers",
            xlabel="Layer Position (Soma → Distal)",
            ylabel="I(Vout;C) (bits)",
            show_legend=True,
            rotate_xticks=True
        )
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(layer_names)

    def _plot_layer_heatmap(
        self, 
        ax: plt.Axes, 
        layer_results: Dict[str, Dict[str, float]], 
        layer_names: List[str]
    ):
        """Plot heatmap of information metrics across layers."""
        metrics = ["I(E;C)_mean", "I(I;C)_mean", "I(Vout;C)_mean", "I(E;I)_mean"]
        metric_labels = ["I(E;C)", "I(I;C)", "I(Vout;C)", "I(E;I)"]
        
        # Create data matrices
        mean_matrix = []
        std_matrix = []
        
        for metric in metrics:
            mean_row = []
            std_row = []
            for layer_name in layer_names:
                mean_row.append(layer_results[layer_name].get(metric, 0))
                std_key = metric.replace("_mean", "_std")
                std_row.append(layer_results[layer_name].get(std_key, 0))
            mean_matrix.append(mean_row)
            std_matrix.append(std_row)
        
        mean_matrix = np.array(mean_matrix)
        std_matrix = np.array(std_matrix)
        
        # Create heatmap
        im = ax.imshow(mean_matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(np.arange(len(layer_names)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(layer_names, rotation=45)
        ax.set_yticklabels(metric_labels)
        ax.set_title('Information Metrics Heatmap (Mean Values)', 
                    fontsize=self.config.fontsize["title"])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='MI (bits)')
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(layer_names)):
                mean_val = mean_matrix[i, j]
                std_val = std_matrix[i, j]
                if mean_val > 0:
                    text = f'{mean_val:.3f}\n±{std_val:.3f}'
                    text_color = "white" if mean_val > np.max(mean_matrix) * 0.6 else "black"
                    ax.text(j, i, text, ha="center", va="center", 
                           color=text_color, fontsize=8, weight='bold')

    def _plot_old_format(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        filename_prefix: str = "information",
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """Plot information metrics in the old flat format."""
        plots = {}
        
        fig, ax = self._create_figure()
        
        # Extract metrics if they exist
        metrics_to_plot = {}
        for key in ['I_E_C', 'I_I_C', 'I_V_C', 'I_E_I', 'I_E_V', 'I_I_V', 'H_C']:
            if key in results:
                if isinstance(results[key], (list, np.ndarray)):
                    metrics_to_plot[key] = np.mean(results[key])
                else:
                    metrics_to_plot[key] = results[key]
        
        if metrics_to_plot:
            keys = list(metrics_to_plot.keys())
            values = list(metrics_to_plot.values())
            
            # Color code different types
            colors = []
            for key in keys:
                if 'E' in key and 'I' not in key and 'V' not in key:
                    colors.append(self.config.colors["excitatory"])
                elif 'I' in key and 'E' not in key and 'V' not in key:
                    colors.append(self.config.colors["inhibitory"])
                elif 'V' in key:
                    colors.append(self.config.colors["combined"])
                else:
                    colors.append(self.config.colors["primary"])
            
            bars = ax.bar(keys, values, color=colors, alpha=self.config.alpha["main"])
            self._add_value_labels(ax, bars, values)
            
            self._apply_styling(
                ax,
                title="Information Analysis Summary",
                ylabel="Information (bits)",
                rotate_xticks=True
            )
        else:
            ax.text(0.5, 0.5, 'Information Analysis Results\n(No plottable metrics found)', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Information Analysis')
        
        plots['summary'] = (fig, ax)
        
        if save_path:
            self._save_plot(fig, save_path, f"{filename_prefix}_summary")
            
        return plots

    def plot_correlation_vs_information(
        self,
        correlation_data: Dict[str, np.ndarray],
        information_data: Dict[str, float],
        save_path: Optional[str] = None,
        filename_prefix: str = "correlation_info",
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """Plot relationship between correlation structure and information content."""
        fig, (ax1, ax2) = self._create_figure(1, 2, figsize=(15, 6))

        # Extract correlation statistics
        layers = []
        mean_corrs = []
        info_values = []

        for layer_name in correlation_data.keys():
            if layer_name in information_data:
                corr_matrix = correlation_data[layer_name]
                # Get upper triangle (excluding diagonal)
                upper_tri = np.triu(corr_matrix, k=1)
                mean_corr = np.mean(np.abs(upper_tri[upper_tri != 0]))

                layers.append(layer_name)
                mean_corrs.append(mean_corr)
                info_values.append(information_data[layer_name])

        if layers:
            # Plot 1: Mean correlation by layer
            bars = ax1.bar(layers, mean_corrs, alpha=self.config.alpha["main"], 
                          color=self.config.colors["primary"])
            self._add_value_labels(ax1, bars, mean_corrs)
            self._apply_styling(
                ax1,
                title="Average Correlation Strength by Layer",
                xlabel="Layer",
                ylabel="Mean Absolute Correlation",
                rotate_xticks=True
            )

            # Plot 2: Correlation vs Information scatter
            ax2.scatter(mean_corrs, info_values, s=100, 
                       alpha=self.config.alpha["main"], 
                       color=self.config.colors["combined"])
            
            for i, layer in enumerate(layers):
                ax2.annotate(
                    layer,
                    (mean_corrs[i], info_values[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9
                )

            # Add trend line if enough points
            if len(mean_corrs) > 2:
                z = np.polyfit(mean_corrs, info_values, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(mean_corrs), max(mean_corrs), 100)
                ax2.plot(
                    x_trend,
                    p(x_trend),
                    "r--",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}",
                )
                ax2.legend()
            
            self._apply_styling(
                ax2,
                title="Correlation vs Information Content",
                xlabel="Mean Absolute Correlation",
                ylabel="Mutual Information (bits)"
            )
        else:
            # No data case
            for ax, title in zip([ax1, ax2], ["Mean Correlations", "Correlation vs Information"]):
                ax.text(0.5, 0.5, 'No correlation data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)

        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path, f"{filename_prefix}_correlation_analysis")
        
        return fig, (ax1, ax2) 
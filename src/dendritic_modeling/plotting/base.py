import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from abc import ABC, abstractmethod
from typing import , , , Optional, Any, Union
import seaborn as sns

class PlotConfig:
    """Configuration for plot appearance and behavior."""

    def __init__(
        self,
        figsize: tuple[float, float] = (10, 6),
        dpi: int = 300,
        style: str = "standard",
        fontsize: dict[str, int] = None,
        colors: dict[str, str] = None,
        save_formats: list[str] = ["png"],
        tight_layout: bool = True,
        grid: bool = True,
        grid_alpha: float = 0.3,
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.fontsize = fontsize or {
            "title": 16,
            "label": 14,
            "tick": 12,
            "legend": 12,
        }
        self.colors = colors or {
            "excitatory": "#e74c3c",
            "inhibitory": "#3498db",
            "combined": "#9b59b6",
            "baseline": "#2c3e50",
            "both": "#9b59b6",
            "neither": "#95a5a6",
        }
        self.save_formats = save_formats
        self.tight_layout = tight_layout
        self.grid = grid
        self.grid_alpha = grid_alpha

class BasePlotter(ABC):
    """Abstract base class for all plotters."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self._setup_style()

    def _setup_style(self):
        """Apply the configured plotting style."""
        if self.config.style == "publication":
            plt.style.use("seaborn-v0_8-paper")
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.size"] = 10
            plt.rcParams["axes.linewidth"] = 0.5
        elif self.config.style == "presentation":
            plt.style.use("seaborn-v0_8-talk")
            plt.rcParams["font.size"] = 14
            plt.rcParams["axes.linewidth"] = 2
        else:  # standard
            plt.style.use("seaborn-v0_8-whitegrid")

    @abstractmethod
    def plot(self, data: Any, **kwargs) -> plt.Figure:
        """Create the plot. Must be implemented by subclasses."""
        pass

    def save(self, fig: plt.Figure, filepath: str, metadata: dict[str, Any] = None):
        """Save the figure with optional metadata."""
        # Save in requested formats
        base_path = os.path.splitext(filepath)[0]
        for fmt in self.config.save_formats:
            save_path = f"{base_path}.{fmt}"
            if self.config.tight_layout:
                fig.tight_layout()
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        # Save metadata if provided
        if metadata:
            import json
            meta_path = f"{base_path}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def format_axis(self, ax: plt.Axes, xlabel: str = None, ylabel: str = None,
                    title: str = None, xlim: tuple = None, ylim: tuple = None):
        """Apply consistent formatting to an axis."""
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.config.fontsize["label"])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.config.fontsize["label"])
        if title:
            ax.set_title(title, fontsize=self.config.fontsize["title"])

        ax.tick_params(labelsize=self.config.fontsize["tick"])

        if self.config.grid:
            ax.grid(True, alpha=self.config.grid_alpha)

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

class LinePlotter(BasePlotter):
    """Plotter for line plots with multiple series."""

    def plot(
        self,
        data: dict[str, list[float]],
        x_values: list[float] = None,
        xlabel: str = "X",
        ylabel: str = "Y",
        title: str = "Line Plot",
        markers: dict[str, str] = None,
        baseline: float = None,
        baseline_label: str = "Baseline",
        xlim: tuple = None,
        ylim: tuple = None,
        ax: plt.Axes = None,
    ) -> Union[plt.Figure, plt.Axes]:
        """
        Create a line plot with multiple series.

        Args:
            data: Dictionary mapping series names to y-values
            x_values: X-axis values (if None, uses indices)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            markers: Dictionary mapping series names to marker styles
            baseline: Optional baseline value to plot as horizontal line
            baseline_label: Label for baseline line
            xlim: X-axis limits
            ylim: Y-axis limits
            ax: Existing axes to plot on (if None, creates new figure)

        Returns:
            Figure if ax is None, otherwise the axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize)
            return_fig = True
        else:
            fig = ax.figure
            return_fig = False

        if x_values is None:
            x_values = list(range(len(next(iter(data.values())))))

        # Default markers
        default_markers = ["o", "s", "^", "v", "D", "*", "p", "h"]
        if markers is None:
            markers = {}

        # Plot each series
        for i, (name, y_values) in enumerate(data.items()):
            color = self.config.colors.get(name.lower(), f"C{i}")
            marker = markers.get(name, default_markers[i % len(default_markers)])

            ax.plot(
                x_values, y_values,
                marker=marker,
                label=name,
                linewidth=3,
                markersize=10,
                color=color,
                alpha=0.8,
            )

        # Add baseline if provided
        if baseline is not None:
            ax.axhline(
                y=baseline,
                color=self.config.colors.get("baseline", "k"),
                linestyle="--",
                linewidth=2,
                label=f"{baseline_label} ({baseline:.3f})"
            )

        # Format the axis
        self.format_axis(ax, xlabel, ylabel, title, xlim, ylim)
        ax.legend(fontsize=self.config.fontsize["legend"])

        return fig if return_fig else ax

class BarPlotter(BasePlotter):
    """Plotter for grouped bar plots."""

    def plot(
        self,
        data: dict[str, list[float]],
        x_labels: list[str],
        xlabel: str = "Category",
        ylabel: str = "Value",
        title: str = "Bar Plot",
        errors: dict[str, list[float]] = None,
        bar_width: float = 0.25,
        show_values: bool = False,
        xlim: tuple = None,
        ylim: tuple = None,
        ax: plt.Axes = None,
    ) -> Union[plt.Figure, plt.Axes]:
        """
        Create a grouped bar plot.

        Args:
            data: Dictionary mapping group names to values
            x_labels: Labels for x-axis categories
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            errors: Dictionary mapping group names to error values
            bar_width: Width of individual bars
            show_values: Whether to show values on top of bars
            xlim: X-axis limits
            ylim: Y-axis limits
            ax: Existing axes to plot on

        Returns:
            Figure if ax is None, otherwise the axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize)
            return_fig = True
        else:
            fig = ax.figure
            return_fig = False

        x = np.arange(len(x_labels))
        n_groups = len(data)

        # Calculate bar positions
        offsets = np.linspace(
            -(n_groups-1)*bar_width/2,
            (n_groups-1)*bar_width/2,
            n_groups
        )

        # Plot each group
        for i, (name, values) in enumerate(data.items()):
            color = self.config.colors.get(name.lower(), f"C{i}")
            error_vals = errors.get(name) if errors else None

            bars = ax.bar(
                x + offsets[i],
                values,
                bar_width,
                label=name,
                color=color,
                alpha=0.7,
                yerr=error_vals,
                capsize=5 if error_vals else 0,
            )

            # Add value labels if requested
            if show_values:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        height,
                        f'{height:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=self.config.fontsize["tick"] - 2
                    )

        # Format the axis
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        self.format_axis(ax, xlabel, ylabel, title, xlim, ylim)
        ax.legend(fontsize=self.config.fontsize["legend"])

        return fig if return_fig else ax

class BoxPlotter(BasePlotter):
    """Plotter for box plots."""

    def plot(
        self,
        data: list[list[float]],
        labels: list[str],
        group_labels: list[str] = None,
        xlabel: str = "Category",
        ylabel: str = "Value",
        title: str = "Box Plot",
        positions: list[float] = None,
        widths: float = 0.5,
        colors: list[str] = None,
        xlim: tuple = None,
        ylim: tuple = None,
        ax: plt.Axes = None,
    ) -> Union[plt.Figure, plt.Axes]:
        """
        Create a box plot.

        Args:
            data: list of data arrays for each box
            labels: Labels for each box
            group_labels: Labels for groups of boxes
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            positions: Custom positions for boxes
            widths: Width of boxes
            colors: Colors for each box
            xlim: X-axis limits
            ylim: Y-axis limits
            ax: Existing axes to plot on

        Returns:
            Figure if ax is None, otherwise the axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize)
            return_fig = True
        else:
            fig = ax.figure
            return_fig = False

        if positions is None:
            positions = list(range(len(data)))

        # Create box plot
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=widths,
            patch_artist=True,
            labels=labels if not group_labels else None,
        )

        # Color the boxes
        if colors:
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        # Set group labels if provided
        if group_labels:
            ax.set_xticks(positions)
            ax.set_xticklabels(group_labels)

        # Format the axis
        self.format_axis(ax, xlabel, ylabel, title, xlim, ylim)

        # Add custom legend if colors provided
        if colors and labels:
            legend_elements = [
                mpatches.Patch(facecolor=color, alpha=0.7, label=label)
                for color, label in zip(colors, labels)
            ]
            ax.legend(handles=legend_elements, fontsize=self.config.fontsize["legend"])

        return fig if return_fig else ax

class ComparisonPlotter(BasePlotter):
    """Plotter for side-by-side comparison plots."""

    def create_subplots(
        self,
        n_plots: int,
        n_cols: int = 2,
        subplot_size: tuple[float, float] = (10, 6),
        **subplot_kwargs
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Create a figure with multiple subplots.

        Args:
            n_plots: Number of subplots needed
            n_cols: Number of columns
            subplot_size: Size of each subplot
            **subplot_kwargs: Additional arguments for subplots

        Returns:
            Figure and list of axes
        """
        n_rows = (n_plots + n_cols - 1) // n_cols
        figsize = (subplot_size[0] * n_cols, subplot_size[1] * n_rows)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=figsize,
            **subplot_kwargs
        )

        # Flatten axes array and remove extra
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        # Remove extra subplots
        for i in range(n_plots, len(axes)):
            fig.delaxes(axes[i])

        return fig, axes[:n_plots]

    def plot(self, plotters_and_data: list[tuple[BasePlotter, Dict]], **kwargs):
        """
        Create comparison plots using multiple plotters.

        Args:
            plotters_and_data: list of (plotter, data_dict) tuples
            **kwargs: Arguments passed to create_subplots

        Returns:
            Figure with comparison plots
        """
        n_plots = len(plotters_and_data)
        fig, axes = self.create_subplots(n_plots, **kwargs)

        for ax, (plotter, data_dict) in zip(axes, plotters_and_data):
            plotter.plot(ax=ax, **data_dict)

        if self.config.tight_layout:
            fig.tight_layout()

        return fig

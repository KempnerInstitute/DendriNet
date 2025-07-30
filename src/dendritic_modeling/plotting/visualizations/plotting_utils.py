"""Shared utilities for plotting functions."""

import logging
import os
from typing import Optional, Any, Tuple, Dict, List
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def save_plot(fig: plt.Figure, save_path: str, filename: str, dpi: int = 300) -> None:
    """Save plot to file with error handling."""
    try:
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, f"{filename}.png")
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")
    except Exception as e:
        logger.warning(f"Error saving plot: {e}")

def create_figure(figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure and axis with standard settings."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def create_subplots(rows: int, cols: int, figsize: Tuple[int, int] = (15, 10)) -> Tuple[plt.Figure, np.ndarray]:
    """Create subplots with standard settings."""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    return fig, axes

def add_error_bars(ax: plt.Axes, x, y, yerr, color: str, label: str, capsize: int = 4, capthick: float = 1.5):
    """Add error bars to a plot with standard styling."""
    return ax.errorbar(x, y, yerr=yerr, marker='o', linewidth=2, color=color, 
                      label=label, capsize=capsize, capthick=capthick)

def annotate_bars(ax: plt.Axes, bars, values, errors=None, offset: float = 0.01, fontsize: int = 8):
    """Annotate bars with values and optional errors."""
    for i, (bar, val) in enumerate(zip(bars, values)):
        if errors is not None and len(errors) > i:
            err = errors[i]
            text = f'{val:.3f}±{err:.3f}'
        else:
            text = f'{val:.3f}'
        
        height = bar.get_height() if hasattr(bar, 'get_height') else val
        x_pos = bar.get_x() + bar.get_width()/2. if hasattr(bar, 'get_x') else i
        
        ax.text(x_pos, height + offset, text, ha='center', va='bottom', fontsize=fontsize)

def setup_basic_plot(ax: plt.Axes, title: str, xlabel: str, ylabel: str, 
                    fontsize_dict: Dict[str, int] = None, grid: bool = True):
    """Set up basic plot formatting."""
    if fontsize_dict is None:
        fontsize_dict = {"title": 16, "label": 14}
    
    ax.set_title(title, fontsize=fontsize_dict.get("title", 16))
    ax.set_xlabel(xlabel, fontsize=fontsize_dict.get("label", 14))
    ax.set_ylabel(ylabel, fontsize=fontsize_dict.get("label", 14))
    
    if grid:
        ax.grid(True, alpha=0.3, axis='y')

def convert_layer_names(layer_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert technical layer names to intuitive format (soma → distal)."""
    if not layer_results:
        return {}
    
    layer_names = list(layer_results.keys())
    indexed_layers = []
    
    for name in layer_names:
        try:
            if "layers." in name:
                parts = name.split("layers.")
                if len(parts) > 1:
                    layer_part = parts[1].split(".")[0]
                    if layer_part.isdigit():
                        layer_idx = int(layer_part)
                        indexed_layers.append((layer_idx, name))
                    else:
                        indexed_layers.append((999, name))
                else:
                    indexed_layers.append((999, name))
            else:
                indexed_layers.append((999, name))
        except:
            indexed_layers.append((999, name))
    
    # Sort by index (reverse order so highest index = soma, lowest = distal)
    indexed_layers.sort(key=lambda x: x[0], reverse=True)
    
    # Create new dictionary with intuitive names
    converted_results = {}
    for i, (original_idx, original_name) in enumerate(indexed_layers):
        if i == 0:
            new_name = "Soma"
        else:
            new_name = f"Layer {i}"
        
        converted_results[new_name] = layer_results[original_name]
    
    return converted_results

def get_color_scheme():
    """Get standard color scheme for consistency."""
    return {
        "excitatory": "#e74c3c",
        "inhibitory": "#3498db", 
        "combined": "#9b59b6",
        "primary": "#2c3e50",
        "baseline": "#2c3e50",
    }

def create_heatmap(ax: plt.Axes, data_matrix: np.ndarray, 
                  xlabels: List[str], ylabels: List[str],
                  title: str, cmap: str = 'viridis',
                  annotate: bool = True, fmt: str = '.3f'):
    """Create a standardized heatmap."""
    import seaborn as sns
    
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto')
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45)
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Value')
    
    # Add text annotations
    if annotate:
        for i in range(len(ylabels)):
            for j in range(len(xlabels)):
                val = data_matrix[i, j]
                if val > 0:
                    text_color = "white" if val > np.max(data_matrix) * 0.6 else "black"
                    ax.text(j, i, f'{val:{fmt}}', ha="center", va="center", 
                           color=text_color, fontsize=8, weight='bold')
    
    return im

def handle_plot_error(func_name: str, error: Exception, create_placeholder: bool = True):
    """Standard error handling for plotting functions."""
    logger.warning(f"Error in {func_name}: {error}")
    
    if create_placeholder:
        try:
            fig, ax = create_figure()
            ax.text(0.5, 0.5, f'{func_name}\n(Error: {str(error)[:50]}...)', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{func_name} - Error')
            return fig, ax
        except:
            return None
    return None 
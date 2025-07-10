"""
Classification analysis plotting utilities.

This module contains plotting functions for visualizing classification analysis results,
including layer-wise ablation and contribution analysis.
"""
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

def plot_layerwise_ablation(
    results: dict[str, list[float]],
    save_dir: Optional[str] = None,
    filename: str = "layerwise_ablation"
) -> plt.Figure:
    """
    Plot layer-wise ablation analysis results.

    Args:
        results: Dictionary containing layer_idx, baseline_acc, inh_ablated_acc, exc_ablated_acc
        save_dir: Directory to save plot
        filename: Filename for saved plot

    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(10, 6))

    plt.plot(results["layer_idx"], results["baseline_acc"], "o-", label="Baseline")
    plt.plot(
        results["layer_idx"],
        results["inh_ablated_acc"],
        "s-",
        label="Inhibitory Ablated",
    )
    plt.plot(
        results["layer_idx"],
        results["exc_ablated_acc"],
        "^-",
        label="Excitatory Ablated",
    )

    plt.xlabel("Layer Index (higher = closer to input)")
    plt.ylabel("Classification Accuracy")
    plt.title("Layer-wise Ablation Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300)
        plt.close()

    return fig

def plot_layerwise_contributions(
    results: dict[str, list[float]],
    save_dir: Optional[str] = None,
    filename: str = "layerwise_contributions"
) -> plt.Figure:
    """
    Plot layer-wise contribution analysis showing accuracy drops.

    Args:
        results: Dictionary containing layer_idx, baseline_acc, inh_ablated_acc, exc_ablated_acc
        save_dir: Directory to save plot
        filename: Filename for saved plot

    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(10, 6))

    # Calculate differences
    inh_diff = [
        baseline - inh_ablated
        for baseline, inh_ablated in zip(
            results["baseline_acc"], results["inh_ablated_acc"]
        )
    ]
    exc_diff = [
        baseline - exc_ablated
        for baseline, exc_ablated in zip(
            results["baseline_acc"], results["exc_ablated_acc"]
        )
    ]

    x = np.array(results["layer_idx"])
    width = 0.35

    plt.bar(
        x - width / 2,
        inh_diff,
        width,
        label="Inhibitory Contribution",
        color="blue",
        alpha=0.7,
    )
    plt.bar(
        x + width / 2,
        exc_diff,
        width,
        label="Excitatory Contribution",
        color="red",
        alpha=0.7,
    )

    plt.xlabel("Layer Index (higher = closer to input)")
    plt.ylabel("Accuracy Drop when Ablated")
    plt.title("Contribution of Inhibitory and Excitatory Inputs per Layer")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300)
        plt.close()

    return fig

def plot_activation_decoding(
    layer_accuracies: dict[int, float],
    save_path: Optional[str] = None,
    filename: str = "activation_decoding"
) -> plt.Figure:
    """
    Plot decoding accuracy from different layer activations.

    Args:
        layer_accuracies: Dictionary mapping layer index to decoding accuracy
        save_path: Path to save plot
        filename: Filename for saved plot

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = sorted(layer_accuracies.keys())
    accuracies = [layer_accuracies[layer] for layer in layers]

    ax.plot(layers, accuracies, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title('Classification Accuracy from Layer Activations')
    ax.grid(True, alpha=0.3)

    # Add value labels on points
    for layer, acc in zip(layers, accuracies):
        ax.annotate(f'{acc:.3f}', (layer, acc), textcoords="offset points",
                   xytext=(0,10), ha='center')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()

    return fig

def plot_ablation_comparison(
    normal_accuracies: dict[int, float],
    ablated_accuracies: dict[str, dict[int, float]],
    save_path: Optional[str] = None,
    filename: str = "ablation_comparison"
) -> plt.Figure:
    """
    Compare decoding accuracies under different ablation conditions.

    Args:
        normal_accuracies: Dictionary mapping layer to normal accuracy
        ablated_accuracies: Dictionary of ablation_type -> layer -> accuracy
        save_path: Path to save plot
        filename: Filename for saved plot

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    layers = sorted(normal_accuracies.keys())

    # Plot normal accuracies
    normal_acc = [normal_accuracies[layer] for layer in layers]
    ax.plot(layers, normal_acc, 'o-', linewidth=2, markersize=8, label='Normal')

    # Plot ablated accuracies
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for idx, (ablation_type, layer_acc) in enumerate(ablated_accuracies.items()):
        acc_values = [layer_acc.get(layer, 0) for layer in layers]
        ax.plot(layers, acc_values, 's--', linewidth=2, markersize=6,
               label=f'{ablation_type} ablated', color=colors[idx % len(colors)])

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title('Decoding Accuracy Under Different Ablation Conditions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()

    return fig

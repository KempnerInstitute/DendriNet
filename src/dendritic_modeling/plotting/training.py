"""
Training-related plotting functions.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curves(
    train_losses: list[float],
    valid_losses: list[float],
    loss_name: str,
    save_dir: Optional[str] = None,
    filename_prefix: str = "",
):
    """
    Plot the negative log-likelihood loss curves for training and validation.

    Args:
        Train_losses: list of training losses
        Valid_losses: list of validation losses
        num_epochs: Number of epochs
        save_dir: Directory to save the plot
    """
    # Convert to list if numpy array
    train_losses = list(train_losses)
    valid_losses = list(valid_losses)

    # Create the x-axis values (epochs)
    # Make sure length matches the loss arrays
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, train_losses, "0.4", label="training")

    if valid_losses:
        # Make sure valid losses have same length as train losses
        if len(valid_losses) != len(train_losses):
            print(
                f"Warning: Valid losses length ({len(valid_losses)}) != Train losses length ({len(train_losses)})"
            )
            # Use the shorter length
            min_len = min(len(valid_losses), len(train_losses))
            valid_losses = valid_losses[:min_len]
            epochs_valid = epochs[:min_len]
        else:
            epochs_valid = epochs

        plt.plot(epochs_valid, valid_losses, "r", label="validation")

    plt.grid()
    plt.title("Train and Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel(loss_name)
    plt.legend()

    # Save the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}loss_curves.png"))

    plt.close()


def plot_ei_sweep_results(results_csv_path, ei_ratio_range, split="test"):
    """Plot E/I ratio sweep results."""
    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific split
    split_df = results_df[results_df["split"] == split]

    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"E/I Ratio Sweep Results ({split} split)", fontsize=16)

    # Plot accuracy vs E/I ratio
    axes[0, 0].plot(split_df["ei_ratio"], split_df["accuracy"], "o-", color="blue")
    axes[0, 0].set_xlabel("E/I Ratio")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Accuracy vs E/I Ratio")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot loss vs E/I ratio
    axes[0, 1].plot(split_df["ei_ratio"], split_df["loss"], "o-", color="red")
    axes[0, 1].set_xlabel("E/I Ratio")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Loss vs E/I Ratio")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot precision vs E/I ratio
    axes[1, 0].plot(split_df["ei_ratio"], split_df["precision"], "o-", color="green")
    axes[1, 0].set_xlabel("E/I Ratio")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision vs E/I Ratio")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot recall vs E/I ratio
    axes[1, 1].plot(split_df["ei_ratio"], split_df["recall"], "o-", color="orange")
    axes[1, 1].set_xlabel("E/I Ratio")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Recall vs E/I Ratio")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(
        os.path.dirname(results_csv_path), f"ei_sweep_results_{split}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {save_path}")


def plot_branch_sweep_results(results_csv_path, split="test"):
    """Plot branch sweep results."""
    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific split
    split_df = results_df[results_df["split"] == split]

    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Branch Sweep Results ({split} split)", fontsize=16)

    # Plot accuracy vs number of branches
    axes[0, 0].plot(split_df["n_branches"], split_df["accuracy"], "o-", color="blue")
    axes[0, 0].set_xlabel("Number of Branches")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Accuracy vs Number of Branches")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot loss vs number of branches
    axes[0, 1].plot(split_df["n_branches"], split_df["loss"], "o-", color="red")
    axes[0, 1].set_xlabel("Number of Branches")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Loss vs Number of Branches")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot precision vs number of branches
    axes[1, 0].plot(split_df["n_branches"], split_df["precision"], "o-", color="green")
    axes[1, 0].set_xlabel("Number of Branches")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision vs Number of Branches")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot recall vs number of branches
    axes[1, 1].plot(split_df["n_branches"], split_df["recall"], "o-", color="orange")
    axes[1, 1].set_xlabel("Number of Branches")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Recall vs Number of Branches")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(
        os.path.dirname(results_csv_path), f"branch_sweep_results_{split}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {save_path}")


def plot_noise_sweep_ei_results(results_csv_path, noise_type="gaussian"):
    """Plot noise sweep results for E/I networks."""
    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific noise type
    noise_df = results_df[results_df["noise_type"] == noise_type]

    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Noise Sweep Results - {noise_type.title()} Noise (E/I Networks)", fontsize=16
    )

    # Plot accuracy vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[0, 0].plot(
            ei_data["noise_level"],
            ei_data["accuracy"],
            "o-",
            label=f"E/I_var = {ei_ratio}",
        )
    axes[0, 0].set_xlabel("Noise Level")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Accuracy vs Noise Level")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot loss vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[0, 1].plot(
            ei_data["noise_level"], ei_data["loss"], "o-", label=f"E/I_var = {ei_ratio}"
        )
    axes[0, 1].set_xlabel("Noise Level")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Loss vs Noise Level")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot precision vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[1, 0].plot(
            ei_data["noise_level"],
            ei_data["precision"],
            "o-",
            label=f"E/I_var = {ei_ratio}",
        )
    axes[1, 0].set_xlabel("Noise Level")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision vs Noise Level")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot recall vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[1, 1].plot(
            ei_data["noise_level"],
            ei_data["recall"],
            "o-",
            label=f"E/I_var = {ei_ratio}",
        )
    axes[1, 1].set_xlabel("Noise Level")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Recall vs Noise Level")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(
        os.path.dirname(results_csv_path), f"noise_sweep_ei_results_{noise_type}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {save_path}")


def plot_noise_sweep_branch_results(results_csv_path, noise_type="gaussian"):
    """Plot noise sweep results for branch networks."""
    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific noise type
    noise_df = results_df[results_df["noise_type"] == noise_type]

    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Noise Sweep Results - {noise_type.title()} Noise (Branch Networks)",
        fontsize=16,
    )

    # Plot accuracy vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[0, 0].plot(
            branch_data["noise_level"],
            branch_data["accuracy"],
            "o-",
            label=f"Branches={n_branches}",
        )
    axes[0, 0].set_xlabel("Noise Level")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Accuracy vs Noise Level")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot loss vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[0, 1].plot(
            branch_data["noise_level"],
            branch_data["loss"],
            "o-",
            label=f"Branches={n_branches}",
        )
    axes[0, 1].set_xlabel("Noise Level")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Loss vs Noise Level")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot precision vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[1, 0].plot(
            branch_data["noise_level"],
            branch_data["precision"],
            "o-",
            label=f"Branches={n_branches}",
        )
    axes[1, 0].set_xlabel("Noise Level")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision vs Noise Level")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot recall vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[1, 1].plot(
            branch_data["noise_level"],
            branch_data["recall"],
            "o-",
            label=f"Branches={n_branches}",
        )
    axes[1, 1].set_xlabel("Noise Level")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Recall vs Noise Level")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(
        os.path.dirname(results_csv_path),
        f"noise_sweep_branch_results_{noise_type}.png",
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {save_path}")

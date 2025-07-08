"""
Basic plotting utilities for dendritic modeling.

This module contains fundamental plotting utilities including basic plot functions,
color management, and loss curve plotting.
"""

import os
from typing import List, Optional, Callable, Any

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotWithColor(data, title: str = "") -> None:
    """
    Plot an image with a scale color bar.
    
    Args:
        data: Data to plot (tensor or array)
        title: Plot title
    """
    if torch.is_tensor(data):
        data = data.cpu().numpy()
        
    ax = plt.subplot()
    im = ax.imshow(data)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    if len(title) > 0:
        plt.title(title)

    plt.show()


def add_colorbars(
    fig,
    ax_iterable,
    im_iterable,
    position: str,
    size: str,
    pad: float,
) -> None:
    """
    Add colorbars to multiple axes.
    
    Args:
        fig: Figure object
        ax_iterable: Iterable of axes
        im_iterable: Iterable of image objects
        position: Position of colorbar ('right', 'left', 'top', 'bottom')
        size: Size of colorbar as percentage string (e.g., '5%')
        pad: Padding between colorbar and axes
    """
    for ax, im in zip(ax_iterable, im_iterable):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position=position, size=size, pad=pad)
        fig.colorbar(im, cax=cax)


def shape_helper(condition: bool, reshape_fn: Optional[Callable], tensors: List[Any]) -> List[Any]:
    """
    Helper function for reshaping tensors based on conditions.
    
    Args:
        condition: Whether to apply reshape function
        reshape_fn: Function to reshape tensors
        tensors: List of tensors to potentially reshape
        
    Returns:
        List of reshaped or processed tensors
    """
    if condition and reshape_fn is not None:
        return [None if tensor is None else reshape_fn(tensor) for tensor in tensors]
    else:
        ret = []
        for tensor in tensors:
            if tensor is None:
                ret.append(None)
            elif tensor.dim() == 1:
                ret.append(tensor.view(1, -1))
            else:
                ret.append(tensor)
        return ret


def plot_NLL_loss_curves(
    Train_losses: List[float], 
    Valid_losses: List[float], 
    num_epochs: int, 
    save_dir: Optional[str]
) -> None:
    """
    Plot the negative log-likelihood loss curves for training and validation.

    Args:
        Train_losses: List of training losses
        Valid_losses: List of validation losses
        num_epochs: Number of epochs
        save_dir: Directory to save the plot
    """
    # Convert to list if numpy array
    Train_losses = list(Train_losses)
    Valid_losses = list(Valid_losses)

    # Create the x-axis values (epochs)
    # Make sure length matches the loss arrays
    actual_epochs = len(Train_losses)
    if actual_epochs != num_epochs:
        print(
            f"Warning: Actual epochs ({actual_epochs}) != expected epochs ({num_epochs})"
        )
        # Use actual number of epochs for x-axis
        epochs = list(range(1, actual_epochs + 1))
    else:
        epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, Train_losses, "0.4", label="training")

    if Valid_losses:
        # Make sure valid losses have same length as train losses
        if len(Valid_losses) != len(Train_losses):
            print(
                f"Warning: Valid losses length ({len(Valid_losses)}) != Train losses length ({len(Train_losses)})"
            )
            # Use the shorter length
            min_len = min(len(Valid_losses), len(Train_losses))
            Valid_losses = Valid_losses[:min_len]
            epochs_valid = epochs[:min_len]
        else:
            epochs_valid = epochs

        plt.plot(epochs_valid, Valid_losses, "r", label="validation")

    plt.grid()
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Save the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "loss_curves.png"))

    plt.close() 
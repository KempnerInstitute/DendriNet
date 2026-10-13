"""
Data Utilities
==============

This module contains utility functions for data handling, serialization,
and basic data transformations.
"""

import json
import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig

from dendritic_modeling.utils.reproducibility import set_seed  # noqa: F401


class Shaper:
    """
    A utility class for reshaping input data to a specified shape.

    This is useful for data preprocessing pipelines where consistent
    reshaping is needed.
    """

    def __init__(self, shape):
        """
        Initialize the Shaper with target shape.

        Args:
            shape: Target shape tuple for reshaping
        """
        self.shape = shape

    def reshape(self, x):
        """
        Reshape input tensor to the target shape.

        Args:
            x: Input tensor to reshape

        Returns:
            Reshaped tensor with target shape
        """
        return x.view(*self.shape)


def convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert OmegaConf objects to standard Python data types.

    This is essential for saving configurations and results to JSON files,
    as OmegaConf objects are not directly JSON serializable.

    Args:
        obj: Object to convert (can be nested dicts, lists, etc.)

    Returns:
        Converted object with standard Python types
    """
    if isinstance(obj, DictConfig):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, ListConfig):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()
    elif isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, torch.dtype):
        return str(obj)
    else:
        return obj


def save_dict(dict_obj: dict[str, Any], save_path: str, fname: str) -> None:
    """
    Save a dictionary to a JSON file with proper serialization.

    This function handles OmegaConf objects and creates directories as needed.

    Args:
        dict_obj: Dictionary to save
        save_path: Directory path to save the file
        fname: Filename for the saved file
    """
    # Convert to serializable format
    dict_obj = convert_to_serializable(dict_obj)

    # Create full path
    full_path = os.path.join(save_path, fname)

    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Save to JSON file
    with open(full_path, "w") as f:
        json.dump(dict_obj, f, indent=4)

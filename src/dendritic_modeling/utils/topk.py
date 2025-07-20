"""
TopK Utilities Module.

This module provides utility functions and classes for TopK operations,
making them reusable across different parts of the codebase.
"""

from typing import Dict, Type, Any

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import TopKLinear
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.stochastic import StochasticTopKLinear
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import VarianceTopKLinear


# Registry of available TopK implementations
TOPK_REGISTRY: Dict[str, Type[TopKLinear]] = {
    "standard": TopKLinear,
    "stochastic": StochasticTopKLinear,
    "variance": VarianceTopKLinear,
}


def get_topk_layer(
    topk_type: str,
    in_features: int,
    out_features: int,
    K: int,
    **kwargs
) -> TopKLinear:
    """
    Factory function to create TopK layers based on type.
    
    Args:
        topk_type: Type of TopK layer ("standard", "stochastic", "variance")
        in_features: Number of input features
        out_features: Number of output features  
        K: Number of connections to keep per output
        **kwargs: Additional arguments specific to the TopK type
        
    Returns:
        Initialized TopK layer
        
    Raises:
        ValueError: If topk_type is not recognized
    """
    if topk_type not in TOPK_REGISTRY:
        available_types = list(TOPK_REGISTRY.keys())
        raise ValueError(f"Unknown TopK type '{topk_type}'. Available types: {available_types}")
    
    topk_class = TOPK_REGISTRY[topk_type]
    return topk_class(
        in_features=in_features,
        out_features=out_features,
        K=K,
        **kwargs
    )


def register_topk_layer(name: str, layer_class: Type[TopKLinear]) -> None:
    """
    Register a new TopK layer type.
    
    Args:
        name: Name to register the layer under
        layer_class: Class implementing the TopK layer
    """
    TOPK_REGISTRY[name] = layer_class


def get_available_topk_types() -> list[str]:
    """Get list of available TopK layer types."""
    return list(TOPK_REGISTRY.keys())


def calculate_effective_sparsity(
    weight_matrix: torch.Tensor,
    K: int,
    dim: int = -1
) -> float:
    """
    Calculate the effective sparsity of a weight matrix after TopK selection.
    
    Args:
        weight_matrix: Weight tensor
        K: Number of connections to keep per output
        dim: Dimension along which to compute TopK
        
    Returns:
        Effective sparsity ratio (0 = dense, 1 = completely sparse)
    """
    total_elements = weight_matrix.numel()
    
    if dim == -1:
        # TopK across last dimension (most common case)
        active_elements = weight_matrix.shape[0] * K
    else:
        active_elements = weight_matrix.shape[dim] * K
    
    sparsity = 1.0 - (active_elements / total_elements)
    return float(sparsity)


def topk_mask_from_weights(
    weights: torch.Tensor,
    K: int,
    dim: int = -1,
    largest: bool = True
) -> torch.Tensor:
    """
    Create a binary mask for TopK selection.
    
    Args:
        weights: Weight tensor
        K: Number of elements to select
        dim: Dimension along which to select TopK
        largest: Whether to select largest or smallest values
        
    Returns:
        Binary mask with 1s for selected elements, 0s elsewhere
    """
    # Get TopK indices
    _, indices = torch.topk(weights, K, dim=dim, largest=largest, sorted=False)
    
    # Create mask
    mask = torch.zeros_like(weights, dtype=torch.bool)
    
    if dim == -1:
        # Handle last dimension case
        batch_indices = torch.arange(weights.shape[0], device=weights.device)[:, None]
        mask[batch_indices, indices] = True
    else:
        # More general case - would need more implementation for arbitrary dims
        raise NotImplementedError("Only dim=-1 currently supported")
    
    return mask.float()


def apply_topk_sparsity(
    weights: torch.Tensor,
    K: int,
    dim: int = -1,
    largest: bool = True
) -> torch.Tensor:
    """
    Apply TopK sparsity to weights by setting non-selected elements to zero.
    
    Args:
        weights: Weight tensor
        K: Number of elements to keep per row/column
        dim: Dimension along which to apply TopK
        largest: Whether to keep largest or smallest values
        
    Returns:
        Sparse weight tensor with only TopK elements preserved
    """
    mask = topk_mask_from_weights(weights, K, dim, largest)
    return weights * mask


def count_active_parameters(topk_layers: list[TopKLinear]) -> int:
    """
    Count the total number of active (non-zero) parameters across TopK layers.
    
    Args:
        topk_layers: List of TopK layer instances
        
    Returns:
        Total number of active parameters
    """
    total_active = 0
    for layer in topk_layers:
        if hasattr(layer, 'K') and hasattr(layer, 'out_features'):
            total_active += layer.out_features * layer.K
    return total_active


def estimate_topk_memory_usage(
    in_features: int,
    out_features: int,
    K: int,
    batch_size: int = 1,
    dtype_bytes: int = 4
) -> Dict[str, int]:
    """
    Estimate memory usage for a TopK layer.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        K: TopK parameter
        batch_size: Batch size for activation memory
        dtype_bytes: Bytes per parameter (4 for float32)
        
    Returns:
        Dictionary with memory estimates in bytes
    """
    # Parameter memory
    param_memory = out_features * in_features * dtype_bytes
    
    # Index memory (for storing TopK indices)
    index_memory = out_features * K * 8  # 8 bytes for int64 indices
    
    # Activation memory
    activation_memory = batch_size * out_features * K * dtype_bytes
    
    return {
        "parameters": param_memory,
        "indices": index_memory,
        "activations": activation_memory,
        "total": param_memory + index_memory + activation_memory
    }


__all__ = [
    "TOPK_REGISTRY",
    "get_topk_layer",
    "register_topk_layer", 
    "get_available_topk_types",
    "calculate_effective_sparsity",
    "topk_mask_from_weights",
    "apply_topk_sparsity",
    "count_active_parameters",
    "estimate_topk_memory_usage",
] 
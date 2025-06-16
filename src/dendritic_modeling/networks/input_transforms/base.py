"""
Base Input Transform for neural networks.

This module implements the base InputNetTransform class that maps
raw inputs to excitatory and inhibitory pathways.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import logging

from dendritic_modeling.networks.input_transforms.index import IndexInputTransform
from dendritic_modeling.networks.input_transforms.mlp import MLPInputTransform
from dendritic_modeling.networks.input_transforms.cnn import CNNInputTransform
from dendritic_modeling.networks.input_transforms.alexnet import AlexNetInputTransform

logger = logging.getLogger(__name__)


class InputNetTransform(nn.Module):
    """
    A transform that maps the raw input x to (excitatory_input, inhibitory_input).
    This is a generalized version that can support different network architectures.
    By default, it uses an MLP with configurable hidden dimensions.
    """

    def __init__(
        self, 
        input_dim: Optional[int] = None, 
        hidden_dims: Optional[List[int]] = None, 
        independent_heads: bool = False,
        excitatory_dim: Optional[int] = None, 
        inhibitory_dim: Optional[int] = None, 
        output_dim: Optional[int] = None,
        input_shape: Optional[Tuple[int, int, int]] = None,
        excitatory_hidden_dims: Optional[List[int]] = None,
        inhibitory_hidden_dims: Optional[List[int]] = None,
        excitatory_kernel_sizes: Optional[List[int]] = None,
        inhibitory_kernel_sizes: Optional[List[int]] = None,
        excitatory_strides: Optional[List[int]] = None,
        inhibitory_strides: Optional[List[int]] = None,
        network_type: str = "MLPInputTransform",
    ):
        super().__init__()
        self.network_type = network_type

        if network_type == "IndexInputTransform":
            self.input_transform = IndexInputTransform(
                input_dim=input_dim,
                excitatory_dim=excitatory_dim,
                inhibitory_dim=inhibitory_dim,
            )

        elif network_type == "MLPInputTransform":
            self.input_transform = MLPInputTransform(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                excitatory_dim=excitatory_dim,
                inhibitory_dim=inhibitory_dim,
            )
        
        elif network_type == "FeatureExtractorMLP":
            from ..mlp import FeatureExtractorMLP
            self.input_transform = FeatureExtractorMLP(
                input_dim = input_dim,
                hidden_dims = hidden_dims,
                output_dim = output_dim
            )
            self.input_transform.set_readout(False)
            self.input_transform.disable_grad()
        elif network_type == "AlexNetTransform":
            self.input_transform = AlexNetInputTransform(
                excitatory_dim=excitatory_dim, inhibitory_dim=inhibitory_dim
            )

        elif network_type == "CNNInputTransform":
            self.input_transform = CNNInputTransform(
                input_shape=input_shape,
                excitatory_hidden_dims=excitatory_hidden_dims,
                inhibitory_hidden_dims=inhibitory_hidden_dims,
                excitatory_kernel_sizes=excitatory_kernel_sizes,
                inhibitory_kernel_sizes=inhibitory_kernel_sizes,
                excitatory_strides=excitatory_strides,
                inhibitory_strides=inhibitory_strides,
            )

        else:
            raise ValueError(f"Unknown network type: {network_type}")

    def forward(self, x):
        excitatory_input, inhibitory_input = self.input_transform(x)
        return excitatory_input, inhibitory_input

    def load_pretrained_weights(self, source, weights_path=None, pytorch_model=None):
        """
        Load pre-trained weights for the input network.

        Args:
            source (str): Source of weights - 'file', 'pytorch', or 'none'
            weights_path (str, optional): Path to weights file if source is 'file'
            pytorch_model (str, optional): Name of PyTorch model if source is 'pytorch'
        """
        if source == "none":
            logger.info("No pre-trained weights to load.")
            return

        elif source == "file":
            if not weights_path:
                raise ValueError("weights_path must be provided when source is 'file'")
            logger.info(f"Loading weights from file: {weights_path}")
            state_dict = torch.load(weights_path)
            self.load_state_dict(state_dict)

        elif source == "pytorch":
            if not pytorch_model:
                raise ValueError(
                    "pytorch_model must be provided when source is 'pytorch'"
                )
            logger.info(f"Loading pre-trained PyTorch model: {pytorch_model}")

            import torchvision.models as models

            if pytorch_model == "alexnet":
                pretrained_model = models.alexnet(pretrained=True)
                # Copy features to shared backbone
                if self.network_type == "AlexNetTransform":
                    # For AlexNet, we can directly copy the features
                    for i, layer in enumerate(list(pretrained_model.features)):
                        if i < len(list(self.shared)):
                            self.shared[i].load_state_dict(layer.state_dict())
            else:
                raise ValueError(f"Unsupported PyTorch model: {pytorch_model}")
        else:
            raise ValueError(f"Unknown source: {source}")


__all__ = ["InputNetTransform"] 
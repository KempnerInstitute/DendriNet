"""
Vision encoder wrapper and baseline builder for Adi.

VisionEncoderWrapper exposes .encoder and .output_dim so the repo's
setup_utils.initialize_model sees the same interface as other encoder types.
output_dim is set from actual encoder(x) shape (single forward on dummy input).
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.factory import get_architecture


class VisionEncoderWrapper(nn.Module):
    """
    Wrapper so the repo sees encoder_network.encoder and encoder_network.output_dim.
    forward(x) returns self.encoder(x).
    output_dim must equal the actual feature dimension emitted by self.encoder(x),
    verified by a single forward pass on dummy input.
    """

    def __init__(self, encoder: nn.Module, output_dim: int):
        super().__init__()
        self.encoder = encoder
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def get_baseline_encoder(parameters: Dict[str, Any]) -> VisionEncoderWrapper:
    """
    Build CNNDownsample from params and wrap in VisionEncoderWrapper.
    Uses only CNNDownsample-supported args: input_shape, hidden_dims, kernel_sizes,
    strides, activation, output_dim (optional).
    Maps config keys encoder_hidden_dims -> hidden_dims, etc.
    Sets wrapper.output_dim from actual encoder(dummy_input).shape[-1].
    """
    # Map EncoderParamsConfig-style keys to CNNDownsample kwargs
    input_shape = parameters.get("input_shape", [3, 32, 32])
    if not isinstance(input_shape, tuple):
        input_shape = tuple(input_shape)
    hidden_dims = parameters.get(
        "encoder_hidden_dims", parameters.get("hidden_dims", [64, 192, 384, 256, 256])
    )
    kernel_sizes = parameters.get(
        "encoder_kernel_sizes", parameters.get("kernel_sizes", [3, 3, 3, 3, 3])
    )
    strides = parameters.get(
        "encoder_strides", parameters.get("strides", [1, 1, 1, 1, 1])
    )
    activation = parameters.get("activation", "relu")
    output_dim = parameters.get("output_dim")  # None = encoder outputs raw feature dim

    cnn_params = {
        "input_shape": input_shape,
        "hidden_dims": hidden_dims,
        "kernel_sizes": kernel_sizes,
        "strides": strides,
        "activation": activation,
        "output_dim": output_dim,
    }
    encoder = get_architecture("cnndownsample", cnn_params)

    # Verify output_dim equals actual feature dimension from encoder(x)
    with torch.no_grad():
        dummy = torch.zeros(1, *input_shape)
        out = encoder(dummy)
    actual_dim = out.shape[-1]
    if output_dim is not None and actual_dim != output_dim:
        raise ValueError(
            f"encoder output_dim mismatch: config output_dim={output_dim}, "
            f"actual encoder(x).shape[-1]={actual_dim}"
        )
    return VisionEncoderWrapper(encoder, actual_dim)


def get_hybrid_encoder(parameters: Dict[str, Any]) -> VisionEncoderWrapper:
    """
    Build hybrid backbone (conv1..conv5, fc6, fc7; fc8 in decoder) from params.
    conv_type: list of "standard"|"dendritic" for conv1..conv5.
    fc_type: list of "standard"|"dendritic" for fc6, fc7. fc8 always standard in decoder.
    """
    from dendritic_modeling.Adi.hybrid_backbone import HybridBackbone

    conv_type = parameters.get("conv_type", ["standard"] * 5)
    fc_type = parameters.get("fc_type", ["standard", "standard"])
    activation = parameters.get("activation", "relu")
    encoder = HybridBackbone(conv_type=conv_type, fc_type=fc_type, activation=activation)
    # output_dim must equal actual encoder(x).shape[-1]
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 32, 32)
        out = encoder(dummy)
    actual_dim = out.shape[-1]
    return VisionEncoderWrapper(encoder, actual_dim)

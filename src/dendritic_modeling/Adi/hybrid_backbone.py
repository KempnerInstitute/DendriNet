"""
Hybrid backbone: conv1..conv5, fc6, fc7, fc8 with per-layer standard|dendritic toggle.
fc8 is always standard (classifier head). Same dimensions as locked baseline.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d

from dendritic_modeling.Adi.dendritic_linear import DendriticLinearWrapper


# Locked dimensions (matches baseline CNNDownsample: 5 conv, then flatten)
CONV_CHANNELS = [64, 192, 384, 256, 256]
CONV_KERNEL = 3
CONV_STRIDE = 1
INPUT_SHAPE = (3, 32, 32)
FLAT_DIM = 256 * 22 * 22  # 123904
FC6_DIM = 4096
FC7_DIM = 4096
NUM_CLASSES = 10


def _make_conv_layer(
    in_ch: int,
    out_ch: int,
    kernel: int,
    stride: int,
    dendritic: bool,
) -> nn.Module:
    if dendritic:
        return DendriteConv2d(
            in_ch, out_ch, kernel, stride=stride, padding=0
        )
    conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=0)
    nn.init.kaiming_normal_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def _make_fc_layer(
    in_features: int,
    out_features: int,
    dendritic: bool,
) -> nn.Module:
    if dendritic:
        return DendriticLinearWrapper(in_features, out_features)
    linear = nn.Linear(in_features, out_features)
    nn.init.kaiming_normal_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


class HybridBackbone(nn.Module):
    """
    Backbone with named layers conv1..conv5, fc6, fc7, fc8.
    conv_type[i] and fc_type[i] select standard vs dendritic per layer.
    fc8 is always standard.
    """

    def __init__(
        self,
        conv_type: Optional[List[str]] = None,
        fc_type: Optional[List[str]] = None,
        activation: str = "relu",
    ):
        super().__init__()
        conv_type = conv_type or ["standard"] * 5
        fc_type = fc_type or ["standard", "standard"]
        assert len(conv_type) == 5
        assert len(fc_type) == 2
        act_factory = ActivationFactory()
        layers = []
        in_ch = INPUT_SHAPE[0]
        for i, out_ch in enumerate(CONV_CHANNELS):
            layers.append(
                _make_conv_layer(
                    in_ch,
                    out_ch,
                    CONV_KERNEL,
                    CONV_STRIDE,
                    dendritic=(conv_type[i].lower() == "dendritic"),
                )
            )
            layers.append(act_factory.create(act_type=activation, output_dim=out_ch))
            in_ch = out_ch
        self.conv_block = nn.Sequential(*layers)
        self.flatten = nn.Flatten(start_dim=-3)
        self.fc6 = _make_fc_layer(FLAT_DIM, FC6_DIM, fc_type[0].lower() == "dendritic")
        self.fc6_act = act_factory.create(act_type=activation, output_dim=FC6_DIM)
        self.fc7 = _make_fc_layer(FC6_DIM, FC7_DIM, fc_type[1].lower() == "dendritic")
        self.fc7_act = act_factory.create(act_type=activation, output_dim=FC7_DIM)
        # fc8 (classifier head) is always in decoder in repo pipeline; encoder outputs features
        self.output_dim = FC7_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc6_act(self.fc6(x))
        x = self.fc7_act(self.fc7(x))
        return x


def get_hybrid_encoder_output_dim() -> int:
    """Feature dim after conv block (before fc6). For use as core input_dim when encoder is hybrid (fc8 in decoder)."""
    return FLAT_DIM

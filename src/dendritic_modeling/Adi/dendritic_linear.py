"""
Dendritic FC drop-in: single in_features -> out_features using DendriticBranchLayer.
Presents the same interface as nn.Linear for use in the hybrid backbone.
"""

from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)


class DendriticLinearWrapper(nn.Module):
    """
    Drop-in for nn.Linear using one DendriticBranchLayer (excitatory only, no inhibitory).
    forward(x) returns (batch, out_features). Same shape contract as nn.Linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: Optional[int] = None,
        use_shunting: bool = True,
        reactivate: bool = True,
        **kwargs,
    ):
        super().__init__()
        K = K or min(64, in_features, out_features)
        self._layer = DendriticBranchLayer(
            output_dim=out_features,
            excitatory_input_dim=in_features,
            excitatory_synapses_per_branch=K,
            inhibitory_input_dim=None,
            inhibitory_synapses_per_branch=None,
            use_shunting=use_shunting,
            reactivate=reactivate,
            topk_init_method=kwargs.get("topk_init_method", "xavier_normal"),
            topk_noise_level=kwargs.get("topk_noise_level", 0.0),
            topk_type=kwargs.get("topk_type", "standard"),
            reactivation_type=kwargs.get("reactivation_type", "param_tanh"),
            reactivation_init_m=kwargs.get("reactivation_init_m", 1.0),
            reactivation_init_b=kwargs.get("reactivation_init_b", 0.5),
            **{k: v for k, v in kwargs.items() if k not in ("topk_init_method", "topk_noise_level", "topk_type", "reactivation_type", "reactivation_init_m", "reactivation_init_b")},
        )
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layer(x, inhibitory_input=None, branch_input=None)

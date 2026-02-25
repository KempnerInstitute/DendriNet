"""
Base model class for dendritic modeling.

This module contains the abstract BaseModel class that provides common functionality
for all model wrappers including parameter grouping and weight decay.
"""

from itertools import chain

import torch
import torch.nn as nn

from dendritic_modeling.config import ParamGroupsConfig
from dendritic_modeling.networks import (
    BaseNetwork,
    BlockLinear,
    ExcitationInhibitionLayer,
    Identity,
    ParametricActivation,
    TopKLinear,
)

try:
    from dendritic_modeling.networks.architectures.recurrent import DendriticRNNNetwork
except (ImportError, ModuleNotFoundError):
    DendriticRNNNetwork = None


class BaseModel(nn.Module):
    def __init__(
        self,
        encoder_network: BaseNetwork,
        core_network: BaseNetwork,
        decoder_network: BaseNetwork,
    ):
        super().__init__()

        self.encoder_network = encoder_network
        self.core_network = core_network
        self.decoder_network = decoder_network
        # Cache once: non-recurrent models avoid per-forward isinstance checks.
        self._is_recurrent_core = DendriticRNNNetwork is not None and isinstance(
            self.core_network, DendriticRNNNetwork
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Recurrent cores expect [batch, seq_len, feature_dim]
        # Encoder may need per-timestep application
        if self._is_recurrent_core:
            if not isinstance(self.encoder_network, Identity) and x.dim() == 3:
                # Apply encoder per timestep: [B, T, D] -> [B, T, D']
                B, T, D = x.shape
                x = x.reshape(B * T, D)
                x = self.encoder_network(x)
                x = x.reshape(B, T, -1)
            else:
                x = self.encoder_network(x)
            x = self.core_network(x)
        else:
            x = self.encoder_network(x)
            x = self.core_network(x)
        x = self.decoder_network(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Predict method must be implemented in subclass.")

    def decay_weights(self, weight_decay: float, weight_boosting: bool = False):
        """
        Forward this call to self.net if it implements decay_weights.
        """
        if hasattr(self.core_network, "decay_weights"):
            self.core_network.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self):
        """
        Forward this call to self.net if it implements apply_rewiring.
        """
        if hasattr(self.core_network, "apply_rewiring"):
            self.core_network.apply_rewiring()

    def get_param_groups(self, param_groups_config: ParamGroupsConfig) -> list[dict]:
        """
        Get parameter groups for optimizer.

        Materializes chain() iterators to lists (one-shot iterators can't be
        reused by the optimizer). Adds a catch-all group for any parameters not
        matched by specific categories (e.g. recurrent nn.Linear, tau params).
        """
        split_params = getattr(param_groups_config, "split_params", True)

        if split_params:
            topk_params = []
            blocklinear_params = []
            reactivation_params = []
            mlp_inh_net_params = []
            decoder_params = []

            for module in self.core_network.modules():
                if isinstance(module, TopKLinear):
                    topk_params.append(module.parameters())
                elif isinstance(module, BlockLinear):
                    blocklinear_params.append(module.parameters())
                elif isinstance(module, ParametricActivation):
                    reactivation_params.append(module.parameters())
                elif isinstance(module, ExcitationInhibitionLayer):
                    if isinstance(module.inhibitory_cells, nn.Sequential):
                        mlp_inh_net_params.append(module.inhibitory_cells.parameters())

            decoder_params.append(self.decoder_network.parameters())
            if hasattr(self, "log_output_scale"):
                self.log_output_scale: nn.Parameter
                decoder_params.append([self.log_output_scale])

            param_groups = []
            categorized_ids: set[int] = set()

            for params_list, lr_key, default_lr in [
                (topk_params, "topk_lr", 0.001),
                (blocklinear_params, "blocklinear_lr", 0.001),
                (reactivation_params, "reactivation_lr", 0.001),
                (mlp_inh_net_params, "lr", 0.001),
                (decoder_params, "decoder_lr", 0.001),
            ]:
                if params_list:
                    materialized = list(chain(*params_list))
                    categorized_ids.update(id(p) for p in materialized)
                    param_groups.append(
                        {
                            "params": materialized,
                            "lr": getattr(param_groups_config, lr_key, default_lr),
                        }
                    )

            # Catch-all: any params not in the above groups
            uncategorized = [
                p
                for p in self.parameters()
                if p.requires_grad and id(p) not in categorized_ids
            ]
            if uncategorized:
                param_groups.append(
                    {
                        "params": uncategorized,
                        "lr": getattr(param_groups_config, "lr", 0.001),
                    }
                )

            return param_groups
        else:
            return [
                {
                    "params": self.parameters(),
                    "lr": getattr(param_groups_config, "lr", 0.001),
                }
            ]


__all__ = ["BaseModel"]

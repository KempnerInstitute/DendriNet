"""
Standard classification models.

This module contains standard classifiers using cross-entropy loss.
"""

from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.models.base import BaseModel
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.base import BaseNetwork


class Classifier(BaseModel):
    """
    A standard classifier that returns raw logits;
    cross-entropy is used for compute_loss.
    """

    def __init__(
        self,
        encoder_network: Optional[BaseNetwork],
        core_network: BaseNetwork,
        decoder_network: Optional[BaseNetwork],
    ):
        super().__init__(encoder_network, core_network, decoder_network)

        if isinstance(self.decoder_network, Identity):
            self.log_output_scale = nn.Parameter(
                torch.zeros((self.core_network.output_dim,)), requires_grad=True
            )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = super().forward(x, **kwargs)

        if hasattr(self, "log_output_scale"):
            x = self.log_output_scale.exp() * x

        return x

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x, **kwargs)
            return logits.argmax(dim=-1)


__all__ = ["Classifier"]

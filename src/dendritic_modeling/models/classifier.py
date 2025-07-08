"""
Standard classification models.

This module contains standard classifiers using cross-entropy loss.
"""

import torch
import torch.nn as nn
from typing import Optional

from dendritic_modeling.models.base import BaseModel
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
        decoder_network: Optional[BaseNetwork]
    ):
        super(Classifier, self).__init__(
            encoder_network,
            core_network,
            decoder_network
        )

        if isinstance(self.decoder_network, nn.Identity):
            self.log_output_scale = nn.Parameter(
                torch.zeros((self.core_network.output_dim,)), requires_grad=True
            )
        
        self.final = nn.LogSoftmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)

        if hasattr(self, 'log_output_scale'):
            x = self.log_output_scale.exp() * x

        logits = self.final(x)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)


__all__ = ["Classifier"]
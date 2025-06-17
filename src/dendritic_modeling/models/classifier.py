"""
Standard classification models.

This module contains standard classifiers using cross-entropy loss.
"""

import torch
import torch.nn as nn

from dendritic_modeling.models.base import BaseModel
from dendritic_modeling.networks.base import BaseNetwork


class Classifier(BaseModel):
    """
    A standard classifier that returns raw logits;
    cross-entropy is used for compute_loss.
    """

    def __init__(
        self,
        net: BaseNetwork,
        output_layer: bool = False,
        output_dim: int = 10
    ):
        super(Classifier, self).__init__(net)

        if output_layer:
            self.final_fc = nn.Linear(net.output_dim, output_dim)
        else:
            self.log_output_scale = nn.Parameter(
                torch.zeros((net.output_dim,)), requires_grad=True
            )
        
        self.final = nn.LogSoftmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if hasattr(self, "final_fc"):
            out = self.final_fc(out)
        else:
            out = self.log_output_scale.exp() * out

        logits = self.final(out)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)


__all__ = ["Classifier"]
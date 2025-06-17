"""
Standard regression models.

This module contains standard regression models using MSE loss.
"""

import torch
import torch.nn as nn

from dendritic_modeling.models.base import BaseModel
from dendritic_modeling.networks.base import BaseNetwork


class Regressor(BaseModel):
    """
    A standard regressor that returns continuous outputs;
    MSE is used for compute_loss.
    """
    def __init__(
        self,
        net: BaseNetwork,
        output_layer: bool = False,
        output_dim: int = 10,
    ):
        super(Regressor, self).__init__(net)
        if output_layer:
            self.final_fc = nn.Linear(net.output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if hasattr(self, 'final_fc'):
            out = self.final_fc(out)
        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


__all__ = ["Regressor"] 
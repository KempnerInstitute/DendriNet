"""
Standard regression models.

This module contains standard regression models using MSE loss.
"""
from typing import Optional

import torch

from dendritic_modeling.models.base import BaseModel
from dendritic_modeling.networks.base import BaseNetwork


class Regressor(BaseModel):
    """
    A standard regressor that returns continuous outputs;
    MSE is used for compute_loss.
    """
    def __init__(
        self,
        encoder_network: Optional[BaseNetwork],
        core_network: BaseNetwork,
        decoder_network: Optional[BaseNetwork]
    ):
        super(Regressor, self).__init__(
            encoder_network,
            core_network,
            decoder_network
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)

__all__ = ["Regressor"]

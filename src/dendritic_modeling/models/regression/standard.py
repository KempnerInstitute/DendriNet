"""
Standard regression models.

This module contains standard regression models using MSE loss.
"""

import torch
import torch.nn as nn

from ..base import BaseModel


class Regressor(BaseModel):
    """
    A standard regressor that returns continuous outputs;
    MSE is used for compute_loss.
    """
    def __init__(self, net, output_dim=None, use_output_layer=False, seed=None):
        super(Regressor, self).__init__(net)
        
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer
        self.seed = seed
        if self.use_output_layer:
            # Use a dummy input with correct input dimension
            dummy_input = torch.zeros(1, net.input_dim)  # First layer size is input dim
            self.final_fc = nn.Linear(
                net.forward(dummy_input).shape[1], 
                self.output_dim
            )
            
        if self.seed is not None:
            self.net.set_seed(self.seed)

    def forward(self, x):
        outputs = self.net(x)
        if hasattr(self, 'final_fc'):
            outputs = self.final_fc(outputs)
        return outputs

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def compute_loss(self, x, y):
        outputs = self.forward(x)
        loss_fn = nn.MSELoss()
        main_loss = loss_fn(outputs, y)
        return main_loss

    def forward_with_branch_outputs(self, x):
        net_output, per_layer_acts = self.net.forward_with_branch_outputs(x)
        if hasattr(self, 'final_fc'):
            net_output = self.final_fc(net_output)
        return net_output, per_layer_acts

    def set_seed(self, seed):
        """Set the seed for the network."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["Regressor"] 
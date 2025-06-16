"""
Deep Sparse Training regression models.

This module contains regression models that use dynamic sparse training (DeepST).
"""

import torch
import torch.nn as nn

from dendritic_modeling.models.base import BaseModel


class DeepstRegressor(BaseModel):
    """
    A regressor model that incorporates the Bellec 2018 dynamic rewiring algorithm.
    Uses MSE for compute_loss like standard Regressor.
    """
    def __init__(self, net, output_dim=None, use_output_layer=False, seed=None, 
                 rewire_frequency=1):
        super(DeepstRegressor, self).__init__(net)
        
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer
        self.seed = seed
        self.rewire_frequency = rewire_frequency
        self.step_counter = 0
        
        if self.use_output_layer:
            # Use a dummy input with correct input dimension
            dummy_input = torch.zeros(1, net.input_dim)  # First layer size is input dim
            self.final_fc = nn.Linear(
                net.forward(dummy_input).shape[1], 
                self.output_dim
            )
            
        if self.seed is not None:
            self.set_seed(self.seed)

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
    
    def training_step(self, batch, optimizer):
        """Custom training step that incorporates rewiring"""
        x, y = batch
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass and compute loss
        loss = self.compute_loss(x, y)
        
        # Backward pass
        loss.backward()
        
        # Apply rewiring based on frequency
        self.step_counter += 1
        if hasattr(self.net, 'apply_rewiring') and self.step_counter % self.rewire_frequency == 0:
            self.net.apply_rewiring()
        
        # Optimizer step
        optimizer.step()
        
        return loss.item()

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


__all__ = ["DeepstRegressor"] 
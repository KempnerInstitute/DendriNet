"""
Standard classification models.

This module contains standard classifiers using cross-entropy loss.
"""

import torch
import torch.nn as nn

from ..base import BaseModel


class Classifier(BaseModel):
    """
    A standard classifier that returns raw logits;
    cross-entropy is used for compute_loss.
    """

    def __init__(self, net, output_dim=10, use_output_layer=False):
        super(Classifier, self).__init__(net)
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer

        # Fix: Use the same approach as ProbabilisticClassifier for final_fc
        if self.use_output_layer:
            # Get the output dimension from the network
            with torch.no_grad():
                if hasattr(net, "n_branch_layers"):
                    test_input = torch.zeros(1, net.n_branch_layers)
                else:
                    test_input = torch.zeros(
                        1, net.input_dim if hasattr(net, "input_dim") else 784
                    )
                net_out_dim = net.forward(test_input).shape[1]
                self.final_fc = nn.Linear(net_out_dim, self.output_dim)
        else:
            self.log_output_scale = nn.Parameter(
                torch.zeros((self.output_dim,)), requires_grad=True
            )
        
    def forward(self, x, y=None):
        logits = self.net(x)
        if hasattr(self, "final_fc"):
            logits = self.final_fc(logits)
        else:
            logits = self.log_output_scale.exp() * logits

        return logits

    def predict(self, x, stochastic=False):
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)

    def compute_loss(self, x, y):
        logits = self.forward(x)
        loss_fn = nn.CrossEntropyLoss()
        main_loss = loss_fn(logits, y)
        return main_loss

    def forward_with_branch_outputs(self, x):
        net_output, per_layer_acts = self.net.forward_with_branch_outputs(x)
        if hasattr(self, "final_fc"):
            net_output = self.final_fc(net_output)
        return net_output, per_layer_acts


__all__ = ["Classifier"] 
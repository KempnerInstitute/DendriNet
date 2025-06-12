"""
Baseline MLP models.

This module contains simple MLP implementations for baseline comparisons.
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    A simple PyTorch MLP that's optimized for MNIST classification.
    This will be used to demonstrate good performance on MNIST.
    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Create hidden layers
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(h_dim))
            prev_dim = h_dim

        # Create output layer
        layers.append(torch.nn.Linear(prev_dim, output_dim))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input is flattened
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        return self.network(x)


class SimpleClassifier(nn.Module):
    """
    A simple classifier that wraps our SimpleMLP for better performance.
    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.net = SimpleMLP(input_dim, hidden_dims, output_dim)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, x, y):
        logits = self.forward(x)
        return self.criterion(logits, y)


class MLPWithClassificationHead(nn.Module):
    """
    A wrapper around an MLP that adds a classification head for training,
    but allows accessing the feature extractor part separately.
    """

    def __init__(self, feature_extractor, num_features, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_head = torch.nn.Linear(num_features, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classification_head(features)

    def extract_features(self, x):
        return self.feature_extractor(x)

    def compute_loss(self, x, y):
        logits = self.forward(x)
        return self.criterion(logits, y)


__all__ = ["SimpleMLP", "SimpleClassifier", "MLPWithClassificationHead"] 
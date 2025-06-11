"""
Simple activation functions.

This module contains basic activation functions like ReLU, Sigmoid, and Tanh.
"""

import torch
import torch.nn as nn


class SimpleReLU(nn.Module):
    def __init__(self):
        super(SimpleReLU, self).__init__()

    def forward(self, x):
        return torch.relu(x)


class SimpleSigmoid(nn.Module):
    def __init__(self):
        super(SimpleSigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class SimpleTanh(nn.Module):
    def __init__(self):
        super(SimpleTanh, self).__init__()

    def forward(self, x):
        return torch.tanh(x)


__all__ = ["SimpleReLU", "SimpleSigmoid", "SimpleTanh"] 
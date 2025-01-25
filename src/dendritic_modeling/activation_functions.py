"""
activation_functions.py
=======================
Provides various activation/ reactivation modules, plus a factory to build
the desired activation function from config.

Example usage:
    act_fn = ActivationFactory.create(
        act_type="param_tanh",  # or "relu", "sigmoid", "tanh", etc.
        output_dim=128,
        init_m=1.5,
        init_b=1.0,
        train_b_only=False,
    )
"""

import torch
import torch.nn as nn
import math


class ParametricTanh(nn.Module):
    """
    A parametric tanh reactivation function with learnable parameters m and b.

    This matches your old 'Reactivation' logic, but is now a self-contained
    activation module. If 'train_b_only' is True, it freezes m.
    """
    def __init__(self, output_dim, init_m=1.5, init_b=1.0, train_b_only=False):
        super(ParametricTanh, self).__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)
        self.log_b = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)

        with torch.no_grad():
            self.log_m.data.fill_(math.log(init_m))
            self.log_b.data.fill_(math.log(init_b))

        if train_b_only:
            self.log_m.requires_grad = False

    def forward(self, V):
        m = self.log_m.exp()  # shape [output_dim]
        b = self.log_b.exp()  # shape [output_dim]
        # shape broadcasting: if V is [batch_size, output_dim],
        # then (V - b) is broadcast per-dim.
        return (torch.tanh(m * (V - b)) + 1) / 2


class SimpleReLU(nn.Module):
    """
    A standard ReLU activation.
    """
    def __init__(self):
        super(SimpleReLU, self).__init__()

    def forward(self, x):
        return torch.relu(x)


class SimpleSigmoid(nn.Module):
    """
    A standard Sigmoid activation.
    """
    def __init__(self):
        super(SimpleSigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class SimpleTanh(nn.Module):
    """
    A standard Tanh activation.
    """
    def __init__(self):
        super(SimpleTanh, self).__init__()

    def forward(self, x):
        return torch.tanh(x)


class ActivationFactory:
    """
    A factory for building different reactivation/activation modules
    depending on the config or arguments.
    """

    @staticmethod
    def create(
        act_type,
        output_dim=None,
        init_m=1.5,
        init_b=1.0,
        train_b_only=False,
    ):
        """
        Create an activation module of the desired type.

        Parameters
        ----------
        act_type : str
            One of ["none", "param_tanh", "relu", "sigmoid", "tanh"].
        output_dim : int or None
            Used by param_tanh (must be set).
        init_m : float
            Initial value for m in parametric Tanh.
        init_b : float
            Initial value for b in parametric Tanh.
        train_b_only : bool
            If True, param_tanh freezes m and only trains b.

        Returns
        -------
        nn.Module
            The activation object.
        """
        act_type = act_type.lower()

        if act_type == "none":
            # Return an identity function if no activation is desired
            return nn.Identity()

        elif act_type == "param_tanh":
            if output_dim is None:
                raise ValueError("ParametricTanh requires output_dim.")
            return ParametricTanh(output_dim, init_m, init_b, train_b_only)

        elif act_type == "relu":
            return SimpleReLU()

        elif act_type == "sigmoid":
            return SimpleSigmoid()

        elif act_type == "tanh":
            return SimpleTanh()

        else:
            raise ValueError(f"Unknown activation type: {act_type}")
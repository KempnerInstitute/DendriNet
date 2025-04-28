"""
activation_functions.py
=======================
Provides various activation/ reactivation modules, plus a factory to build
the desired activation function from config.
"""

import math

import torch
from torch import nn


class ParametricTanh(nn.Module):
    """
    A parametric tanh reactivation function with learnable parameters m and b.
    Both parameters are trainable.
    """

    def __init__(self, output_dim, init_m=1.5, init_b=1.0):
        super(ParametricTanh, self).__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)
        self.log_b = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)

        nn.init.constant_(self.log_m, math.log(init_m))
        nn.init.constant_(self.log_b, math.log(init_b))

    def forward(self, V):
        m = self.log_m.exp()  # slope
        b = self.log_b.exp()  # midpoint
        return (torch.tanh(m * (V - b)) + 1) / 2

    def initialize(self, slope_val, log_mid):
        self.log_m.data.fill_(math.log(slope_val))
        self.log_b.data.fill_(log_mid)


class ParametricTanhOnlyM(nn.Module):
    """
    A parametric tanh activation function where b is fixed and only m is
    trainable. The fixed value of b is provided via the fixed_b argument.
    """

    def __init__(self, output_dim, init_m=1.5, fixed_b=0.5):
        super(ParametricTanhOnlyM, self).__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)
        self.fixed_b = fixed_b  # fixed midpoint value
        with torch.no_grad():
            self.log_m.data.fill_(math.log(init_m))

    def forward(self, V):
        m = self.log_m.exp()  # slope
        b = self.fixed_b  # fixed midpoint
        return (torch.tanh(m * (V - b)) + 1) / 2


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


class ParametricLinearSigmoid(nn.Module):
    """
    A parametric activation function that interpolates between linear and sigmoid functions.
    The function is: h(x)*g(x) + f(x)*(1-g(x)) where:
    - h(x) = x (linear function)
    - g(x) = exp(x/2) (exponential weighting)
    - f(x) = 4/(1 + exp(-(mx + 2))) (sigmoid function)
    
    Parameters
    ----------
    output_dim : int
        Dimension of the output
    init_m : float
        Initial value for the slope parameter
    init_b : float or None
        Initial value for the bias parameter
    trainable : bool
        Whether to train the slope parameter (default: True)
    """
    def __init__(self, output_dim, init_m=4, init_b=None, trainable=True,
                 activation_wrapper=None):
        super(ParametricLinearSigmoid, self).__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), 
                                requires_grad=trainable)
        
        # Initialize trainable parameter
        if init_m == 0:
            init_m = 1e-8
        nn.init.constant_(self.log_m, math.log(init_m))

        # Fixed parameters for sigmoid
        self.fixed_L = 1.0
        if init_b is None:
            self.fixed_b = 2.0
        else:
            self.fixed_b = init_b
        
        # Wrapper for the activation function using ActivationFactory
        self.activation_wrapper = activation_wrapper
        
    def forward(self, x):
        m = self.log_m.exp()  # slope
        
        # Calculate the three component functions
        h_x = x  # linear function
        g_x = torch.exp(-3*x)  # exponential weighting
        f_x = self.fixed_L / (1 + torch.exp(-(m * x + self.fixed_b)))  # sigmoid
        
        # Combine using the interpolation formula
        if self.activation_wrapper is not None:
            act_wrap = ActivationFactory.create(self.activation_wrapper)
            return act_wrap(h_x * g_x + f_x * (1 - g_x))
        else:
            return h_x * g_x + f_x * (1 - g_x)


class ActivationFactory:
    """
    A factory for building different activation/ reactivation modules.
    """

    @staticmethod
    def create(
        act_type, output_dim=None, init_m=1.5, init_b=1.0, fixed_b=0.5, **kwargs
    ):
        """
        Create an activation module of the desired type.

        Parameters
        ----------
        act_type : str
            One of ["none", "param_tanh", "param_tanh_only_m", "relu", "sigmoid", "tanh"].
        output_dim : int or None
            Used by param_tanh and param_tanh_only_m (must be set if act_type is "param_tanh" or "param_tanh_only_m").
        init_m : float
            Initial value for m.
        init_b : float
            Initial value for b (used only for param_tanh).
        fixed_b : float
            The fixed value of b when using param_tanh_only_m.
        **kwargs : dict
            Additional keyword arguments are accepted but ignored (for backwards compatibility).

        Returns
        -------
        nn.Module
            The activation module.
        """
        act_type = act_type.lower()
        if act_type == "none":
            return nn.Identity()
        elif act_type == "param_tanh":
            if output_dim is None:
                raise ValueError("ParametricTanh requires output_dim.")
            return ParametricTanh(output_dim, init_m, init_b)
        elif act_type == "param_tanh_only_m":
            if output_dim is None:
                raise ValueError("ParametricTanhOnlyM requires output_dim.")
            return ParametricTanhOnlyM(output_dim, init_m, fixed_b)
        elif act_type == "relu":
            return SimpleReLU()
        elif act_type == "sigmoid":
            return SimpleSigmoid()
        elif act_type == "tanh":
            return SimpleTanh()
        elif act_type == "param_linear_sigmoid":
            if output_dim is None:
                raise ValueError("ParametricLinearSigmoid requires output_dim.")
            return ParametricLinearSigmoid(output_dim, init_m=init_m, init_b=init_b, **kwargs)
        else:
            raise ValueError(f"Unknown activation type: {act_type}")

"""
Activation Factory.

This module provides a factory for building different activation/reactivation modules.
"""

import torch.nn as nn

from dendritic_modeling.networks.activations.parametric import (
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricLinearTanhTransition,
    ParametricTanh,
    ParametricTanhOnlyM,
)


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
        if act_type is None:
            raise ValueError(
                "Activation type must be a string, got None. "
                "Use 'none' for an identity/no-op activation."
            )
        act_type = str(act_type).lower()
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
            return nn.ReLU()
        elif act_type == "sigmoid":
            return nn.Sigmoid()
        elif act_type == "tanh":
            return nn.Tanh()
        elif act_type == "param_linear_sigmoid":
            if output_dim is None:  # trainable init_m
                raise ValueError("ParametricLinearSigmoid requires output_dim.")
            return ParametricLinearSigmoid(
                output_dim, init_m=init_m, init_b=init_b, trainable=True, **kwargs
            )
        elif act_type == "linear_sigmoid":  # not trainable
            if output_dim is None:
                raise ValueError("ParametricLinearSigmoid requires output_dim.")
            return ParametricLinearSigmoid(
                output_dim, init_m=init_m, init_b=init_b, trainable=False, **kwargs
            )
        elif act_type == "param_linear_tanh":  # trainable init_m
            if output_dim is None:
                raise ValueError("ParametricLinearTanh requires output_dim.")
            return ParametricLinearTanh(
                output_dim, init_m=init_m, init_b=init_b, trainable=True, **kwargs
            )
        elif act_type == "linear_tanh":  # not trainable
            if output_dim is None:
                raise ValueError("ParametricLinearTanh requires output_dim.")
            return ParametricLinearTanh(
                output_dim, init_m=init_m, init_b=init_b, trainable=False, **kwargs
            )
        elif act_type == "param_linear_tanh_transition":  # trainable init_m, init_b
            # Map factory parameters to ParametricLinearTanhTransition parameters:
            # - init_b maps to init_threshold (the point where transition happens)
            # - init_m maps to init_transition_slope (controls transition curve steepness)
            return ParametricLinearTanhTransition(
                output_dim,
                init_threshold=init_b,
                linear_slope=1.0,
                init_transition_slope=init_m,
                trainable_threshold=True,
                trainable_transition_slope=True,
            )
        elif act_type == "linear_tanh_transition":  # not trainable
            # Map factory parameters to ParametricLinearTanhTransition parameters:
            # - init_b maps to init_threshold (the point where transition happens)
            # - init_m maps to init_transition_slope (controls transition curve steepness)
            return ParametricLinearTanhTransition(
                output_dim,
                init_threshold=init_b,
                linear_slope=1.0,
                init_transition_slope=init_m,
                trainable_threshold=False,
                trainable_transition_slope=False,
            )
        else:
            raise ValueError(f"Unknown activation type: {act_type}")


__all__ = ["ActivationFactory"]

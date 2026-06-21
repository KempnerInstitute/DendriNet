"""
Activation Factory.

This module provides a factory for building different activation/reactivation modules.
"""

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from dendritic_modeling.networks.activations.parametric import (
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricLinearTanhTransition,
    ParametricTanh,
    ParametricTanhOnlyM,
)

ActivationBuilder = Callable[..., nn.Module]


def _require_output_dim(output_dim: int | None, activation_name: str) -> int:
    if output_dim is None:
        raise ValueError(f"{activation_name} requires output_dim.")
    return output_dim


def _identity_builder(**_: Any) -> nn.Module:
    return nn.Identity()


def _relu_builder(**_: Any) -> nn.Module:
    return nn.ReLU()


def _sigmoid_builder(**_: Any) -> nn.Module:
    return nn.Sigmoid()


def _tanh_builder(**_: Any) -> nn.Module:
    return nn.Tanh()


def _param_tanh_builder(
    *, output_dim: int | None, init_m: float, init_b: float, **_: Any
) -> nn.Module:
    return ParametricTanh(
        _require_output_dim(output_dim, "ParametricTanh"),
        init_m,
        init_b,
    )


def _param_tanh_only_m_builder(
    *, output_dim: int | None, init_m: float, fixed_b: float, **_: Any
) -> nn.Module:
    return ParametricTanhOnlyM(
        _require_output_dim(output_dim, "ParametricTanhOnlyM"),
        init_m,
        fixed_b,
    )


def _param_linear_sigmoid_builder(
    *,
    output_dim: int | None,
    init_m: float,
    init_b: float,
    fixed_b: float,
    **kwargs: Any,
) -> nn.Module:
    return ParametricLinearSigmoid(
        _require_output_dim(output_dim, "ParametricLinearSigmoid"),
        init_m=init_m,
        init_b=init_b,
        trainable=True,
        **kwargs,
    )


def _linear_sigmoid_builder(
    *,
    output_dim: int | None,
    init_m: float,
    init_b: float,
    fixed_b: float,
    **kwargs: Any,
) -> nn.Module:
    return ParametricLinearSigmoid(
        _require_output_dim(output_dim, "ParametricLinearSigmoid"),
        init_m=init_m,
        init_b=init_b,
        trainable=False,
        **kwargs,
    )


def _param_linear_tanh_builder(
    *,
    output_dim: int | None,
    init_m: float,
    init_b: float,
    fixed_b: float,
    **kwargs: Any,
) -> nn.Module:
    return ParametricLinearTanh(
        _require_output_dim(output_dim, "ParametricLinearTanh"),
        init_m=init_m,
        init_b=init_b,
        trainable=True,
        **kwargs,
    )


def _linear_tanh_builder(
    *,
    output_dim: int | None,
    init_m: float,
    init_b: float,
    fixed_b: float,
    **kwargs: Any,
) -> nn.Module:
    return ParametricLinearTanh(
        _require_output_dim(output_dim, "ParametricLinearTanh"),
        init_m=init_m,
        init_b=init_b,
        trainable=False,
        **kwargs,
    )


def _param_linear_tanh_transition_builder(
    *, output_dim: int | None, init_m: float, init_b: float, **_: Any
) -> nn.Module:
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


def _linear_tanh_transition_builder(
    *, output_dim: int | None, init_m: float, init_b: float, **_: Any
) -> nn.Module:
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


_ACTIVATION_REGISTRY: dict[str, ActivationBuilder] = {
    "linear_sigmoid": _linear_sigmoid_builder,
    "linear_tanh": _linear_tanh_builder,
    "linear_tanh_transition": _linear_tanh_transition_builder,
    "none": _identity_builder,
    "param_linear_sigmoid": _param_linear_sigmoid_builder,
    "param_linear_tanh": _param_linear_tanh_builder,
    "param_linear_tanh_transition": _param_linear_tanh_transition_builder,
    "param_tanh": _param_tanh_builder,
    "param_tanh_only_m": _param_tanh_only_m_builder,
    "relu": _relu_builder,
    "sigmoid": _sigmoid_builder,
    "tanh": _tanh_builder,
}


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
        builder = _ACTIVATION_REGISTRY.get(act_type)
        if builder is None:
            raise ValueError(f"Unknown activation type: {act_type}")
        return builder(
            output_dim=output_dim,
            init_m=init_m,
            init_b=init_b,
            fixed_b=fixed_b,
            **kwargs,
        )


__all__ = ["ActivationFactory"]

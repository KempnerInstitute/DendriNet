"""
Activation functions module.

This module provides various activation/reactivation functions and a factory
to build the desired activation function from config.
"""

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.activations.parametric import (
    ParametricActivation,
    ParametricTanh,
    ParametricTanhOnlyM,
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricLinearTanhTransition,
)

__all__ = [
    # Factory
    "ActivationFactory",
    # Parametric activations
    "ParametricActivation",
    "ParametricTanh",
    "ParametricTanhOnlyM",
    "ParametricLinearSigmoid",
    "ParametricLinearTanh",
    "ParametricLinearTanhTransition"
] 
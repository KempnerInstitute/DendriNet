"""
Activation functions module.

This module provides various activation/reactivation functions and a factory
to build the desired activation function from config.
"""

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.activations.parametric import (
    ParametricActivation,
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricLinearTanhTransition,
    ParametricTanh,
    ParametricTanhOnlyM,
)

__all__ = [
    # Factory
    "ActivationFactory"
    # Parametric activations
    "ParametricActivation"
    "ParametricLinearSigmoid"
    "ParametricLinearTanh"
    "ParametricLinearTanhTransition"
    "ParametricTanh"
    "ParametricTanhOnlyM"
]

"""
Activation functions module.

This module provides various activation/reactivation functions and a factory
to build the desired activation function from config.
"""

from .factory import ActivationFactory
from .parametric import (
    ParametricTanh,
    ParametricTanhOnlyM,
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricLinearTanhTransition,
)
from .simple import SimpleReLU, SimpleSigmoid, SimpleTanh

__all__ = [
    # Factory
    "ActivationFactory",
    # Parametric activations
    "ParametricTanh",
    "ParametricTanhOnlyM",
    "ParametricLinearSigmoid",
    "ParametricLinearTanh",
    "ParametricLinearTanhTransition",
    # Simple activations
    "SimpleReLU",
    "SimpleSigmoid",
    "SimpleTanh",
] 
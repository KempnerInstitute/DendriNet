"""
Networks module.

This module contains various neural network components including layers,
architectures, activations, and input transforms.
"""

# Activations
from dendritic_modeling.networks.activations import (
    ActivationFactory,
    ParametricActivation,
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricLinearTanhTransition,
    ParametricTanh,
    ParametricTanhOnlyM,
)
from dendritic_modeling.networks.architectures import (
    MLP,
    AlexNet,
    BlockLinear,
    CNNDownsample,
    CNNUpsample,
    DeepstLinear,
    DendriNet,
    DendriticBranchLayer,
    ExcitationInhibitionLayer,
    ExcitationInhibitionNetwork,
    GradientScaler,
    Identity,
    IdentityInputTransform,
    IndexInputTransform,
    MatchedTotalParamMLP,
    MatchedActiveParamMLP,
    StochasticTopKLinear,
    TopKLinear,
    TransferLayer,
    VarianceTopKLinear,
    analytical_expectation_dbl_init,
)
from dendritic_modeling.networks.architectures.classical.autoencoder.base import (
    BaseAutoencoder,
    CNNAutoencoder,
    MLPAutoencoder,
)
from dendritic_modeling.networks.architectures.classical.autoencoder.vae import (
    CNNVariationalAutoencoder,
    MLPVariationalAutoencoder,
)
from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.input import InputNetTransform, SigmoidOutputNetwork

__all__ = [
    "ActivationFactory",
    "AlexNet",
    "BaseAutoencoder",
    "BaseNetwork",
    "BlockLinear",
    "CNNAutoencoder",
    "CNNDownsample",
    "CNNUpsample",
    "CNNVariationalAutoencoder",
    "DeepstLinear",
    "DendriNet",
    "DendriticBranchLayer",
    "ExcitationInhibitionLayer",
    "ExcitationInhibitionNetwork",
    "GradientScaler",
    "Identity",
    "IdentityInputTransform",
    "IndexInputTransform",
    "analytical_expectation_dbl_init",
    "InputNetTransform",
    "MatchedTotalParamMLP",
    "MatchedActiveParamMLP",
    "MLP",
    "MLPAutoencoder",
    "MLPVariationalAutoencoder",
    "ParametricActivation",
    "ParametricLinearSigmoid",
    "ParametricLinearTanh",
    "ParametricLinearTanhTransition",
    "ParametricTanh",
    "ParametricTanhOnlyM",
    "SigmoidOutputNetwork",
    "StochasticTopKLinear",
    "TopKLinear",
    "TransferLayer",
    "VarianceTopKLinear",
]

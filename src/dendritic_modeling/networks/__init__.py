"""
Networks module.

This module contains various neural network components including layers,
architectures, activations, and input transforms.
"""

# Activations
from dendritic_modeling.networks.activations import (
    ActivationFactory,
    ParametricActivation,
    ParametricTanh,
    ParametricTanhOnlyM,
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricLinearTanhTransition
)

from dendritic_modeling.networks.architectures import (
    Identity,
    AlexNet,
    CNNUpsample,
    CNNDownsample,
    MLP,
    MatchedParamMLP,
    ExcitationInhibitionLayer,
    ExcitationInhibitionNetwork,
    IdentityInputTransform,
    IndexInputTransform,
    TransferLayer,
    TopKLinear,
    StochasticTopKLinear,
    VarianceTopKLinear,
    DeepstLinear,
    BlockLinear,
    DendriticBranchLayer,
    DendriNet,
    GradientScaler,
    initialize_dendritic_branch_layer
)

from dendritic_modeling.networks.input_network import (
    InputNetTransform,
    SigmoidOutputNetwork
)

from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.architectures.classical.autoencoder.base import (
    BaseAutoencoder,
    MLPAutoencoder,
    CNNAutoencoder,
    MLPVariationalAutoencoder,
    CNNVariationalAutoencoder
)

__all__ = [
    "ActivationFactory",
    "ParametricActivation",
    "ParametricTanh",
    "ParametricTanhOnlyM",
    "ParametricLinearSigmoid",
    "ParametricLinearTanh",
    "ParametricLinearTanhTransition",
    "BaseNetwork",
    "Identity",
    "AlexNet",
    "CNNUpsample",
    "CNNDownsample",
    "MLP",
    "MatchedParamMLP",
    "ExcitationInhibitionLayer",
    "ExcitationInhibitionNetwork",
    "IdentityInputTransform",
    "IndexInputTransform",
    "TransferLayer",
    "TopKLinear",
    "StochasticTopKLinear",
    "VarianceTopKLinear",
    "DeepstLinear",
    "BlockLinear",
    "DendriticBranchLayer",
    "DendriNet",
    "GradientScaler",
    "initialize_dendritic_branch_layer",
    "BaseAutoencoder",
    "MLPAutoencoder",
    "CNNAutoencoder",
    "MLPVariationalAutoencoder",
    "CNNVariationalAutoencoder",
    "InputNetTransform",
    "SigmoidOutputNetwork"
]
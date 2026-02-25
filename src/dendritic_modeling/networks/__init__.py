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
    MatchedActiveParamMLP,
    MatchedTotalParamMLP,
    StochasticTopKLinear,
    TopKLinear,
    TransferLayer,
    VarianceTopKLinear,
    analytical_expectation_dbl_init,
)

try:
    from dendritic_modeling.networks.architectures import (
        DendriticRNNCell,
        DendriticRNNNetwork,
        RecurrentState,
    )
except ImportError:
    pass
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
    "MLP",
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
    "DendriticRNNCell",
    "DendriticRNNNetwork",
    "ExcitationInhibitionLayer",
    "ExcitationInhibitionNetwork",
    "GradientScaler",
    "Identity",
    "IdentityInputTransform",
    "IndexInputTransform",
    "InputNetTransform",
    "MLPAutoencoder",
    "MLPVariationalAutoencoder",
    "MatchedActiveParamMLP",
    "MatchedTotalParamMLP",
    "ParametricActivation",
    "ParametricLinearSigmoid",
    "ParametricLinearTanh",
    "ParametricLinearTanhTransition",
    "ParametricTanh",
    "ParametricTanhOnlyM",
    "RecurrentState",
    "SigmoidOutputNetwork",
    "StochasticTopKLinear",
    "TopKLinear",
    "TransferLayer",
    "VarianceTopKLinear",
    "analytical_expectation_dbl_init",
]

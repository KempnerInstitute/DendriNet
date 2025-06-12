"""
Networks module.

This module contains various neural network components including layers,
architectures, activations, and input transforms.
"""

# Activations
from .activations import (
    ActivationFactory,
    ParametricTanh,
    ParametricTanhOnlyM,
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricLinearTanhTransition,
    SimpleReLU,
    SimpleSigmoid,
    SimpleTanh,
)

# Input transforms
from .input_transforms import (
    InputNetTransform,
    IndexInputTransform,
    MLPInputTransform,
    CNNInputTransform,
    AlexNetInputTransform,
    SigmoidInputNetTransform,
)

# Layers
from .layers import (
    # Linear layers
    StochasticSynapseLinear,
    BlockLinear,
    BlockLinearDeepST,
    BlockLinearVariance,
    # Dendritic layers
    DendriticBranchLayer,
    ExcitationInhibitionLayer,
)

# Architectures
from .architectures import (
    # Dendritic
    DendriNet,
    # Excitation-Inhibition
    ExcitationInhibitionNetwork,
    MLPExcInhNetwork,
    # MLP
    MatchedParamMLP,
    SimpleMLPNet,
)

# MLP components
from .mlp import (
    FeatureExtractorMLP,
    SimpleMLPLayer,
)

__all__ = [
    # Activations
    "ActivationFactory",
    "ParametricTanh",
    "ParametricTanhOnlyM",
    "ParametricLinearSigmoid",
    "ParametricLinearTanh",
    "ParametricLinearTanhTransition",
    "SimpleReLU",
    "SimpleSigmoid",
    "SimpleTanh",
    # Input transforms
    "InputNetTransform",
    "IndexInputTransform",
    "MLPInputTransform",
    "CNNInputTransform",
    "AlexNetInputTransform",
    "SigmoidInputNetTransform",
    # Linear layers
    "StochasticSynapseLinear",
    "BlockLinear",
    "BlockLinearDeepST",
    "BlockLinearVariance",
    # Dendritic layers
    "DendriticBranchLayer",
    "ExcitationInhibitionLayer",
    # Architectures
    "DendriNet",
    "ExcitationInhibitionNetwork",
    "MLPExcInhNetwork",
    "MatchedParamMLP",
    "SimpleMLPNet",
    # MLP components
    "FeatureExtractorMLP",
    "SimpleMLPLayer",
] 
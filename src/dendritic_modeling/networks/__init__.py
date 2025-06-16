"""
Networks module.

This module contains various neural network components including layers,
architectures, activations, and input transforms.
"""

# Activations
from dendritic_modeling.networks.activations import (
    ActivationFactory,
    ParametricTanh,
    ParametricTanhOnlyM,
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricLinearTanhTransition,
    SimpleReLU,
    SimpleSigmoid,
    SimpleTanh
)

# Input transforms
from dendritic_modeling.networks.input_transforms import (
    InputNetTransform,
    IndexInputTransform,
    MLPInputTransform,
    CNNInputTransform,
    AlexNetInputTransform,
    SigmoidInputNetTransform
)

# Layers
from dendritic_modeling.networks.layers import (
    # Linear layers
    StochasticSynapseLinear,
    BlockLinear,
    BlockLinearDeepST,
    BlockLinearVariance,
    # Dendritic layers
    DendriticBranchLayer,
    ExcitationInhibitionLayer
)

# Architectures
from dendritic_modeling.networks.architectures import (
    # Dendritic
    DendriNet,
    # Excitation-Inhibition
    ExcitationInhibitionNetwork,
    MLPExcInhNetwork,
    # MLP
    MatchedParamMLP,
    SimpleMLPNet
)

# MLP components
from dendritic_modeling.networks.mlp import (
    FeatureExtractorMLP,
    SimpleMLPLayer
)

# Initialization utilities
from dendritic_modeling.networks.initialize import (
    compute_expectation_truncated_log_normal,
    initialize_dendritic_branch_layer
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
    # Initialization utilities
    "compute_expectation_truncated_log_normal",
    "initialize_dendritic_branch_layer",
] 
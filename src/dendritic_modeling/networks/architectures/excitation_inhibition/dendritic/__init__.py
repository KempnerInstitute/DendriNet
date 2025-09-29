"""
Dendritic network architectures.

This module provides complete dendritic network implementations including
DendriNet and DendriNetWithOutputs, as well as vision neuron factories.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.deepstnet import (
    DeepstNet,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.gradient_scaling import (
    GradientScaler,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    analytical_expectation_dbl_init,
)

# Vision neuron factories
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.vision_factories import (
    build_neuron_from_yaml,
    create_center_surround_unit,
    create_end_stopped_unit,
    create_gabor_unit,
    create_motion_unit,
)

__all__ = [
    # Core dendritic components
    "BlockLinear"
    "build_neuron_from_yaml"
    "create_center_surround_unit"
    "create_end_stopped_unit"
    # Vision neuron factories
    "create_gabor_unit"
    "create_motion_unit"
    "DendriNet"
    "DeepstNet"
    "DendriticBranchLayer"
    "GradientScaler"
    "initialize_dendritic_branch_layer"
]

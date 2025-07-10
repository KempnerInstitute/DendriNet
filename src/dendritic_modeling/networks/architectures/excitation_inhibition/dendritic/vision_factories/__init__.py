"""
Vision neuron factories for dendritic modeling.

This module provides factory functions for creating biologically-inspired
vision neurons using dendritic computation, including:
- Gabor orientation-selective units
- Center-surround receptive fields
- Motion-selective units
- End-stopped cells
"""

from .vision_units import (
    create_gabor_unit,
    create_center_surround_unit,
    create_motion_unit,
    create_end_stopped_unit,
    rf_on_center,
    rf_off_center,
    rf_gabor,
)

from .neuron_factory import (
    build_neuron_from_yaml,
    load_properties_config,
    get_stimulus_params,
    get_criterion,
)

__all__ = [
    # Vision units
    "create_gabor_unit",
    "create_center_surround_unit",
    "create_motion_unit",
    "create_end_stopped_unit",
    "rf_on_center",
    "rf_off_center",
    "rf_gabor",
    # Factory functions
    "build_neuron_from_yaml",
    "load_properties_config",
    "get_stimulus_params",
    "get_criterion",
]

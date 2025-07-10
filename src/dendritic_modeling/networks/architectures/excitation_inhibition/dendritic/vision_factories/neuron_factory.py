"""Factory for building dendritic neurons from configuration."""

import yaml
from pathlib import Path
from typing import Dict, Any

from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d
from .vision_units import (
    create_gabor_unit,
    create_center_surround_unit,
    create_motion_unit,
    create_end_stopped_unit,
)

__all__ = ["build_neuron_from_yaml", "load_properties_config"]


def load_properties_config(config_path: str = None) -> Dict[str, Any]:
    """Load visual properties configuration.
    
    Parameters
    ----------
    config_path : str or None
        Path to properties.yaml. If None, uses default.
        
    Returns
    -------
    config : dict
        Properties configuration
    """
    if config_path is None:
        # Use default config - go up to project root then into configs
        config_path = Path(__file__).parent.parent.parent.parent.parent.parent.parent.parent / "configs" / "properties.yaml"
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config


def build_neuron_from_yaml(property_tag: str, config: Dict[str, Any] = None) -> DendriteConv2d:
    """Build a dendritic neuron from property configuration.
    
    Parameters
    ----------
    property_tag : str
        Property tag from properties.yaml (e.g., 'V1_OSI', 'retina_dog')
    config : dict or None
        Properties config dict. If None, loads default.
        
    Returns
    -------
    neuron : DendriteConv2d
        Configured dendritic neuron
    """
    if config is None:
        config = load_properties_config()
        
    if property_tag not in config:
        raise ValueError(f"Unknown property tag: {property_tag}")
        
    prop_config = config[property_tag]
    template = prop_config["weight_template"]
    params = prop_config.get("params", {})
    
    # Build based on template type
    if template == "gabor":
        neuron = create_gabor_unit(
            orientation=params.get("orientation", 0),
            wavelength=params.get("wavelength", 8),
            sigma=params.get("sigma", 3.0)
        )
        
    elif template == "center_surround":
        neuron = create_center_surround_unit(
            kind=params.get("cell_type", "on-center"),
            center_size=params.get("center_size", 5),
            surround_size=params.get("surround_size", 11)
        )
        
    elif template == "motion_ds":
        neuron = create_motion_unit(
            direction=params.get("preferred_direction", "rightward"),
            kernel_size=params.get("kernel_size", 15)
        )
        
    elif template == "end_stopped":
        neuron = create_end_stopped_unit(
            preferred_length=params.get("preferred_length", 20),
            orientation=params.get("orientation", 90)
        )
        
    elif template == "gabor_pair":
        # Complex cell with multiple phase Gabors
        # For now, return single Gabor (extend later for full complex cell)
        neuron = create_gabor_unit(
            orientation=params.get("orientation", 0),
            wavelength=params.get("wavelength", 8),
            sigma=params.get("sigma", 3.0)
        )
        
    else:
        raise ValueError(f"Unknown weight template: {template}")
        
    return neuron


def get_stimulus_params(property_tag: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get stimulus parameters for a property.
    
    Parameters
    ----------
    property_tag : str
        Property tag
    config : dict or None
        Properties config
        
    Returns
    -------
    stim_params : dict
        Stimulus type and parameters
    """
    if config is None:
        config = load_properties_config()
        
    prop_config = config[property_tag]
    return prop_config["stimulus"]


def get_criterion(property_tag: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get success criterion for a property.
    
    Parameters
    ----------
    property_tag : str
        Property tag
    config : dict or None
        Properties config
        
    Returns
    -------
    criterion : dict
        Metric name and threshold
    """
    if config is None:
        config = load_properties_config()
        
    prop_config = config[property_tag]
    return prop_config["criterion"] 
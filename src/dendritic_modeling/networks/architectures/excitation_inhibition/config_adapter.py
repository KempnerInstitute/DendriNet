"""Config adapter for ExcitationInhibitionNetwork initialization."""

from typing import Any, Union

from omegaconf import DictConfig, OmegaConf

from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork,
)


def prepare_ei_network_params(
    config: Union[dict, DictConfig], input_dim: int
) -> dict[str, Any]:
    """
    Prepare parameters for ExcitationInhibitionNetwork from structured config.

    Args:
        config: Either a dict or DictConfig with structured parameters
        input_dim: Input dimension from encoder

    Returns:
        Flattened parameter dictionary ready for EINet initialization
    """
    # Convert to regular dict if needed
    if isinstance(config, DictConfig):
        architecture_dict = OmegaConf.to_container(config.architecture, resolve=True)
        connectivity_dict = OmegaConf.to_container(config.connectivity, resolve=True)
        transfer_dict = OmegaConf.to_container(config.transfer, resolve=True)
        morphology_dict = OmegaConf.to_container(config.morphology, resolve=True)
        implementation_dict = OmegaConf.to_container(
            config.implementation, resolve=True
        )
        sparsity_dict = OmegaConf.to_container(config.sparsity, resolve=True)
        reactivation_dict = OmegaConf.to_container(config.reactivation, resolve=True)
    else:
        architecture_dict = config.get("architecture", {})
        connectivity_dict = config.get("connectivity", {})
        transfer_dict = config.get("transfer", {})
        morphology_dict = config.get("morphology", {})
        implementation_dict = config.get("implementation", {})
        sparsity_dict = config.get("sparsity", {})
        reactivation_dict = config.get("reactivation", {})

    # Build base parameters
    params = {
        "input_dim": input_dim,
        **architecture_dict,
        **connectivity_dict,
        "transfer_params": transfer_dict,
        **morphology_dict,
        **implementation_dict,
    }

    # Add sparsity parameters with proper naming
    for key, value in sparsity_dict.items():
        if key == "deepst":
            for deepst_key, deepst_value in value.items():
                params[deepst_key] = deepst_value
        elif key == "gradient_scaling":
            params["topk_strategy"] = value
        else:
            params[f"topk_{key}"] = value

    # Add reactivation parameters with proper naming
    params["reactivate"] = reactivation_dict.pop("enabled", True)
    for key, value in reactivation_dict.items():
        if key == "type":
            params["reactivation_type"] = value
        elif key == "gradient_scaling":
            params["reactivation_strategy"] = value
        elif key.startswith("init_"):
            params[f"reactivation_{key}"] = value
        else:
            params[f"reactivation_{key}"] = value

    # Add blocklinear parameters
    if isinstance(config, DictConfig):
        blocklinear_dict = OmegaConf.to_container(config.blocklinear, resolve=True)
    else:
        blocklinear_dict = config.get("blocklinear", {})

    for key, value in blocklinear_dict.items():
        if key == "efficient":
            params["efficient_blocklinear"] = value
        elif key == "gradient_scaling":
            params["blocklinear_strategy"] = value
        else:
            params[f"blocklinear_{key}"] = value

    return params


class ConfigurableEINetwork(ExcitationInhibitionNetwork):
    """
    ExcitationInhibitionNetwork that is initialized from structured config.

    This is now the standard way to create EINet instances, providing a clean
    separation between configuration structure and implementation details.
    """

    def __init__(self, config: Union[dict, DictConfig], input_dim: int):
        """
        Initialize from structured config.

        Args:
            config: Structured configuration (e.g., core_config from YAML)
                   Must contain 'architecture', 'connectivity', 'transfer', etc. sections
            input_dim: Input dimension from encoder
        """
        params = prepare_ei_network_params(config, input_dim)
        super().__init__(**params)

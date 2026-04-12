from typing import Any, Optional, Union

import torch.nn as nn
from omegaconf import DictConfig

from dendritic_modeling.networks.architectures.classical.autoencoder import (
    CNNAutoencoder,
    CNNVariationalAutoencoder,
    MLPAutoencoder,
    MLPVariationalAutoencoder,
)
from dendritic_modeling.networks.architectures.classical.cnn import (
    AlexNet,
    CNNDownsample,
    CNNUpsample,
)
from dendritic_modeling.networks.architectures.classical.identity import Identity
from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP,
    EffectiveMatchedParamMLP,
    MatchedActiveParamMLP,
    MatchedTotalParamMLP,
    PointMatchedParamMLP,
    SparseStructuredMLP,
)
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ConfigurableEINetwork,
)

# All EINet type strings handled via ConfigurableEINetwork (+ dendritic_mlp special-cased)
_EINET_TYPES: set[str] = {
    "einet",
    "dendritic_shunting",
    "dendritic_additive",
    "dendritic_mlp",
    "dendritic_signed",
    "flat_shunting",
    "flat_additive",
    "flat_signed",
    "flat_mlp",
}


def get_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int] = None,
) -> nn.Module:
    """
    Factory method for creating network architectures.

    Args:
        type: Architecture type name
        parameters: Either flat parameters dict or structured config
        input_dim: Optional input dimension (needed for EINet from config)

    Returns:
        Initialized network module
    """
    type = (type or "").lower()

    # ------------------------------------------------------------------
    # EINet variants (require structured config)
    # ------------------------------------------------------------------
    if type in _EINET_TYPES:
        if not (
            isinstance(parameters, DictConfig)
            or (
                isinstance(parameters, dict)
                and "architecture" in parameters
                and "connectivity" in parameters
            )
        ):
            raise ValueError(
                "EINet now requires structured config with 'architecture', 'connectivity', etc. "
                "sections. Flat parameters are no longer supported."
            )
        if input_dim is None:
            raise ValueError(
                "input_dim required when passing structured config to EINet"
            )

        synapse_mode = None
        use_shunting = False
        weight_transform = None
        flatten_dendrites = False

        if "shunting" in type:
            use_shunting = True
        
        if "flat" in type:
            flatten_dendrites = True
        
        if "signed" in type:
            weight_transform = "identity"

        if "mlp" in type:
            synapse_mode = "mlp"
            weight_transform = "identity"

        # type == "einet": use arguments from config directly

        return ConfigurableEINetwork(
            config=parameters,
            input_dim=input_dim,
            synapse_mode=synapse_mode,
            use_shunting=use_shunting,
            weight_transform=weight_transform,
            flatten_dendrites=flatten_dendrites,
        )

    # ------------------------------------------------------------------
    # Point-neuron MLP baseline (parameter-matched, structured config)
    # ------------------------------------------------------------------
    if type in ["point_mlp", "mlp"] and (
        isinstance(parameters, DictConfig)
        or (
            isinstance(parameters, dict)
            and "architecture" in parameters
            and "connectivity" in parameters
        )
    ):
        if input_dim is None:
            raise ValueError("input_dim required when passing structured config to mlp")

        from dendritic_modeling.scripts.script_utils.config_utils import (
            prepare_ei_network_params,
        )

        params = prepare_ei_network_params(parameters, input_dim)
        return PointMatchedParamMLP(**params)

    if type == "total_param_mlp":
        return MatchedTotalParamMLP(parameters, input_dim)

    if type == "active_param_mlp":
        return MatchedActiveParamMLP(parameters, input_dim)

    if type == "ss_mlp":
        return SparseStructuredMLP(parameters, input_dim)

    if type == "ss_mlp_flat":
        return SparseStructuredMLP(parameters, input_dim, flatten_dendrites=True)

    # ------------------------------------------------------------------
    # All other architectures expect flat parameters
    # ------------------------------------------------------------------
    if isinstance(parameters, DictConfig):
        raise ValueError(
            f"Architecture {type} expects flat parameters, not structured config"
        )

    if type == "alexnet":
        return AlexNet(**parameters)
    elif type in ["cnn", "cnn_down", "cnndownsample", "cnn_downsample"]:
        return CNNDownsample(**parameters)
    elif type in ["cnn_up", "cnnupsample", "cnn_upsample"]:
        return CNNUpsample(**parameters)
    elif type == "mlp":
        return MLP(**parameters)
    elif type == "effectivematchedparammlp":
        return EffectiveMatchedParamMLP(**parameters)
    elif type == "mlpautoencoder":
        return MLPAutoencoder(**parameters)
    elif type == "cnnautoencoder":
        return CNNAutoencoder(**parameters)
    elif type == "mlpvariationalautoencoder":
        return MLPVariationalAutoencoder(**parameters)
    elif type == "cnnvariationalautoencoder":
        return CNNVariationalAutoencoder(**parameters)
    elif type in ["identity", "none", None]:
        return Identity(**parameters)
    else:
        raise ValueError(f"Invalid architecture type: {type}")

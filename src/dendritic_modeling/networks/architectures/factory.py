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
    MatchedParamMLP,
    PointMatchedParamMLP,
)
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ConfigurableEINetwork,
)


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
    # Structured (DictConfig) architectures / core-network variants
    # ------------------------------------------------------------------
    # We support selecting EI/shunting variants via `model.core.type`, since these
    # are architectural comparisons (not just a morphology toggle).
    if type in ["einet", "dendritic_shunting", "dendritic_additive", "dendritic_mlp"]:
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
        if type == "einet":
            return ConfigurableEINetwork(parameters, input_dim)

        # For the dendritic_* variants, flatten the config once and override
        # only the behavioral flags we intend to compare.
        from dendritic_modeling.networks.architectures.excitation_inhibition import (
            ExcitationInhibitionNetwork,
        )
        from dendritic_modeling.scripts.script_utils.config_utils import (
            prepare_ei_network_params,
        )

        params = prepare_ei_network_params(parameters, input_dim)
        params["synapse_mode"] = "ei"

        if type == "dendritic_shunting":
            params["use_shunting"] = True
        elif type == "dendritic_additive":
            params["use_shunting"] = False
        elif type == "dendritic_mlp":
            # Remove explicit E/I synapse constraint: merge inputs and allow signed weights.
            params["use_shunting"] = False
            params["synapse_mode"] = "mlp"
            params["weight_transform"] = "identity"
        else:
            raise ValueError(f"Unhandled structured dendritic type: {type}")

        return ExcitationInhibitionNetwork(**params)

    # Structured point-neuron MLP baseline (parameter-matched).
    # Preferred name: "point_mlp". We also accept "mlp" for backward compatibility.
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
        # Use the parameter-matched *point-neuron* baseline.
        return PointMatchedParamMLP(**params)

    # All other architectures expect flat parameters
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
    elif type == "matchedparammlp":
        return MatchedParamMLP(**parameters)
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

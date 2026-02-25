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

# Backward-compatible aliases: MR branch names -> canonical names
_EINET_ALIASES: dict[str, str] = {
    "einet_sh": "dendritic_shunting",
    "einet_ns": "dendritic_additive",
    "einet_sh_flat": "flat_shunting",
    "einet_ns_flat": "flat_additive",
}

# All EINet type strings handled via ConfigurableEINetwork (+ dendritic_mlp special-cased)
_EINET_TYPES: set[str] = {
    "einet",
    "dendritic_shunting",
    "dendritic_additive",
    "dendritic_mlp",
    "flat_shunting",
    "flat_additive",
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

    # Normalize MR aliases to canonical names
    type = _EINET_ALIASES.get(type, type)

    # ------------------------------------------------------------------
    # Recurrent architectures
    # ------------------------------------------------------------------
    if type in ("dendritic_rnn", "dendrnn", "vanilla_rnn"):
        from dendritic_modeling.config.recurrent import RecurrentConfig
        from dendritic_modeling.networks.architectures.recurrent import (
            DendriticRNNNetwork,
        )

        # Extract recurrent config from structured parameters
        if isinstance(parameters, DictConfig):
            from omegaconf import OmegaConf

            rec_cfg = OmegaConf.to_container(
                parameters.get("recurrent", {}), resolve=True
            )
        elif hasattr(parameters, "recurrent"):
            rec_cfg = parameters.recurrent
        elif isinstance(parameters, dict) and "recurrent" in parameters:
            rec_cfg = parameters["recurrent"]
        else:
            rec_cfg = {}

        # Build RecurrentConfig
        if isinstance(rec_cfg, RecurrentConfig):
            recurrent_config = rec_cfg
        elif isinstance(rec_cfg, dict):
            recurrent_config = RecurrentConfig(**rec_cfg)
        else:
            recurrent_config = rec_cfg

        if not recurrent_config.enabled:
            raise ValueError(
                f"type='{type}' requires recurrent.enabled=true. "
                f"For feedforward models, use type='dendritic_shunting' or similar."
            )

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating recurrent architecture (type='{type}')"
            )

        return DendriticRNNNetwork(recurrent_config, input_dim)

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

        # dendritic_mlp: signed weights via ExcitationInhibitionNetwork directly
        if type == "dendritic_mlp":
            from dendritic_modeling.networks.architectures.excitation_inhibition import (
                ExcitationInhibitionNetwork,
            )
            from dendritic_modeling.scripts.script_utils.config_utils import (
                prepare_ei_network_params,
            )

            params = prepare_ei_network_params(parameters, input_dim)
            params["synapse_mode"] = "mlp"
            params["use_shunting"] = False
            params["weight_transform"] = "identity"
            return ExcitationInhibitionNetwork(**params)

        # All other EINet variants go through ConfigurableEINetwork
        use_shunting: Optional[bool] = None
        flatten_dendrites: bool = False

        if type == "dendritic_shunting":
            use_shunting = True
        elif type == "dendritic_additive":
            use_shunting = False
        elif type == "flat_shunting":
            use_shunting = True
            flatten_dendrites = True
        elif type == "flat_additive":
            use_shunting = False
            flatten_dendrites = True
        # type == "einet": use_shunting=None reads from config

        return ConfigurableEINetwork(
            config=parameters,
            input_dim=input_dim,
            use_shunting=use_shunting,
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

    # ------------------------------------------------------------------
    # MR-style structured-config MLP baselines
    # ------------------------------------------------------------------
    if type in ["matchedtotalparammlp", "total_param_mlp"]:
        return MatchedTotalParamMLP(parameters, input_dim)

    if type in ["matchedactiveparammlp", "active_param_mlp"]:
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

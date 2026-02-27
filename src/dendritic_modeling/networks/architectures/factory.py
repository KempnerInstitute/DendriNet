from dataclasses import asdict, is_dataclass
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

_UNIFIED_EI_TYPES: set[str] = {
    "ei_unified",
    "unified_ei",
    "ei_net",
    "unified_einet",
}

_BASELINE_RNN_TYPES: dict[str, str] = {
    "gru": "gru",
    "lstm": "lstm",
    "vanilla_rnn": "vanilla",
    "rnn": "vanilla",
}


def _to_plain_mapping(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of config-like objects into a plain mapping."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, DictConfig):
        from omegaconf import OmegaConf

        converted = OmegaConf.to_container(obj, resolve=True)
        return converted if isinstance(converted, dict) else {}
    if is_dataclass(obj):
        converted = asdict(obj)
        return converted if isinstance(converted, dict) else {}
    # Fallback for custom config objects exposing attrs.
    try:
        return dict(obj)
    except Exception:
        return {}


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
    # Unified E-I architecture (opt-in, isolated from legacy code paths)
    # ------------------------------------------------------------------
    if type in _UNIFIED_EI_TYPES:
        from dendritic_modeling.networks.architectures.recurrent import (
            EILayerConfig,
            EINetwork,
            EINetworkConfig,
            PopulationConfig,
        )

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating unified E-I architecture (type='{type}')"
            )

        raw_params = _to_plain_mapping(parameters)
        unified_cfg = raw_params.get("unified_ei", raw_params.get("ei_unified", {}))
        if not unified_cfg and "layers" in raw_params:
            # Allow passing EINetworkConfig-shaped dict directly.
            unified_cfg = raw_params

        if not isinstance(unified_cfg, dict):
            raise ValueError(
                "Unified E-I config must be a mapping, got "
                f"{unified_cfg.__class__.__name__}"
            )

        unified_cfg = dict(unified_cfg)
        unified_cfg["input_dim"] = input_dim

        layers_cfg = []
        for layer in unified_cfg.get("layers", []):
            if isinstance(layer, EILayerConfig):
                layers_cfg.append(layer)
                continue
            if not isinstance(layer, dict):
                raise ValueError(
                    "Each unified E-I layer config must be a dict or EILayerConfig"
                )
            layer = dict(layer)
            exc = layer.get("excitatory")
            inh = layer.get("inhibitory")
            if isinstance(exc, dict):
                layer["excitatory"] = PopulationConfig(**exc)
            if isinstance(inh, dict):
                layer["inhibitory"] = PopulationConfig(**inh)
            layers_cfg.append(EILayerConfig(**layer))

        if not layers_cfg:
            raise ValueError(
                "Unified E-I architecture requires at least one layer in unified_ei.layers"
            )

        unified_cfg["layers"] = layers_cfg
        return EINetwork(EINetworkConfig(**unified_cfg))

    # ------------------------------------------------------------------
    # Baseline RNN models (GRU, LSTM, Vanilla RNN)
    # ------------------------------------------------------------------
    if type in _BASELINE_RNN_TYPES:
        from dendritic_modeling.networks.architectures.recurrent import (
            BaselineRNN,
            BaselineRNNConfig,
        )

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating baseline RNN (type='{type}')"
            )

        raw_params = _to_plain_mapping(parameters)
        baseline_cfg = raw_params.get("baseline_rnn", raw_params)
        if not isinstance(baseline_cfg, dict):
            baseline_cfg = {}
        baseline_cfg = dict(baseline_cfg)
        baseline_cfg["cell_type"] = _BASELINE_RNN_TYPES[type]
        baseline_cfg["input_dim"] = input_dim
        return BaselineRNN(BaselineRNNConfig(**baseline_cfg))

    # ------------------------------------------------------------------
    # Legacy recurrent type aliases (removed — use unified E-I instead)
    # ------------------------------------------------------------------
    if type in ("dendritic_rnn", "dendrnn"):
        raise ValueError(
            f"Architecture type '{type}' has been removed. "
            f"Use 'ei_unified' (or 'unified_ei', 'ei_net') with the new "
            f"EINetworkConfig format instead."
        )

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

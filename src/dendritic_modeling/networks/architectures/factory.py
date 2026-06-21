import logging
from collections.abc import Callable
from typing import Any, Optional, Union

import torch.nn as nn
from omegaconf import DictConfig

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.legacy import (
    canonicalize_unified_ei_config,
    normalize_transfer_config,
)
from dendritic_modeling.config.model_aliases import (
    get_core_morphology_alias_overrides,
    get_unified_ei_alias_overrides,
)
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
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ConfigurableEINetwork,
)
from dendritic_modeling.networks.architectures.factory_unified import (
    build_unified_ei_layer_configs,
)
from dendritic_modeling.networks.architectures.recurrent.structured_einet_factory import (
    _build_recurrent_einet_from_structured_config,
)
from dendritic_modeling.networks.architectures.registry import (
    build_registered_architecture,
    get_registered_architecture_names,
    reserve_architecture_names,
)

logger = logging.getLogger(__name__)

# All EINet type strings handled via ConfigurableEINetwork (+ dendritic_mlp special-cased)
_EINET_TYPES: set[str] = {
    "einet",
    "dendritic_shunting",
    "dendritic_additive",
    "dendritic_normalized_additive",
    "dendritic_mlp",
    "dendritic_signed",
    "flat_shunting",
    "flat_additive",
    "flat_normalized_additive",
    "flat_mlp",
    "flat_signed",
}

_UNIFIED_EI_TYPES: set[str] = {
    "ei_unified",
    "unified_ei",
    "ei_net",
    "unified_einet",
    # Parallel aliases that mirror the BP / feedforward vocabulary
    # (`dendritic_shunting`, `dendritic_additive`, `flat_shunting`,
    # `flat_additive`). An `rnn_` prefix disambiguates them from the
    # non-recurrent BP types. The factory below canonicalizes each alias
    # to the appropriate `use_shunting` / `use_additive_normalization`
    # pair on every configured population, leaving the `ei_unified` path
    # unchanged for backward compatibility.
    "rnn_dendritic_shunting",
    "rnn_dendritic_additive",
    "rnn_dendritic_normalized_additive",
    "rnn_flat_shunting",
    "rnn_flat_additive",
    "rnn_flat_normalized_additive",
}

# Spatial dendritic cores: flatten encoder spatial output → EINet → project.
# Tensor-map baseline: flatten → DendriNet → project (no spatial locality).
_SPATIAL_DENDRITIC_TYPES: set[str] = {
    "hierarchical_dendritic_tensor_map",
    "dendritic_tensor_map",
    "tensor_map",
}

# Spatial dendritic conv: F.unfold → shared DendriNet on local patches → fold.
# Preserves spatial structure (B, C_out, H_out, W_out).
# Canonical name: hierarchical_dendritic_conv
_SPATIAL_DENDRITIC_CONV_TYPES: set[str] = {
    "hierarchical_dendritic_conv",
    "spatial_dendritic_conv",
}

_BASELINE_RNN_TYPES: dict[str, str] = {
    "gru": "gru",
    "lstm": "lstm",
    "vanilla_rnn": "vanilla",
    "rnn": "vanilla",
}

_POPULATION_NETWORK_TYPES: set[str] = {
    "population_network",
}

_BUILTIN_ARCHITECTURE_TYPES: frozenset[str] = frozenset(
    set(_EINET_TYPES)
    | set(_UNIFIED_EI_TYPES)
    | set(_BASELINE_RNN_TYPES)
    | set(_POPULATION_NETWORK_TYPES)
    | set(_SPATIAL_DENDRITIC_TYPES)
    | set(_SPATIAL_DENDRITIC_CONV_TYPES)
    | {
        "active_param_mlp",
        "alexnet",
        "cnn",
        "cnn_down",
        "cnn_downsample",
        "cnn_up",
        "cnn_upsample",
        "cnnautoencoder",
        "cnnvariationalautoencoder",
        "cnndownsample",
        "cnnupsample",
        "effectivematchedparammlp",
        "identity",
        "mlp",
        "mlpautoencoder",
        "mlpvariationalautoencoder",
        "none",
        "pathway_router",
        "point_mlp",
        "router",
        "ss_mlp",
        "ss_mlp_flat",
        "total_param_mlp",
    }
)

reserve_architecture_names(_BUILTIN_ARCHITECTURE_TYPES)


def get_available_architectures() -> list[str]:
    """Return built-in and extension architecture names."""
    return sorted(
        set(_BUILTIN_ARCHITECTURE_TYPES) | set(get_registered_architecture_names())
    )


def _to_plain_mapping(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of config-like objects into a plain mapping."""
    return to_plain_dict(obj)


ArchitectureBuilder = Callable[
    [str, Union[dict[str, Any], DictConfig], Optional[int], Optional[int]],
    nn.Module,
]


def _build_unified_ei_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build the unified recurrent E/I architecture family."""
    del suffix_input_dim
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

    unified_cfg = canonicalize_unified_ei_config(unified_cfg)
    unified_cfg["input_dim"] = input_dim

    alias_overrides = get_unified_ei_alias_overrides(type)
    layers_cfg = build_unified_ei_layer_configs(
        layers=unified_cfg.get("layers", []),
        ei_layer_config_cls=EILayerConfig,
        population_config_cls=PopulationConfig,
        alias_overrides=alias_overrides,
        core_type=type,
    )

    if not layers_cfg:
        raise ValueError(
            "Unified E-I architecture requires at least one layer in unified_ei.layers"
        )

    unified_cfg["layers"] = layers_cfg
    return EINetwork(EINetworkConfig(**unified_cfg))


def _build_baseline_rnn_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build baseline GRU/LSTM/vanilla RNN cores."""
    del suffix_input_dim
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


def _build_population_network_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build canonical population-network layers."""
    del suffix_input_dim
    from dendritic_modeling.networks.architectures.recurrent import (
        PopulationLayerConfig,
        PopulationNetwork,
        PopulationNetworkConfig,
    )

    if input_dim is None:
        raise ValueError(
            f"input_dim required when creating population-network architecture (type='{type}')"
        )

    raw_params = _to_plain_mapping(parameters)
    mp_cfg = raw_params.get("population_network") or raw_params
    if not isinstance(mp_cfg, dict):
        mp_cfg = {}
    mp_cfg = dict(mp_cfg)
    if "transfer_params" in mp_cfg:
        mp_cfg["transfer_params"] = normalize_transfer_config(mp_cfg["transfer_params"])
    mp_cfg["input_dim"] = input_dim
    mp_cfg["layers"] = [
        (
            layer
            if isinstance(layer, PopulationLayerConfig)
            else PopulationLayerConfig(**layer)
        )
        for layer in mp_cfg.get("layers", [])
    ]
    return PopulationNetwork(PopulationNetworkConfig(**mp_cfg))


def _build_spatial_dendritic_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build flattened spatial dendritic tensor-map cores."""
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
        HierarchicalDendriticTensorMap,
    )

    if input_dim is None:
        raise ValueError(
            f"input_dim required when creating spatial dendritic core (type='{type}')"
        )

    return HierarchicalDendriticTensorMap(
        config=parameters,
        input_dim=input_dim,
        suffix_input_dim=suffix_input_dim,
    )


def _build_spatial_dendritic_conv_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build shared-patch spatial dendritic convolution cores."""
    del suffix_input_dim
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
        HierarchicalDendriticConv,
    )

    raw_params = _to_plain_mapping(parameters)
    spatial_cfg = _to_plain_mapping(raw_params.get("spatial", {}))
    kernel_size = spatial_cfg.get("kernel_size", 3)
    stride = spatial_cfg.get("stride", 1)
    padding = spatial_cfg.get("padding", 0)

    if input_dim is None:
        raise ValueError(
            f"input_dim required when creating spatial dendritic conv (type='{type}')"
        )

    # input_dim here is the number of input channels (C_in). For spatial
    # cores, input_dim = prefix_output_spec.channels.
    return HierarchicalDendriticConv(
        config=parameters,
        in_channels=input_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def _build_einet_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build structured feedforward/recurrent E/I architectures."""
    del suffix_input_dim
    raw_params = _to_plain_mapping(parameters)
    if not ("architecture" in raw_params and "connectivity" in raw_params):
        raise ValueError(
            "EINet now requires structured config with 'architecture', "
            "'connectivity', etc. sections. Flat parameters are no longer supported."
        )
    if input_dim is None:
        raise ValueError("input_dim required when passing structured config to EINet")

    recurrent_einet = _build_recurrent_einet_from_structured_config(
        type=type, raw_params=raw_params, input_dim=input_dim
    )
    if recurrent_einet is not None:
        return recurrent_einet

    synapse_mode: Optional[str] = None
    use_shunting: Optional[bool] = None
    weight_transform: Optional[str] = None
    flatten_dendrites = False
    core_morphology_overrides = get_core_morphology_alias_overrides(type)

    if "use_shunting" in core_morphology_overrides:
        use_shunting = bool(core_morphology_overrides["use_shunting"])

    if "flat" in type:
        flatten_dendrites = True

    if "signed" in type:
        weight_transform = "identity"
        use_shunting = False

    if "mlp" in type:
        synapse_mode = "mlp"
        weight_transform = "identity"
        use_shunting = False

    params_for_build = _to_plain_mapping(parameters)
    params_for_build.setdefault("type", type)

    return ConfigurableEINetwork(
        config=params_for_build,
        input_dim=input_dim,
        synapse_mode=synapse_mode,
        use_shunting=use_shunting,
        weight_transform=weight_transform,
        flatten_dendrites=flatten_dendrites,
    )


def _build_point_mlp_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build the point-neuron MLP baseline from structured E/I config."""
    del type, suffix_input_dim
    raw_params = _to_plain_mapping(parameters)
    if not ("architecture" in raw_params and "connectivity" in raw_params):
        raise ValueError(
            "point_mlp requires structured architecture/connectivity config"
        )
    if input_dim is None:
        raise ValueError(
            "input_dim required when passing structured config to point_mlp"
        )

    from dendritic_modeling.scripts.script_utils.config_utils import (
        prepare_ei_network_params,
    )

    params = prepare_ei_network_params(raw_params, input_dim)
    return PointMatchedParamMLP(**params)


def _ensure_flat_parameters(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
) -> dict[str, Any]:
    if isinstance(parameters, DictConfig):
        raise ValueError(
            f"Architecture {type} expects flat parameters, not structured config"
        )
    return parameters


def _build_flat_mlp_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build flat MLP, or point-MLP when passed structured config."""
    del suffix_input_dim
    raw_params = _to_plain_mapping(parameters)
    if "architecture" in raw_params and "connectivity" in raw_params:
        return _build_point_mlp_architecture(type, parameters, input_dim, None)
    return MLP(**_ensure_flat_parameters(type, parameters))


def _build_flat_architecture(
    cls: type[nn.Module],
) -> ArchitectureBuilder:
    def _builder(
        type: str,
        parameters: Union[dict[str, Any], DictConfig],
        input_dim: Optional[int],
        suffix_input_dim: Optional[int],
    ) -> nn.Module:
        del input_dim, suffix_input_dim
        return cls(**_ensure_flat_parameters(type, parameters))

    return _builder


def _build_total_param_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del type, suffix_input_dim
    return MatchedTotalParamMLP(parameters, input_dim)


def _build_active_param_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del type, suffix_input_dim
    return MatchedActiveParamMLP(parameters, input_dim)


def _build_sparse_structured_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del type, suffix_input_dim
    return SparseStructuredMLP(parameters, input_dim)


def _build_sparse_structured_flat_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del type, suffix_input_dim
    return SparseStructuredMLP(parameters, input_dim, flatten_dendrites=True)


def _build_pathway_router(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del input_dim, suffix_input_dim
    return PathwayRouter(**_ensure_flat_parameters(type, parameters))


def _build_identity(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del input_dim, suffix_input_dim
    return Identity(**_ensure_flat_parameters(type, parameters))


def _build_builtin_architecture_registry() -> dict[str, ArchitectureBuilder]:
    builders: dict[str, ArchitectureBuilder] = {}
    builders.update(dict.fromkeys(_UNIFIED_EI_TYPES, _build_unified_ei_architecture))
    builders.update(
        dict.fromkeys(_BASELINE_RNN_TYPES, _build_baseline_rnn_architecture)
    )
    builders.update(
        dict.fromkeys(_POPULATION_NETWORK_TYPES, _build_population_network_architecture)
    )
    builders.update(
        dict.fromkeys(_SPATIAL_DENDRITIC_TYPES, _build_spatial_dendritic_architecture)
    )
    builders.update(
        dict.fromkeys(
            _SPATIAL_DENDRITIC_CONV_TYPES, _build_spatial_dendritic_conv_architecture
        )
    )
    builders.update(dict.fromkeys(_EINET_TYPES, _build_einet_architecture))
    builders.update(
        {
            "point_mlp": _build_point_mlp_architecture,
            "mlp": _build_flat_mlp_architecture,
            "total_param_mlp": _build_total_param_mlp,
            "active_param_mlp": _build_active_param_mlp,
            "ss_mlp": _build_sparse_structured_mlp,
            "ss_mlp_flat": _build_sparse_structured_flat_mlp,
            "alexnet": _build_flat_architecture(AlexNet),
            "cnn": _build_flat_architecture(CNNDownsample),
            "cnn_down": _build_flat_architecture(CNNDownsample),
            "cnndownsample": _build_flat_architecture(CNNDownsample),
            "cnn_downsample": _build_flat_architecture(CNNDownsample),
            "cnn_up": _build_flat_architecture(CNNUpsample),
            "cnnupsample": _build_flat_architecture(CNNUpsample),
            "cnn_upsample": _build_flat_architecture(CNNUpsample),
            "effectivematchedparammlp": _build_flat_architecture(
                EffectiveMatchedParamMLP
            ),
            "mlpautoencoder": _build_flat_architecture(MLPAutoencoder),
            "cnnautoencoder": _build_flat_architecture(CNNAutoencoder),
            "mlpvariationalautoencoder": _build_flat_architecture(
                MLPVariationalAutoencoder
            ),
            "cnnvariationalautoencoder": _build_flat_architecture(
                CNNVariationalAutoencoder
            ),
            "pathway_router": _build_pathway_router,
            "router": _build_pathway_router,
            "identity": _build_identity,
            "none": _build_identity,
        }
    )
    return builders


_BUILTIN_ARCHITECTURE_BUILDERS: dict[str, ArchitectureBuilder] = (
    _build_builtin_architecture_registry()
)


def get_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int] = None,
    suffix_input_dim: Optional[int] = None,
) -> nn.Module:
    """
    Factory method for creating network architectures.

    Args:
        type: Architecture type name
        parameters: Either flat parameters dict or structured config
        input_dim: Optional input dimension (needed for EINet from config)
        suffix_input_dim: Optional suffix input dimension. Used by spatial
            dendritic cores (e.g., ``hierarchical_dendritic_tensor_map``) to
            add an output projection when the EINet output dim doesn't match
            the downstream decoder's expected input.

    Returns:
        Initialized network module
    """
    type = (type or "").lower()
    if type in {"ei_net", "unified_einet"}:
        logger.warning(
            "Architecture type %r is deprecated; use 'unified_ei' instead.",
            type,
        )
    builder = _BUILTIN_ARCHITECTURE_BUILDERS.get(type)
    if builder is not None:
        return builder(type, parameters, input_dim, suffix_input_dim)

    registered = build_registered_architecture(
        type,
        parameters,
        input_dim=input_dim,
        suffix_input_dim=suffix_input_dim,
    )
    if registered is not None:
        return registered

    raise ValueError(f"Invalid architecture type: {type}")

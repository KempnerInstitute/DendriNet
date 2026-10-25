import dataclasses
import logging
import math
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from dendritic_modeling.config.affine_mlp import AffineMLPConfig
from dendritic_modeling.config.biological import resolve_biological_neuron
from dendritic_modeling.config.conversion import to_plain_dict as _to_plain_mapping
from dendritic_modeling.config.core_sections import BUILTIN_CORE_TYPES
from dendritic_modeling.config.model_aliases import get_core_morphology_alias_overrides
from dendritic_modeling.config.recurrence import _STRUCTURED_EI_TYPES, _UNIFIED_EI_TYPES
from dendritic_modeling.config.seeds import derive_seed
from dendritic_modeling.config.shared_projection import SharedProjectionShuntingConfig
from dendritic_modeling.networks.architectures.biological_constraints import (
    enforce_biological_neuron,
)
from dendritic_modeling.networks.architectures.classical.autoencoder import (
    BaseAutoencoder,
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
    DirectActiveMatchedPointBottleneck,
    EffectiveMatchedParamMLP,
    MatchedActiveParamMLP,
    MatchedTotalParamMLP,
    PointMatchedParamMLP,
    SparseActiveMatchedPointAffine,
    SparseStructuredMLP,
    build_affine_mlp,
)
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ConfigurableEINetwork,
)
from dendritic_modeling.networks.architectures.factory_recurrent import (
    _BASELINE_RNN_TYPES,
    _HETEROGENEOUS_LEAK_CTRNN_TYPES,
    _LEGENDRE_MEMORY_TYPES,
    _POPULATION_NETWORK_TYPES,
    _build_baseline_rnn_architecture,
    _build_heterogeneous_leak_ctrnn_architecture,
    _build_legendre_memory_architecture,
    _build_population_network_architecture,
)
from dendritic_modeling.networks.architectures.factory_spatial import (
    SPATIAL_DENDRITIC_CONV_TYPES,
    SPATIAL_DENDRITIC_TYPES,
    SPATIAL_PATCH_POINT_TYPES,
    _build_spatial_dendritic_architecture,
    _build_spatial_dendritic_conv_architecture,
    _build_spatial_patch_point_architecture,
)
from dendritic_modeling.networks.architectures.factory_unified import (
    _build_unified_ei_architecture,
)
from dendritic_modeling.networks.architectures.recurrent.structured_einet_factory import (
    _build_recurrent_einet_from_structured_config,
    reject_recurrent_request_for_feedforward_core,
)
from dendritic_modeling.networks.architectures.registry import (
    build_registered_architecture,
    get_registered_architecture_names,
    reserve_architecture_names,
)

logger = logging.getLogger(__name__)

# Structured EINet types (ConfigurableEINetwork, or the recurrent EINetwork
# when connectivity.rec_* counts are set) and unified E/I types. The registries
# live in config.recurrence so configuration code can use them without torch.
_EINET_TYPES = _STRUCTURED_EI_TYPES

# The configuration package lists the built-in names without importing torch;
# every one of them has a builder below.
_BUILTIN_ARCHITECTURE_TYPES: frozenset[str] = BUILTIN_CORE_TYPES

reserve_architecture_names(_BUILTIN_ARCHITECTURE_TYPES)


def get_available_architectures() -> list[str]:
    """Return built-in and extension architecture names."""
    return sorted(
        set(_BUILTIN_ARCHITECTURE_TYPES) | set(get_registered_architecture_names())
    )


ArchitectureBuilder = Callable[
    [str, Union[dict[str, Any], DictConfig], Optional[int], Optional[int]],
    nn.Module,
]


@contextmanager
def _keyed_initialization_stream(seed, core_type):
    """Construct a control core from a CPU stream keyed by an explicit seed.

    ``None`` keeps the caller's global stream. The key keeps a seed equal to
    the run's model seed from replaying the draws of an encoder built from it.
    """
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(
            derive_seed(int(seed), f"core.{core_type}.initialization")
        )
        yield


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
    if raw_params.get("population_network"):
        reference = _build_population_network_architecture(
            "population_network", raw_params, input_dim, None
        )
        return _build_reference_matched_mlp(
            reference=reference,
            input_dim=input_dim,
            match_mode="effective",
        )
    if not ("architecture" in raw_params and "connectivity" in raw_params):
        raise ValueError(
            "point_mlp requires structured architecture/connectivity config or "
            "a population_network reference config"
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


def _build_direct_active_matched_point_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build a one-affine-layer point control matched to active E/I params."""

    del type
    raw_params = _to_plain_mapping(parameters)
    if not ("architecture" in raw_params and "connectivity" in raw_params):
        raise ValueError(
            "direct_active_matched_point requires structured "
            "architecture/connectivity reference config"
        )
    if input_dim is None:
        raise ValueError("input_dim required when building direct_active_matched_point")

    reference_output_sizes = raw_params["architecture"].get(
        "excitatory_layer_sizes", []
    )
    if not reference_output_sizes:
        raise ValueError("direct_active_matched_point requires excitatory_layer_sizes")
    configured_output_dim = int(reference_output_sizes[-1])
    if suffix_input_dim is not None and configured_output_dim != int(suffix_input_dim):
        raise ValueError(
            "direct_active_matched_point zero-padded interface does not match "
            f"the suffix: configured={configured_output_dim}, "
            f"suffix_input_dim={int(suffix_input_dim)}"
        )

    from dendritic_modeling.scripts.script_utils.config_utils import (
        prepare_ei_network_params,
    )

    params = prepare_ei_network_params(raw_params, input_dim)
    return DirectActiveMatchedPointBottleneck(**params)


def _build_sparse_active_matched_point_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build an all-output-active fixed-index affine point control."""

    del type
    raw_params = _to_plain_mapping(parameters)
    if not ("architecture" in raw_params and "connectivity" in raw_params):
        raise ValueError(
            "sparse_active_matched_point requires structured "
            "architecture/connectivity reference config"
        )
    if input_dim is None:
        raise ValueError("input_dim required when building sparse_active_matched_point")

    reference_output_sizes = raw_params["architecture"].get(
        "excitatory_layer_sizes", []
    )
    if not reference_output_sizes:
        raise ValueError("sparse_active_matched_point requires excitatory_layer_sizes")
    configured_output_dim = int(reference_output_sizes[-1])
    if suffix_input_dim is not None and configured_output_dim != int(suffix_input_dim):
        raise ValueError(
            "sparse_active_matched_point output does not match the suffix: "
            f"configured={configured_output_dim}, "
            f"suffix_input_dim={int(suffix_input_dim)}"
        )

    from dendritic_modeling.scripts.script_utils.config_utils import (
        prepare_ei_network_params,
    )

    params = prepare_ei_network_params(raw_params, input_dim)
    if "target_active_parameters" in raw_params:
        params["target_active_parameters"] = raw_params["target_active_parameters"]
    return SparseActiveMatchedPointAffine(**params)


def _build_reference_matched_mlp(
    *,
    reference: nn.Module,
    input_dim: Optional[int],
    match_mode: str,
    activation: str = "relu",
) -> nn.Module:
    if input_dim is None:
        raise ValueError("input_dim required when building matched MLP baselines")

    output_dim = int(reference.output_dim)
    if match_mode == "effective" and hasattr(reference, "get_effective_params"):
        target_params = int(reference.get_effective_params())
    elif match_mode in {"effective", "total"}:
        target_params = sum(
            p.numel() for p in reference.parameters() if p.requires_grad
        )
    else:
        raise ValueError(f"Unknown matched MLP mode: {match_mode!r}")

    # One hidden layer, matching PointMatchedParamMLP semantics:
    # (input_dim * H + H) + (H * output_dim + output_dim) >= target_params.
    denom = int(input_dim) + output_dim + 1
    hidden_width = max(1, math.ceil(max(0, target_params - output_dim) / denom))
    return MLP(
        input_dim=int(input_dim),
        hidden_dims=[hidden_width],
        activation=activation,
        output_dim=output_dim,
    )


def _ensure_flat_parameters(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
) -> dict[str, Any]:
    if isinstance(parameters, DictConfig):
        raise ValueError(
            f"Architecture {type} expects flat parameters, not structured config"
        )
    if dataclasses.is_dataclass(parameters):
        raise ValueError(
            f"Architecture {type!r} is an encoder/decoder building block built "
            "from flat parameters; model.core cannot configure it. Use it as "
            "model.encoder.type or model.decoder.type, or choose a core type "
            "(affine_mlp builds a plain dense or sparse MLP core)."
        )
    return parameters


def _build_feature_readout_architecture(type, parameters, input_dim, suffix_input_dim):
    from dendritic_modeling.config.core_sections import reject_unread_core_sections
    from dendritic_modeling.config.feature_readout import FeatureReadoutConfig
    from dendritic_modeling.config.keys import build_section
    from dendritic_modeling.config.model import CoreConfig
    from dendritic_modeling.networks.architectures.classical.feature_readout import (
        build_feature_readout,
    )

    raw = _to_plain_mapping(parameters)
    if "feature_readout" in raw:
        if str(raw.get("type", type)).strip().lower() != type:
            raise ValueError(
                "feature_readout builder and configured core type disagree"
            )
        core = (
            parameters
            if isinstance(parameters, CoreConfig)
            else CoreConfig(**{"type": type, **raw})
        )
        reject_unread_core_sections(core, raw)
    payload = raw.get("feature_readout", raw)
    if payload is None:
        raise ValueError("feature_readout requires its configuration section")
    config = build_section(FeatureReadoutConfig, payload, section="feature_readout")
    config.validate()
    if input_dim is not None and input_dim != math.prod(config.input_shape):
        raise ValueError("feature_readout input dimension does not match the encoder")
    if (
        suffix_input_dim is not None
        and suffix_input_dim != config.readout.dimensions[-1]
    ):
        raise ValueError("feature_readout output dimension does not match the decoder")
    seed = raw.get("initialization_seed")
    if seed is None:
        return build_feature_readout(config)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        return build_feature_readout(config)


def _build_affine_mlp_architecture(type, parameters, input_dim, suffix_input_dim):
    del type
    raw = _to_plain_mapping(parameters)
    payload = raw.get("affine_mlp", raw)
    if payload is None:
        raise ValueError("affine_mlp requires an explicit configuration")
    config = AffineMLPConfig(**payload)
    config.validate()
    if input_dim is not None and input_dim != config.dimensions[0]:
        raise ValueError("affine_mlp input dimension does not match the encoder")
    if suffix_input_dim is not None and suffix_input_dim != config.dimensions[-1]:
        raise ValueError("affine_mlp output dimension does not match the decoder")
    return build_affine_mlp(config)


def _build_shared_projection_architecture(
    type, parameters, input_dim, suffix_input_dim
):
    from dendritic_modeling.config.core_sections import reject_unread_core_sections
    from dendritic_modeling.networks.architectures.excitation_inhibition.shared_projection import (
        build_shared_projection_shunting,
    )

    raw = _to_plain_mapping(parameters)
    payload = raw.get("shared_projection_shunting", raw)
    if payload is None:
        raise ValueError(
            "shared_projection_shunting requires an explicit configuration"
        )
    if "shared_projection_shunting" in raw:
        # Direct structured construction must reject ignored options too.
        from dendritic_modeling.config.model import CoreConfig

        core = CoreConfig(**{"type": type, **raw})
        reject_unread_core_sections(core, raw)
    config = SharedProjectionShuntingConfig(**payload)
    config.validate()
    if input_dim is not None and input_dim != config.input_dim:
        raise ValueError(
            "shared_projection_shunting input dimension does not match the encoder"
        )
    if suffix_input_dim is not None and suffix_input_dim != config.output_dim:
        raise ValueError(
            "shared_projection_shunting output dimension does not match the decoder"
        )
    return build_shared_projection_shunting(config)


def _build_routed_ei_architecture(type, parameters, input_dim, suffix_input_dim):
    from dendritic_modeling.config.core_sections import reject_unread_core_sections
    from dendritic_modeling.config.routed_ei import RoutedEIConfig
    from dendritic_modeling.networks.architectures.excitation_inhibition.routed_ei import (
        build_routed_ei,
    )

    raw = _to_plain_mapping(parameters)
    if "routed_ei" in raw:
        from dendritic_modeling.config.model import CoreConfig

        if str(raw.get("type", type)).strip().lower() != type:
            raise ValueError("routed_ei builder and configured core type disagree")
        # Keep typed nested sections so the shared validator recognizes the
        # topology seeds filled by training setup, rather than warning that
        # those default-only sections were explicitly requested by the user.
        core = (
            parameters
            if isinstance(parameters, CoreConfig)
            else CoreConfig(**{"type": type, **raw})
        )
        reject_unread_core_sections(core, raw)
    payload = raw.get("routed_ei", raw)
    if payload is None:
        raise ValueError("routed_ei requires its configuration section")
    config = RoutedEIConfig(**payload)
    config.validate()
    if input_dim is not None and input_dim != 2 * config.features:
        raise ValueError("routed_ei encoder output must contain concatenated [E, I]")
    if suffix_input_dim is not None and suffix_input_dim != config.output_dim:
        raise ValueError("routed_ei output dimension does not match the decoder")
    seed = raw.get("initialization_seed")
    with _keyed_initialization_stream(seed, "routed_ei"):
        return build_routed_ei(config)


def _build_flat_mlp_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build a plain MLP from flat parameters (decoders and direct calls)."""
    del input_dim, suffix_input_dim
    raw_params = _to_plain_mapping(parameters)
    if "architecture" in raw_params and "connectivity" in raw_params:
        raise ValueError(
            "type 'mlp' takes flat MLP parameters, not a structured core config "
            "(which used to be replaced silently by the parameter-matched point "
            "MLP). Use type: point_mlp for that control, or type: affine_mlp "
            "with model.core.affine_mlp for a plain dense or sparse MLP."
        )
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

    # Every build returns this class (see builtin_architecture_class).
    _builder.architecture_class = cls  # type: ignore[attr-defined]
    return _builder


def _build_total_param_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del type, suffix_input_dim
    raw_params = _to_plain_mapping(parameters)
    if raw_params.get("population_network"):
        reference = _build_population_network_architecture(
            "population_network", raw_params, input_dim, None
        )
        return _build_reference_matched_mlp(
            reference=reference,
            input_dim=input_dim,
            match_mode="total",
        )
    return MatchedTotalParamMLP(parameters, input_dim)


def _build_active_param_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del type, suffix_input_dim
    raw_params = _to_plain_mapping(parameters)
    if raw_params.get("population_network"):
        reference = _build_population_network_architecture(
            "population_network", raw_params, input_dim, None
        )
        return _build_reference_matched_mlp(
            reference=reference,
            input_dim=input_dim,
            match_mode="effective",
        )
    return MatchedActiveParamMLP(parameters, input_dim)


def _reject_feedforward_core_options(
    type: str, parameters: Union[dict[str, Any], DictConfig]
) -> None:
    raw_params = _to_plain_mapping(parameters)
    reject_recurrent_request_for_feedforward_core(type, raw_params)
    if raw_params.get("initialization_seed") is not None:
        # ss_mlp copies only shapes from a reference EINet; its own layers are
        # initialized from the global stream, so a keyed seed cannot apply.
        raise ValueError(
            f"core type {type!r} does not support initialization_seed; its "
            "sparse-structured layers use the global torch RNG"
        )


def _build_sparse_structured_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del suffix_input_dim
    _reject_feedforward_core_options(type, parameters)
    return SparseStructuredMLP(parameters, input_dim)


def _build_sparse_structured_flat_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del suffix_input_dim
    _reject_feedforward_core_options(type, parameters)
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
    del suffix_input_dim
    if dataclasses.is_dataclass(parameters):
        # An identity core passes the encoder output to the decoder.
        if input_dim is None:
            raise ValueError(f"core type {type!r} needs the encoder output size")
        return Identity(input_dim=input_dim)
    return Identity(**_ensure_flat_parameters(type, parameters))


def _build_dendritic_alexnet_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build an end-to-end AlexNet-shaped dendritic vision network."""

    del suffix_input_dim
    from dendritic_modeling.networks.architectures.vision import DendriticAlexNet

    raw_params = _to_plain_mapping(parameters)
    reject_recurrent_request_for_feedforward_core(type, raw_params)
    input_channels = int(
        raw_params.get("vision", {}).get(
            "input_channels", 3 if input_dim is None else input_dim
        )
    )
    return DendriticAlexNet(raw_params, input_channels=input_channels)


def _build_builtin_architecture_registry() -> dict[str, ArchitectureBuilder]:
    builders: dict[str, ArchitectureBuilder] = {}
    builders.update(dict.fromkeys(_UNIFIED_EI_TYPES, _build_unified_ei_architecture))
    builders.update(
        dict.fromkeys(_BASELINE_RNN_TYPES, _build_baseline_rnn_architecture)
    )
    builders.update(
        dict.fromkeys(
            _HETEROGENEOUS_LEAK_CTRNN_TYPES,
            _build_heterogeneous_leak_ctrnn_architecture,
        )
    )
    builders.update(
        dict.fromkeys(_LEGENDRE_MEMORY_TYPES, _build_legendre_memory_architecture)
    )
    builders.update(
        dict.fromkeys(_POPULATION_NETWORK_TYPES, _build_population_network_architecture)
    )
    builders.update(
        dict.fromkeys(SPATIAL_DENDRITIC_TYPES, _build_spatial_dendritic_architecture)
    )
    builders.update(
        dict.fromkeys(
            SPATIAL_DENDRITIC_CONV_TYPES, _build_spatial_dendritic_conv_architecture
        )
    )
    builders.update(
        dict.fromkeys(
            SPATIAL_PATCH_POINT_TYPES, _build_spatial_patch_point_architecture
        )
    )
    builders.update(dict.fromkeys(_EINET_TYPES, _build_einet_architecture))
    builders.update(
        {
            "direct_active_matched_point": (
                _build_direct_active_matched_point_architecture
            ),
            "dendritic_alexnet": _build_dendritic_alexnet_architecture,
            "sparse_active_matched_point": (
                _build_sparse_active_matched_point_architecture
            ),
            "point_mlp": _build_point_mlp_architecture,
            "mlp": _build_flat_mlp_architecture,
            "affine_mlp": _build_affine_mlp_architecture,
            "feature_readout": _build_feature_readout_architecture,
            "shared_projection_shunting": _build_shared_projection_architecture,
            "routed_ei": _build_routed_ei_architecture,
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
    biological = resolve_biological_neuron(
        type, _declared_biological_neuron(parameters)
    )
    if type in {"ei_net", "unified_einet"}:
        logger.warning(
            "Architecture type %r is deprecated; use 'unified_ei' instead.",
            type,
        )
    builder = _BUILTIN_ARCHITECTURE_BUILDERS.get(type)
    if builder is not None:
        network = builder(type, parameters, input_dim, suffix_input_dim)
    else:
        network = build_registered_architecture(
            type,
            parameters,
            input_dim=input_dim,
            suffix_input_dim=suffix_input_dim,
        )
        if network is None:
            raise ValueError(f"Invalid architecture type: {type}")
    if biological:
        enforce_biological_neuron(type, network)
    return network


def builtin_architecture_class(name: str) -> Optional[type[nn.Module]]:
    """The class that built-in architecture ``name`` always builds, or None.

    None means that only building the architecture tells its class: the name
    is not built in (an extension registers it), or its builder chooses among
    classes from the parameters.
    """
    builder = _BUILTIN_ARCHITECTURE_BUILDERS.get((name or "").lower())
    return getattr(builder, "architecture_class", None)


def builds_autoencoder(name: str) -> Optional[bool]:
    """Whether architecture ``name`` builds an autoencoder (``BaseAutoencoder``).

    Answers for every built-in architecture from its name alone, so a run can
    reject an autoencoder request before it builds anything. None for a name
    an extension registers, whose class is known only once it is built.
    """
    if (name or "").lower() not in _BUILTIN_ARCHITECTURE_BUILDERS:
        return None
    cls = builtin_architecture_class(name)
    return cls is not None and issubclass(cls, BaseAutoencoder)


def _declared_biological_neuron(parameters: Any) -> Any:
    """Return the core's ``biological_neuron`` declaration, or None."""
    if isinstance(parameters, (dict, DictConfig)):
        return parameters.get("biological_neuron")
    return getattr(parameters, "biological_neuron", None)

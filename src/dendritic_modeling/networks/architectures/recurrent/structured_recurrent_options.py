"""Build-option validation for structured recurrent E/I factory configs."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.inhibitory_wiring import check_inhibitory_wiring
from dendritic_modeling.config.model_aliases import (
    get_core_morphology_alias_overrides,
    get_structured_recurrent_alias_overrides,
    warn_alias_conflicts,
)
from dendritic_modeling.networks.activations import resolve_dendritic_activation
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_types import (
    _RecurrentReactivationOptions,
    _RecurrentSynapseCounts,
    _StructuredRecurrentBuildOptions,
    _StructuredRecurrentSections,
)


def _validate_structured_recurrent_type(core_type: str) -> None:
    # The recurrent bridge builds positive-weight dendritic populations; it has
    # no signed-weight, merged-MLP-bank or flattened-tree variant, so these
    # aliases would silently lose their defining semantics.
    if core_type in {
        "dendritic_mlp",
        "dendritic_signed",
        "flat_mlp",
        "flat_signed",
        "flat_shunting",
        "flat_additive",
        "flat_normalized_additive",
    }:
        raise ValueError(
            f"Recurrent mode is not supported for architecture type '{core_type}'. "
            "Use 'EINet', 'dendritic_shunting', 'dendritic_additive', or "
            "'dendritic_normalized_additive'."
        )


def _reject_feedforward_only_options(sections: _StructuredRecurrentSections) -> None:
    """Reject EINet options the recurrent population bridge does not translate."""
    reactivation = sections.reactivation
    unsupported = []
    if reactivation.get("soma_type") is not None:
        unsupported.append(f"reactivation.soma_type={reactivation['soma_type']!r}")
    for key, default in (("soma_init_m", 1.0), ("soma_init_b", 0.0)):
        if key in reactivation and float(reactivation[key]) != default:
            unsupported.append(f"reactivation.{key}={reactivation[key]!r}")
    network_type = str(
        sections.architecture.get("inhibitory_network_type", "dendritic")
    ).lower()
    if network_type != "dendritic":
        unsupported.append(f"architecture.inhibitory_network_type={network_type!r}")
    if bool(sections.implementation.get("compile_forward", False)):
        unsupported.append("implementation.compile_forward=true")
    if unsupported:
        raise ValueError(
            "Recurrent EINet (connectivity.rec_* > 0 or stateful dynamics) "
            "builds dendritic E/I populations and does not apply "
            + ", ".join(unsupported)
            + ". These options are feedforward-EINet only; remove them or use a "
            "feedforward configuration."
        )


def _require_recurrent_excitatory_sizes(
    architecture: dict[str, Any],
) -> list[Any]:
    excitatory_sizes = list(architecture.get("excitatory_layer_sizes", []))
    if not excitatory_sizes:
        raise ValueError(
            "Recurrent mode requires non-empty architecture.excitatory_layer_sizes"
        )
    return excitatory_sizes


def _resolve_recurrent_use_transfer(recurrent_cfg: dict[str, Any]) -> bool:
    # Keep legacy EINet behavior by default: TransferLayer is on unless disabled.
    if "use_transfer" in recurrent_cfg:
        return bool(recurrent_cfg["use_transfer"])
    return True


def _resolve_recurrent_morphology_options(
    *,
    core_type: str,
    morphology: dict[str, Any],
) -> tuple[dict[str, object], bool, bool]:
    core_morphology_overrides = get_core_morphology_alias_overrides(core_type)
    warn_alias_conflicts(
        core_type,
        morphology,
        core_morphology_overrides,
        context="model.core.morphology",
    )
    structured_alias_overrides = get_structured_recurrent_alias_overrides(core_type)

    use_shunting = bool(morphology.get("use_shunting", True))
    if "use_shunting" in core_morphology_overrides:
        use_shunting = bool(core_morphology_overrides["use_shunting"])

    use_additive_normalization = bool(
        morphology.get("use_additive_normalization", False)
    )
    if "use_additive_normalization" in core_morphology_overrides:
        use_additive_normalization = bool(
            core_morphology_overrides["use_additive_normalization"]
        )
    if "use_additive_normalization" in structured_alias_overrides:
        use_additive_normalization = bool(
            structured_alias_overrides["use_additive_normalization"]
        )
    return structured_alias_overrides, use_shunting, use_additive_normalization


def _resolve_recurrent_reactivation_options(
    *,
    core_type: str,
    reactivation: dict[str, Any],
    structured_alias_overrides: dict[str, object],
) -> _RecurrentReactivationOptions:
    reactivation_init_m = float(reactivation.get("init_m", 1.5))
    reactivation_init_b = float(reactivation.get("init_b", 0.5))
    reactivation_init_policy = str(reactivation.get("init_policy", "analytical"))
    reactivate, reactivation_type = resolve_dendritic_activation(
        reactivation.get("dendritic_activation"),
        bool(reactivation.get("enabled", True)),
        reactivation.get("type", "param_tanh"),
    )
    if structured_alias_overrides:
        # Only the reactivation settings the alias actually fixes can conflict.
        alias_reactivation = {
            key: structured_alias_overrides[source]
            for key, source in (
                ("init_m", "reactivation_init_m"),
                ("init_b", "reactivation_init_b"),
                ("init_policy", "reactivation_init_policy"),
            )
            if source in structured_alias_overrides
        }
        warn_alias_conflicts(
            core_type,
            reactivation,
            alias_reactivation,
            context="model.core.reactivation",
        )
        reactivation_init_m = float(
            structured_alias_overrides.get("reactivation_init_m", reactivation_init_m)
        )
        reactivation_init_b = float(
            structured_alias_overrides.get("reactivation_init_b", reactivation_init_b)
        )
        reactivation_init_policy = str(
            structured_alias_overrides.get(
                "reactivation_init_policy", reactivation_init_policy
            )
        )
    return _RecurrentReactivationOptions(
        reactivate=reactivate,
        reactivation_type=reactivation_type,
        init_m=reactivation_init_m,
        init_b=reactivation_init_b,
        init_policy=reactivation_init_policy,
    )


def _resolve_structured_recurrent_build_options(
    *,
    core_type: str,
    sections: _StructuredRecurrentSections,
    counts: _RecurrentSynapseCounts,
) -> _StructuredRecurrentBuildOptions:
    """Resolve validated build options before constructing recurrent layers."""
    recurrent_cfg = sections.recurrent_cfg
    architecture = sections.architecture

    _validate_structured_recurrent_type(core_type)
    _reject_feedforward_only_options(sections)
    excitatory_sizes = _require_recurrent_excitatory_sizes(architecture)
    use_transfer = _resolve_recurrent_use_transfer(recurrent_cfg)
    check_inhibitory_wiring(
        architecture,
        {
            **sections.connectivity,
            "rec_ei_synapses_per_branch": counts.rec_ei,
            "rec_ie_synapses_per_branch": counts.rec_ie,
            "rec_ii_synapses_per_branch": counts.rec_ii,
        },
        has_transfer_stream=use_transfer,
    )
    (
        structured_alias_overrides,
        use_shunting,
        use_additive_normalization,
    ) = _resolve_recurrent_morphology_options(
        core_type=core_type,
        morphology=sections.morphology,
    )
    reactivation_options = _resolve_recurrent_reactivation_options(
        core_type=core_type,
        reactivation=sections.reactivation,
        structured_alias_overrides=structured_alias_overrides,
    )

    # Per-layer recurrence follows the synapse counts unless listed explicitly.
    explicit_recurrent_layers = recurrent_cfg.get("recurrent_layers")

    return _StructuredRecurrentBuildOptions(
        excitatory_sizes=excitatory_sizes,
        inhibitory_sizes=list(architecture.get("inhibitory_layer_sizes", [])),
        use_transfer=use_transfer,
        explicit_recurrent_layers=explicit_recurrent_layers,
        use_shunting=use_shunting,
        use_additive_normalization=use_additive_normalization,
        reactivation_options=reactivation_options,
    )


__all__ = [
    "_require_recurrent_excitatory_sizes",
    "_resolve_recurrent_morphology_options",
    "_resolve_recurrent_reactivation_options",
    "_resolve_recurrent_use_transfer",
    "_resolve_structured_recurrent_build_options",
    "_validate_structured_recurrent_type",
]

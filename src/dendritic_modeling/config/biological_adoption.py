"""Explicit adoption of biological constraints in an experiment configuration.

Unlike the version upgrader, this operation can change the experiment. Callers
choose the input and source contracts; nothing infers a biological circuit from
the dataset name. Build the returned model to check its complete constraint set.
"""

from __future__ import annotations

from typing import Any


def prepare_biological_config(
    mapping: dict[str, Any],
    *,
    biological: bool = True,
    input_activation: str | None = None,
    input_source_identity: str | None = None,
    positive_input_projection: bool = False,
) -> dict[str, Any]:
    """Upgrade one expanded experiment and apply explicitly chosen contracts.

    ``None`` preserves an existing input/source setting. ``nonnegative`` is a
    checked identity, not rectification. Positive projection changes parameters
    and initialization and requires retraining. This function neither alters
    output gates nor invents inhibitory neurons. ``biological=False`` provides
    the explicit exploratory/control path. Unsupported models fail validation.
    """
    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.config.recurrence import _POPULATION_NETWORK_TYPES
    from dendritic_modeling.scripts.tools.upgrade_config import upgrade_mapping

    if "base_config" in mapping or "model" not in mapping:
        raise ValueError("Expand the sweep before preparing each experiment")
    if not isinstance(biological, bool) or not isinstance(
        positive_input_projection, bool
    ):
        raise ValueError("biological and positive_input_projection must be bools")
    if not biological and any(
        (
            input_activation is not None,
            input_source_identity is not None,
            positive_input_projection,
        )
    ):
        raise ValueError(
            "Exploratory configurations must not request biological repairs"
        )
    data, _, unresolved = upgrade_mapping(mapping)
    if unresolved:
        raise ValueError(
            f"Resolve historical configuration ambiguities first: {unresolved}"
        )
    core = data["model"].setdefault("core", {})
    core["biological_neuron"] = biological
    if biological:
        population = str(core.get("type", "")).lower() in _POPULATION_NETWORK_TYPES
        if population:
            network = core.setdefault("population_network", {})
            if network.get("use_transfer", False):
                transfer = network.setdefault("transfer_params", {})
                if input_activation is not None:
                    transfer["output_activation"] = input_activation
            else:
                transfer = None
                if input_activation is not None:
                    network["input_transform"] = input_activation
            if positive_input_projection:
                if not network.get("input_projection_dims"):
                    raise ValueError(
                        "Positive projection requires input_projection_dims"
                    )
                network["input_projection_weight_transform"] = "softplus"
        else:
            if positive_input_projection:
                raise ValueError("Positive projection requires a population network")
            transfer = core.setdefault("transfer", {})
            if input_activation is not None:
                transfer["output_activation"] = input_activation
        if input_source_identity is not None:
            if transfer is None:
                raise ValueError(
                    "An external E/I source declaration requires a transfer layer"
                )
            transfer["input_source_identity"] = input_source_identity
    config_from_mapping(data)
    return data

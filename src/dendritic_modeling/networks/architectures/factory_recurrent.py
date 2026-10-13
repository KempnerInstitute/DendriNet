"""Recurrent architecture factory helpers."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from omegaconf import DictConfig

from dendritic_modeling.config.conversion import (
    set_derived_input_dim,
    to_plain_dict as _to_plain_mapping,
)
from dendritic_modeling.config.keys import config_from_payload
from dendritic_modeling.config.legacy import normalize_transfer_config
from dendritic_modeling.config.recurrence import (
    _HETEROGENEOUS_LEAK_CTRNN_TYPES,
    _LEGENDRE_MEMORY_TYPES,
    _POPULATION_NETWORK_TYPES,
)

# Cell implementation for each baseline RNN core type.
_BASELINE_RNN_TYPES: dict[str, str] = {
    "gru": "gru",
    "lstm": "lstm",
    "vanilla_rnn": "vanilla",
    "rnn": "vanilla",
}


def _payload(raw_params: dict[str, Any], name: str) -> tuple[dict[str, Any], str]:
    """The payload of a recurrent baseline and the section its errors name.

    A core mapping holds it under ``name`` (``model.core.<name>``); a mapping
    without that key is taken as the payload itself.
    """
    if name in raw_params:
        payload, section = raw_params.get(name), f"model.core.{name}"
    else:
        payload, section = raw_params, f"{name} parameters"
    return (dict(payload) if isinstance(payload, dict) else {}), section


def _build_baseline_rnn_architecture(
    type: str,
    parameters: dict[str, Any] | DictConfig,
    input_dim: int | None,
    suffix_input_dim: int | None,
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
    baseline_cfg, section = _payload(raw_params, "baseline_rnn")
    cell_type = _BASELINE_RNN_TYPES[type]
    written = baseline_cfg.get("cell_type")
    if written is not None and str(written).lower() != cell_type:
        raise ValueError(
            f"{section}.cell_type={written!r} conflicts with model.core.type="
            f"{type!r}, which builds the {cell_type!r} cell; remove cell_type"
        )
    baseline_cfg["cell_type"] = cell_type
    set_derived_input_dim(baseline_cfg, input_dim, section=section)
    return BaselineRNN(
        config_from_payload(BaselineRNNConfig, baseline_cfg, section=section)
    )


def _build_population_network_architecture(
    type: str,
    parameters: dict[str, Any] | DictConfig,
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> nn.Module:
    """Build canonical population-network layers."""

    del suffix_input_dim
    from dendritic_modeling.networks.architectures.recurrent import (
        PopulationLayerConfig,
        PopulationNetwork,
        PopulationNetworkConfig,
    )
    from dendritic_modeling.networks.architectures.recurrent.population_configs import (
        transfer_options,
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
    # A block holding only seeds stays as written while the transfer layer is
    # off: normalizing it would fill in defaults that look like options.
    if "transfer_params" in mp_cfg and (
        mp_cfg.get("use_transfer", False) or transfer_options(mp_cfg["transfer_params"])
    ):
        mp_cfg["transfer_params"] = normalize_transfer_config(mp_cfg["transfer_params"])
    set_derived_input_dim(mp_cfg, input_dim, section="model.core.population_network")
    mp_cfg["layers"] = [
        (
            layer
            if isinstance(layer, PopulationLayerConfig)
            else config_from_payload(
                PopulationLayerConfig,
                layer,
                section=f"model.core.population_network.layers[{index}]",
            )
        )
        for index, layer in enumerate(mp_cfg.get("layers", []))
    ]
    return PopulationNetwork(
        config_from_payload(
            PopulationNetworkConfig, mp_cfg, section="model.core.population_network"
        )
    )


def _build_heterogeneous_leak_ctrnn_architecture(
    type: str,
    parameters: dict[str, Any] | DictConfig,
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> nn.Module:
    """Build the sparse heterogeneous-leak point recurrent control."""

    del suffix_input_dim
    from dendritic_modeling.networks.architectures.recurrent import (
        HeterogeneousLeakCTRNN,
        HeterogeneousLeakCTRNNConfig,
    )

    if input_dim is None:
        raise ValueError(f"input_dim required when creating CTRNN (type='{type}')")
    raw_params = _to_plain_mapping(parameters)
    ctrnn_cfg, section = _payload(raw_params, "heterogeneous_leak_ctrnn")
    set_derived_input_dim(ctrnn_cfg, input_dim, section=section)
    return HeterogeneousLeakCTRNN(
        config_from_payload(HeterogeneousLeakCTRNNConfig, ctrnn_cfg, section=section)
    )


def _build_legendre_memory_architecture(
    type: str,
    parameters: dict[str, Any] | DictConfig,
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> nn.Module:
    """Build a fixed Legendre Delay Network memory core."""

    del suffix_input_dim
    from dendritic_modeling.networks.architectures.recurrent import (
        FixedLegendreMemory,
        FixedLegendreMemoryConfig,
    )

    if input_dim is None:
        raise ValueError(
            f"input_dim required when creating fixed Legendre memory (type='{type}')"
        )
    raw_params = _to_plain_mapping(parameters)
    memory_cfg, section = _payload(raw_params, "legendre_memory")
    set_derived_input_dim(memory_cfg, input_dim, section=section)
    return FixedLegendreMemory(
        config_from_payload(FixedLegendreMemoryConfig, memory_cfg, section=section)
    )


__all__ = [
    "_BASELINE_RNN_TYPES",
    "_HETEROGENEOUS_LEAK_CTRNN_TYPES",
    "_LEGENDRE_MEMORY_TYPES",
    "_POPULATION_NETWORK_TYPES",
    "_build_baseline_rnn_architecture",
    "_build_heterogeneous_leak_ctrnn_architecture",
    "_build_legendre_memory_architecture",
    "_build_population_network_architecture",
]

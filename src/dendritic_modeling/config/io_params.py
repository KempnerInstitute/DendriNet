"""Which ``model.encoder.params`` and ``model.decoder.params`` each type reads.

Encoder and decoder params are one flat schema shared by every type, and each
builder reads only its own fields. A field written for a type that never reads
it would be accepted and ignored, so the loader rejects it unless it still
holds its default value (saved ``config.json`` files contain every field).
Types without an entry here (registered extensions) are not checked.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

from dendritic_modeling.config.keys import field_default

logger = logging.getLogger(__name__)

IDENTITY_TYPES = frozenset({"identity", "none"})

# Training setup writes input_dim and input_shape from the data.
_ENCODER_DERIVED = frozenset({"input_dim", "input_shape"})
_ROUTER = frozenset(
    {
        "router_mode",
        "pathway_groups",
        "shared_indices",
        "n_pathways",
        "pathway_dim",
        "pathway_dims",
        "router_activation",
        "learned_router_temperature",
        "learned_router_init_scale",
    }
)
_MLP_AUTOENCODER = frozenset(
    {"encoder_hidden_dims", "latent_dim", "decoder_hidden_dims", "activation"}
)
_CNN_AUTOENCODER = frozenset(
    {"encoder_hidden_dims", "latent_shape", "decoder_hidden_dims", "activation"}
)

ENCODER_PARAMS_READ: dict[str, frozenset[str]] = {
    **dict.fromkeys(IDENTITY_TYPES, _ENCODER_DERIVED),
    **dict.fromkeys(("pathway_router", "router"), _ROUTER | _ENCODER_DERIVED),
    **dict.fromkeys(
        ("mlpautoencoder", "mlpvariationalautoencoder"),
        _MLP_AUTOENCODER | _ENCODER_DERIVED,
    ),
    **dict.fromkeys(
        ("cnnautoencoder", "cnnvariationalautoencoder"),
        _CNN_AUTOENCODER | _ENCODER_DERIVED,
    ),
}

DECODER_PARAMS_READ: dict[str, frozenset[str]] = {
    "mlp": frozenset(
        {
            "input_dim",
            "hidden_dims",
            "activation",
            "output_dim",
            "input_normalization",
            "input_norm_eps",
            "output_init",
        }
    ),
    # An identity decoder passes the core output through. Its input_dim is set
    # from the core, and an explicit output_dim is checked against the core
    # output dimension when the model is built.
    **dict.fromkeys(IDENTITY_TYPES, frozenset({"input_dim", "output_dim"})),
}


def is_saved_record(params: Mapping[str, Any], params_cls: type) -> bool:
    """Return True for a params mapping that lists every field (``config.json``)."""
    return {item.name for item in dataclasses.fields(params_cls)} <= set(params)


def unread_params(
    kind: str, type_name: Any, params: Mapping[str, Any], params_cls: type
) -> list[str]:
    """Return the explicit ``model.<kind>.params`` fields the type does not read.

    ``kind`` is ``"encoder"`` or ``"decoder"``. Fields holding their default,
    unknown keys and types without an entry in the tables are not listed.
    """
    lowered = str(type_name or "").strip().lower()
    table = ENCODER_PARAMS_READ if kind == "encoder" else DECODER_PARAMS_READ
    read = table.get(lowered)
    if read is None:
        return []
    fields = {item.name for item in dataclasses.fields(params_cls)}
    unread = []
    for name, value in params.items():
        # Unknown and retired keys are reported by the section builder.
        if name in read or name not in fields:
            continue
        if kind == "decoder" and lowered in IDENTITY_TYPES and name == "hidden_dims":
            if not value:
                continue  # an identity decoder has no hidden layers
        if value == field_default(params_cls, name):
            continue
        unread.append(name)
    return unread


def reject_unread_params(
    kind: str, type_name: Any, params: Mapping[str, Any], params_cls: type
) -> None:
    """Raise when explicit ``model.<kind>.params`` fields would be ignored.

    ``kind`` is ``"encoder"`` or ``"decoder"``. A field that still holds its
    default is accepted. In a saved record the unread fields are reported as a
    warning, because the record lists every field.
    """
    unread = unread_params(kind, type_name, params, params_cls)
    if not unread:
        return
    section = f"model.{kind}"
    message = f"{section}.type={type_name!r} does not read " + ", ".join(
        f"{section}.params.{name}" for name in unread
    )
    if is_saved_record(params, params_cls):
        logger.warning("Ignoring params of a saved config: %s.", message)
        return
    raise ValueError(message + ". Remove them.")


__all__ = [
    "DECODER_PARAMS_READ",
    "ENCODER_PARAMS_READ",
    "IDENTITY_TYPES",
    "is_saved_record",
    "reject_unread_params",
    "unread_params",
]

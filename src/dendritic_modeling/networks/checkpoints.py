"""Safe helpers for strict model-state checkpoint loading."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_model_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Extract a tensor-only model state from a standard checkpoint payload."""

    candidate = payload
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                candidate = nested
                break
    if not isinstance(candidate, Mapping) or not candidate:
        raise TypeError("Checkpoint does not contain a non-empty model state mapping")
    state: dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Model state must map string keys to tensors only")
        normalized = key[7:] if key.startswith("module.") else key
        if normalized in state:
            raise ValueError(
                f"Checkpoint contains duplicate normalized key {normalized!r}"
            )
        state[normalized] = value
    return state


def load_model_state_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state dictionary with PyTorch's restricted weights-only loader."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    return extract_model_state_dict(payload)


def load_initial_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Strictly load a model warm start and return an auditable record."""

    resolved = Path(path).expanduser().resolve()
    observed_sha256 = sha256_file(resolved)
    if expected_sha256 is not None:
        normalized_expected = str(expected_sha256).strip().lower()
        if observed_sha256 != normalized_expected:
            raise ValueError(
                "Initial checkpoint SHA-256 mismatch: "
                f"expected {normalized_expected}, observed {observed_sha256}"
            )
    state = load_model_state_checkpoint(resolved)
    incompatibilities = model.load_state_dict(state, strict=bool(strict))
    return {
        "schema_version": "initial_model_checkpoint_v1",
        "path": str(resolved),
        "sha256": observed_sha256,
        "strict": bool(strict),
        "state_dict_key_count": len(state),
        "missing_keys": list(incompatibilities.missing_keys),
        "unexpected_keys": list(incompatibilities.unexpected_keys),
    }

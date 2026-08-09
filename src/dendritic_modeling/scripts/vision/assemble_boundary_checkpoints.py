"""Assemble independently trained AlexNet boundaries into one warm start."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from dendritic_modeling.networks.checkpoints import (
    load_model_state_checkpoint,
    sha256_file,
)

SCHEMA_VERSION = "dendritic_vision_checkpoint_assembly_v1"
COMPONENTS = ("conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7")

_SPATIAL_FORWARD_INDICES = (0, 3, 6, 8, 10)
_SPATIAL_ADAPTER_INDICES = (1, 4, 7, 9, 11)
_CLASSIFIER_FORWARD_INDICES = (1, 4)
_CLASSIFIER_ADAPTER_INDICES = (2, 5)


def _component_prefixes() -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for index, name in enumerate(COMPONENTS[:5]):
        result[name] = (
            f"core_network.spatial_blocks.{index}.",
            f"core_network.spatial_output_adapters.{index}.",
            f"core_network.features.{_SPATIAL_FORWARD_INDICES[index]}.",
            f"core_network.features.{_SPATIAL_ADAPTER_INDICES[index]}.",
        )
    for index, name in enumerate(COMPONENTS[5:]):
        result[name] = (
            f"core_network.feedforward_blocks.{index}.",
            f"core_network.feedforward_output_adapters.{index}.",
            f"core_network.classifier.{_CLASSIFIER_FORWARD_INDICES[index]}.",
            f"core_network.classifier.{_CLASSIFIER_ADAPTER_INDICES[index]}.",
        )
    return result


COMPONENT_PREFIXES = _component_prefixes()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, Mapping):
        raise TypeError("Assembly manifest must contain a YAML mapping")
    manifest = dict(payload)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    return manifest


def _resolve_path(value: Any, *, base: Path, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(text).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _source_specs(
    manifest: Mapping[str, Any], *, base: Path
) -> dict[str, dict[str, str]]:
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, Mapping):
        raise TypeError(
            "sources must map every component and remainder to a checkpoint"
        )
    required = {*COMPONENTS, "remainder"}
    observed = {str(key) for key in raw_sources}
    if observed != required:
        raise ValueError(
            "sources must contain exactly "
            f"{sorted(required)}; observed {sorted(observed)}"
        )
    result: dict[str, dict[str, str]] = {}
    for component in (*COMPONENTS, "remainder"):
        raw = raw_sources[component]
        if not isinstance(raw, Mapping):
            raise TypeError(f"sources.{component} must contain path and sha256")
        path = _resolve_path(
            raw.get("path"), base=base, field=f"sources.{component}.path"
        )
        expected = str(raw.get("sha256", "")).strip().lower()
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise ValueError(
                f"sources.{component}.sha256 must be a 64-character hexadecimal digest"
            )
        observed_digest = sha256_file(path)
        if observed_digest != expected:
            raise ValueError(
                f"sources.{component} SHA-256 mismatch: expected {expected}, "
                f"observed {observed_digest}"
            )
        result[component] = {"path": str(path), "sha256": observed_digest}
    return result


def _validate_inventory(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    component: str,
) -> None:
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        unexpected = sorted(candidate_keys - reference_keys)
        raise ValueError(
            f"sources.{component} has a different key inventory; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for key in reference:
        expected = reference[key]
        observed = candidate[key]
        if expected.shape != observed.shape or expected.dtype != observed.dtype:
            raise ValueError(
                f"sources.{component} tensor contract differs at {key!r}: "
                f"expected shape={tuple(expected.shape)} dtype={expected.dtype}, "
                f"observed shape={tuple(observed.shape)} dtype={observed.dtype}"
            )


def _keys_for_component(state: Mapping[str, torch.Tensor], component: str) -> set[str]:
    prefixes = COMPONENT_PREFIXES[component]
    keys = {key for key in state if key.startswith(prefixes)}
    if not keys:
        raise ValueError(f"No state keys matched the declared {component} prefixes")
    return keys


def _validate_alias_pair(
    state: Mapping[str, torch.Tensor],
    canonical_prefix: str,
    forward_prefix: str,
    *,
    component: str,
) -> None:
    canonical = {
        key[len(canonical_prefix) :]: value
        for key, value in state.items()
        if key.startswith(canonical_prefix)
    }
    forwarded = {
        key[len(forward_prefix) :]: value
        for key, value in state.items()
        if key.startswith(forward_prefix)
    }
    if set(canonical) != set(forwarded):
        raise ValueError(f"{component} inventory and forward-path aliases do not match")
    for suffix, tensor in canonical.items():
        if not torch.equal(tensor, forwarded[suffix]):
            raise ValueError(f"{component} aliases differ at {suffix!r}")


def _validate_component_aliases(
    state: Mapping[str, torch.Tensor], component: str
) -> None:
    prefixes = COMPONENT_PREFIXES[component]
    _validate_alias_pair(state, prefixes[0], prefixes[2], component=component)
    _validate_alias_pair(state, prefixes[1], prefixes[3], component=component)


def _semantic_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def assemble_boundary_checkpoints(
    manifest_path: str | Path,
    *,
    output_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble boundary-specific state and write a hash-bound provenance record."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    sources = _source_specs(manifest, base=manifest_path.parent)
    destination = _resolve_path(
        (
            output_checkpoint
            if output_checkpoint is not None
            else manifest.get("output_checkpoint")
        ),
        base=manifest_path.parent,
        field="output_checkpoint",
    )
    provenance_path = destination.with_suffix(destination.suffix + ".provenance.json")
    if destination.exists() or provenance_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an assembled checkpoint or provenance record: "
            f"{destination}, {provenance_path}"
        )

    states = {
        component: load_model_state_checkpoint(spec["path"])
        for component, spec in sources.items()
    }
    reference = states["remainder"]
    for component in COMPONENTS:
        _validate_inventory(reference, states[component], component=component)
        _validate_component_aliases(states[component], component)

    component_keys = {
        component: _keys_for_component(reference, component) for component in COMPONENTS
    }
    claimed: set[str] = set()
    for component, keys in component_keys.items():
        overlap = claimed & keys
        if overlap:
            raise ValueError(
                f"Component key sets overlap for {component}: {sorted(overlap)[:5]}"
            )
        claimed.update(keys)
    remainder_keys = set(reference) - claimed
    if not remainder_keys:
        raise ValueError(
            "No unassigned remainder state remains after boundary partitioning"
        )

    assembled = {key: tensor.clone() for key, tensor in reference.items()}
    records: dict[str, Any] = {}
    for component in COMPONENTS:
        keys = component_keys[component]
        for key in keys:
            assembled[key] = states[component][key].clone()
        records[component] = {
            **sources[component],
            "state_dict_key_count": len(keys),
            "tensor_bytes": sum(
                states[component][key].numel() * states[component][key].element_size()
                for key in keys
            ),
        }
    records["remainder"] = {
        **sources["remainder"],
        "state_dict_key_count": len(remainder_keys),
        "tensor_bytes": sum(
            reference[key].numel() * reference[key].element_size()
            for key in remainder_keys
        ),
    }

    semantic_digest = _semantic_state_sha256(assembled)
    _atomic_torch_save(assembled, destination)
    result = {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "output_checkpoint": str(destination),
        "output_checkpoint_sha256": sha256_file(destination),
        "output_semantic_state_sha256": semantic_digest,
        "state_dict_key_count": len(assembled),
        "components": records,
        "metadata": dict(manifest.get("metadata", {})),
    }
    provenance_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = assemble_boundary_checkpoints(
        args.manifest,
        output_checkpoint=args.output_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

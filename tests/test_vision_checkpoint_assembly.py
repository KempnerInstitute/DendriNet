from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from dendritic_modeling.networks.checkpoints import (
    load_initial_model_checkpoint,
    load_model_state_checkpoint,
    sha256_file,
)
from dendritic_modeling.scripts.vision.assemble_boundary_checkpoints import (
    COMPONENT_PREFIXES,
    COMPONENTS,
    SCHEMA_VERSION,
    assemble_boundary_checkpoints,
)


def _state(fill: float = 0.0) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for component_index, component in enumerate(COMPONENTS, start=1):
        prefixes = COMPONENT_PREFIXES[component]
        block_value = torch.tensor([fill + component_index], dtype=torch.float32)
        adapter_value = torch.tensor([fill + component_index + 0.5])
        state[prefixes[0] + "weight"] = block_value.clone()
        state[prefixes[2] + "weight"] = block_value.clone()
        state[prefixes[1] + "log_scale"] = adapter_value.clone()
        state[prefixes[3] + "log_scale"] = adapter_value.clone()
    state["core_network.feedforward_blocks.2.weight"] = torch.tensor([fill + 90])
    state["core_network.classifier.6.weight"] = torch.tensor([fill + 90])
    state["output_scale"] = torch.tensor([fill + 100])
    return state


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    paths: dict[str, Path] = {}
    sources = {}
    for index, component in enumerate((*COMPONENTS, "remainder"), start=1):
        path = tmp_path / f"{component}.pt"
        torch.save(_state(fill=100.0 * index), path)
        paths[component] = path
        sources[component] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sources": sources,
        "output_checkpoint": str(tmp_path / "assembled.pt"),
        "metadata": {"study": "unit_test"},
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path, paths


def test_assemble_boundary_checkpoints_preserves_component_ownership(tmp_path):
    manifest, source_paths = _manifest(tmp_path)

    result = assemble_boundary_checkpoints(manifest)

    output = Path(result["output_checkpoint"])
    assembled = load_model_state_checkpoint(output)
    remainder = load_model_state_checkpoint(source_paths["remainder"])
    for component in COMPONENTS:
        source = load_model_state_checkpoint(source_paths[component])
        prefixes = COMPONENT_PREFIXES[component]
        keys = {key for key in assembled if key.startswith(prefixes)}
        assert keys
        assert all(torch.equal(assembled[key], source[key]) for key in keys)
    assert torch.equal(assembled["output_scale"], remainder["output_scale"])
    assert result["metadata"] == {"study": "unit_test"}
    assert output.with_suffix(".pt.provenance.json").is_file()


def test_assembly_rejects_changed_source_hash(tmp_path):
    manifest, source_paths = _manifest(tmp_path)
    torch.save(_state(fill=-1.0), source_paths["conv3"])

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        assemble_boundary_checkpoints(manifest)


def test_assembly_rejects_different_tensor_contract(tmp_path):
    manifest, source_paths = _manifest(tmp_path)
    state = load_model_state_checkpoint(source_paths["conv4"])
    key = next(iter(state))
    state[key] = torch.zeros(2)
    torch.save(state, source_paths["conv4"])
    payload = yaml.safe_load(manifest.read_text())
    payload["sources"]["conv4"]["sha256"] = sha256_file(source_paths["conv4"])
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="tensor contract differs"):
        assemble_boundary_checkpoints(manifest)


def test_initial_checkpoint_loader_is_hash_bound_and_strict(tmp_path):
    source = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "linear.pt"
    torch.save(source.state_dict(), checkpoint)
    destination = torch.nn.Linear(3, 2)

    record = load_initial_model_checkpoint(
        destination,
        checkpoint,
        expected_sha256=sha256_file(checkpoint),
        strict=True,
    )

    assert record["strict"] is True
    assert record["missing_keys"] == []
    assert record["unexpected_keys"] == []
    assert all(
        torch.equal(expected, observed)
        for expected, observed in zip(
            source.state_dict().values(), destination.state_dict().values(), strict=True
        )
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_initial_model_checkpoint(
            destination,
            checkpoint,
            expected_sha256="0" * 64,
        )

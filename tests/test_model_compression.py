"""Tests for lossless model deployment compaction."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.analysis.utils.model_resources import (
    _dense_candidate_topk_resource_counts,
)
from dendritic_modeling.config.compression import (
    CompressionVerificationConfig,
    ModelCompressionConfig,
    SparseTopologyCompressionConfig,
)
from dendritic_modeling.deployment.compression import (
    compact_model_checkpoint,
    freeze_sparse_topology_,
    load_compact_model_checkpoint,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    DenseToSparseLinear,
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
    TopKLinear,
)
from dendritic_modeling.scripts.deployment.compact_checkpoint import (
    _write_report_atomically,
)


def _topk_model() -> nn.Sequential:
    model = nn.Sequential(
        TopKLinear(12, 5, 3, weight_transform="identity"),
        nn.ReLU(),
    )
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        model[0].pre_w.copy_(torch.randn(model[0].pre_w.shape, generator=generator))
    return model


def test_default_verification_tolerance_allows_float32_reduction_reordering() -> None:
    verification = CompressionVerificationConfig()

    assert verification.atol == 1e-5
    assert verification.rtol == 1e-5


def test_dense_candidate_resource_count_does_not_require_a_dense_active_mask() -> None:
    layer = TopKLinear(1000, 7, 3)

    counts = _dense_candidate_topk_resource_counts(layer)

    assert counts["candidate_slots"] == 7000
    assert counts["active_synapses"] == 21
    assert counts["realized_k_min"] == 3
    assert counts["realized_k_max"] == 3
    assert counts["selection_policy"] == "topk"


def test_compact_export_report_is_atomic_and_never_overwritten(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"

    _write_report_atomically({"schema_version": "test", "value": 3}, report_path)

    assert report_path.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        _write_report_atomically({"value": 4}, report_path)


def test_freeze_sparse_topology_is_numerically_equivalent() -> None:
    model = _topk_model().eval()
    sample = torch.randn(7, 12, generator=torch.Generator().manual_seed(12))
    expected = model(sample)

    records = freeze_sparse_topology_(model)

    assert len(records) == 1
    assert records[0].path == "0"
    assert records[0].source_type == "TopKLinear"
    assert isinstance(model[0], IndexedSparseLinear)
    assert model[0].pre_w.numel() == 5 * 3
    torch.testing.assert_close(model(sample), expected)


def test_dense_to_sparse_must_reach_final_schedule_by_default() -> None:
    model = nn.Sequential(
        DenseToSparseLinear(
            12,
            4,
            3,
            initial_k=12,
            end_step=100,
            weight_transform="identity",
        )
    )

    with pytest.raises(ValueError, match="has not reached target K"):
        freeze_sparse_topology_(model)

    config = ModelCompressionConfig(
        sparse_topology=SparseTopologyCompressionConfig(
            require_final_dense_to_sparse_step=False
        )
    )
    records = freeze_sparse_topology_(model, config)
    assert records[0].synapses_per_output == 12


def test_noisy_or_stochastic_topology_fails_closed() -> None:
    noisy = nn.Sequential(TopKLinear(8, 3, 2, noise_level=0.1))
    with pytest.raises(ValueError, match="noisy TopKLinear"):
        freeze_sparse_topology_(noisy)

    stochastic = nn.Sequential(
        IndexedDynamicTopKLinear(
            8,
            3,
            2,
            candidate_size=4,
            selection="stochastic",
            seed=3,
        )
    )
    with pytest.raises(ValueError, match="deterministic standard selection"):
        freeze_sparse_topology_(stochastic)


@pytest.mark.parametrize("sparse_type", ["stochastic", "variance", "deepst"])
def test_sparse_layers_without_a_lossless_freeze_fail_closed(sparse_type) -> None:
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.factory import (
        get_sparse_layer,
    )

    model = nn.Sequential(get_sparse_layer(sparse_type, 8, 3, 2))
    with pytest.raises(ValueError, match="cannot freeze these sparse layers"):
        freeze_sparse_topology_(model)
    allowed = ModelCompressionConfig(
        sparse_topology=SparseTopologyCompressionConfig(allow_unconverted=True)
    )
    assert freeze_sparse_topology_(model, allowed) == []


def test_shared_dynamic_sparse_module_fails_closed() -> None:
    shared = TopKLinear(8, 3, 2, weight_transform="identity")
    model = nn.ModuleDict({"first": shared, "second": shared})

    with pytest.raises(ValueError, match="Shared dynamic sparse modules"):
        freeze_sparse_topology_(model)
    assert model["first"] is shared
    assert model["second"] is shared


def test_deterministic_indexed_dynamic_freezes_current_support() -> None:
    model = nn.Sequential(
        IndexedDynamicTopKLinear(
            10,
            4,
            3,
            candidate_size=6,
            selection="standard",
            noise_level=0.0,
            weight_transform="identity",
            seed=7,
        )
    ).eval()
    sample = torch.randn(5, 10, generator=torch.Generator().manual_seed(8))
    expected = model(sample)

    records = freeze_sparse_topology_(model)

    assert records[0].source_type == "IndexedDynamicTopKLinear"
    assert isinstance(model[0], IndexedSparseLinear)
    torch.testing.assert_close(model(sample), expected)


@pytest.mark.parametrize("encoding", ["uint", "bitmask"])
def test_compact_checkpoint_round_trip(tmp_path: Path, encoding: str) -> None:
    source_model = _topk_model().eval()
    sample = torch.randn(4, 12, generator=torch.Generator().manual_seed(13))
    expected = source_model(sample)
    source = tmp_path / "source.pt"
    output = tmp_path / f"compact_{encoding}.pt"
    torch.save({"model_state_dict": source_model.state_dict()}, source)
    policy = ModelCompressionConfig(
        sparse_topology=SparseTopologyCompressionConfig(topology_encoding=encoding)
    )

    report = compact_model_checkpoint(source_model, source, output, policy)

    assert report["converted_modules"] == 1
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["format"] == "dendritic_modeling.compact_checkpoint"
    assert payload["source"]["sha256"]
    if encoding == "bitmask":
        assert "0.connection_indices.__bitmask" in payload["state_dict"]

    restored = _topk_model().eval()
    load_report = load_compact_model_checkpoint(restored, output)
    assert load_report["converted_modules"] == 1
    assert isinstance(restored[0], IndexedSparseLinear)
    torch.testing.assert_close(restored(sample), expected)


def test_compact_checkpoint_never_overwrites_inputs(tmp_path: Path) -> None:
    model = _topk_model()
    source = tmp_path / "source.pt"
    torch.save(model.state_dict(), source)

    with pytest.raises(ValueError, match="must differ"):
        compact_model_checkpoint(model, source, source)

    output = tmp_path / "existing.pt"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        compact_model_checkpoint(model, source, output)
    assert output.read_bytes() == b"keep"


def test_compression_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "compression.yaml"
    config.write_text("unexpected: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown compression config"):
        ModelCompressionConfig.load(config)


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_explicit_fixed_storage_and_execution_options_round_trip(index_dtype):
    from dataclasses import asdict

    from dendritic_modeling.deployment.compression import (
        prepare_model_for_compact_state_,
    )

    model = _topk_model().eval()
    expected = model(torch.eye(12))
    policy = ModelCompressionConfig(
        sparse_topology=SparseTopologyCompressionConfig(
            index_dtype=index_dtype,
            projection_backend="eager",
            workspace_mb=128,
        )
    )
    records = freeze_sparse_topology_(model, policy)
    assert model[0].connection_indices.dtype == getattr(torch, index_dtype)
    assert model[0].projection_backend == "eager"
    assert model[0].workspace_mb == 128
    assert records[0].index_dtype == index_dtype
    restored = _topk_model().eval()
    prepare_model_for_compact_state_(
        restored, [asdict(r) for r in records], model.state_dict()
    )
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored[0].connection_indices.dtype == getattr(torch, index_dtype)
    assert torch.equal(restored(torch.eye(12)), model(torch.eye(12)))
    torch.testing.assert_close(restored(torch.eye(12)), expected)


def test_legacy_compact_manifest_still_uses_int64():
    from dataclasses import asdict

    from dendritic_modeling.deployment.compression import (
        prepare_model_for_compact_state_,
    )

    model = _topk_model().eval()
    records = [asdict(r) for r in freeze_sparse_topology_(model)]
    for record in records:
        record.pop("index_dtype")
    restored = _topk_model().eval()
    prepare_model_for_compact_state_(restored, records, model.state_dict())
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored[0].connection_indices.dtype == torch.int64
    assert torch.equal(restored(torch.eye(12)), model(torch.eye(12)))


@pytest.mark.parametrize(
    "options",
    [
        {"index_dtype": "int16"},
        {"workspace_mb": 0},
        {"workspace_mb": float("nan")},
        {"projection_backend": ""},
    ],
)
def test_invalid_topology_target_options_are_rejected(options):
    with pytest.raises(ValueError):
        ModelCompressionConfig(
            sparse_topology=SparseTopologyCompressionConfig(**options)
        )


def test_topk_inventory_counts_controls_and_frozen_parameters():
    from dendritic_modeling.analysis.utils.model_resources import (
        topk_parameter_inventory,
    )
    from dendritic_modeling.config import AffineMLPConfig
    from dendritic_modeling.networks.architectures.classical.mlp import build_affine_mlp

    model = build_affine_mlp(
        AffineMLPConfig(dimensions=[12, 7, 3], mode="topk", fanin=[4, 2])
    )
    before = topk_parameter_inventory(model)
    assert before["retained_parameters"] == 7 * 4 + 7 + 3 * 2 + 3
    freeze_sparse_topology_(model)
    after = topk_parameter_inventory(model)
    assert after["candidate_weights"] == 0
    assert after["training_parameters"] == before["retained_parameters"]
    model.requires_grad_(False)
    assert topk_parameter_inventory(model)["retained_parameters"] == 0


@pytest.mark.parametrize(
    "params, processing, expected",
    [
        ({"input_dim": 784}, {}, (784,)),
        ({"input_dim": 3072}, {"flatten": True}, (3072,)),
        ({"input_dim": 3072}, {"flatten": False}, (3, 32, 32)),
        (
            {"input_dim": 3, "input_shape": [3, 224, 224]},
            {"flatten": False},
            (3, 224, 224),
        ),
    ],
)
def test_full_model_verification_uses_the_configured_input_shape(
    params, processing, expected
) -> None:
    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.scripts.deployment.compact_checkpoint import (
        _verification_input_shape,
    )

    config = config_from_mapping(
        {
            "data": {"dataset_name": "cifar10", "processing": processing},
            "model": {"encoder": {"type": "identity", "params": params}},
        }
    )
    assert _verification_input_shape(config) == expected

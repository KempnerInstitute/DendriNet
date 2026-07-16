"""Contracts for the all-active fixed-index sparse point control."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from dendritic_modeling.config import load_config
from dendritic_modeling.networks.architectures.classical.mlp import (
    SparseActiveMatchedPointAffine,
)
from dendritic_modeling.networks.architectures.factory import get_architecture

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "drafts/dendritic-information-processing/empirical/configs/fig5"
    / "empirical_publication_v2_fig5_alexnet_fc7_sparse_point_seed41.yaml"
)


def _control(
    *,
    input_dim: int = 8,
    output_dim: int = 8,
    target: int = 27,
    seed: int = 41,
) -> SparseActiveMatchedPointAffine:
    return SparseActiveMatchedPointAffine(
        input_dim=input_dim,
        excitatory_layer_sizes=[output_dim],
        inhibitory_layer_sizes=[2],
        excitatory_branch_factors=[2],
        inhibitory_branch_factors=[],
        ee_synapses_per_branch_per_layer=[1],
        ei_synapses_per_branch_per_layer=[1],
        ie_synapses_per_branch_per_layer=[1],
        target_active_parameters=target,
        indexed_seed=seed,
        indexed_output_chunk_size=3,
        transfer_params={"output_activation": "relu"},
    )


def _row_support(
    control: SparseActiveMatchedPointAffine, output_row: int
) -> torch.Tensor:
    support = control.base_connection_indices[output_row]
    try:
        extra_position = control.high_fan_in_output_rows.index(output_row)
    except ValueError:
        return support
    extra_support = control.extra_connection_indices[
        extra_position : extra_position + 1
    ]
    return torch.cat((support, extra_support))


def test_exact_publication_parameter_and_storage_contract() -> None:
    control = _control(
        input_dim=4096,
        output_dim=4096,
        target=1_376_320,
        seed=41,
    )
    accounting = control.resource_accounting()

    assert control.base_weight.shape == (4096, 335)
    assert control.extra_weight.shape == (64,)
    assert control.bias.shape == (4096,)
    assert sum(parameter.numel() for parameter in control.parameters()) == 1_376_320
    assert control.get_effective_params() == 1_376_320
    assert accounting == {
        "target_active_parameters": 1_376_320,
        "active_parameters": 1_376_320,
        "stored_parameters": 1_376_320,
        "weight_parameters": 1_372_224,
        "bias_parameters": 4_096,
        "source_index_entries": 1_372_224,
        "source_index_buffer_bytes": 10_977_792,
        "destination_index_entries": 64,
        "destination_index_buffer_bytes": 512,
        "connection_index_entries": 1_372_288,
        "connection_index_buffer_bytes": 10_978_304,
        "persistent_buffer_bytes": 10_978_304,
        "other_buffer_bytes": 0,
        "all_outputs_active": True,
        "output_fan_in_min": 335,
        "output_fan_in_max": 336,
        "higher_fan_in_output_count": 64,
        "weight_parameterization": "signed_affine",
        "support_sha256": (
            "45799526ace49bb3c7bdb11a4dce7e474ee8ee2eeb30c1b4b1530d13b86756ab"
        ),
    }
    assert set(dict(control.named_buffers())) == {
        "base_connection_indices",
        "extra_connection_indices",
        "extra_output_connection_indices",
    }
    assert set(control.state_dict()) == {
        "base_weight",
        "extra_weight",
        "bias",
        "base_connection_indices",
        "extra_connection_indices",
        "extra_output_connection_indices",
    }
    assert list(dict(control.named_children())) == ["activation"]
    assert torch.equal(control.bias, torch.zeros_like(control.bias))
    assert bool((control.base_weight < 0).any())
    assert bool((control.base_weight > 0).any())

    high_rows = torch.tensor(control.high_fan_in_output_rows)
    high_mask = torch.zeros(control.output_dim, dtype=torch.bool)
    high_mask[high_rows] = True
    low_standard_deviation = float(control.base_weight[~high_mask].detach().std())
    high_weights = torch.cat(
        (control.base_weight[high_mask].reshape(-1), control.extra_weight)
    )
    high_standard_deviation = float(high_weights.detach().std())
    assert low_standard_deviation == pytest.approx(math.sqrt(2.0 / 335), rel=0.02)
    assert high_standard_deviation == pytest.approx(math.sqrt(2.0 / 336), rel=0.03)

    # The replacement has the same scalar count as the dendritic core, hence
    # the same AlexNet stored/full and trainable/full scalar totals.
    assert 40_222_528 + 1_376_320 + 4_097_000 == 45_695_848
    assert 1_376_320 + 4_097_000 == 5_473_320
    assert 1_376_320 * 4 + 10_978_304 == 16_483_584


def test_support_is_unique_balanced_seed_keyed_and_label_independent() -> None:
    torch.manual_seed(1)
    first = _control(seed=41)
    torch.manual_seed(999)
    repeated = _control(seed=41)
    changed = _control(seed=42)

    assert torch.equal(first.base_connection_indices, repeated.base_connection_indices)
    assert torch.equal(
        first.extra_connection_indices, repeated.extra_connection_indices
    )
    assert first.high_fan_in_output_rows == repeated.high_fan_in_output_rows
    assert first.connectivity_support_sha256() == repeated.connectivity_support_sha256()
    assert first.connectivity_support_sha256() != changed.connectivity_support_sha256()

    output_degrees = []
    input_degree = torch.zeros(first.input_dim, dtype=torch.long)
    for output_row in range(first.output_dim):
        support = _row_support(first, output_row)
        assert support.unique().numel() == support.numel()
        output_degrees.append(support.numel())
        input_degree.scatter_add_(
            0,
            support,
            torch.ones_like(support),
        )
    assert sorted(output_degrees) == [2] * 5 + [3] * 3
    assert int(input_degree.max() - input_degree.min()) <= 1


def test_strict_cross_seed_load_restores_destination_rows_and_function() -> None:
    source = _control(seed=41)
    destination = _control(seed=42)
    assert source.connectivity_support_sha256() != (
        destination.connectivity_support_sha256()
    )
    assert not torch.equal(
        source.extra_output_connection_indices,
        destination.extra_output_connection_indices,
    )
    inputs = torch.randn(9, 8)
    with torch.no_grad():
        expected = source(inputs)

    incompatible = destination.load_state_dict(source.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert destination.support_seed == 42
    assert torch.equal(
        destination.extra_output_connection_indices,
        source.extra_output_connection_indices,
    )
    assert destination.connectivity_support_sha256() == (
        source.connectivity_support_sha256()
    )
    with torch.no_grad():
        observed = destination(inputs)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_forward_uses_signed_weights_relu_and_every_output() -> None:
    control = _control()
    with torch.no_grad():
        control.base_weight.fill_(1.0)
        control.extra_weight.fill_(1.0)
        control.bias.zero_()
    x = torch.ones(4, 8, requires_grad=True)
    output = control(x)
    assert output.shape == (4, 8)
    assert torch.all(output > 0)
    output.sum().backward()
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in control.parameters())

    with torch.no_grad():
        control.base_weight.fill_(-1.0)
        control.extra_weight.fill_(-1.0)
    assert torch.equal(control(torch.ones(8)), torch.zeros(8))


def test_autocast_prefix_forward_backward_matches_fp32_reference() -> None:
    """Reduced prefix activations must share one sparse accumulator dtype."""

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_dtype = torch.float16 if use_cuda else torch.bfloat16
    control = _control().to(device)
    prefix = torch.nn.Linear(8, 8).to(device)
    raw_input = torch.randn(6, 8, device=device, requires_grad=True)

    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        encoded = torch.relu(prefix(raw_input))
        mixed_precision_output = control(encoded)
    assert encoded.dtype == amp_dtype
    assert torch.isfinite(mixed_precision_output).all()

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        fp32_reference = control(encoded.detach().float())
    tolerance = 0.02 if use_cuda else 0.08
    assert torch.allclose(
        mixed_precision_output.float(),
        fp32_reference,
        rtol=tolerance,
        atol=tolerance,
    )

    mixed_precision_output.float().square().mean().backward()
    assert raw_input.grad is not None
    assert torch.isfinite(raw_input.grad).all()
    for parameter in (*prefix.parameters(), *control.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_factory_builds_sparse_point_and_fails_closed_on_interface_or_seed() -> None:
    structured = {
        "architecture": {
            "excitatory_layer_sizes": [8],
            "inhibitory_layer_sizes": [2],
            "excitatory_branch_factors": [2],
            "inhibitory_branch_factors": [],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [1],
            "ei_synapses_per_branch_per_layer": [1],
            "ie_synapses_per_branch_per_layer": [1],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {"output_activation": "relu"},
        "sparsity": {"indexed": {"seed": 41, "output_chunk_size": 3}},
        "target_active_parameters": 27,
    }
    built = get_architecture(
        "sparse_active_matched_point",
        structured,
        input_dim=8,
        suffix_input_dim=8,
    )
    assert isinstance(built, SparseActiveMatchedPointAffine)
    assert built(torch.randn(2, 8)).shape == (2, 8)

    with pytest.raises(ValueError, match="does not match the suffix"):
        get_architecture(
            "sparse_active_matched_point",
            structured,
            input_dim=8,
            suffix_input_dim=7,
        )
    missing_seed = dict(structured)
    missing_seed["sparsity"] = {}
    with pytest.raises(ValueError, match=r"requires sparsity\.indexed\.seed"):
        get_architecture(
            "sparse_active_matched_point",
            missing_seed,
            input_dim=8,
            suffix_input_dim=8,
        )


def test_seed41_config_changes_only_the_control_and_output_identity() -> None:
    sparse = load_config(str(CONFIG))
    direct_path = CONFIG.with_name(
        "empirical_publication_v2_fig5_alexnet_fc7_direct_point_seed41.yaml"
    )
    direct = load_config(str(direct_path))

    assert sparse.model.core.type == "sparse_active_matched_point"
    assert sparse.model.core.sparsity.type == "indexed"
    assert sparse.model.core.sparsity.indexed["seed"] == 41
    assert sparse.model.core.sparsity.indexed["output_chunk_size"] == 2048
    assert sparse.model.core.transfer.output_activation == "relu"
    assert sparse.experiment.record_dataset_fingerprints is False
    assert sparse.outputs.run_name == "fig5_sparse_point_affine_active_matched_seed41"
    assert "fig5_sparse_point_affine_active_matched_seed41" in (
        sparse.outputs.results_dir
    )

    sparse_experiment = asdict(sparse.experiment)
    direct_experiment = asdict(direct.experiment)
    sparse_experiment["checkpointing"].pop("checkpoint_dir")
    direct_experiment["checkpointing"].pop("checkpoint_dir")
    assert sparse_experiment == direct_experiment
    assert sparse.data == direct.data
    assert sparse.model.vision_replacement == direct.model.vision_replacement
    assert sparse.model.core.architecture == direct.model.core.architecture
    assert sparse.model.core.connectivity == direct.model.core.connectivity
    assert sparse.model.core.transfer == direct.model.core.transfer
    assert sparse.model.core.morphology == direct.model.core.morphology
    assert sparse.model.core.sparsity == direct.model.core.sparsity
    assert sparse.model.core.reactivation == direct.model.core.reactivation
    assert sparse.model.core.blocklinear == direct.model.core.blocklinear
    assert sparse.model.core.implementation == direct.model.core.implementation
    assert sparse.training == direct.training
    assert sparse.analysis == direct.analysis
    assert sparse.distributed == direct.distributed

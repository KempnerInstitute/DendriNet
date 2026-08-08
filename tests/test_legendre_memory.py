"""Contracts for the fixed Legendre Delay Network memory core."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from scipy.linalg import expm

from dendritic_modeling.analysis.utils.model_resources import model_resource_summary
from dendritic_modeling.config import load_config
from dendritic_modeling.config.model import CoreConfig
from dendritic_modeling.config.recurrence import is_recurrent_core_config
from dendritic_modeling.networks.architectures.factory import (
    get_architecture,
    get_available_architectures,
)
from dendritic_modeling.networks.architectures.recurrent import (
    FixedLegendreMemory,
    FixedLegendreMemoryConfig,
    LegendreMemoryState,
    RecurrentCore,
    SteppableRecurrentCore,
    legendre_delay_continuous_matrices,
    zero_order_hold_discretization,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model


def _small_config(**overrides: object) -> FixedLegendreMemoryConfig:
    payload: dict[str, object] = {
        "input_dim": 3,
        "output_dim": 2,
        "memory_order": 4,
        "n_memory_channels": 2,
        "theta": 7.5,
        "dt": 0.25,
        "input_contacts": 5,
        "readout_contacts": 9,
        "topology_seed": 101,
        "initialization_seed": 211,
        "output_mode": "all",
    }
    payload.update(overrides)
    return FixedLegendreMemoryConfig(**payload)


def _independent_continuous_reference(
    order: int,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    A = np.empty((order, order), dtype=np.float64)
    B = np.empty(order, dtype=np.float64)
    for i in range(order):
        scale = (2 * i + 1) / theta
        B[i] = scale * ((-1.0) ** i)
        for j in range(order):
            sign = -1.0 if i < j else (-1.0) ** (i - j + 1)
            A[i, j] = scale * sign
    return A, B


def _independent_zoh_reference(
    A: np.ndarray,
    B: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    order = A.shape[0]
    augmented = np.zeros((order + 1, order + 1), dtype=np.float64)
    augmented[:order, :order] = A
    augmented[:order, order] = B
    discrete = expm(augmented * dt)
    return discrete[:order, :order], discrete[:order, order]


def test_primary_equations_and_zoh_match_independent_scipy_reference() -> None:
    order = 5
    theta = 9.25
    dt = 0.4
    expected_A, expected_B = _independent_continuous_reference(order, theta)
    expected_A_bar, expected_B_bar = _independent_zoh_reference(
        expected_A,
        expected_B,
        dt,
    )

    A, B = legendre_delay_continuous_matrices(order, theta)
    A_bar, B_bar = zero_order_hold_discretization(A, B, dt=dt)

    assert A.dtype == B.dtype == torch.float64
    assert A_bar.dtype == B_bar.dtype == torch.float64
    np.testing.assert_array_equal(A.numpy(), expected_A)
    np.testing.assert_array_equal(B.numpy(), expected_B)
    np.testing.assert_allclose(A_bar.numpy(), expected_A_bar, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(B_bar.numpy(), expected_B_bar, rtol=2e-14, atol=2e-14)


def test_registered_buffers_are_derived_in_float64_then_cast() -> None:
    config = _small_config(transition_buffer_dtype="float32")
    model = FixedLegendreMemory(config)
    expected_A, expected_B = legendre_delay_continuous_matrices(
        config.memory_order,
        config.theta,
    )
    expected_A_bar, expected_B_bar = zero_order_hold_discretization(
        expected_A,
        expected_B,
        dt=config.dt,
    )

    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    assert {"theta", "A", "B", "A_bar", "B_bar"} <= set(named_buffers)
    assert all(
        name not in named_parameters for name in ("theta", "A", "B", "A_bar", "B_bar")
    )
    assert all(
        named_buffers[name].dtype == torch.float32
        for name in ("theta", "A", "B", "A_bar", "B_bar")
    )
    torch.testing.assert_close(model.A, expected_A.float(), rtol=0, atol=0)
    torch.testing.assert_close(model.B, expected_B.float(), rtol=0, atol=0)
    torch.testing.assert_close(model.A_bar, expected_A_bar.float(), rtol=0, atol=0)
    torch.testing.assert_close(model.B_bar, expected_B_bar.float(), rtol=0, atol=0)


def test_impulse_rollout_matches_equation_and_hidden_state_carries() -> None:
    model = FixedLegendreMemory(_small_config()).double()
    sequence = torch.zeros((2, 7, 3), dtype=torch.float64)
    sequence[:, 0] = torch.tensor(
        [[1.0, -0.25, 0.5], [-0.5, 0.75, 0.2]],
        dtype=torch.float64,
    )
    sequence[:, 3] = torch.tensor(
        [[0.1, 0.3, -0.2], [0.4, -0.1, 0.6]],
        dtype=torch.float64,
    )

    outputs, final_state = model(sequence, return_hidden=True)
    manual_state = torch.zeros((2, 2, 4), dtype=torch.float64)
    manual_outputs = []
    for timestep in range(sequence.shape[1]):
        drive = model.input_mixer(sequence[:, timestep])
        manual_state = manual_state @ model.A_bar.T
        manual_state = manual_state + drive.unsqueeze(-1) * model.B_bar
        manual_outputs.append(model.state_readout(manual_state.flatten(start_dim=1)))
    expected_outputs = torch.stack(manual_outputs, dim=1)

    torch.testing.assert_close(outputs, expected_outputs, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(final_state.memory, manual_state, rtol=1e-13, atol=1e-13)

    first_outputs, carried = model(sequence[:, :3], return_hidden=True)
    second_outputs, split_final = model(
        sequence[:, 3:],
        hidden=carried,
        return_hidden=True,
    )
    torch.testing.assert_close(
        torch.cat((first_outputs, second_outputs), dim=1),
        outputs,
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(split_final.memory, final_state.memory)


def test_sparse_supports_and_initialization_are_seed_deterministic() -> None:
    torch.manual_seed(999)
    expected_next_draw = torch.rand(5)
    torch.manual_seed(999)
    first = FixedLegendreMemory(_small_config())
    actual_next_draw = torch.rand(5)
    repeated = FixedLegendreMemory(_small_config())
    changed_topology = FixedLegendreMemory(_small_config(topology_seed=102))

    assert torch.equal(actual_next_draw, expected_next_draw)
    for name in ("input_mixer", "state_readout"):
        first_map = getattr(first, name)
        repeated_map = getattr(repeated, name)
        changed_map = getattr(changed_topology, name)
        assert torch.equal(
            first_map.connection_indices_padded(),
            repeated_map.connection_indices_padded(),
        )
        assert not torch.equal(
            first_map.connection_indices_padded(),
            changed_map.connection_indices_padded(),
        )
    for (first_name, first_tensor), (repeated_name, repeated_tensor) in zip(
        first.state_dict().items(),
        repeated.state_dict().items(),
        strict=True,
    ):
        assert first_name == repeated_name
        assert torch.equal(first_tensor, repeated_tensor)


def test_gradients_flow_only_through_sparse_learned_maps() -> None:
    model = FixedLegendreMemory(_small_config())
    x = torch.randn(4, 6, 3, generator=torch.Generator().manual_seed(42))
    x.requires_grad_()
    loss = model(x).square().mean()
    loss.backward()

    parameters = dict(model.named_parameters())
    assert parameters
    assert all(
        name.startswith(("input_mixer.", "state_readout.")) for name in parameters
    )
    for parameter in parameters.values():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert int(torch.count_nonzero(parameter.grad)) > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name in ("theta", "A", "B", "A_bar", "B_bar"):
        assert getattr(model, name).requires_grad is False


@pytest.mark.parametrize("output_mode", ["all", "last", "mean"])
def test_sequence_output_modes_and_state_api(output_mode: str) -> None:
    model = FixedLegendreMemory(_small_config(output_mode=output_mode))
    x = torch.randn(2, 5, 3)
    lengths = torch.tensor([3, 5])
    output, state = model(x, return_hidden=True, seq_lengths=lengths)

    expected_shape = (2, 5, 2) if output_mode == "all" else (2, 2)
    assert output.shape == expected_shape
    assert state.memory.shape == (2, 2, 4)
    assert isinstance(model, RecurrentCore)
    assert isinstance(model, SteppableRecurrentCore)
    clone = state.clone()
    detached = state.detach()
    assert torch.equal(clone.memory, state.memory)
    assert clone.memory.data_ptr() != state.memory.data_ptr()
    assert detached.memory.requires_grad is False
    moved = state.to("cpu", dtype=torch.float64)
    assert moved.memory.dtype == torch.float64

    if output_mode == "all":
        assert torch.equal(output[0, 3:], torch.zeros_like(output[0, 3:]))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"memory_order": True}, "memory_order"),
        ({"n_memory_channels": 0}, "n_memory_channels"),
        ({"theta": True}, "theta"),
        ({"theta": float("nan")}, "theta"),
        ({"dt": 0.0}, "dt"),
        ({"transition_buffer_dtype": "bfloat16"}, "transition_buffer_dtype"),
        ({"trainable_transition": True}, "transition is fixed"),
        ({"trainable_theta": True}, "theta horizon is fixed"),
        ({"input_contacts": 2}, "input_contacts"),
        ({"readout_contacts": 7}, "readout_contacts"),
        (
            {
                "indexed_recompute_backward": True,
                "indexed_projection_backend": "triton",
            },
            "cannot be combined",
        ),
    ],
)
def test_config_validation_fails_closed(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _small_config(**overrides)


def test_protocol_construction_has_exact_state_contact_and_fixed_coupling_ledgers() -> (
    None
):
    model = FixedLegendreMemory(
        FixedLegendreMemoryConfig(
            input_dim=20,
            output_dim=16,
            memory_order=20,
            n_memory_channels=30,
            theta=337.0,
            input_contacts=580,
            readout_contacts=700,
            topology_seed=7727,
            initialization_seed=8837,
            output_mode="all",
        )
    )
    state = model.init_state(1, torch.device("cpu"))
    summary = model_resource_summary(model)
    ledger = summary["mechanism_ledger"]

    assert state.memory.shape == (1, 30, 20)
    assert summary["persistent_state_scalars_per_sample"] == 600
    assert summary["total_parameters"] == 1280
    assert summary["trainable_parameters"] == 1280
    assert summary["synapse_candidate_parameters"] == 1280
    assert summary["active_synapses"] == 1280
    assert summary["candidate_synapse_slots"] == 10200
    assert ledger["memory_state_scalars_per_sample"] == 600
    assert ledger["learned_input_mixer_contacts"] == 580
    assert ledger["learned_state_readout_contacts"] == 700
    assert ledger["active_learned_contacts"] == 1280
    assert ledger["fixed_transition_coefficients_declared"] == 12000
    assert ledger["fixed_input_injection_coefficients_declared"] == 600
    assert ledger["fixed_couplings_declared"] == 12600
    assert ledger["shared_fixed_discrete_operator_scalars"] == 420
    assert ledger["registered_fixed_buffer_scalars"] == 841
    assert ledger["input_source_coverage"] is True
    assert ledger["state_source_coverage"] is True
    assert model.input_mixer.total_contacts == 580
    assert model.state_readout.total_contacts == 700
    assert set(model.input_mixer.degrees.tolist()) == {19, 20}
    assert set(model.state_readout.degrees.tolist()) == {43, 44}


def test_factory_config_and_recurrent_detection_are_registered() -> None:
    parameters = {
        "legendre_memory": {
            "output_dim": 2,
            "memory_order": 4,
            "n_memory_channels": 2,
            "theta": 8.0,
            "input_contacts": 5,
            "readout_contacts": 9,
        }
    }
    model = get_architecture("legendre_memory", parameters, input_dim=3)

    assert isinstance(model, FixedLegendreMemory)
    assert "legendre_memory" in get_available_architectures()
    assert is_recurrent_core_config(
        CoreConfig(
            type="legendre_memory", legendre_memory=parameters["legendre_memory"]
        )
    )
    with pytest.raises(ValueError, match="input_dim required"):
        get_architecture("legendre_memory", parameters)


def test_standard_yaml_preserves_and_builds_legendre_memory(tmp_path) -> None:
    memory_payload = {
        "output_dim": 2,
        "memory_order": 4,
        "n_memory_channels": 2,
        "theta": 8.0,
        "input_contacts": 5,
        "readout_contacts": 9,
        "output_mode": "all",
    }
    payload = {
        "data": {"dataset_name": "mnist"},
        "model": {
            "task": "regression",
            "encoder": {
                "type": "identity",
                "params": {"input_dim": 3, "input_shape": None},
            },
            "core": {
                "type": "legendre_memory",
                "legendre_memory": memory_payload,
            },
            "decoder": {
                "type": "MLP",
                "params": {
                    "hidden_dims": [],
                    "activation": "relu",
                    "output_dim": 1,
                },
            },
        },
        "training": {"main": {"common": {"epochs": 1}}},
    }
    config_path = tmp_path / "legendre_memory.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    load_config.cache_clear()
    config = load_config(str(config_path))

    assert config.model.core.legendre_memory == memory_payload
    assert is_recurrent_core_config(config.model.core)
    model, _ = initialize_model(config.model)
    assert isinstance(model.core_network, FixedLegendreMemory)
    assert model.core_network.output_dim == 2


def test_state_and_input_validation_rejects_silent_mismatch() -> None:
    model = FixedLegendreMemory(_small_config())
    x = torch.zeros(2, 3)
    with pytest.raises(TypeError, match="LegendreMemoryState"):
        model.step(x, torch.zeros(2, 2, 4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        model.step(x, LegendreMemoryState(torch.zeros(1, 2, 4)))
    with pytest.raises(ValueError, match="share device and dtype"):
        model.step(x, LegendreMemoryState(torch.zeros(2, 2, 4, dtype=torch.float64)))
    with pytest.raises(ValueError, match="input dimension"):
        model(torch.zeros(2, 4, 2))
    with pytest.raises(ValueError, match="seq_lengths"):
        model(torch.zeros(2, 4, 3), seq_lengths=torch.tensor([4]))
    with pytest.raises(TypeError, match="integer dtype"):
        model(torch.zeros(2, 4, 3), seq_lengths=torch.tensor([4.0, 4.0]))

"""Contracts for the mechanism-matched heterogeneous-leak point RNN."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict

import pytest
import torch
import yaml

from dendritic_modeling.analysis.utils.model_resources import model_resource_summary
from dendritic_modeling.analysis.utils.state_transition import (
    HeterogeneousLeakStateCodec,
    StateTransitionLinearization,
)
from dendritic_modeling.config import load_config
from dendritic_modeling.config.recurrence import is_recurrent_core_config
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.degree_grouped_indexed import (
    DegreeGroupedIndexedLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent import (
    HeterogeneousLeakCTRNN,
    HeterogeneousLeakCTRNNConfig,
    HeterogeneousLeakCTRNNState,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model


def _small_config(**overrides) -> HeterogeneousLeakCTRNNConfig:
    payload = {
        "input_dim": 2,
        "input_projection_dims": [3],
        "n_excitatory_outputs": 2,
        "n_inhibitory_outputs": 1,
        "channels_per_output": 4,
        "tau_values": [9.0, 3.0],
        "owner_tau_patterns": [
            {
                "owner_count": 3,
                "tau_counts": [2, 2],
                "excitatory_owner_count": 2,
            }
        ],
        "input_contacts": 30,
        "recurrent_excitatory_contacts": 18,
        "recurrent_inhibitory_contacts": 9,
        "readout_contacts": 4,
        "input_scale": 0.7,
        "recurrent_excitatory_scale": 0.5,
        "recurrent_inhibitory_scale": 0.4,
        "readout_scale": 1.0,
        "topology_seed": 711,
        "initialization_seed": 919,
    }
    payload.update(overrides)
    return HeterogeneousLeakCTRNNConfig(**payload)


def _structured_point_config(**overrides) -> HeterogeneousLeakCTRNNConfig:
    payload = {
        "input_dim": 5,
        "input_projection_dims": [],
        "input_contacts": 2800,
        "recurrent_excitatory_contacts": 2800,
        "recurrent_inhibitory_contacts": 656,
        "readout_contacts": 64,
        "input_routing": {
            "method": "tau_feature_blocks",
            "require_source_coverage": True,
            "ranges_by_tau": [
                [[2, 4]],
                [[0, 2], [4, 5]],
                [[0, 2], [4, 5]],
            ],
        },
    }
    payload.update(overrides)
    return HeterogeneousLeakCTRNNConfig(**payload)


@pytest.fixture(scope="module")
def paper_control() -> HeterogeneousLeakCTRNN:
    return HeterogeneousLeakCTRNN(HeterogeneousLeakCTRNNConfig())


@pytest.fixture(scope="module")
def structured_point_control() -> HeterogeneousLeakCTRNN:
    return HeterogeneousLeakCTRNN(_structured_point_config())


def test_degree_grouped_indexed_is_exact_deterministic_and_covered() -> None:
    degrees = torch.tensor([0, 1, 2, 3, 2, 1, 3])
    first = DegreeGroupedIndexedLinear(
        5,
        7,
        degrees,
        seed=17,
        require_source_coverage=True,
        row_sum_init=0.9,
    )
    repeated = DegreeGroupedIndexedLinear(
        5,
        7,
        degrees,
        seed=17,
        require_source_coverage=True,
        row_sum_init=0.9,
    )
    changed = DegreeGroupedIndexedLinear(
        5,
        7,
        degrees,
        seed=19,
        require_source_coverage=True,
        row_sum_init=0.9,
    )

    first_indices = first.connection_indices_padded()
    assert torch.equal(first_indices, repeated.connection_indices_padded())
    assert not torch.equal(first_indices, changed.connection_indices_padded())
    assert first.total_contacts == int(degrees.sum()) == 12
    assert int(first.source_outdegrees().min()) > 0
    assert int(first.source_outdegrees().max() - first.source_outdegrees().min()) <= 1
    for row, degree in enumerate(degrees.tolist()):
        selected = first_indices[row, :degree]
        assert selected.unique().numel() == degree
        assert torch.equal(
            first_indices[row, degree:],
            torch.full_like(first_indices[row, degree:], -1),
        )

    x = torch.arange(10, dtype=torch.float32).reshape(2, 5).requires_grad_()
    output = first(x)
    expected = torch.zeros(2, 7)
    for row, degree in enumerate(degrees.tolist()):
        if degree:
            expected[:, row] = 0.9 * x[:, first_indices[row, :degree]].mean(dim=-1)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in first.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_degree_grouped_indexed_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        DegreeGroupedIndexedLinear(3, 2, [1], seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        DegreeGroupedIndexedLinear(3, 2, [1, -1], seed=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        DegreeGroupedIndexedLinear(3, 2, [1, 4], seed=1)
    with pytest.raises(ValueError, match="source coverage"):
        DegreeGroupedIndexedLinear(
            5,
            2,
            [1, 1],
            seed=1,
            require_source_coverage=True,
        )


def test_degree_grouped_indexed_hard_mask_is_exact_balanced_and_deterministic() -> None:
    degrees = torch.tensor([1, 2, 1, 0, 2, 1, 1])
    mask = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    first = DegreeGroupedIndexedLinear(
        5,
        7,
        degrees,
        seed=23,
        connection_mask=mask,
        require_source_coverage=True,
        row_sum_init=0.8,
    )
    repeated = DegreeGroupedIndexedLinear(
        5,
        7,
        degrees,
        seed=23,
        connection_mask=mask,
        require_source_coverage=True,
        row_sum_init=0.8,
    )

    indices = first.connection_indices_padded()
    assert torch.equal(indices, repeated.connection_indices_padded())
    for row, degree in enumerate(degrees.tolist()):
        selected = indices[row, :degree]
        assert selected.unique().numel() == degree
        assert bool(mask[row, selected].all())
        assert torch.equal(
            indices[row, degree:],
            torch.full_like(indices[row, degree:], -1),
        )
    assert bool((first.source_outdegrees() > 0).all())
    counts = first.connectivity_resource_counts()
    assert counts["candidate_slots"] == int(mask.sum())
    assert counts["active_synapses"] == int(degrees.sum())
    assert counts["mask_source"] == "configured_mask_seeded_balanced_indices"

    x = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    output = first(x)
    expected = torch.zeros(2, 7)
    for row, degree in enumerate(degrees.tolist()):
        if degree:
            expected[:, row] = 0.8 * x[:, indices[row, :degree]].mean(dim=-1)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)


def test_degree_grouped_indexed_hard_mask_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        DegreeGroupedIndexedLinear(
            3,
            2,
            [1, 1],
            seed=1,
            connection_mask=torch.ones(2, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="requested degree"):
        DegreeGroupedIndexedLinear(
            3,
            2,
            [2, 1],
            seed=1,
            connection_mask=torch.tensor([[1, 0, 0], [0, 1, 0]]),
        )
    with pytest.raises(ValueError, match="excludes sources"):
        DegreeGroupedIndexedLinear(
            3,
            3,
            [1, 1, 1],
            seed=1,
            connection_mask=torch.tensor(
                [[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.bool
            ),
            require_source_coverage=True,
        )


@pytest.mark.parametrize(
    "degrees",
    (
        [1.5, 2],
        [1.0, 2.0],
        [True, 2],
        torch.tensor([1.0, 2.0]),
        torch.tensor([True, False]),
    ),
)
def test_degree_grouped_indexed_rejects_non_integer_degrees(degrees) -> None:
    with pytest.raises(TypeError, match="integer values"):
        DegreeGroupedIndexedLinear(3, 2, degrees, seed=1)


def test_paper_control_exact_state_tau_owner_and_contact_ledger(
    paper_control: HeterogeneousLeakCTRNN,
) -> None:
    model = paper_control
    state = model.init_state(1, torch.device("cpu"))
    assert state.voltage.shape == (1, 2800)
    assert state.outputs.shape == (1, 80)

    assert Counter(model.taus.tolist()) == Counter({125.0: 1728, 25.0: 768, 5.0: 304})
    owner_tau = model.tau_indices.reshape(80, 35)
    owner_counts = [
        tuple(torch.bincount(row, minlength=3).tolist()) for row in owner_tau
    ]
    assert Counter(owner_counts[:64]) == Counter({(22, 10, 3): 38, (21, 9, 5): 26})
    assert Counter(owner_counts[64:]) == Counter({(22, 10, 3): 10, (21, 9, 5): 6})

    expected_degrees = {
        "input_drive": Counter({2: 1616, 3: 1184}),
        "recurrent_excitation": Counter({1: 544, 2: 2256}),
        "recurrent_inhibition": Counter({0: 272, 1: 2528}),
    }
    for name, expected in expected_degrees.items():
        projection = getattr(model, name)
        assert Counter(projection.degrees.tolist()) == expected
        assert int(projection.source_outdegrees().min()) > 0
        assert (
            int(
                projection.source_outdegrees().max()
                - projection.source_outdegrees().min()
            )
            <= 1
        )

    assert model.exc_readout.connection_indices.shape == (64, 12)
    for owner, indices in enumerate(model.exc_readout.connection_indices):
        assert bool((indices >= owner * 35).all())
        assert bool((indices < (owner + 1) * 35).all())
        assert indices.unique().numel() == 12

    summary = model_resource_summary(model)
    assert summary["total_parameters"] == 18704
    assert summary["trainable_parameters"] == 18704
    assert summary["synapse_candidate_parameters"] == 15136
    assert summary["non_synaptic_parameters"] == 3568
    assert summary["candidate_synapse_slots"] == 405440
    assert summary["active_synapses"] == 15136
    assert summary["persistent_state_scalars_per_sample"] == 2880
    ledger = summary["mechanism_ledger"]
    assert ledger["leaky_state_scalars_per_sample"] == 2800
    assert ledger["delayed_output_scalars_per_sample"] == 80
    assert ledger["tau_counts"] == {"125.0": 1728, "25.0": 768, "5.0": 304}
    assert ledger["fixed_channel_to_owner_wires"] == 2800
    assert (
        ledger["input_contacts"],
        ledger["recurrent_excitatory_contacts"],
        ledger["recurrent_inhibitory_contacts"],
        ledger["learned_readout_contacts"],
    ) == (6784, 5056, 2528, 768)
    assert ledger["excitatory_readout_mode"] == "positive_linear"
    assert ledger["excitatory_readout_bounded"] is False


def test_unrestricted_default_fixed_support_is_bitwise_unchanged(
    paper_control: HeterogeneousLeakCTRNN,
) -> None:
    expected = {
        "input_drive": "829ca9334039f5d766b05fe19e7c879f36a19d4a544b30fc27667559347e18b0",
        "recurrent_excitation": "1fb6b752fc14d6733b8e13f824b795ce2a20245e8a97f3e9954e2baec6590f97",
        "recurrent_inhibition": "f6e147c7bf59f4f911a9ce8c11206d02a39bf0ce71f40270d3bc9afc323a4aac",
    }
    for name, expected_sha256 in expected.items():
        indices = (
            getattr(paper_control, name)
            .connection_indices_padded()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )
        assert hashlib.sha256(indices).hexdigest() == expected_sha256
    ledger = paper_control.resource_ledger()
    assert ledger["input_projection_mode"] == "mlp"
    assert ledger["input_routing_method"] == "unrestricted"
    assert ledger["input_routing_hard"] is False
    assert ledger["input_routing_candidate_slots"] == 2800 * 64
    assert ledger["input_source_ranges_by_tau"] is None


def test_structured_point_exact_state_tau_contact_and_routing_ledger(
    structured_point_control: HeterogeneousLeakCTRNN,
) -> None:
    model = structured_point_control
    assert isinstance(model.input_projection, torch.nn.Identity)
    assert model.init_state(1, torch.device("cpu")).voltage.shape == (1, 2800)
    assert model.init_state(1, torch.device("cpu")).outputs.shape == (1, 80)
    assert Counter(model.taus.tolist()) == Counter({125.0: 1728, 25.0: 768, 5.0: 304})

    input_indices = model.input_drive.connection_indices_padded()
    assert input_indices.shape == (2800, 1)
    allowed_by_tau = ({2, 3}, {0, 1, 4}, {0, 1, 4})
    for tau_index, allowed in enumerate(allowed_by_tau):
        rows = model.tau_indices == tau_index
        selected = input_indices[rows, 0]
        assert set(selected.tolist()).issubset(allowed)
        assert set(selected.tolist()) == allowed
    eligibility_classes = (
        (model.tau_indices == 0, {2, 3}),
        (model.tau_indices != 0, {0, 1, 4}),
    )
    for rows, allowed in eligibility_classes:
        selected = input_indices[rows, 0]
        per_source = torch.bincount(selected, minlength=5)[sorted(allowed)]
        assert int(per_source.max() - per_source.min()) <= 1
    assert bool((model.input_drive.source_outdegrees() > 0).all())

    summary = model_resource_summary(model)
    assert summary["total_parameters"] == 9120
    assert summary["synapse_candidate_parameters"] == 6320
    assert summary["non_synaptic_parameters"] == 2800
    assert summary["candidate_synapse_slots"] == 232912
    assert summary["active_synapses"] == 6320
    assert summary["persistent_state_scalars_per_sample"] == 2880
    assert [record["active_synapses"] for record in summary["synapses"]] == [
        2800,
        2800,
        656,
        64,
    ]
    ledger = summary["mechanism_ledger"]
    assert ledger["reachable_carried_state_scalars_per_sample"] == 2880
    assert ledger["tau_counts"] == {"125.0": 1728, "25.0": 768, "5.0": 304}
    assert ledger["active_learned_contacts"] == 6320
    assert ledger["input_projection_mode"] == "identity"
    assert ledger["projected_input_dim"] == 5
    assert ledger["input_routing_method"] == "tau_feature_blocks"
    assert ledger["input_routing_hard"] is True
    assert ledger["input_routing_require_source_coverage"] is True
    assert ledger["input_routing_candidate_slots"] == 6672
    assert ledger["input_source_ranges_by_tau"] == {
        "125.0": [[2, 4]],
        "25.0": [[0, 2], [4, 5]],
        "5.0": [[0, 2], [4, 5]],
    }


def test_structured_point_forbidden_raw_features_have_zero_direct_derivative(
    structured_point_control: HeterogeneousLeakCTRNN,
) -> None:
    model = structured_point_control
    state = model.init_state(1, torch.device("cpu"))
    x = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6]], requires_grad=True)

    def next_voltage(raw_input: torch.Tensor) -> torch.Tensor:
        return model.step(raw_input, state)[1].voltage

    jacobian = torch.autograd.functional.jacobian(next_voltage, x)[0, :, 0, :]
    assert bool((jacobian.abs().sum(dim=1) > 0).all())
    allowed_by_tau = ({2, 3}, {0, 1, 4}, {0, 1, 4})
    for tau_index, allowed in enumerate(allowed_by_tau):
        rows = model.tau_indices == tau_index
        forbidden = sorted(set(range(5)) - allowed)
        assert torch.count_nonzero(jacobian[rows][:, forbidden]) == 0

    def next_outputs(raw_input: torch.Tensor) -> torch.Tensor:
        return model.step(raw_input, state)[1].outputs

    output_jacobian = torch.autograd.functional.jacobian(next_outputs, x)[0, :, 0, :]
    assert bool((output_jacobian.abs().sum(dim=1) > 0).all())


def test_structured_and_unrestricted_point_keep_state_recurrence_and_readout_fixed(
    structured_point_control: HeterogeneousLeakCTRNN,
) -> None:
    structured = structured_point_control
    unrestricted = HeterogeneousLeakCTRNN(_structured_point_config(input_routing=None))
    for name in ("taus", "tau_indices", "owner_polarity", "owner_indices"):
        assert torch.equal(getattr(structured, name), getattr(unrestricted, name))
    for name in ("recurrent_excitation", "recurrent_inhibition"):
        assert torch.equal(
            getattr(structured, name).connection_indices_padded(),
            getattr(unrestricted, name).connection_indices_padded(),
        )
    assert torch.equal(
        structured.exc_readout.connection_indices,
        unrestricted.exc_readout.connection_indices,
    )
    assert not torch.equal(
        structured.input_drive.connection_indices_padded(),
        unrestricted.input_drive.connection_indices_padded(),
    )
    structured_parameters = dict(structured.named_parameters())
    unrestricted_parameters = dict(unrestricted.named_parameters())
    assert structured_parameters.keys() == unrestricted_parameters.keys()
    for name, parameter in structured_parameters.items():
        assert torch.equal(parameter, unrestricted_parameters[name]), name
    for model in (structured, unrestricted):
        ledger = model.resource_ledger()
        assert ledger["reachable_carried_state_scalars_per_sample"] == 2880
        assert ledger["active_learned_contacts"] == 6320


def test_small_control_manual_step_reconstructs_true_equation() -> None:
    model = HeterogeneousLeakCTRNN(_small_config())
    state = model.init_state(2, torch.device("cpu"))
    state.voltage.copy_(torch.linspace(-0.2, 0.3, 24).reshape(2, 12))
    state.outputs.copy_(torch.linspace(0.1, 0.6, 6).reshape(2, 3))
    x = torch.tensor([[0.2, 0.7], [0.8, 0.1]])

    output, next_state = model.step(x, state)
    projected = model.input_projection(x)
    drive = (
        model.input_drive(projected)
        + model.recurrent_excitation(state.outputs[:, :2])
        - model.recurrent_inhibition(state.outputs[:, 2:])
        + model.bias
    )
    expected_voltage = model.decays * state.voltage + (1.0 - model.decays) * drive
    rates = model._activation(expected_voltage)
    owner_means = rates.reshape(2, 3, 4).mean(dim=-1)
    expected_exc = 0.5 * (owner_means[:, :2] + model.exc_readout(rates))
    expected_outputs = torch.cat((expected_exc, owner_means[:, 2:]), dim=-1)

    assert torch.equal(next_state.voltage, expected_voltage)
    assert torch.equal(next_state.outputs, expected_outputs)
    assert torch.equal(output, expected_exc)


def test_convex_excitatory_readout_matches_normalized_sparse_equation() -> None:
    model = HeterogeneousLeakCTRNN(_small_config(excitatory_readout_mode="convex"))
    state = model.init_state(2, torch.device("cpu"))
    state.voltage.copy_(torch.linspace(-0.2, 0.3, 24).reshape(2, 12))
    state.outputs.copy_(torch.linspace(0.1, 0.6, 6).reshape(2, 3))
    x = torch.tensor([[0.2, 0.7], [0.8, 0.1]])

    output, next_state = model.step(x, state)
    projected = model.input_projection(x)
    drive = (
        model.input_drive(projected)
        + model.recurrent_excitation(state.outputs[:, :2])
        - model.recurrent_inhibition(state.outputs[:, 2:])
        + model.bias
    )
    expected_voltage = model.decays * state.voltage + (1.0 - model.decays) * drive
    rates = model._activation(expected_voltage)
    owner_means = rates.reshape(2, 3, 4).mean(dim=-1)
    selected = rates[:, model.exc_readout.connection_indices]
    normalized_weights = torch.softmax(model.exc_readout.pre_w, dim=-1)
    normalized_learned = (selected * normalized_weights.unsqueeze(0)).sum(dim=-1)
    expected_exc = 0.5 * (owner_means[:, :2] + normalized_learned)

    assert torch.equal(output, expected_exc)
    assert torch.equal(next_state.outputs[:, :2], expected_exc)
    ledger = model.resource_ledger()
    assert ledger["excitatory_readout_mode"] == "convex"
    assert ledger["excitatory_readout_bounded"] is True


def test_convex_excitatory_readout_is_bounded_under_extreme_weights() -> None:
    model = HeterogeneousLeakCTRNN(_small_config(excitatory_readout_mode="convex"))
    with torch.no_grad():
        model.exc_readout.pre_w.copy_(torch.tensor([[-40.0, 40.0], [40.0, -40.0]]))
    state = model.init_state(2, torch.device("cpu"))
    state.voltage.copy_(torch.linspace(-50.0, 50.0, 24).reshape(2, 12))
    state.outputs.copy_(torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]]))

    output, next_state = model.step(
        torch.tensor([[100.0, -100.0], [-100.0, 100.0]]),
        state,
    )

    assert torch.isfinite(output).all()
    assert bool(((0.0 <= output) & (output <= 1.0)).all())
    assert torch.isfinite(next_state.outputs).all()
    assert bool(((0.0 <= next_state.outputs) & (next_state.outputs <= 1.0)).all())


def test_paper_convex_readout_is_stable_at_huge_equal_raw_weights() -> None:
    config = HeterogeneousLeakCTRNNConfig(excitatory_readout_mode="convex")
    model = HeterogeneousLeakCTRNN(config)
    with torch.no_grad():
        model.exc_readout.pre_w.fill_(100.0)
    rates = torch.linspace(
        0.01,
        0.99,
        config.hidden_dim,
        dtype=model.exc_readout.pre_w.dtype,
    ).unsqueeze(0)
    rates.requires_grad_()

    output = model._convex_excitatory_readout(rates)
    selected = rates[:, model.exc_readout.connection_indices]
    expected = selected.mean(dim=-1)

    torch.testing.assert_close(output, expected)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    gradient = model.exc_readout.pre_w.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_convex_excitatory_readout_receives_finite_nonzero_gradients() -> None:
    model = HeterogeneousLeakCTRNN(_small_config(excitatory_readout_mode="convex"))
    state = None
    output = None
    for timestep in range(3):
        x = torch.tensor(
            [
                [0.2 + 0.1 * timestep, 0.8 - 0.05 * timestep],
                [0.7 - 0.05 * timestep, 0.1 + 0.1 * timestep],
            ]
        )
        output, state = model.step(x, state)
    assert output is not None
    output.square().sum().backward()

    gradient = model.exc_readout.pre_w.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_small_control_uses_previous_outputs_and_all_paths_receive_gradients() -> None:
    model = HeterogeneousLeakCTRNN(_small_config())
    x = torch.full((2, 2), 0.4)
    state = model.init_state(2, torch.device("cpu"))
    changed = state.clone()
    changed.outputs.fill_(0.3)
    base_output, base_next = model.step(x, state)
    changed_output, changed_next = model.step(x, changed)
    assert not torch.equal(base_next.voltage, changed_next.voltage)
    assert not torch.equal(base_output, changed_output)

    rollout_state = None
    for timestep in range(4):
        step_input = x + 0.03 * timestep
        output, rollout_state = model.step(step_input, rollout_state)
    output.sum().backward()
    sparse_modules = (
        model.input_drive,
        model.recurrent_excitation,
        model.recurrent_inhibition,
    )
    for module in sparse_modules:
        for group in module.groups:
            gradient = group.projection.pre_w.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient) > 0
    readout_gradient = model.exc_readout.pre_w.grad
    assert readout_gradient is not None
    assert torch.isfinite(readout_gradient).all()
    assert torch.count_nonzero(readout_gradient) > 0
    assert model.bias.grad is not None
    assert torch.isfinite(model.bias.grad).all()
    assert torch.count_nonzero(model.bias.grad) == model.config.hidden_dim


def test_small_control_is_seed_deterministic_without_global_rng_dependence() -> None:
    torch.manual_seed(1)
    caller_state = torch.random.get_rng_state().clone()
    first = HeterogeneousLeakCTRNN(_small_config())
    assert torch.equal(torch.random.get_rng_state(), caller_state)

    torch.manual_seed(999)
    caller_state = torch.random.get_rng_state().clone()
    repeated = HeterogeneousLeakCTRNN(_small_config())
    assert torch.equal(torch.random.get_rng_state(), caller_state)
    assert first.state_dict().keys() == repeated.state_dict().keys()
    for key, value in first.state_dict().items():
        assert torch.equal(value, repeated.state_dict()[key]), key

    changed = HeterogeneousLeakCTRNN(_small_config(topology_seed=712))
    assert not torch.equal(
        first.input_drive.connection_indices_padded(),
        changed.input_drive.connection_indices_padded(),
    )
    assert first.resource_ledger() == changed.resource_ledger()
    assert Counter(first.taus.tolist()) == Counter(changed.taus.tolist())

    changed_initialization = HeterogeneousLeakCTRNN(
        _small_config(initialization_seed=920)
    )
    assert torch.equal(
        first.input_drive.connection_indices_padded(),
        changed_initialization.input_drive.connection_indices_padded(),
    )
    assert not torch.equal(
        first.input_projection[0].weight,
        changed_initialization.input_projection[0].weight,
    )


def test_sequence_reductions_and_factory_contract() -> None:
    model = HeterogeneousLeakCTRNN(_small_config(output_mode="all"))
    sequence = torch.linspace(0.1, 0.9, 20).reshape(2, 5, 2)
    all_output, hidden = model(sequence, return_hidden=True)
    assert all_output.shape == (2, 5, 2)
    assert isinstance(hidden, HeterogeneousLeakCTRNNState)

    lengths = torch.tensor([3, 5])
    masked = model(sequence, seq_lengths=lengths)
    assert torch.equal(masked[0, 3:], torch.zeros_like(masked[0, 3:]))
    assert torch.equal(masked[1], all_output[1])

    model._output_mode = "last"
    last = model(sequence, seq_lengths=lengths)
    assert torch.equal(last[0], all_output[0, 2])
    assert torch.equal(last[1], all_output[1, 4])
    model._output_mode = "mean"
    mean = model(sequence, seq_lengths=lengths)
    assert torch.equal(mean[0], all_output[0, :3].mean(dim=0))
    assert torch.equal(mean[1], all_output[1].mean(dim=0))

    factory_model = get_architecture(
        "heterogeneous_leak_ctrnn",
        {"heterogeneous_leak_ctrnn": _small_config().__dict__},
        input_dim=2,
    )
    assert isinstance(factory_model, HeterogeneousLeakCTRNN)
    assert factory_model.output_dim == 2


@pytest.mark.parametrize("output_mode", ["all", "last", "mean"])
def test_sequence_reductions_map_zero_length_to_zero(output_mode: str) -> None:
    model = HeterogeneousLeakCTRNN(_small_config(output_mode=output_mode))
    outputs = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)

    reduced = model._reduce_outputs(outputs, torch.tensor([0, 2]))

    assert torch.equal(reduced[0], torch.zeros_like(reduced[0]))
    if output_mode == "all":
        assert torch.equal(reduced[1, :2], outputs[1, :2])
        assert torch.equal(reduced[1, 2:], torch.zeros_like(reduced[1, 2:]))
    elif output_mode == "last":
        assert torch.equal(reduced[1], outputs[1, 1])
    else:
        assert torch.equal(reduced[1], outputs[1, :2].mean(dim=0))


@pytest.mark.parametrize("output_mode", ["all", "last", "mean"])
@pytest.mark.parametrize(
    ("seq_lengths", "exception", "match"),
    [
        (torch.tensor([2]), ValueError, "shape"),
        (torch.tensor([[2], [4]]), ValueError, "shape"),
        ([2, 4], ValueError, "shape"),
        (torch.tensor([2.0, 4.0]), TypeError, "integer dtype"),
        (torch.tensor([True, False]), TypeError, "integer dtype"),
        (torch.tensor([-1, 4]), ValueError, r"\[0, time\]"),
        (torch.tensor([2, 5]), ValueError, r"\[0, time\]"),
    ],
)
def test_sequence_reductions_reject_invalid_lengths(
    output_mode: str,
    seq_lengths: object,
    exception: type[Exception],
    match: str,
) -> None:
    model = HeterogeneousLeakCTRNN(_small_config(output_mode=output_mode))
    outputs = torch.zeros(2, 4, 2)

    with pytest.raises(exception, match=match):
        model._reduce_outputs(outputs, seq_lengths)  # type: ignore[arg-type]


def test_standard_yaml_config_preserves_and_builds_ctrnn_payload(tmp_path) -> None:
    config_path = tmp_path / "heterogeneous_ctrnn.yaml"
    ctrnn_payload = asdict(_small_config())
    payload = {
        "data": {"dataset_name": "mnist"},
        "model": {
            "task": "regression",
            "encoder": {
                "type": "identity",
                "params": {"input_dim": 2, "input_shape": None},
            },
            "core": {
                "type": "heterogeneous_leak_ctrnn",
                "heterogeneous_leak_ctrnn": ctrnn_payload,
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
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    load_config.cache_clear()
    config = load_config(str(config_path))

    assert config.model.core.heterogeneous_leak_ctrnn == ctrnn_payload
    assert is_recurrent_core_config(config.model.core)
    regressor, _ = initialize_model(config.model)
    assert isinstance(regressor.core_network, HeterogeneousLeakCTRNN)
    assert regressor.core_network.output_dim == 2


def test_standard_yaml_builds_identity_tau_routed_ctrnn(tmp_path) -> None:
    config_path = tmp_path / "structured_point_ctrnn.yaml"
    ctrnn_payload = asdict(_structured_point_config())
    routing = ctrnn_payload["input_routing"]
    assert routing is not None
    routing["ranges_by_tau"] = [
        [[int(start), int(stop)] for start, stop in tau_ranges]
        for tau_ranges in routing["ranges_by_tau"]
    ]
    payload = {
        "data": {"dataset_name": "mnist"},
        "model": {
            "task": "regression",
            "encoder": {
                "type": "identity",
                "params": {"input_dim": 5, "input_shape": None},
            },
            "core": {
                "type": "heterogeneous_leak_ctrnn",
                "heterogeneous_leak_ctrnn": ctrnn_payload,
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
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    load_config.cache_clear()
    config = load_config(str(config_path))
    regressor, _ = initialize_model(config.model)

    core = regressor.core_network
    assert isinstance(core, HeterogeneousLeakCTRNN)
    assert isinstance(core.input_projection, torch.nn.Identity)
    assert core.config.input_projection_dims == []
    assert core.config.input_routing is not None
    assert core.config.input_routing.method == "tau_feature_blocks"
    assert core.resource_ledger()["active_learned_contacts"] == 6320


def test_true_carried_state_codec_and_matrix_free_jvp_vjp() -> None:
    model = HeterogeneousLeakCTRNN(_small_config())
    input_t = torch.tensor([[0.3, 0.7]])
    _, state = model.step(input_t)
    codec = HeterogeneousLeakStateCodec(model, state)
    assert codec.dimension == 15
    assert [spec.field for spec in codec.specs] == ["voltage", "outputs"]
    vector = codec.flatten(state)
    rebuilt = codec.unflatten(vector)
    assert torch.equal(rebuilt.voltage, state.voltage)
    assert torch.equal(rebuilt.outputs, state.outputs)

    linearization = StateTransitionLinearization(model, state, input_t)
    assert linearization.dimension == 15
    jacobian = torch.autograd.functional.jacobian(
        linearization.transition,
        linearization.state_vector,
    )
    assert jacobian.shape == (15, 15)
    assert bool((jacobian.abs().sum(dim=0) > 0).all())

    direction = torch.linspace(-0.4, 0.6, 15)
    cotangent = torch.linspace(0.7, -0.2, 15)
    jvp = linearization.jvp(direction)
    vjp = linearization.vjp(cotangent)
    torch.testing.assert_close(jvp, jacobian @ direction)
    torch.testing.assert_close(vjp, jacobian.T @ cotangent)
    torch.testing.assert_close(torch.dot(cotangent, jvp), torch.dot(vjp, direction))


def test_flat_point_voltage_transition_has_no_serial_cross_channel_jacobian() -> None:
    model = HeterogeneousLeakCTRNN(
        _small_config(
            input_projection_dims=[],
            input_contacts=12,
            input_routing={
                "method": "tau_feature_blocks",
                "ranges_by_tau": [[[0, 1]], [[1, 2]]],
                "require_source_coverage": True,
            },
        )
    )
    state = model.init_state(1, torch.device("cpu"))
    state.voltage.copy_(torch.linspace(-0.2, 0.3, 12).reshape(1, 12))
    state.outputs.copy_(torch.linspace(0.1, 0.4, 3).reshape(1, 3))
    x = torch.tensor([[0.25, 0.75]])

    def voltage_transition(voltage: torch.Tensor) -> torch.Tensor:
        replaced = HeterogeneousLeakCTRNNState(voltage=voltage, outputs=state.outputs)
        return model.step(x, replaced)[1].voltage

    jacobian = torch.autograd.functional.jacobian(
        voltage_transition,
        state.voltage,
    )[0, :, 0, :]
    expected = torch.diag(model.decays)
    torch.testing.assert_close(jacobian, expected)
    assert torch.count_nonzero(jacobian - torch.diag(torch.diagonal(jacobian))) == 0


def test_config_validation_rejects_inexact_or_unreachable_controls() -> None:
    with pytest.raises(ValueError, match="positive values"):
        _small_config(tau_values=[9.0, 0.0])
    with pytest.raises(ValueError, match="every output owner"):
        _small_config(
            owner_tau_patterns=[
                {
                    "owner_count": 2,
                    "tau_counts": [2, 2],
                    "excitatory_owner_count": 1,
                }
            ]
        )
    with pytest.raises(ValueError, match="input_contacts"):
        _small_config(input_contacts=11)
    with pytest.raises(ValueError, match="positive integer"):
        _small_config(recurrent_inhibitory_contacts=0)
    with pytest.raises(ValueError, match="divisible"):
        _small_config(readout_contacts=3)
    with pytest.raises(ValueError, match="excitatory_readout_mode"):
        _small_config(excitatory_readout_mode="normalized")
    with pytest.raises(ValueError, match=r"readout_scale must equal 1\.0"):
        _small_config(excitatory_readout_mode="convex", readout_scale=0.5)
    with pytest.raises(ValueError, match=r"requires input_projection_dims=\[\]"):
        _small_config(
            input_routing={
                "method": "tau_feature_blocks",
                "ranges_by_tau": [[[0, 1]], [[1, 2]]],
            }
        )
    with pytest.raises(ValueError, match="one entry per tau_values"):
        _small_config(
            input_projection_dims=[],
            input_routing={
                "method": "tau_feature_blocks",
                "ranges_by_tau": [[[0, 2]]],
            },
        )
    with pytest.raises(ValueError, match="must satisfy"):
        _small_config(
            input_projection_dims=[],
            input_routing={
                "method": "tau_feature_blocks",
                "ranges_by_tau": [[[0, 3]], [[1, 4]]],
            },
        )
    with pytest.raises(ValueError, match="must not overlap"):
        _small_config(
            input_projection_dims=[],
            input_routing={
                "method": "tau_feature_blocks",
                "ranges_by_tau": [
                    [[0, 2], [1, 2]],
                    [[0, 2]],
                ],
            },
        )
    with pytest.raises(ValueError, match="excludes raw features"):
        _small_config(
            input_projection_dims=[],
            input_contacts=12,
            input_routing={
                "method": "tau_feature_blocks",
                "ranges_by_tau": [[[0, 1]], [[0, 1]]],
            },
        )
    with pytest.raises(ValueError, match="maximum input row degree"):
        _small_config(
            input_projection_dims=[],
            input_contacts=24,
            input_routing={
                "method": "tau_feature_blocks",
                "ranges_by_tau": [[[0, 1]], [[0, 2]]],
                "require_source_coverage": True,
            },
        )


@pytest.mark.parametrize(
    ("value", "error", "message"),
    (
        (True, TypeError, "real numbers"),
        ("9.0", TypeError, "real numbers"),
        (9.0 + 0.0j, TypeError, "real numbers"),
        (float("nan"), ValueError, "positive values"),
        (float("inf"), ValueError, "positive values"),
        (-1.0, ValueError, "positive values"),
    ),
)
def test_config_tau_values_reject_lossy_or_nonfinite_values(
    value,
    error,
    message,
) -> None:
    with pytest.raises(error, match=message):
        _small_config(tau_values=[9.0, value])

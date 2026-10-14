"""Correctness tests for the actual recurrent-state Jacobian."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.analysis.tools.state_transition_jacobian import (
    StateTransitionJacobianAnalyzer,
)
from dendritic_modeling.analysis.utils.state_transition import (
    PopulationStateCodec,
    StateTransitionLinearization,
    StateTransitionProduct,
    UnsupportedPopulationStateError,
    leading_state_transition_singular_value,
    leading_state_transition_singular_value_block_subspace,
)
from dendritic_modeling.networks.architectures.recurrent import (
    PopulationLayerConfig,
    PopulationNetwork,
    PopulationNetworkConfig,
)


def _tiny_rate_network() -> PopulationNetwork:
    torch.manual_seed(5)
    network = PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=2,
            output_mode="last",
            layers=[
                PopulationLayerConfig(
                    name="microcircuit",
                    recurrent=True,
                    readout_population="pyr",
                    population_defaults={
                        "ff_excitatory_synapses": 2,
                        "ff_inhibitory_synapses": 0,
                        "rec_excitatory_synapses": 2,
                        "rec_inhibitory_synapses": 0,
                        "somatic_synapses": True,
                        "reactivate": False,
                        "dbl_init_method": "default",
                        "weight_transform": "identity",
                        "topk_type": "standard",
                        "allow_self_recurrence": True,
                        "level_taus": [3.0],
                    },
                    populations=[
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 2,
                            "branch_factors": [],
                        }
                    ],
                    connections=[
                        {"source": "input", "target": "pyr"},
                        {
                            "source": "pyr",
                            "target": "pyr",
                            "timing": "delayed",
                        },
                    ],
                )
            ],
        )
    )
    return network.double()


def _trajectory_point() -> tuple[PopulationNetwork, list, torch.Tensor]:
    network = _tiny_rate_network()
    input_t = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    state = network.init_state(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    with torch.no_grad():
        for _ in range(3):
            _, state = network(input_t, hidden=state, return_hidden=True)
    return network, state, input_t


def test_rate_state_codec_includes_five_trace_banks_and_population_output() -> None:
    network, state, _ = _trajectory_point()
    codec = PopulationStateCodec(network, state)

    assert codec.dimension == 12  # (five traces + one output) * two neurons
    assert [spec.field for spec in codec.specs] == [
        "trace_E",
        "trace_E_rec",
        "trace_I",
        "trace_I_rec",
        "trace_branch",
        "outputs",
    ]
    vector = codec.flatten(state)
    rebuilt = codec.unflatten(vector)
    assert torch.equal(codec.flatten(rebuilt), vector)


def test_rate_state_codec_fails_closed_on_optional_state() -> None:
    network, state, _ = _trajectory_point()
    state[0].population_states["pyr"].typed_traces = {
        "ampa": [torch.zeros(1, 2, dtype=torch.float64)]
    }

    try:
        PopulationStateCodec(network, state)
    except UnsupportedPopulationStateError as exc:
        assert "typed_traces" in str(exc)
    else:  # pragma: no cover - assertion message is clearer than pytest.raises here
        raise AssertionError("optional typed state must not be silently omitted")


def test_jvp_matches_dense_jacobian_and_centered_finite_difference() -> None:
    network, state, input_t = _trajectory_point()
    linearization = StateTransitionLinearization(network, state, input_t)
    vector = torch.randn(
        linearization.dimension,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(11),
    )

    dense_jacobian = torch.autograd.functional.jacobian(
        linearization.transition,
        linearization.state_vector,
    )
    matrix_free = linearization.jvp(vector)
    assert torch.allclose(matrix_free, dense_jacobian @ vector, atol=1e-11, rtol=1e-11)

    epsilon = 1e-6
    with torch.no_grad():
        finite_difference = (
            linearization.transition(linearization.state_vector + epsilon * vector)
            - linearization.transition(linearization.state_vector - epsilon * vector)
        ) / (2.0 * epsilon)
    assert torch.allclose(matrix_free, finite_difference, atol=5e-9, rtol=5e-9)


def test_vjp_satisfies_adjoint_identity_and_matches_dense_jacobian() -> None:
    network, state, input_t = _trajectory_point()
    linearization = StateTransitionLinearization(network, state, input_t)
    generator = torch.Generator().manual_seed(17)
    vector = torch.randn(
        linearization.dimension, dtype=torch.float64, generator=generator
    )
    cotangent = torch.randn(
        linearization.dimension, dtype=torch.float64, generator=generator
    )

    dense_jacobian = torch.autograd.functional.jacobian(
        linearization.transition,
        linearization.state_vector,
    )
    jvp = linearization.jvp(vector)
    vjp = linearization.vjp(cotangent)
    assert torch.allclose(vjp, dense_jacobian.T @ cotangent, atol=1e-11, rtol=1e-11)
    assert torch.allclose(
        torch.dot(cotangent, jvp),
        torch.dot(vjp, vector),
        atol=1e-11,
        rtol=1e-11,
    )


def test_input_jvp_and_vjp_match_dense_input_jacobian() -> None:
    network, state, input_t = _trajectory_point()
    linearization = StateTransitionLinearization(network, state, input_t)
    generator = torch.Generator().manual_seed(23)
    direction = torch.randn(input_t.shape, dtype=torch.float64, generator=generator)
    cotangent = torch.randn(
        linearization.dimension, dtype=torch.float64, generator=generator
    )

    dense_jacobian = torch.autograd.functional.jacobian(
        linearization.transition_from_input,
        linearization.input_t,
    ).reshape(linearization.dimension, -1)
    jvp = linearization.input_jvp(direction)
    vjp = linearization.input_vjp(cotangent)

    assert torch.allclose(jvp, dense_jacobian @ direction.reshape(-1), atol=1e-11)
    assert torch.allclose(vjp.reshape(-1), dense_jacobian.T @ cotangent, atol=1e-11)
    assert torch.allclose(
        torch.dot(cotangent, jvp),
        torch.dot(vjp.reshape(-1), direction.reshape(-1)),
        atol=1e-11,
    )


def test_power_iteration_matches_dense_leading_singular_value() -> None:
    network, state, input_t = _trajectory_point()
    linearization = StateTransitionLinearization(network, state, input_t)
    dense_jacobian = torch.autograd.functional.jacobian(
        linearization.transition,
        linearization.state_vector,
    )
    expected = torch.linalg.svdvals(dense_jacobian)[0].item()

    estimate = leading_state_transition_singular_value(
        linearization,
        max_iterations=100,
        relative_tolerance=1e-7,
        absolute_tolerance=1e-10,
        seed=7,
    )
    assert abs(estimate.singular_value - expected) < 2e-6


def test_block_krylov_matches_dense_state_jacobian_with_residual_certificate() -> None:
    network, state, input_t = _trajectory_point()
    linearization = StateTransitionLinearization(network, state, input_t)
    dense_jacobian = torch.autograd.functional.jacobian(
        linearization.transition,
        linearization.state_vector,
    )
    expected = torch.linalg.svdvals(dense_jacobian)[0].item()

    estimate = leading_state_transition_singular_value_block_subspace(
        linearization,
        block_size=3,
        max_iterations=4,
        relative_residual_tolerance=1e-9,
        absolute_residual_tolerance=1e-11,
        seed=37,
    )

    assert estimate.converged
    assert estimate.singular_value == pytest.approx(expected, rel=1e-9, abs=1e-10)
    assert estimate.relative_residual_norm <= 1.1e-9
    assert estimate.matvec_calls == 3 * estimate.iterations
    assert estimate.rmatvec_calls == 3 * estimate.iterations


def test_block_krylov_public_defaults_cover_a_small_state_space() -> None:
    network, state, input_t = _trajectory_point()
    linearization = StateTransitionLinearization(network, state, input_t)

    estimate = leading_state_transition_singular_value_block_subspace(linearization)

    assert estimate.block_size == 6
    assert estimate.maximum_subspace_dimension == linearization.dimension == 12
    assert estimate.maximum_matvec_calls == 12
    assert estimate.maximum_rmatvec_calls == 12


def test_state_transition_product_rejects_a_different_network_object() -> None:
    network, state, input_t = _trajectory_point()
    first = StateTransitionLinearization(network, state, input_t)

    other_network, other_state, other_input = _trajectory_point()
    with torch.no_grad():
        next(other_network.parameters()).add_(0.25)
    second = StateTransitionLinearization(other_network, other_state, other_input)

    with pytest.raises(ValueError, match="same network object"):
        StateTransitionProduct([first, second])


def test_state_transition_product_rejects_nonconsecutive_points() -> None:
    network, state, input_t = _trajectory_point()
    first = StateTransitionLinearization(network, state, input_t)
    repeated_point = StateTransitionLinearization(network, state, input_t)

    with pytest.raises(ValueError, match="must be consecutive"):
        StateTransitionProduct([first, repeated_point])

    consecutive_state = first.codec.unflatten(first.next_state_vector)
    consecutive = StateTransitionLinearization(network, consecutive_state, input_t)
    product = StateTransitionProduct([first, consecutive])
    assert product.dimension == first.dimension


def test_analyzer_metadata_distinguishes_state_jacobian_from_weight_spectrum() -> None:
    network, state, input_t = _trajectory_point()
    result = StateTransitionJacobianAnalyzer(
        max_iterations=5,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        seed=3,
    ).analyze(network, state=state, input_t=input_t)

    assert result["analysis_type"] == "state_transition_jacobian"
    assert result["is_state_transition_jacobian"] is True
    assert result["operator_materialized"] is False
    assert result["transition_definition"] == "q_t = F(q_{t-1}, x_t)"
    assert result["state_dimension"] == 12
    assert result["method"] == "two_sided_power_iteration_with_jvp_vjp"


def test_inconsistent_forward_products_fail_closed() -> None:
    network, state, input_t = _trajectory_point()
    linearization = StateTransitionLinearization(network, state, input_t)
    exact = linearization.jvp
    linearization.jvp = lambda vector, **kwargs: 1.5 * exact(vector, **kwargs)
    with pytest.raises(RuntimeError, match="inconsistent with its transpose"):
        linearization._check_adjoint_consistency()


def _indexed_network(backend: str) -> PopulationNetwork:
    torch.manual_seed(5)
    return PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=8,
            output_mode="last",
            layers=[
                PopulationLayerConfig(
                    name="microcircuit",
                    recurrent=True,
                    readout_population="pyr",
                    population_defaults={
                        "ff_excitatory_synapses": 4,
                        "ff_inhibitory_synapses": 0,
                        "rec_excitatory_synapses": 4,
                        "rec_inhibitory_synapses": 0,
                        "somatic_synapses": True,
                        "reactivate": False,
                        "dbl_init_method": "default",
                        "weight_transform": "identity",
                        "topk_type": "indexed",
                        "indexed_projection_backend": backend,
                        "allow_self_recurrence": True,
                        "level_taus": [3.0, 2.0],
                    },
                    populations=[
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 16,
                            "branch_factors": [2],
                        }
                    ],
                    connections=[
                        {"source": "input", "target": "pyr"},
                        {"source": "pyr", "target": "pyr", "timing": "delayed"},
                    ],
                )
            ],
        )
    ).cuda()


@pytest.mark.requires_gpu
def test_triton_indexed_projections_give_the_exact_state_jacobian() -> None:
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.backend import (
        indexed_projection_backend_available,
    )

    if not indexed_projection_backend_available("triton_transposed", device="cuda"):
        pytest.skip("triton_transposed is unavailable")
    products = {}
    for backend in ("eager", "triton_transposed"):
        network = _indexed_network(backend).eval()
        x = torch.linspace(-1.0, 1.0, 8, device="cuda").unsqueeze(0)
        state = network.init_state(
            batch_size=1, device=torch.device("cuda"), dtype=torch.float32
        )
        with torch.no_grad():
            for _ in range(3):
                _, state = network(x, hidden=state, return_hidden=True)
        linearization = StateTransitionLinearization(network, state, x)
        generator = torch.Generator(device="cuda").manual_seed(0)
        direction = torch.randn(
            linearization.dimension, generator=generator, device="cuda"
        )
        products[backend] = linearization.jvp(direction)
        configured = {
            layer.projection_backend
            for layer in network.modules()
            if hasattr(layer, "_last_resolved_projection_backend")
        }
        assert configured == {backend}
    torch.testing.assert_close(
        products["triton_transposed"], products["eager"], rtol=1e-4, atol=1e-4
    )

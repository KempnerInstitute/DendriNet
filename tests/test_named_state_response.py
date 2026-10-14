"""Tests for named, matrix-free state-coordinate response operators."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.analysis.utils.named_state_response import (
    InputFeatureCoordinate,
    StateCoordinateSelector,
    leading_input_to_named_state_gain,
    leading_named_state_gain,
    resolve_state_coordinate,
)
from dendritic_modeling.analysis.utils.state_transition import (
    PopulationStateCodec,
    StateTransitionLinearization,
    StateTransitionProduct,
)
from dendritic_modeling.networks.architectures.recurrent import (
    PopulationLayerConfig,
    PopulationNetwork,
    PopulationNetworkConfig,
)


def _tree_network() -> PopulationNetwork:
    torch.manual_seed(31)
    return PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=3,
            output_mode="last",
            layers=[
                PopulationLayerConfig(
                    name="tree_layer",
                    recurrent=True,
                    readout_population="pyr",
                    population_defaults={
                        "ff_excitatory_synapses": 3,
                        "ff_inhibitory_synapses": 0,
                        "rec_excitatory_synapses": 1,
                        "rec_inhibitory_synapses": 0,
                        "somatic_synapses": True,
                        "reactivate": False,
                        "dbl_init_method": "default",
                        "weight_transform": "identity",
                        "topk_type": "standard",
                        "allow_self_recurrence": True,
                        "level_taus": [2.0, 4.0, 8.0],
                    },
                    populations=[
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 1,
                            "branch_factors": [2, 2],
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
    ).double()


def _trajectory() -> tuple[PopulationNetwork, list[StateTransitionLinearization]]:
    network = _tree_network()
    state = network.init_state(1, torch.device("cpu"), torch.float64)
    inputs = (
        torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64),
        torch.tensor([[0.1, 0.4, -0.2]], dtype=torch.float64),
    )
    linearizations = []
    for input_t in inputs:
        linearization = StateTransitionLinearization(network, state, input_t)
        linearizations.append(linearization)
        state = linearization.codec.unflatten(linearization.next_state_vector)
    return network, linearizations


def _dense_operator(
    matvec,
    input_dimension: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    identity = torch.eye(input_dimension, dtype=dtype)
    return torch.stack(
        [matvec(identity[:, index]) for index in range(input_dimension)], 1
    )


def test_level_from_soma_resolves_distal_to_soma_convention() -> None:
    _, linearizations = _trajectory()
    codec = PopulationStateCodec(
        linearizations[0].network,
        linearizations[0].codec.unflatten(linearizations[0].state_vector),
    )
    soma = resolve_state_coordinate(
        codec,
        StateCoordinateSelector(0, "pyr", "trace_E", level_from_soma=0),
    )
    proximal = resolve_state_coordinate(
        codec,
        StateCoordinateSelector(0, "pyr", "trace_E", level_from_soma=1),
    )
    distal = resolve_state_coordinate(
        codec,
        StateCoordinateSelector(0, "pyr", "trace_E", level_from_soma=2),
    )

    assert soma.spec.level_index == 2 and soma.dimension == 1
    assert proximal.spec.level_index == 1 and proximal.dimension == 2
    assert distal.spec.level_index == 0 and distal.dimension == 4


def test_selector_fails_closed_on_ambiguous_or_invalid_levels() -> None:
    _, linearizations = _trajectory()
    codec = linearizations[0].codec
    with pytest.raises(ValueError, match="level-indexed"):
        resolve_state_coordinate(codec, StateCoordinateSelector(0, "pyr", "trace_E"))
    with pytest.raises(ValueError, match="only one"):
        resolve_state_coordinate(
            codec,
            StateCoordinateSelector(
                0, "pyr", "trace_E", level_index=0, level_from_soma=0
            ),
        )
    with pytest.raises(ValueError, match="resolved to 0"):
        resolve_state_coordinate(
            codec,
            StateCoordinateSelector(0, "pyr", "trace_E", level_from_soma=7),
        )
    with pytest.raises(TypeError, match="level_index must be an integer"):
        resolve_state_coordinate(
            codec,
            StateCoordinateSelector(0, "pyr", "trace_E", level_index=True),
        )
    with pytest.raises(ValueError, match="level_index must be non-negative"):
        resolve_state_coordinate(
            codec,
            StateCoordinateSelector(0, "pyr", "trace_E", level_index=-1),
        )


def test_named_state_gain_matches_dense_selected_jacobian() -> None:
    _, linearizations = _trajectory()
    product = StateTransitionProduct(linearizations)
    source_selector = StateCoordinateSelector(
        0, "pyr", "trace_E_rec", level_from_soma=2
    )
    target_selector = StateCoordinateSelector(0, "pyr", "trace_E", level_from_soma=0)
    source = resolve_state_coordinate(linearizations[0].codec, source_selector)
    target = resolve_state_coordinate(linearizations[-1].codec, target_selector)
    dense = _dense_operator(
        lambda vector: target.project(product.jvp(source.embed(vector))),
        source.dimension,
    )
    expected = torch.linalg.svdvals(dense)[0].item()

    result = leading_named_state_gain(
        product,
        source=source_selector,
        target=target_selector,
        block_size=source.dimension,
        max_iterations=1,
        relative_residual_tolerance=1e-10,
        absolute_residual_tolerance=1e-12,
        seed=3,
    )

    assert result.estimate.converged
    assert result.estimate.singular_value == pytest.approx(expected, abs=1e-11)
    assert result.horizon_steps == 2
    assert result.target_metadata["level_index_distal_to_soma"] == 2


def test_input_to_named_state_gain_matches_dense_selected_jacobian() -> None:
    _, linearizations = _trajectory()
    source_selector = InputFeatureCoordinate((0, 2))
    target_selector = StateCoordinateSelector(0, "pyr", "trace_E", level_from_soma=1)
    source = source_selector.resolve(linearizations[0])
    target = resolve_state_coordinate(linearizations[-1].codec, target_selector)

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        propagated = linearizations[0].input_jvp(source.embed(vector))
        propagated = linearizations[1].jvp(propagated)
        return target.project(propagated)

    dense = _dense_operator(matvec, source.dimension)
    expected = torch.linalg.svdvals(dense)[0].item()
    result = leading_input_to_named_state_gain(
        linearizations,
        source=source_selector,
        target=target_selector,
        block_size=source.dimension,
        max_iterations=1,
        relative_residual_tolerance=1e-10,
        absolute_residual_tolerance=1e-12,
        seed=4,
    )

    assert result.estimate.converged
    assert result.estimate.singular_value == pytest.approx(expected, abs=1e-11)
    assert result.source_metadata["feature_indices"] == [0, 2]

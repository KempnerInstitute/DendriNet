"""Regression tests for realized sparse-operator analysis."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_release import (
    IndexedReleaseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.stochastic import (
    StochasticTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import (
    VarianceTopKLinear,
)


def test_snapshot_matches_the_deterministic_topk_forward_operator() -> None:
    layer = TopKLinear(
        in_features=4,
        out_features=2,
        K=2,
        weight_transform="identity",
    )
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor([[1.0, 4.0, 2.0, 3.0], [8.0, 5.0, 7.0, 6.0]]))

    snapshot = effective_synapse_snapshot(layer)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    torch.testing.assert_close(layer(x), x @ snapshot.effective_weight.T)
    torch.testing.assert_close(snapshot.realized_k, torch.tensor([2, 2]))
    assert snapshot.mask_source == "deterministic_current"
    assert torch.count_nonzero(snapshot.candidate_weight) == 8
    assert torch.count_nonzero(snapshot.effective_weight) == 4


def test_large_inactive_candidate_cannot_enter_effective_snapshot() -> None:
    allowed = torch.tensor([[True, True, False]])
    layer = TopKLinear(
        in_features=3,
        out_features=1,
        K=1,
        weight_transform="identity",
        connection_mask=allowed,
    )
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor([[1.0, 2.0, 1_000.0]]))

    snapshot = effective_synapse_snapshot(layer)

    assert snapshot.candidate_weight[0, 2] == 1_000.0
    assert snapshot.candidate_mask[0, 2] == 0
    assert snapshot.active_mask[0, 2] == 0
    assert snapshot.effective_weight[0, 2] == 0
    assert snapshot.effective_weight[0, 1] == 2.0


def test_stochastic_snapshot_requires_and_reuses_realized_forward_mask() -> None:
    layer = StochasticTopKLinear(
        in_features=5,
        out_features=2,
        K=2,
        weight_transform="identity",
        temperature=0.5,
    )
    with pytest.raises(RuntimeError, match="realized forward mask"):
        effective_synapse_snapshot(layer)

    layer.cache_mask = True
    x = torch.ones(1, 5)
    output = layer(x)
    snapshot = effective_synapse_snapshot(layer)

    assert snapshot.mask_source == "cached_forward"
    torch.testing.assert_close(output, x @ snapshot.effective_weight.T)


@pytest.mark.parametrize("transform", ["exp", "softplus"])
@pytest.mark.parametrize("rounding", ["none", "float16", "bfloat16"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_release_snapshot_uses_mean_transmission_without_changing_state(
    transform, rounding, device
):
    layer = (
        IndexedReleaseLinear(
            4,
            2,
            2,
            seed=7,
            weight_transform=transform,
            release_p_min=0.2,
            release_p_max=0.9,
            release_weight_scale=0.3,
            release_max_conductance=0.8,
            signal_rounding=rounding,
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor([[-2.0, 0.7], [-1.0, 0.1]]))
    before = {k: v.clone() for k, v in layer.state_dict().items()}
    rng = torch.get_rng_state().clone()
    snapshot = effective_synapse_snapshot(layer)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(value, layer.state_dict()[key]) for key, value in before.items()
    )
    weights = (
        layer.pre_w.exp()
        if transform == "exp"
        else torch.nn.functional.softplus(layer.pre_w)
    ).clamp(max=0.8)

    def rounded(value):
        return (
            value
            if rounding == "none"
            else value.to(getattr(torch, rounding)).to(value.dtype)
        )

    weights = rounded(weights)
    probabilities = rounded(0.2 + 0.7 * weights / (weights + 0.075))
    expected = torch.zeros(2, 4, device=device).scatter_add(
        1, layer.connection_indices.long(), weights * probabilities
    )
    torch.testing.assert_close(snapshot.effective_weight, expected)
    assert snapshot.transmission_mode == "mean_release"
    assert snapshot.signal_rounding == rounding
    assert snapshot.realized_k.tolist() == [2, 2]
    x = torch.tensor([[0.1, 0.5, 0.9, 1.3], [1.1, 0.3, 0.7, 1.7]], device=device)
    # Matrix multiplication can change reduction order; rounding is outside
    # the matrix and must be replayed when comparing the actual forward.
    expected_output = rounded(
        (rounded(x)[:, layer.connection_indices.long()] * weights * probabilities).sum(
            -1
        )
    )
    torch.testing.assert_close(layer(x), expected_output, rtol=0, atol=0)
    if rounding == "none":
        torch.testing.assert_close(layer(x), x @ snapshot.effective_weight.T)


def test_release_training_realizations_are_not_mislabelled_as_one_matrix():
    layer = IndexedReleaseLinear(4, 2, 2)
    with pytest.raises(RuntimeError, match="require eval mode"):
        effective_synapse_snapshot(layer)


@pytest.mark.parametrize("layer_class", [IndexedSparseLinear, IndexedReleaseLinear])
def test_indexed_snapshot_respects_the_eligible_source_pool(layer_class):
    allowed = torch.tensor([[True, True, True, False], [False, True, True, True]])
    layer = layer_class(
        4,
        2,
        1,
        seed=2,
        connection_mask=allowed,
        forbidden_input_index_per_output=torch.tensor([0, 3]),
    ).eval()
    snapshot = effective_synapse_snapshot(layer)
    expected = allowed.clone()
    expected[0, 0] = expected[1, 3] = False
    assert torch.equal(snapshot.candidate_mask.bool(), expected)
    assert not (snapshot.active_mask.bool() & ~expected).any()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_variance_snapshot_uses_frozen_training_statistics(device):
    layer = VarianceTopKLinear(
        3, 1, 1, weight_transform="identity", noise_level=0.1
    ).to(device)
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor([[5.0, 1.0, 0.1]]))
    layer.update_stats(torch.tensor([[1.0, 0.0, 0.0], [1.0, 8.0, 0.0]], device=device))
    layer.eval()
    before = {k: v.clone() for k, v in layer.state_dict().items()}
    x = torch.tensor([[9.0, 2.0, 8.0]], device=device)
    snapshot = effective_synapse_snapshot(layer, prefer_cached_forward_mask=False)
    assert snapshot.active_mask.tolist() == [[0.0, 1.0, 0.0]]
    torch.testing.assert_close(layer(x), x @ snapshot.effective_weight.T)
    assert all(
        torch.equal(value, layer.state_dict()[key]) for key, value in before.items()
    )


def test_stochastic_snapshot_in_eval_mode_is_the_deterministic_topk() -> None:
    layer = StochasticTopKLinear(
        in_features=5,
        out_features=2,
        K=2,
        weight_transform="identity",
        temperature=0.5,
    ).eval()
    snapshot = effective_synapse_snapshot(layer)
    torch.testing.assert_close(snapshot.active_mask, TopKLinear.weight_mask(layer))
    assert snapshot.mask_source == "deterministic_current"


def test_resource_summary_counts_stochastic_selectors_in_eval_mode() -> None:
    from test_biological_constraints import _einet

    from dendritic_modeling.analysis.utils.model_resources import model_resource_summary
    from dendritic_modeling.networks.architectures.factory import get_architecture

    core = _einet()
    core["sparsity"]["type"] = "stochastic"
    torch.manual_seed(0)
    network = get_architecture("einet", core, input_dim=8).train()

    summary = model_resource_summary(network)

    assert summary["active_synapses"] > 0
    assert network.training and all(module.training for module in network.modules())

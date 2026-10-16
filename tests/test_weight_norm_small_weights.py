"""Weight normalization stays exact for norms below machine epsilon.

The norm floor only keeps the unselected branch of ``torch.where`` finite, so
a row of small positive weights still normalizes to ``gamma``; rows whose norm
is at least machine epsilon are normalized exactly as before.
"""

import pytest
import torch

from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_dynamic import (
    IndexedDynamicTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)

GAMMA = 2.0
KINDS = ["indexed", "topk", "dynamic"]


def _layer(kind, dtype=torch.float32):
    torch.manual_seed(0)
    options = {"weight_norm_order": 1, "gamma": GAMMA}
    if kind == "indexed":
        layer = IndexedSparseLinear(6, 3, 2, weight_transform="softplus", **options)
    elif kind == "topk":
        layer = TopKLinear(6, 3, 2, weight_transform="exp", **options)
    else:
        layer = IndexedDynamicTopKLinear(6, 3, 2, weight_transform="exp", **options)
    return layer.to(dtype)


def _normalized(layer):
    weights = [
        layer.pruned_weight(),
        effective_synapse_snapshot(layer).effective_weight,
    ]
    for name in ("_normalized_sparse_weight", "dense_weight"):
        if hasattr(layer, name):
            weights.append(getattr(layer, name)())
    return weights


def _former(weight):
    norm = torch.norm(weight, p=1, dim=-1, keepdim=True)
    floor = norm.clamp_min(torch.finfo(weight.dtype).eps)
    return torch.where(norm > 0, weight / floor * GAMMA, weight)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rows_below_machine_epsilon_still_normalize_to_gamma(kind, dtype):
    layer = _layer(kind, dtype)
    with torch.no_grad():
        layer.pre_w.fill_(-40.0)  # transformed weights near 4e-18
    expected = torch.full((3,), GAMMA, dtype=dtype)
    for weight in _normalized(layer):
        torch.testing.assert_close(weight.sum(-1), expected)


@pytest.mark.parametrize("kind", KINDS)
def test_rows_above_machine_epsilon_are_normalized_as_before(kind):
    layer = _layer(kind)
    if kind == "topk":
        weight = layer.weight()
        assert torch.equal(layer._apply_weight_norm(weight), _former(weight))
    else:
        weight = layer.sparse_weight()
        if kind == "dynamic":
            weight = weight * layer.active_sparse_mask()
        assert torch.equal(layer._normalized_sparse_weight(), _former(weight))


def test_zero_rows_stay_zero_with_finite_gradients():
    layer = _layer("indexed")
    with torch.no_grad():
        layer.pre_w.fill_(-200.0)  # softplus underflows to exactly zero
    weight = layer._normalized_sparse_weight()
    assert torch.count_nonzero(weight) == 0
    weight.sum().backward()
    assert torch.isfinite(layer.pre_w.grad).all()

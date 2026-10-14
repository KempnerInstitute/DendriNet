"""Qualify existing sparse backends with positive conductance initialization."""

import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import (
    VarianceTopKLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


@pytest.mark.parametrize("family", ["einet", "population_network"])
@pytest.mark.parametrize(
    "activation", ["endpoint_tanh", "param_tanh", "tanh", "relu", "sigmoid"]
)
@pytest.mark.parametrize(
    "backend", ["standard", "variance", "indexed", "indexed_release"]
)
def test_native_builders_preserve_sparse_semantics_and_checkpoint(
    family, backend, activation
):
    config = _config(family, "shunting", activation)
    options = (
        config["morphology"]
        if family == "einet"
        else config["population_network"]["layers"][0]["population_defaults"]
    )
    options.update(
        weight_transform="exp",
        dbl_init_method="fan_in_lognormal",
        leak_conductance=0.03,
        learnable_leak=True,
    )
    if family == "einet":
        config["sparsity"] = {"type": backend}
    else:
        options["topk_type"] = backend
    torch.manual_seed(16)
    model = get_architecture(family, config, input_dim=6).eval()
    x = torch.rand(4, 6, requires_grad=True)
    y = model(x)
    y.square().sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    banks = [
        m for m in model.modules() if isinstance(m, (TopKLinear, IndexedSparseLinear))
    ]
    assert banks
    for bank in banks:
        if backend in {"standard", "variance"}:
            assert type(bank) is (
                TopKLinear if backend == "standard" else VarianceTopKLinear
            )
            mask = bank.weight_mask().bool()
            assert mask.sum(1).eq(bank.K).all()
            assert bank.pre_w.grad[~mask].eq(0).all()
        else:
            assert bank.pre_w.shape == (bank.out_features, bank.K)
    replay = get_architecture(family, config, input_dim=6).eval()
    replay.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(replay(x), y, rtol=0, atol=0)

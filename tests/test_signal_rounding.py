"""Opt-in reduced-precision rounding of dendritic signals and conductances."""

from copy import deepcopy

import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_release import (
    IndexedReleaseLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.utils.signal_rounding import (
    round_signal,
    signal_rounding_dtype,
)

RELEASE = {
    "weight_transform": "exp",
    "dbl_init_method": "fan_in_lognormal",
    "leak_conductance": 0.03,
    "learnable_leak": True,
}


def _model(family, rounding=None, seed=0):
    config = _config(family, "shunting", "endpoint_tanh")
    if family == "population_network":
        options = config["population_network"]["layers"][0]["population_defaults"]
        options.update(RELEASE, topk_type="indexed_release")
    else:
        options = config["morphology"]
        options.update(RELEASE)
        config["sparsity"] = {"type": "indexed_release"}
    if rounding is not None:
        options["signal_rounding"] = rounding
    torch.manual_seed(seed)
    return get_architecture(family, deepcopy(config), input_dim=6)


def test_names_and_unsupported_combinations_are_rejected():
    assert signal_rounding_dtype("none") is None
    assert signal_rounding_dtype("float16") is torch.float16
    assert signal_rounding_dtype("BFloat16") is torch.bfloat16
    with pytest.raises(ValueError, match="signal_rounding"):
        signal_rounding_dtype("fp16")
    with pytest.raises(ValueError, match="signal_rounding"):
        PopulationConfig(signal_rounding="fp16")
    branch = {
        "output_dim": 2,
        "excitatory_input_dim": 4,
        "excitatory_synapses_per_branch": 2,
        "signal_rounding": "float16",
    }
    with pytest.raises(ValueError, match="indexed_release"):
        DendriticBranchLayer(**branch)
    with pytest.raises(ValueError, match="signal_rounding"):
        DendriticBranchLayer(**branch, topk_type="indexed_release", use_shunting=False)


def test_gradients_through_a_rounded_value_are_rounded():
    x = torch.tensor([1e-9, 1.0], requires_grad=True)
    y = round_signal(x, torch.float16)
    assert y.tolist() == [0.0, 1.0]
    y.backward(torch.tensor([1e-9, 3.0001]))
    assert x.grad.tolist() == [0.0, 3.0]
    exact = torch.tensor(0.03, dtype=torch.float64)
    assert round_signal(0.03, torch.float16) == float(exact.half())
    assert round_signal(x, None) is x


def test_default_keeps_state_output_and_random_stream():
    a = _model("population_network")
    state = torch.get_rng_state()
    b = _model("population_network", rounding="none")
    assert torch.equal(state, torch.get_rng_state())
    assert list(a.state_dict()) == list(b.state_dict())
    x = torch.rand(4, 6)
    a.eval()
    b.eval()
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)


@pytest.mark.parametrize("family", ["population_network", "einet"])
@pytest.mark.parametrize("rounding", ["float16", "bfloat16"])
def test_rounding_reaches_every_module_and_bounds_every_signal(family, rounding):
    dtype = signal_rounding_dtype(rounding)
    model = _model(family, rounding)
    branches = [m for m in model.modules() if isinstance(m, DendriticBranchLayer)]
    synapses = [m for m in model.modules() if isinstance(m, IndexedReleaseLinear)]
    couplings = [m for m in model.modules() if isinstance(m, BlockLinear)]
    assert branches and synapses and couplings
    assert all(m._signal_round_dtype is dtype for m in branches)
    assert all(m._round_dtype is dtype for m in synapses + couplings)

    outputs = []
    for module in branches + synapses:
        module.register_forward_hook(lambda m, i, o: outputs.append(o))
    x = torch.rand(5, 6)
    model.eval()
    output = model(x)
    assert len(outputs) == len(branches) + len(synapses)
    for value in outputs:
        assert torch.equal(value, round_signal(value, dtype))
    for module in branches:
        assert torch.equal(
            module.effective_leak, round_signal(module.effective_leak, dtype)
        )
    for module in couplings:
        assert torch.equal(module.weight(), round_signal(module.weight(), dtype))

    reference = _model(family)
    assert list(reference.state_dict()) == list(model.state_dict())
    reference.load_state_dict(model.state_dict(), strict=True)
    reference.eval()
    expected = reference(x)
    assert not torch.equal(output, expected)
    torch.testing.assert_close(output, expected, rtol=0, atol=0.05)

    model.train()
    model(x).square().sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

"""Recurrent unrolls read their nonnegative-input checks once per sequence."""

import dataclasses

import pytest
import torch
from test_unified_biological_input import recipe as unified_recipe

from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.config.biological_adoption import prepare_biological_config
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.utils import input_domain
from dendritic_modeling.networks.utils.input_domain import (
    deferred_nonnegative_checks,
    require_nonnegative_input,
)

STEPS = 6


@pytest.fixture
def reads(monkeypatch):
    """Record each host read of check flags, as the number of flags read."""
    calls = []
    read = input_domain._raise_if_invalid

    def counted(flags):
        calls.append(len(flags))
        read(flags)

    monkeypatch.setattr(input_domain, "_raise_if_invalid", counted)
    return calls


class _Immediate:
    """Thread state that never holds a scope: every check is read at once."""

    flags = property(lambda self: None, lambda self, value: None)


def _population(input_transform, projection_dims=(6, 5)):
    defaults = {
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 0,
        "rec_excitatory_synapses": 1,
        "rec_inhibitory_synapses": 1,
        "weight_transform": "softplus",
        "use_shunting": True,
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "dbl_init_method": "default",
    }

    def connection(source, target, pathway, timing="same_step"):
        return {
            "source": source,
            "target": target,
            "pathway": pathway,
            "timing": timing,
        }

    core = {
        "type": "population_network",
        "population_network": {
            "input_transform": input_transform,
            "input_projection_dims": list(projection_dims),
            "input_projection_weight_transform": "softplus",
            "output_mode": "all",
            "layers": [
                {
                    "name": "layer0",
                    "recurrent": True,
                    "readout_population": "e",
                    "population_defaults": defaults,
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [2],
                        },
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                        },
                    ],
                    "connections": [
                        connection("input", "e", "ff_excitatory"),
                        connection("input", "i", "ff_excitatory"),
                        connection("e", "e", "rec_excitatory", "delayed"),
                        connection("e", "i", "rec_excitatory", "delayed"),
                        connection("i", "e", "rec_inhibitory", "delayed"),
                    ],
                }
            ],
        },
    }
    return config_from_mapping({"model": {"core": core}}).model.core


def _unified():
    data = prepare_biological_config(
        unified_recipe(True),
        input_activation="signed_split",
        positive_input_projection=True,
    )
    data["model"]["core"]["unified_ei"]["input_projection_dims"] = [5, 4]
    return config_from_mapping(data).model.core


# Each family: its typed core, input width and the checks read per step (the
# checked input of a population network and each positive projection).
FAMILIES = {
    "population_checked_input": (lambda: _population("nonnegative"), 4, 3),
    "population_signed_split": (lambda: _population("signed_split"), 4, 2),
    "unified_signed_split": (_unified, 3, 2),
}


def _build(family, dtype=torch.float64):
    core, input_dim, _ = FAMILIES[family]
    core = core()
    torch.manual_seed(0)
    model = get_architecture(core.type, core, input_dim=input_dim)
    return model.to(dtype)


def _sequence(family, dtype=torch.float64):
    _, input_dim, _ = FAMILIES[family]
    torch.manual_seed(1)
    x = torch.rand(2, STEPS, input_dim, dtype=dtype)
    if "signed_split" in family:
        x = x - 0.5  # negative raw rates are valid before an ON/OFF split
    x[:, 2, 0] = 0.0  # the identity derivative at zero is part of the contract
    return x


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _tensors(getattr(value, field.name))
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from _tensors(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def _run(model, x):
    x = x.clone().requires_grad_()
    out, hidden = model(x, return_hidden=True)
    states = list(_tensors(hidden))
    loss = out.sum() + sum(state.sum() for state in states if state.requires_grad)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, [x, *params], allow_unused=True)
    return [out, *states, *(g for g in grads if g is not None)]


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_unroll_reads_every_step_check_once(family, reads):
    model = _build(family)
    model(_sequence(family))
    assert reads == [FAMILIES[family][2] * STEPS]


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_deferred_reads_keep_outputs_states_and_gradients(family, reads, monkeypatch):
    model = _build(family)
    x = _sequence(family)
    deferred = _run(model, x)
    assert reads == [FAMILIES[family][2] * STEPS]
    monkeypatch.setattr(input_domain, "_pending", _Immediate())
    immediate = _run(model, x)
    assert reads[1:] == [1] * (FAMILIES[family][2] * STEPS)
    assert len(deferred) == len(immediate) > 3
    for new, old in zip(deferred, immediate):
        assert torch.equal(new, old)


@pytest.mark.parametrize("bad", [-0.25, float("nan"), float("inf")])
@pytest.mark.parametrize("step", [0, 3, STEPS - 1])
def test_invalid_checked_input_at_any_step_raises_before_returning(bad, step):
    model = _build("population_checked_input")
    x = _sequence("population_checked_input")
    x[1, step, 2] = bad
    with pytest.raises(ValueError, match="nonnegative input"):
        model(x)
    with pytest.raises(ValueError, match="nonnegative input"):
        require_nonnegative_input(torch.tensor([bad]))  # the scope was closed


@pytest.mark.parametrize("family", ["population_signed_split", "unified_signed_split"])
def test_on_off_encoding_accepts_negative_raw_rates(family):
    model = _build(family)
    x = _sequence(family)
    assert (x < 0).any()
    assert torch.isfinite(model(x)).all()


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_each_projection_still_checks_its_own_input(family):
    model = _build(family)
    with torch.no_grad():
        model.input_projection[0].bias[1] = float("nan")
    with pytest.raises(ValueError, match="nonnegative input"):
        model(_sequence(family))


def test_scope_reads_pending_checks_once_at_its_end(reads):
    with pytest.raises(ValueError, match="nonnegative input"):
        with deferred_nonnegative_checks():
            require_nonnegative_input(torch.tensor([-1.0]))
            require_nonnegative_input(torch.tensor([1.0]))
            assert reads == []
    assert reads == [2]
    with pytest.raises(ValueError, match="nonnegative input"):
        require_nonnegative_input(torch.tensor([-1.0]))
    assert reads == [2, 1]


def test_nested_scope_joins_the_enclosing_scope(reads):
    with pytest.raises(ValueError, match="nonnegative input"):
        with deferred_nonnegative_checks():
            with deferred_nonnegative_checks():
                require_nonnegative_input(torch.tensor([-1.0]))
            assert reads == []
            require_nonnegative_input(torch.tensor([2.0]))
    assert reads == [2]


def test_error_inside_scope_discards_pending_checks(reads):
    with pytest.raises(RuntimeError, match="inner"):
        with deferred_nonnegative_checks():
            require_nonnegative_input(torch.tensor([-1.0]))
            raise RuntimeError("inner")
    assert reads == []
    with pytest.raises(ValueError, match="nonnegative input"):
        require_nonnegative_input(torch.tensor([-1.0]))


def test_valid_scope_reads_once_and_returns_inputs_unchanged(reads):
    x = torch.tensor([0.0, 0.5, 3.0], requires_grad=True)
    with deferred_nonnegative_checks():
        y = require_nonnegative_input(x)
    assert y is x and reads == [1]
    with deferred_nonnegative_checks():
        pass
    assert reads == [1]


@pytest.mark.requires_gpu
def test_unroll_reads_cuda_checks_once_per_device(reads):
    model = _build("population_checked_input", torch.float32).cuda()
    model(_sequence("population_checked_input", torch.float32).cuda())
    assert reads == [FAMILIES["population_checked_input"][2] * STEPS]

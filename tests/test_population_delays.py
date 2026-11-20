"""Travel times affect which source sample each existing synapse receives."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import yaml

from dendritic_modeling.analysis.utils.state_transition import PopulationStateCodec
from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.examples import example_text
from dendritic_modeling.networks.architectures.recurrent import (
    PopulationNetwork,
    PopulationNetworkConfig,
)


def _population(name, polarity="excitatory", factors=()):
    return {
        "name": name,
        "polarity": polarity,
        "n_neurons": 1,
        "branch_factors": list(factors),
    }


def _network(delays=True):
    defaults = {
        "ff_excitatory_synapses": 1,
        "ff_inhibitory_synapses": 0,
        "rec_excitatory_synapses": 0,
        "rec_inhibitory_synapses": 0,
        "somatic_synapses": True,
        "reactivate": False,
        "weight_transform": "softplus",
        "use_shunting": True,
        "tau_base": 2.0,
        "tau_ratio": 2.0,
    }
    upstream = {
        "name": "up",
        "readout_population": "e",
        "population_defaults": defaults,
        "populations": [
            _population("e"),
            _population("other"),
            _population("i", "inhibitory"),
        ],
        "connections": [
            {"source": "input", "target": target} for target in ("e", "other", "i")
        ],
    }
    downstream = {
        "name": "down",
        "recurrent": True,
        "population_defaults": {
            **defaults,
            "rec_excitatory_synapses": 2,
            "rec_inhibitory_synapses": 1,
        },
        "populations": [_population("pyr", factors=(1, 1))],
        "connections": [
            {"source": "input", "target": "pyr"},
            {"source": "up.e", "target": "pyr", "timing": "delayed"},
            {"source": "up.other", "target": "pyr", "timing": "delayed"},
            {"source": "up.i", "target": "pyr", "timing": "delayed"},
        ],
    }
    if delays:
        downstream["connections"][1]["delay_steps_by_level"] = [3, 2, 1]
        downstream["connections"][2]["delay_steps"] = 2
        downstream["connections"][3]["delay_steps_by_level"] = [1, 2, 3]
    torch.manual_seed(17)
    return PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=1, output_mode="all", layers=[upstream, downstream]
        )
    ).double()


def test_pulse_arrives_at_each_depth_and_source_at_its_configured_update():
    net = _network()
    x = torch.tensor(
        [[[1.0], [0.0], [0.0], [0.0], [0.0]]], dtype=torch.float64, requires_grad=True
    )
    hidden = None
    sources = []
    arrivals = []

    def sample(source, step, age):
        return sources[step - age][source] if step >= age else x.new_zeros((1, 1))

    with ExitStack() as resources:
        reads = [
            resources.enter_context(
                patch.object(
                    layer, "compute_raw_currents", wraps=layer.compute_raw_currents
                )
            )
            for layer in net.layers[1].populations["pyr"].branch_layers
        ]
        for t in range(x.shape[1]):
            _, hidden = net(x[:, t : t + 1], hidden, return_hidden=True)
            sources.append(hidden[0].outputs)
            arrivals.append(reads[0].call_args.kwargs["recurrent_input"][:, 0])
            for idx, read in enumerate(reads):
                exc = torch.cat(
                    [sample("e", t, (3, 2, 1)[idx]), sample("other", t, 2)], dim=-1
                )
                inh = sample("i", t, (1, 2, 3)[idx])
                torch.testing.assert_close(
                    read.call_args.kwargs["recurrent_input"], exc, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    read.call_args.kwargs["rec_inhibitory_input"], inh, rtol=0, atol=0
                )
    assert arrivals[3].item() > 0 and all(v.item() == 0 for v in arrivals[:3])
    grad = torch.autograd.grad(arrivals[3].sum(), x)[0]
    assert grad[0, 0, 0] != 0 and grad[0, 1:].count_nonzero() == 0


def test_history_carries_across_chunks_and_state_codec():
    net = _network()
    x = torch.rand(2, 7, 1, dtype=torch.float64)
    full = net(x)
    start, hidden = net(x[:, :2], return_hidden=True)
    assert hidden[0].delay_history is None
    histories = hidden[1].delay_history
    assert {k: v.shape[1] for k, v in histories.items()} == {
        "up.e": 2,
        "up.other": 1,
        "up.i": 2,
    }
    clone = hidden[1].clone()
    assert all(
        clone.delay_history[k].data_ptr() != v.data_ptr() for k, v in histories.items()
    )
    assert all(not v.requires_grad for v in hidden[1].detach().delay_history.values())
    codec = PopulationStateCodec(net, hidden)
    for state in (
        hidden,
        codec.unflatten(codec.flatten(hidden)),
        [hidden[0], clone.to(x.device)],
    ):
        tail = net(x[:, 2:], state)
        torch.testing.assert_close(
            torch.cat([start, tail], dim=1), full, rtol=0, atol=0
        )


@pytest.mark.parametrize("source", ["e", "down.pyr"])
def test_local_recurrence_and_feedback_from_a_later_layer(source):
    config = _network(delays=False).config
    first = config.layers[0]
    first["recurrent"] = True
    first["populations"][0]["population"] = {"rec_excitatory_synapses": 1}
    first["connections"].append(
        {"source": source, "target": "e", "timing": "delayed", "delay_steps": 3}
    )
    net = PopulationNetwork(config).double()
    x = torch.rand(1, 5, 1, dtype=torch.float64)
    states = []
    hidden = None
    branch = net.layers[0].populations["e"].branch_layers[0]
    with patch.object(
        branch, "compute_raw_currents", wraps=branch.compute_raw_currents
    ) as read:
        for t in range(5):
            _, hidden = net(x[:, t : t + 1], hidden, return_hidden=True)
            states.append(
                hidden[0].outputs["e"] if source == "e" else hidden[1].outputs["pyr"]
            )
            expected = states[t - 3] if t >= 3 else torch.zeros_like(states[0])
            torch.testing.assert_close(
                read.call_args.kwargs["recurrent_input"], expected, rtol=0, atol=0
            )
    assert states[0].abs().sum() > 0


def test_masked_depths_and_disabled_routes_use_no_unnecessary_history():
    config = _network(delays=False).config
    target = config.layers[1]
    target["connections"][1].update(
        target_levels=["soma"], delay_steps_by_level=[300, 200, 1]
    )
    target["connections"][2].update(delay_steps=42, enabled=False)
    target["population_defaults"]["rec_excitatory_synapses_by_level"] = [0, 0, 1]
    net = PopulationNetwork(config)
    assert not net.layers[1].delay_history_lengths


def test_checkpoint_reload_and_compiled_layer_preserve_travel_delays(tmp_path):
    net = _network()
    path = tmp_path / "weights.pt"
    torch.save(net.state_dict(), path)
    restored = _network()
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.layers[1].compile(backend="eager")
    x = torch.rand(1, 5, 1, dtype=torch.float64)
    results = []
    for model in (net, restored):
        inp = x.clone().requires_grad_()
        out = model(inp)
        results.append((out, torch.autograd.grad(out.sum(), inp)[0]))
    for actual, expected in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_explicit_one_step_preserves_parameters_outputs_and_gradients():
    baseline = _network(delays=False)
    configs = baseline.layer_configs
    configs[1].connections[1]["delay_steps_by_level"] = [1, 1, 1]
    configs[1].connections[2]["delay_steps"] = 1
    torch.manual_seed(17)
    explicit = PopulationNetwork(
        PopulationNetworkConfig(input_dim=1, output_mode="all", layers=configs)
    ).double()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(explicit.state_dict()[key], value, rtol=0, atol=0)
    x = torch.rand(2, 5, 1, dtype=torch.float64)
    observed = []
    for net in (baseline, explicit):
        inp = x.clone().requires_grad_()
        y, hidden = net(inp, return_hidden=True)
        assert all(state.delay_history is None for state in hidden)
        y.sum().backward()
        observed.append(
            [y, inp.grad, *(p.grad for p in net.parameters() if p.grad is not None)]
        )
    for first, second in zip(*observed, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options",
    [
        {"delay_steps": 0},
        {"delay_steps": -1},
        {"delay_steps": True},
        {"delay_steps": 1.5},
        {"delay_steps_by_level": "321"},
        {"delay_steps_by_level": []},
        {"delay_steps_by_level": [1, 2]},
        {"delay_steps_by_level": [3, False, 1]},
        {"delay_steps": 2, "delay_steps_by_level": [3, 2, 1]},
        {"timing": "same_step", "delay_steps": 1},
    ],
)
def test_invalid_delay_is_rejected_by_the_loader(options):
    raw = yaml.safe_load(example_text("recurrent_decision_population"))
    route = raw["model"]["core"]["population_network"]["layers"][0]["connections"][-2]
    route.update(options)
    with pytest.raises(ValueError, match="delay_steps"):
        config_from_mapping(raw)

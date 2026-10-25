"""Per-level DendriNet overrides, explicit re-initialization and level outputs."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
    _reject_structured_contacts,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet_config import (
    DendriNetConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.level_interventions import (
    CurrentInjection,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.level_overrides import (
    DendriticLevelOverride,
    contacts_mask,
    declare_contacts,
    remove_synaptic_pathway,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.reinitialization import (
    LogNormalReinitialization,
    LogOffset,
)

SOMA, STREAMS = 3, 2
LEAVES = 2 * STREAMS
DIM = 2 * LEAVES + STREAMS
BASE = {
    "n_soma": SOMA,
    "branch_factors": [STREAMS, 2],
    "excitatory_input_dim": DIM,
    "excitatory_synapses_per_branch": DIM,
    "inhibitory_input_dim": DIM,
    "inhibitory_synapses_per_branch": DIM,
    "reactivate": False,
    "somatic_synapses": False,
    "use_shunting": True,
    "weight_transform": "exp",
    "use_noise": False,
    "topk_noise_level": 0.0,
    "topk_strategy": "none",
    "blocklinear_strategy": "none",
}
# Terminal compartments see one feature pair (and, excitatory, every cue);
# each proximal compartment's inhibition sees one cue.
TERMINAL_E = [[f, f + LEAVES, *range(2 * LEAVES, DIM)] for f in range(LEAVES)]
TERMINAL_I = [[f, f + LEAVES] for f in range(LEAVES)]
PROXIMAL_I = [[2 * LEAVES + r] for r in range(STREAMS)]
RECIPE = LogNormalReinitialization(
    synapse_log_sd=0.4,
    coupling_log_sd=0.25,
    log_offsets=(LogOffset(1, "inhibitory", math.log(3.0)),),
)


def _overrides(tanh=True):
    proximal = {"depth": 1, "excitatory": False, "inhibitory_contacts": PROXIMAL_I}
    if tanh:
        proximal["activation"] = "tanh"
    return [
        {
            "depth": 2,
            "excitatory_contacts": TERMINAL_E,
            "inhibitory_contacts": TERMINAL_I,
        },
        proximal,
    ]


def _mutated(seed, tanh=True):
    """The same structure made by changing a default DendriNet after construction."""
    torch.manual_seed(seed)
    net = DendriNet(**BASE)
    terminal, proximal, _ = net.branch_layers
    for layer in net.branch_layers:
        layer.epsilon = 0.0
    for synapse, contacts in (
        (terminal.branch_excitation, TERMINAL_E),
        (terminal.branch_inhibition, TERMINAL_I),
        (proximal.branch_inhibition, PROXIMAL_I),
    ):
        mask = torch.zeros_like(synapse.pre_w, dtype=torch.bool)
        for row in range(mask.shape[0]):
            mask[row, contacts[row % len(contacts)]] = True
        synapse.connection_mask = mask
    proximal.branch_excitation = None
    proximal.input_excitatory = False
    if tanh:
        proximal.reactivation = nn.Tanh()
    return net, torch.get_rng_state()


def _typed(seed, tanh=True, **kwargs):
    torch.manual_seed(seed)
    net = DendriNet(**BASE, epsilon=0.0, level_overrides=_overrides(tanh), **kwargs)
    return net, torch.get_rng_state()


def _inputs(seed=0, batch=5):
    generator = torch.Generator().manual_seed(seed)
    excitation = 2 * torch.rand(batch, DIM, generator=generator, dtype=torch.float64)
    inhibition = 2 * torch.rand(batch, DIM, generator=generator, dtype=torch.float64)
    return excitation, inhibition


def _same_state(left, right):
    a, b = left.state_dict(), right.state_dict()
    return list(a) == list(b) and all(torch.equal(a[key], b[key]) for key in a)


@pytest.mark.parametrize("tanh", [False, True])
def test_typed_levels_equal_the_mutated_reference(tanh):
    reference, reference_rng = _mutated(11, tanh)
    typed, typed_rng = _typed(11, tanh)
    assert torch.equal(reference_rng, typed_rng)
    assert [name for name, _ in reference.named_parameters()] == [
        name for name, _ in typed.named_parameters()
    ]
    assert _same_state(reference, typed)
    assert not any("connection_mask" in key for key in typed.state_dict())
    for (_, a), (_, b) in zip(reference.named_modules(), typed.named_modules()):
        if hasattr(a, "connection_mask"):
            assert torch.equal(a.connection_mask, b.connection_mask)
    assert [layer.epsilon for layer in typed.branch_layers] == [0.0, 0.0, 0.0]
    expected = ["Identity", "Tanh" if tanh else "Identity", "Identity"]
    assert [type(layer.reactivation).__name__ for layer in typed.branch_layers] == (
        expected
    )
    reference, typed = reference.double(), typed.double()
    excitation, inhibition = _inputs()
    assert torch.equal(reference(excitation, inhibition), typed(excitation, inhibition))


def test_default_and_identity_overrides_change_nothing():
    torch.manual_seed(5)
    plain = DendriNet(**BASE)
    plain_rng = torch.get_rng_state()
    for overrides in ((), [{"depth": 1}], [DendriticLevelOverride(depth=0)]):
        torch.manual_seed(5)
        other = DendriNet(**BASE, level_overrides=overrides)
        assert torch.equal(torch.get_rng_state(), plain_rng)
        assert _same_state(plain, other)
        excitation, inhibition = _inputs()
        assert torch.equal(
            plain.double()(excitation, inhibition),
            other.double()(excitation, inhibition),
        )
        plain = plain.float()


def test_removed_pathway_leaves_a_level_without_it():
    net, _ = _typed(3)
    proximal = net.branch_layers[1]
    assert proximal.branch_excitation is None and not proximal.input_excitatory
    assert not hasattr(proximal, "excitatory_input_dim")
    assert proximal.branch_config.excitatory_input_dim is None
    assert proximal.branch_config.excitatory_synapses_per_branch is None
    assert net.branch_configs[1] is proximal.branch_config
    assert not any(
        key.startswith("branch_layers.1.branch_excitation") for key in net.state_dict()
    )
    # Shunting without excitation: V = C / (1 + G + I), then the level's tanh.
    net = net.double()
    for layer in net.branch_layers:
        layer.set_branch_diagnostics(True)
    excitation, inhibition = _inputs()
    outputs = net.forward_levels(excitation, inhibition)
    child = proximal.branches_to_output(outputs[0])
    denominator = (
        1
        + proximal.branches_to_output.sum_conductances()
        + proximal.branch_inhibition(inhibition)
    )
    assert torch.allclose(
        outputs[1], torch.tanh(child / denominator), rtol=0, atol=1e-15
    )
    with pytest.raises(ValueError, match="no excitatory pathway"):
        remove_synaptic_pathway(proximal, "excitatory")
    with pytest.raises(ValueError, match="Unknown pathway"):
        remove_synaptic_pathway(proximal, "recurrent")


def test_contacts_given_per_neuron_or_per_compartment_agree():
    template = contacts_mask(
        TERMINAL_I, rows=SOMA * LEAVES, in_features=DIM, n_soma=SOMA, name="c"
    )
    full = contacts_mask(
        TERMINAL_I * SOMA, rows=SOMA * LEAVES, in_features=DIM, n_soma=SOMA, name="c"
    )
    assert torch.equal(template, full)
    assert template.sum().item() == 2 * SOMA * LEAVES
    with pytest.raises(ValueError, match="lists 3 compartments"):
        contacts_mask(
            TERMINAL_I[:3], rows=SOMA * LEAVES, in_features=DIM, n_soma=SOMA, name="c"
        )
    with pytest.raises(ValueError, match="outside"):
        contacts_mask(
            [[DIM]] * LEAVES, rows=SOMA * LEAVES, in_features=DIM, n_soma=SOMA, name="c"
        )


@pytest.mark.parametrize(
    "override, error",
    [
        ({"depth": -1}, "depth must be >= 0"),
        ({"depth": 1.0}, "depth must be an integer"),
        ({"depth": 1, "excitatory": "no"}, "true or false"),
        ({"depth": 1, "activation": "cubic_spline"}, "Unknown level activation"),
        ({"depth": 1, "inhibitory_contacts": [[0, 0]]}, "repeats an input"),
        ({"depth": 1, "inhibitory_contacts": [[]]}, "at least one contact"),
        ({"depth": 1, "inhibitory_contacts": [[0.5]]}, "integer input indices"),
        ({"depth": 1, "inhibitory_contacts": [0, 1]}, "list of input-index lists"),
        (
            {"depth": 1, "inhibitory": False, "inhibitory_contacts": [[0]]},
            "also declares",
        ),
        ({"depth": 1, "synapses": 3}, "Unknown level override keys"),
        ({"excitatory": False}, "needs a depth"),
    ],
)
def test_invalid_overrides_are_rejected(override, error):
    with pytest.raises((TypeError, ValueError), match=error):
        DendriticLevelOverride.from_config(override)


def test_overrides_that_do_not_fit_the_network_are_rejected():
    with pytest.raises(ValueError, match="repeats a depth"):
        DendriNet(**BASE, level_overrides=[{"depth": 1}, {"depth": 1}])
    with pytest.raises(ValueError, match="exceeds the terminal"):
        DendriNet(**BASE, level_overrides=[{"depth": 3}])
    with pytest.raises(ValueError, match="somatic_synapses: false"):
        DendriNet(**BASE, level_overrides=[{"depth": 0, "inhibitory": False}])
    with pytest.raises(ValueError, match="no excitatory pathway"):
        DendriNet(
            **{
                **BASE,
                "excitatory_input_dim": None,
                "excitatory_synapses_per_branch": None,
            },
            level_overrides=[{"depth": 1, "excitatory": False}],
        )
    with pytest.raises(ValueError, match="lists 1 compartments"):
        DendriNet(**BASE, level_overrides=[{"depth": 2, "inhibitory_contacts": [[0]]}])
    with pytest.raises(ValueError, match="soma_type"):
        DendriNet(
            **BASE,
            reactivation_soma_type="tanh",
            level_overrides=[{"depth": 0, "activation": "identity"}],
        )


def test_declared_contacts_exclude_structured_wiring_and_compact_synapses():
    override = DendriticLevelOverride(depth=2, inhibitory_contacts=TERMINAL_I)
    with pytest.raises(ValueError, match="structured_connectivity"):
        _reject_structured_contacts(
            override, masks=(None, torch.ones(1)), indices=(None, None)
        )
    _reject_structured_contacts(
        override, masks=(torch.ones(1), None), indices=(None, None)
    )
    with pytest.raises(ValueError, match="TopK family"):
        declare_contacts(nn.Linear(2, 2), torch.ones(2, 2, dtype=torch.bool))


def test_config_and_keyword_construction_agree():
    config = DendriNetConfig.from_kwargs(
        {**BASE, "epsilon": 0.0, "level_overrides": list(reversed(_overrides()))}
    )
    assert [override.depth for override in config.level_overrides] == [1, 2]
    assert DendriNetConfig.from_kwargs(config.to_kwargs()) == config
    torch.manual_seed(2)
    from_config = DendriNet(config=config)
    typed, _ = _typed(2)
    assert from_config.level_overrides == typed.level_overrides
    assert _same_state(from_config, typed)
    excitatory, inhibitory = _inputs()
    assert torch.equal(
        from_config.double()(excitatory, inhibitory),
        typed.double()(excitatory, inhibitory),
    )
    with pytest.raises(ValueError, match="contradict it: level_overrides"):
        DendriNet(config=config, level_overrides=[{"depth": 1}])
    restored = DendriticLevelOverride.from_config(config.level_overrides[1].to_config())
    assert restored == config.level_overrides[1]


def test_checkpoints_rebuild_declared_contacts_from_configuration():
    source, _ = _typed(4)
    target, _ = _typed(9)
    target.load_state_dict(source.state_dict(), strict=True)
    assert _same_state(source, target)
    terminal = target.branch_layers[0]
    assert terminal.branch_inhibition.connection_mask.sum().item() == 2 * SOMA * LEAVES
    excitation, inhibition = _inputs()
    assert torch.equal(
        source.double()(excitation, inhibition), target.double()(excitation, inhibition)
    )


def test_reinitialize_draws_in_the_documented_order():
    net, _ = _typed(6)
    reference, _ = _typed(6)
    torch.manual_seed(42)
    net.reinitialize(RECIPE)
    after = torch.get_rng_state()
    torch.manual_seed(42)
    terminal, proximal, soma = reference.branch_layers
    with torch.no_grad():
        terminal.branch_excitation.pre_w.normal_(0.0, 0.4)
        terminal.branch_inhibition.pre_w.normal_(0.0, 0.4)
        proximal.branch_inhibition.pre_w.normal_(0.0, 0.4)
        proximal.branches_to_output.log_weight.normal_(0.0, 0.25)
        soma.branches_to_output.log_weight.normal_(0.0, 0.25)
        proximal.branch_inhibition.pre_w.add_(math.log(3.0))
    assert torch.equal(torch.get_rng_state(), after)
    assert _same_state(net, reference)


def test_reinitialize_rejects_what_it_cannot_redraw():
    net, _ = _typed(1)
    with pytest.raises(ValueError, match="no excitatory pathway"):
        net.reinitialize(
            LogNormalReinitialization(
                0.4,
                0.25,
                log_offsets=[{"depth": 1, "target": "excitatory", "value": 1.0}],
            )
        )
    with pytest.raises(ValueError, match="no couplings"):
        net.reinitialize(
            LogNormalReinitialization(
                0.4, 0.25, log_offsets=[LogOffset(2, "coupling", 1.0)]
            )
        )
    with pytest.raises(ValueError, match="not a level"):
        net.reinitialize(
            LogNormalReinitialization(
                0.4, 0.25, log_offsets=[LogOffset(5, "coupling", 1.0)]
            )
        )
    with pytest.raises(ValueError, match="target must be"):
        LogOffset(1, "readout", 1.0)
    with pytest.raises(ValueError, match=">= 0"):
        LogNormalReinitialization(-0.1, 0.25)
    with pytest.raises(TypeError, match="LogNormalReinitialization"):
        net.reinitialize({"synapse_log_sd": 0.4})
    torch.manual_seed(1)
    softplus = DendriNet(**{**BASE, "weight_transform": "softplus"})
    with pytest.raises(ValueError, match="weight_transform: exp"):
        softplus.reinitialize(RECIPE)


def test_forward_levels_returns_every_level_and_can_detach_children():
    net, _ = _typed(8)
    net = net.double()
    excitation, inhibition = _inputs()
    outputs = net.forward_levels(excitation, inhibition)
    assert [output.shape[-1] for output in outputs] == [
        SOMA * LEAVES,
        SOMA * STREAMS,
        SOMA,
    ]
    assert torch.equal(outputs[-1], net(excitation, inhibition))
    terminal_weight = net.branch_layers[0].branch_inhibition.pre_w
    (connected,) = torch.autograd.grad(
        outputs[1].sum(), terminal_weight, allow_unused=True
    )
    assert connected is not None and connected.abs().sum() > 0
    detached = net.forward_levels(excitation, inhibition, detach_child_inputs=True)
    (cut,) = torch.autograd.grad(detached[1].sum(), terminal_weight, allow_unused=True)
    assert cut is None
    assert torch.equal(detached[-1], outputs[-1])
    with pytest.raises(ValueError, match="do not exist"):
        net.forward_levels(excitation, inhibition, interventions={4: None})


def test_current_injection_replaces_one_level_output():
    net, _ = _typed(10)
    net = net.double()
    for layer in net.branch_layers:
        layer.set_branch_diagnostics(True)
    proximal = net.branch_layers[1]
    excitation, inhibition = _inputs()
    reference_input = inhibition.clone()
    reference_input[:, 2 * LEAVES :] = 1.5
    anchor = torch.linspace(-0.2, 0.3, SOMA * STREAMS, dtype=torch.float64)
    plain = net.forward_levels(excitation, inhibition)
    plain_voltage = proximal.get_branch_diagnostics()["V"]
    state = {key: value.clone() for key, value in net.state_dict().items()}
    intervention = CurrentInjection(reference_input, anchor)
    outputs = net.forward_levels(
        excitation, inhibition, interventions={1: intervention}
    )
    child = outputs[0]
    expected = torch.tanh(
        (
            proximal.branches_to_output(child)
            - (
                proximal.branch_inhibition(inhibition)
                - proximal.branch_inhibition(reference_input)
            )
            * anchor
        )
        / (
            1
            + proximal.branches_to_output.sum_conductances()
            + proximal.branch_inhibition(reference_input)
        )
    )
    assert torch.equal(outputs[0], plain[0])
    assert torch.allclose(outputs[1], expected, rtol=0, atol=1e-15)
    # The level's ordinary pass ran; its diagnostics describe the unperturbed state.
    assert torch.equal(proximal.get_branch_diagnostics()["V"], plain_voltage)
    assert not torch.equal(outputs[2], plain[2])
    assert all(
        torch.equal(state[key], value) for key, value in net.state_dict().items()
    )
    # The intervention is differentiable in the level's own parameters.
    (gradient,) = torch.autograd.grad(
        outputs[1].sum(), proximal.branch_inhibition.pre_w
    )
    assert gradient.abs().sum() > 0
    with pytest.raises(ValueError, match="inhibitory input"):
        net.forward_levels(excitation, None, interventions={1: intervention})

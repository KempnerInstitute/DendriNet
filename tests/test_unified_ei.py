import pytest
import torch
from omegaconf import OmegaConf

from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent import (
    BaselineRNN,
    BaselineRNNConfig,
    EILayer,
    EILayerConfig,
    EINetwork,
    EINetworkConfig,
    PopulationConfig,
    StatefulDendriNet,
)

# ---------------------------------------------------------------------------
# StatefulDendriNet tests
# ---------------------------------------------------------------------------


def test_stateful_dendrinet_feedforward_shape():
    pop = PopulationConfig(
        n_neurons=8,
        branch_factors=[2],
        ff_excitatory_synapses=4,
        ff_inhibitory_synapses=2,
        rec_excitatory_synapses=0,
        rec_inhibitory_synapses=0,
    )
    net = StatefulDendriNet(
        pop_config=pop,
        excitatory_input_dim=16,
        inhibitory_input_dim=5,
        recurrent_excitatory_input_dim=None,
        recurrent_inhibitory_input_dim=None,
        dt=1.0,
    )
    x = torch.randn(3, 16)
    inh = torch.randn(3, 5)
    y = net(x, inh)
    assert y.shape == (3, 8)


def test_stateful_dendrinet_recurrent_step_shape():
    pop = PopulationConfig(
        n_neurons=8,
        branch_factors=[2, 2],
        ff_excitatory_synapses=4,
        ff_inhibitory_synapses=2,
        rec_excitatory_synapses=3,
        rec_inhibitory_synapses=2,
    )
    net = StatefulDendriNet(
        pop_config=pop,
        excitatory_input_dim=12,
        inhibitory_input_dim=4,
        recurrent_excitatory_input_dim=8,
        recurrent_inhibitory_input_dim=4,
        dt=1.0,
    )
    state = net.init_state(batch_size=2, device=torch.device("cpu"))
    x = torch.randn(2, 12)
    inh = torch.randn(2, 4)
    rec_e = torch.randn(2, 8)
    rec_i = torch.randn(2, 4)
    y, new_state = net.step(x, inh, rec_e, rec_i, state)

    assert y.shape == (2, 8)
    assert len(new_state.trace_E) == 3
    assert new_state.trace_E[0].shape == (2, 32)
    assert new_state.trace_E[1].shape == (2, 16)
    assert new_state.trace_E[2].shape == (2, 8)


# ---------------------------------------------------------------------------
# EILayer tests
# ---------------------------------------------------------------------------


def test_unified_ei_layer_feedforward():
    cfg = EILayerConfig(
        excitatory=PopulationConfig(
            n_neurons=10,
            branch_factors=[2],
            ff_excitatory_synapses=4,
            ff_inhibitory_synapses=3,
        ),
        inhibitory=PopulationConfig(
            n_neurons=4,
            branch_factors=[1],
            ff_excitatory_synapses=4,
            ff_inhibitory_synapses=2,
        ),
        excitatory_input_dim=12,
        inhibitory_input_dim=5,
        recurrent=False,
        dt=1.0,
    )
    layer = EILayer(cfg)
    x = torch.randn(3, 12)
    ff_i = torch.randn(3, 5)
    s_e, h_i, state = layer(x, ff_i)
    assert s_e.shape == (3, 10)
    assert h_i.shape == (3, 4)
    assert state is None  # Feedforward mode returns no state


def test_unified_ei_layer_recurrent():
    cfg = EILayerConfig(
        excitatory=PopulationConfig(
            n_neurons=10,
            branch_factors=[2],
            ff_excitatory_synapses=4,
            ff_inhibitory_synapses=3,
            rec_excitatory_synapses=3,
            rec_inhibitory_synapses=2,
        ),
        inhibitory=PopulationConfig(
            n_neurons=4,
            branch_factors=[1],
            ff_excitatory_synapses=4,
            ff_inhibitory_synapses=2,
            rec_excitatory_synapses=3,
            rec_inhibitory_synapses=2,
        ),
        excitatory_input_dim=12,
        inhibitory_input_dim=5,
        recurrent=True,
        dt=1.0,
    )
    layer = EILayer(cfg)
    state = layer.init_state(batch_size=3, device=torch.device("cpu"))
    x = torch.randn(3, 12)
    ff_i = torch.randn(3, 5)

    s_e, h_i, new_state = layer(x, ff_i, state)
    assert s_e.shape == (3, 10)
    assert h_i.shape == (3, 4)
    assert new_state.s_E.shape == (3, 10)
    assert new_state.h_I.shape == (3, 4)


# ---------------------------------------------------------------------------
# EINetwork tests
# ---------------------------------------------------------------------------


def test_unified_ei_network_feedforward():
    cfg = EINetworkConfig(
        input_dim=12,
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=10, branch_factors=[2], ff_excitatory_synapses=4
                ),
                inhibitory=PopulationConfig(
                    n_neurons=4, branch_factors=[1], ff_excitatory_synapses=4
                ),
                recurrent=False,
                dt=1.0,
            ),
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=6, branch_factors=[2], ff_excitatory_synapses=4
                ),
                inhibitory=PopulationConfig(
                    n_neurons=3, branch_factors=[1], ff_excitatory_synapses=4
                ),
                recurrent=False,
                dt=1.0,
            ),
        ],
    )
    net = EINetwork(cfg)
    x = torch.randn(5, 12)
    y = net(x)
    assert y.shape == (5, 6)


def test_unified_ei_network_recurrent_all_mode():
    cfg = EINetworkConfig(
        input_dim=12,
        output_mode="all",
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=10,
                    branch_factors=[2],
                    ff_excitatory_synapses=4,
                    rec_excitatory_synapses=3,
                    rec_inhibitory_synapses=2,
                ),
                inhibitory=PopulationConfig(
                    n_neurons=4,
                    branch_factors=[1],
                    ff_excitatory_synapses=4,
                    rec_excitatory_synapses=3,
                    rec_inhibitory_synapses=2,
                ),
                recurrent=True,
                dt=1.0,
            ),
        ],
    )
    net = EINetwork(cfg)
    x = torch.randn(3, 7, 12)
    y = net(x)
    assert y.shape == (3, 7, 10)


def test_unified_ei_network_recurrent_last_mode():
    cfg = EINetworkConfig(
        input_dim=12,
        output_mode="last",
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=10,
                    branch_factors=[1],
                    ff_excitatory_synapses=4,
                    rec_excitatory_synapses=3,
                ),
                inhibitory=None,
                recurrent=True,
                dt=1.0,
            ),
        ],
    )
    net = EINetwork(cfg)
    x = torch.randn(3, 5, 12)
    y = net(x)
    assert y.shape == (3, 10)


def test_unified_ei_network_return_hidden():
    """EINetwork can return final hidden state."""
    cfg = EINetworkConfig(
        input_dim=8,
        output_mode="last",
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=6,
                    branch_factors=[1],
                    ff_excitatory_synapses=4,
                    rec_excitatory_synapses=3,
                ),
                inhibitory=None,
                recurrent=True,
                dt=1.0,
            ),
        ],
    )
    net = EINetwork(cfg)
    x = torch.randn(2, 5, 8)

    # Without return_hidden
    y = net(x)
    assert y.shape == (2, 6)

    # With return_hidden
    y, hidden = net(x, return_hidden=True)
    assert y.shape == (2, 6)
    assert len(hidden) == 1
    assert hidden[0].s_E.shape == (2, 6)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_factory_builds_unified_ei_feedforward():
    params = OmegaConf.create(
        {
            "unified_ei": {
                "output_mode": "last",
                "layers": [
                    {
                        "excitatory": {
                            "n_neurons": 7,
                            "branch_factors": [1],
                            "ff_excitatory_synapses": 3,
                        },
                        "inhibitory": None,
                        "recurrent": False,
                        "dt": 1.0,
                    }
                ],
            }
        }
    )
    net = get_architecture("ei_unified", params, input_dim=9)
    assert isinstance(net, EINetwork)
    y = net(torch.randn(4, 9))
    assert y.shape == (4, 7)


def test_factory_builds_unified_ei_recurrent():
    params = OmegaConf.create(
        {
            "unified_ei": {
                "output_mode": "last",
                "layers": [
                    {
                        "excitatory": {
                            "n_neurons": 6,
                            "branch_factors": [1],
                            "ff_excitatory_synapses": 3,
                            "rec_excitatory_synapses": 2,
                        },
                        "inhibitory": None,
                        "recurrent": True,
                        "dt": 1.0,
                    }
                ],
            }
        }
    )
    net = get_architecture("ei_unified", params, input_dim=8)
    assert isinstance(net, EINetwork)
    y = net(torch.randn(2, 5, 8))
    assert y.shape == (2, 6)


def test_factory_requires_layers_for_unified_ei():
    params = {"unified_ei": {"layers": []}}
    with pytest.raises(ValueError, match="requires at least one layer"):
        get_architecture("ei_unified", params, input_dim=8)


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------


def test_gradient_flow_recurrent():
    """All parameters receive gradients in recurrent mode."""
    cfg = EINetworkConfig(
        input_dim=8,
        output_mode="last",
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=6,
                    branch_factors=[2],
                    ff_excitatory_synapses=4,
                    ff_inhibitory_synapses=3,
                    rec_excitatory_synapses=3,
                    rec_inhibitory_synapses=2,
                ),
                inhibitory=PopulationConfig(
                    n_neurons=3,
                    branch_factors=[1],
                    ff_excitatory_synapses=4,
                    ff_inhibitory_synapses=0,
                    rec_excitatory_synapses=3,
                    rec_inhibitory_synapses=2,
                ),
                recurrent=True,
                dt=1.0,
            ),
        ],
    )
    net = EINetwork(cfg)
    x = torch.randn(2, 4, 8)
    y = net(x)
    loss = y.sum()
    loss.backward()

    params_without_grad = []
    for name, p in net.named_parameters():
        if p.requires_grad and p.grad is None:
            params_without_grad.append(name)

    assert (
        len(params_without_grad) == 0
    ), f"Parameters without gradients: {params_without_grad}"


def test_gradient_flow_feedforward():
    """All parameters receive gradients in feedforward mode."""
    cfg = EINetworkConfig(
        input_dim=8,
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=6,
                    branch_factors=[2],
                    ff_excitatory_synapses=4,
                    ff_inhibitory_synapses=3,
                ),
                inhibitory=PopulationConfig(
                    n_neurons=3,
                    branch_factors=[1],
                    ff_excitatory_synapses=4,
                ),
                recurrent=False,
            ),
        ],
    )
    net = EINetwork(cfg)
    x = torch.randn(2, 8)
    y = net(x)
    loss = y.sum()
    loss.backward()

    params_without_grad = []
    for name, p in net.named_parameters():
        if p.requires_grad and p.grad is None:
            params_without_grad.append(name)

    assert (
        len(params_without_grad) == 0
    ), f"Parameters without gradients: {params_without_grad}"


# ---------------------------------------------------------------------------
# Multi-step trace accumulation test
# ---------------------------------------------------------------------------


def test_trace_accumulation_over_steps():
    """Traces should accumulate over multiple recurrent steps (not reset)."""
    pop = PopulationConfig(
        n_neurons=4,
        branch_factors=[1],
        ff_excitatory_synapses=4,
        ff_inhibitory_synapses=0,
        rec_excitatory_synapses=3,
        rec_inhibitory_synapses=0,
    )
    net = StatefulDendriNet(
        pop_config=pop,
        excitatory_input_dim=6,
        recurrent_excitatory_input_dim=4,
        dt=1.0,
    )
    state = net.init_state(batch_size=2, device=torch.device("cpu"))

    x = torch.randn(2, 6)
    rec = torch.randn(2, 4)

    # Step 1
    y1, state = net.step(x, recurrent_input=rec, state=state)
    trace_after_step1 = state.trace_E[0].clone()

    # Step 2 (same input)
    y2, state = net.step(x, recurrent_input=rec, state=state)
    trace_after_step2 = state.trace_E[0].clone()

    # Traces should differ between steps (accumulation)
    assert not torch.allclose(
        trace_after_step1, trace_after_step2
    ), "Traces should accumulate over steps"

    # Output should differ between steps due to trace dynamics
    assert not torch.allclose(
        y1, y2
    ), "Outputs should differ across steps due to trace dynamics"


# ---------------------------------------------------------------------------
# State operations tests
# ---------------------------------------------------------------------------


def test_state_detach():
    """State.detach() breaks gradient graph."""
    pop = PopulationConfig(n_neurons=4, branch_factors=[1], ff_excitatory_synapses=4)
    net = StatefulDendriNet(
        pop_config=pop, excitatory_input_dim=6, recurrent_excitatory_input_dim=4
    )
    state = net.init_state(batch_size=2, device=torch.device("cpu"))
    x = torch.randn(2, 6, requires_grad=True)
    rec = torch.randn(2, 4)

    _, state = net.step(x, recurrent_input=rec, state=state)
    detached = state.detach()

    for t in detached.trace_E:
        assert not t.requires_grad


def test_state_clone():
    """State.clone() creates independent copy."""
    from dendritic_modeling.networks.architectures.recurrent.ei_state import (
        DendriNetState,
    )

    state = DendriNetState.zeros(
        level_dims=[8, 4], batch_size=2, device=torch.device("cpu")
    )
    state.trace_E[0].fill_(1.0)

    cloned = state.clone()
    cloned.trace_E[0].fill_(2.0)

    # Original should not be modified
    assert torch.allclose(state.trace_E[0], torch.ones(2, 8))


# ---------------------------------------------------------------------------
# Auto tau test
# ---------------------------------------------------------------------------


def test_auto_tau_geometric_spacing():
    """Auto-generated taus should be geometrically spaced."""
    pop = PopulationConfig(
        n_neurons=4,
        branch_factors=[3, 2],
        ff_excitatory_synapses=4,
        tau_base=10.0,
        tau_ratio=3.0,
    )
    net = StatefulDendriNet(
        pop_config=pop,
        excitatory_input_dim=8,
        dt=1.0,
    )
    # 3 levels: distal=10*3^2=90, mid=10*3=30, soma=10
    import math

    expected_decays = [
        math.exp(-1.0 / 90.0),
        math.exp(-1.0 / 30.0),
        math.exp(-1.0 / 10.0),
    ]
    for i, expected in enumerate(expected_decays):
        assert (
            abs(net.decays[i].item() - expected) < 1e-6
        ), f"Level {i}: expected decay={expected}, got {net.decays[i].item()}"


# ---------------------------------------------------------------------------
# Routing tracking tests
# ---------------------------------------------------------------------------


def test_routing_info_populated():
    """store_routing=True populates per-level routing info."""
    cfg = EINetworkConfig(
        input_dim=8,
        output_mode="last",
        store_routing=True,
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=6,
                    branch_factors=[2],
                    ff_excitatory_synapses=4,
                    ff_inhibitory_synapses=2,
                    rec_excitatory_synapses=3,
                    rec_inhibitory_synapses=2,
                ),
                inhibitory=PopulationConfig(
                    n_neurons=3,
                    branch_factors=[1],
                    ff_excitatory_synapses=4,
                    rec_excitatory_synapses=2,
                    rec_inhibitory_synapses=2,
                ),
                recurrent=True,
                dt=1.0,
            ),
        ],
    )
    net = EINetwork(cfg)
    x = torch.randn(4, 5, 8)
    net(x)

    # Check routing info exists
    assert len(net._routing_info) == 5  # one per timestep
    info_t0 = net._routing_info[0]
    assert "layer_0" in info_t0
    assert "excitatory" in info_t0["layer_0"]
    assert "inhibitory" in info_t0["layer_0"]

    # Excitatory has 2 levels (branch_factors=[2])
    exc_info = info_t0["layer_0"]["excitatory"]
    assert "level_contributions" in exc_info
    assert "ff_rec_ratios" in exc_info
    assert exc_info["level_contributions"].shape == (4, 2)
    assert exc_info["ff_rec_ratios"].shape == (4, 2)

    # Inhibitory has 2 levels (branch_factors=[1])
    inh_info = info_t0["layer_0"]["inhibitory"]
    assert inh_info["level_contributions"].shape == (4, 2)


def test_routing_info_ff_ratio_starts_at_one():
    """At t=0, FF/REC ratio should be ~1.0 (no recurrent signal yet)."""
    cfg = EINetworkConfig(
        input_dim=8,
        output_mode="last",
        store_routing=True,
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=6,
                    branch_factors=[3],
                    ff_excitatory_synapses=4,
                    rec_excitatory_synapses=3,
                    rec_inhibitory_synapses=2,
                ),
                inhibitory=None,
                recurrent=True,
                dt=1.0,
            ),
        ],
    )
    net = EINetwork(cfg)
    x = torch.randn(2, 3, 8)
    net(x)

    # At t=0 with zero initial state, recurrent compartments contribute nothing
    ff_ratio_t0 = net._routing_info[0]["layer_0"]["excitatory"]["ff_rec_ratios"]
    assert torch.all(
        ff_ratio_t0 > 0.99
    ), f"Expected FF ratio ~1.0 at t=0, got {ff_ratio_t0}"


def test_routing_store_toggle():
    """store_routing can be toggled on/off."""
    cfg = EINetworkConfig(
        input_dim=8,
        output_mode="last",
        store_routing=False,
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=6,
                    branch_factors=[1],
                    ff_excitatory_synapses=4,
                    rec_excitatory_synapses=3,
                ),
                recurrent=True,
                dt=1.0,
            ),
        ],
    )
    net = EINetwork(cfg)
    x = torch.randn(2, 3, 8)

    # Initially off
    net(x)
    assert len(net._routing_info) == 0

    # Turn on
    net.store_routing = True
    net(x)
    assert len(net._routing_info) == 3

    # Turn off
    net.store_routing = False
    net(x)
    assert len(net._routing_info) == 0


# ---------------------------------------------------------------------------
# Baseline RNN tests
# ---------------------------------------------------------------------------


def test_baseline_rnn_gru():
    """GRU baseline produces correct output shapes."""
    cfg = BaselineRNNConfig(
        cell_type="gru", input_dim=8, hidden_dim=16, output_mode="last"
    )
    net = BaselineRNN(cfg)
    assert net.is_recurrent
    assert net.output_dim == 16

    x = torch.randn(3, 10, 8)
    y = net(x)
    assert y.shape == (3, 16)


def test_baseline_rnn_lstm():
    """LSTM baseline produces correct output shapes."""
    cfg = BaselineRNNConfig(
        cell_type="lstm", input_dim=8, hidden_dim=32, output_mode="mean"
    )
    net = BaselineRNN(cfg)
    x = torch.randn(3, 10, 8)
    y = net(x)
    assert y.shape == (3, 32)


def test_baseline_rnn_vanilla():
    """Vanilla RNN baseline produces correct output shapes."""
    cfg = BaselineRNNConfig(
        cell_type="vanilla", input_dim=8, hidden_dim=24, output_mode="all"
    )
    net = BaselineRNN(cfg)
    x = torch.randn(3, 10, 8)
    y = net(x)
    assert y.shape == (3, 10, 24)


def test_baseline_rnn_return_hidden():
    """Baseline RNN can return hidden state."""
    cfg = BaselineRNNConfig(
        cell_type="gru", input_dim=8, hidden_dim=16, output_mode="last"
    )
    net = BaselineRNN(cfg)
    x = torch.randn(2, 5, 8)
    y, h = net(x, return_hidden=True)
    assert y.shape == (2, 16)
    assert h.shape == (1, 2, 16)  # [num_layers, batch, hidden]


def test_baseline_rnn_gradient_flow():
    """All baseline RNN parameters receive gradients."""
    cfg = BaselineRNNConfig(
        cell_type="gru", input_dim=8, hidden_dim=16, output_mode="last"
    )
    net = BaselineRNN(cfg)
    x = torch.randn(2, 5, 8)
    y = net(x)
    loss = y.sum()
    loss.backward()

    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_factory_builds_baseline_rnn():
    """Factory creates baseline RNNs from type strings."""
    for rnn_type in ["gru", "lstm", "vanilla_rnn"]:
        net = get_architecture(
            rnn_type,
            {"baseline_rnn": {"hidden_dim": 16, "output_mode": "last"}},
            input_dim=8,
        )
        assert isinstance(net, BaselineRNN)
        assert net.is_recurrent
        x = torch.randn(2, 5, 8)
        y = net(x)
        assert y.shape == (2, 16)

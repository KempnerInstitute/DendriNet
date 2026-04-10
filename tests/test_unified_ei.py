import pytest
import torch
from omegaconf import OmegaConf

from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
)
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
from dendritic_modeling.scripts.script_utils.setup_utils import (
    _is_recurrent_core_config,
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


def test_stateful_dendrinet_can_mask_same_population_self_recurrence():
    pop = PopulationConfig(
        n_neurons=4,
        branch_factors=[2],
        ff_excitatory_synapses=4,
        rec_excitatory_synapses=3,
        allow_self_recurrence=False,
    )
    net = StatefulDendriNet(
        pop_config=pop,
        excitatory_input_dim=6,
        recurrent_excitatory_input_dim=4,
        recurrent_excitatory_is_self_population=True,
        dt=1.0,
    )

    for level_idx, layer in enumerate(net.branch_layers):
        mask = layer.branch_recurrent.weight_mask()
        owner_idx = net._output_owner_index_per_level[level_idx]
        for out_idx, forbidden_in_idx in enumerate(owner_idx.tolist()):
            assert mask[out_idx, forbidden_in_idx].item() == 0


def test_unified_ei_layer_masks_only_same_population_self_recurrence():
    cfg = EILayerConfig(
        excitatory=PopulationConfig(
            n_neurons=6,
            branch_factors=[2],
            ff_excitatory_synapses=4,
            ff_inhibitory_synapses=2,
            rec_excitatory_synapses=3,
            rec_inhibitory_synapses=2,
            allow_self_recurrence=False,
        ),
        inhibitory=PopulationConfig(
            n_neurons=3,
            branch_factors=[2],
            ff_excitatory_synapses=3,
            ff_inhibitory_synapses=2,
            rec_excitatory_synapses=2,
            rec_inhibitory_synapses=2,
            allow_self_recurrence=False,
        ),
        excitatory_input_dim=8,
        inhibitory_input_dim=4,
        recurrent=True,
        dt=1.0,
    )
    layer = EILayer(cfg)

    e_self_mask = layer.e_population.branch_layers[-1].branch_recurrent.weight_mask()
    assert torch.all(e_self_mask.diag() == 0)

    e_cross_mask = layer.e_population.branch_layers[
        -1
    ].branch_rec_inhibition.weight_mask()
    assert torch.all(e_cross_mask.sum(dim=-1) > 0)

    i_self_mask = layer.i_population.branch_layers[
        -1
    ].branch_rec_inhibition.weight_mask()
    assert torch.all(i_self_mask.diag() == 0)

    i_cross_mask = layer.i_population.branch_layers[-1].branch_recurrent.weight_mask()
    assert torch.all(i_cross_mask.sum(dim=-1) > 0)


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


def test_unified_ei_network_transfer_input_mode0_first_layer_gets_transfer_inhibitory():
    cfg = EINetworkConfig(
        input_dim=6,
        use_transfer=True,
        transfer_params={"input_mode": 0, "independent_pathways": False},
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=5,
                    branch_factors=[1],
                    ff_excitatory_synapses=3,
                    ff_inhibitory_synapses=2,
                ),
                inhibitory=PopulationConfig(
                    n_neurons=3,
                    branch_factors=[1],
                    ff_excitatory_synapses=3,
                    ff_inhibitory_synapses=2,
                ),
                recurrent=False,
                dt=1.0,
            ),
        ],
    )
    net = EINetwork(cfg)

    captured: dict[str, torch.Tensor | None] = {"ff_inhibitory": None}

    def _capture_ff_inhibitory(_, args):
        captured["ff_inhibitory"] = args[1] if len(args) > 1 else None

    handle = net.layers[0].register_forward_pre_hook(_capture_ff_inhibitory)
    x = torch.randn(4, 6)
    y = net(x)
    handle.remove()

    assert y.shape == (4, 5)
    assert captured["ff_inhibitory"] is not None
    assert torch.allclose(captured["ff_inhibitory"], x)


def test_unified_ei_network_recurrent_transfer_input_mode0_first_step_gets_transfer_inhibitory():
    cfg = EINetworkConfig(
        input_dim=6,
        output_mode="all",
        use_transfer=True,
        transfer_params={"input_mode": 0, "independent_pathways": False},
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(
                    n_neurons=5,
                    branch_factors=[1],
                    ff_excitatory_synapses=3,
                    ff_inhibitory_synapses=2,
                    rec_excitatory_synapses=2,
                ),
                inhibitory=PopulationConfig(
                    n_neurons=3,
                    branch_factors=[1],
                    ff_excitatory_synapses=3,
                    ff_inhibitory_synapses=2,
                    rec_excitatory_synapses=2,
                ),
                recurrent=True,
                dt=1.0,
            ),
        ],
    )
    net = EINetwork(cfg)

    seen_ff_inhibitory: list[torch.Tensor] = []

    def _capture_ff_inhibitory(_, args):
        ff_inhibitory = args[1] if len(args) > 1 else None
        if ff_inhibitory is not None:
            seen_ff_inhibitory.append(ff_inhibitory.detach().clone())

    handle = net.layers[0].register_forward_pre_hook(_capture_ff_inhibitory)
    x = torch.randn(2, 3, 6)
    y = net(x)
    handle.remove()

    assert y.shape == (2, 3, 5)
    assert len(seen_ff_inhibitory) == 3
    assert torch.allclose(seen_ff_inhibitory[0], x[:, 0, :])


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


def test_factory_builds_einet_recurrent_option_with_transfer():
    """New-style: rec synapses in connectivity, no explicit enabled flag."""
    params = OmegaConf.create(
        {
            "architecture": {
                "excitatory_layer_sizes": [6],
                "inhibitory_layer_sizes": [],
                "excitatory_branch_factors": [2],
                "inhibitory_branch_factors": [1],
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [3],
                "ei_synapses_per_branch_per_layer": [0],
                "ie_synapses_per_branch_per_layer": [2],
                "ii_synapses_per_branch_per_layer": [0],
                "rec_ee_synapses_per_branch": [2],
                "rec_ie_synapses_per_branch": [1],
            },
            "transfer": {
                "input_mode": 1,
                "independent_pathways": False,
            },
            "morphology": {
                "use_shunting": True,
                "weight_transform": "exp",
                "somatic_synapses": True,
                "dbl_init_method": "analytical_expectation",
            },
            "sparsity": {
                "init_method": "xavier_normal",
                "noise_level": 0.0,
                "type": "standard",
                "weight_norm_order": None,
                "gamma": 1.0,
                "temperature": 0.0,
                "gradient_scaling": "none",
                "deepst": {
                    "excitatory_target_density": 0.1,
                    "inhibitory_target_density": 0.1,
                    "use_noise": True,
                    "sigma": 0.05,
                    "rewiring_mode": "global",
                    "rewire_frequency": 1,
                    "synapses_per_branch": None,
                    "freeze_excitatory_connectivity": False,
                    "freeze_inhibitory_connectivity": False,
                    "init_method": "xavier_normal",
                    "weight_threshold": 1e-6,
                },
            },
            "reactivation": {
                "enabled": True,
                "type": "param_tanh",
                "init_m": 1.0,
                "init_b": 0.5,
                "gradient_scaling": "none",
            },
            "blocklinear": {"gradient_scaling": "none", "efficient": False},
            "implementation": {"adaptive_initialization": False, "print_hooks": False},
            "recurrent_ei": {
                "dt": 1.0,
                "output_mode": "last",
            },
        }
    )
    net = get_architecture("EINet", params, input_dim=8)
    assert isinstance(net, EINetwork)
    assert net.is_recurrent
    assert net.layers[0].config.direct_ff_inhibitory_to_excitatory is True
    y = net(torch.randn(2, 4, 8))
    assert y.shape == (2, 6)


def test_factory_recurrent_einet_input_mode0_uses_transfer_by_default():
    params = OmegaConf.create(
        {
            "architecture": {
                "excitatory_layer_sizes": [5],
                "inhibitory_layer_sizes": [3],
                "excitatory_branch_factors": [1],
                "inhibitory_branch_factors": [1],
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [3],
                "ei_synapses_per_branch_per_layer": [2],
                "ie_synapses_per_branch_per_layer": [2],
                "ii_synapses_per_branch_per_layer": [1],
                "rec_ee_synapses_per_branch": [1],
                "rec_ie_synapses_per_branch": [1],
                "rec_ei_synapses_per_branch": [1],
                "rec_ii_synapses_per_branch": [1],
            },
            "transfer": {
                "input_mode": 0,
                "independent_pathways": False,
            },
            "morphology": {
                "use_shunting": True,
                "weight_transform": "softplus",
                "somatic_synapses": False,
                "dbl_init_method": "analytical_expectation",
            },
            "sparsity": {
                "init_method": "xavier_normal",
                "noise_level": 0.0,
                "type": "standard",
                "gamma": 1.0,
                "temperature": 0.0,
                "gradient_scaling": "none",
                "deepst": {},
            },
            "reactivation": {
                "enabled": True,
                "type": "param_tanh",
                "init_m": 1.0,
                "init_b": 0.5,
                "gradient_scaling": "none",
            },
            "blocklinear": {"gradient_scaling": "none", "efficient": False},
            "implementation": {"adaptive_initialization": False, "print_hooks": False},
            "recurrent_ei": {"dt": 1.0, "output_mode": "all"},
        }
    )
    net = get_architecture("EINet", params, input_dim=6)
    assert isinstance(net, EINetwork)
    assert net.transfer_fn is not None
    y = net(torch.randn(2, 3, 6))
    assert y.shape == (2, 3, 5)


def test_factory_builds_einet_recurrent_old_style_backward_compat():
    """Old-style: rec synapses in recurrent_ei with enabled flag (backward compat)."""
    params = OmegaConf.create(
        {
            "architecture": {
                "excitatory_layer_sizes": [6],
                "inhibitory_layer_sizes": [],
                "excitatory_branch_factors": [2],
                "inhibitory_branch_factors": [1],
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [3],
                "ei_synapses_per_branch_per_layer": [0],
                "ie_synapses_per_branch_per_layer": [2],
                "ii_synapses_per_branch_per_layer": [0],
            },
            "transfer": {"input_mode": 1, "independent_pathways": False},
            "morphology": {
                "use_shunting": True,
                "weight_transform": "exp",
                "somatic_synapses": True,
                "dbl_init_method": "analytical_expectation",
            },
            "sparsity": {
                "init_method": "xavier_normal",
                "type": "standard",
                "deepst": {},
            },
            "reactivation": {
                "enabled": True,
                "type": "param_tanh",
                "init_m": 1.0,
                "init_b": 0.5,
            },
            "blocklinear": {},
            "implementation": {},
            "recurrent_ei": {
                "enabled": True,
                "use_transfer": True,
                "rec_excitatory_synapses_per_layer": [2],
                "rec_inhibitory_synapses_per_layer": [1],
                "recurrent_layers": [True],
                "dt": 1.0,
                "output_mode": "last",
            },
        }
    )
    net = get_architecture("EINet", params, input_dim=8)
    assert isinstance(net, EINetwork)
    assert net.is_recurrent
    y = net(torch.randn(2, 4, 8))
    assert y.shape == (2, 6)


def test_factory_einet_without_recurrent_option_stays_legacy():
    params = OmegaConf.create(
        {
            "architecture": {
                "excitatory_layer_sizes": [7],
                "inhibitory_layer_sizes": [],
                "excitatory_branch_factors": [2],
                "inhibitory_branch_factors": [1],
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [3],
                "ei_synapses_per_branch_per_layer": [0],
                "ie_synapses_per_branch_per_layer": [3],
                "ii_synapses_per_branch_per_layer": [0],
            },
            "transfer": {"input_mode": 1, "independent_pathways": False},
            "morphology": {
                "use_shunting": True,
                "weight_transform": "exp",
                "somatic_synapses": True,
                "dbl_init_method": "analytical_expectation",
            },
            "sparsity": {
                "init_method": "xavier_normal",
                "noise_level": 0.0,
                "type": "standard",
                "deepst": {
                    "excitatory_target_density": 0.1,
                    "inhibitory_target_density": 0.1,
                    "use_noise": True,
                    "sigma": 0.05,
                    "rewiring_mode": "global",
                    "rewire_frequency": 1,
                    "synapses_per_branch": None,
                    "freeze_excitatory_connectivity": False,
                    "freeze_inhibitory_connectivity": False,
                    "init_method": "xavier_normal",
                    "weight_threshold": 1e-6,
                },
            },
            "reactivation": {
                "enabled": True,
                "type": "param_tanh",
                "init_m": 1.0,
                "init_b": 0.5,
            },
            "blocklinear": {"efficient": False},
            "implementation": {"adaptive_initialization": False, "print_hooks": False},
        }
    )
    net = get_architecture("einet", params, input_dim=8)
    assert isinstance(net, ConfigurableEINetwork)
    y = net(torch.randn(3, 8))
    assert y.shape == (3, 7)


def test_setup_utils_detects_recurrent_einet_option():
    # New style: rec synapses in connectivity
    core_cfg_new = {
        "type": "EINet",
        "connectivity": {"rec_ee_synapses_per_branch": [8]},
    }
    assert _is_recurrent_core_config(core_cfg_new) is True

    # Old style (backward compat)
    core_cfg_old = {
        "type": "EINet",
        "recurrent_ei": {"enabled": True, "recurrent_layers": [False, True]},
    }
    assert _is_recurrent_core_config(core_cfg_old) is True

    # No recurrence (all zeros)
    core_cfg_none = {
        "type": "EINet",
        "connectivity": {
            "rec_ee_synapses_per_branch": [0],
            "rec_ie_synapses_per_branch": [0],
        },
    }
    assert _is_recurrent_core_config(core_cfg_none) is False


def test_factory_einet_recurrent_per_layer_tau():
    """Per-layer tau_base and tau_ratio lists are supported."""
    params = OmegaConf.create(
        {
            "architecture": {
                "excitatory_layer_sizes": [6, 4],
                "excitatory_branch_factors": [2],
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [3],
                "ie_synapses_per_branch_per_layer": [2],
                "rec_ee_synapses_per_branch": [2, 3],
            },
            "transfer": {"input_mode": 1},
            "morphology": {
                "use_shunting": True,
                "weight_transform": "exp",
                "somatic_synapses": True,
            },
            "sparsity": {
                "init_method": "xavier_normal",
                "type": "standard",
                "deepst": {},
            },
            "reactivation": {
                "enabled": True,
                "type": "param_tanh",
                "init_m": 1.0,
                "init_b": 0.5,
            },
            "blocklinear": {},
            "implementation": {},
            "recurrent_ei": {
                "dt": 1.0,
                "output_mode": "last",
                "tau_base": [50.0, 30.0],
                "tau_ratio": [3.0, 2.0],
            },
        }
    )
    net = get_architecture("EINet", params, input_dim=8)
    assert isinstance(net, EINetwork)
    assert net.is_recurrent
    assert len(net.layers) == 2
    # Layer 0: tau_base=50, layer 1: tau_base=30
    assert net.layers[0].config.excitatory.tau_base == 50.0
    assert net.layers[1].config.excitatory.tau_base == 30.0
    assert net.layers[0].config.excitatory.tau_ratio == 3.0
    assert net.layers[1].config.excitatory.tau_ratio == 2.0
    y = net(torch.randn(2, 4, 8))
    assert y.shape == (2, 4)


def test_factory_requires_layers_for_unified_ei():
    params = {"unified_ei": {"layers": []}}
    with pytest.raises(ValueError, match="requires at least one layer"):
        get_architecture("ei_unified", params, input_dim=8)


# ---------------------------------------------------------------------------
# Inhibitory routing tests
# ---------------------------------------------------------------------------


def _make_2layer_recurrent_params(inhibitory_mode="first"):
    """Helper: 2-layer recurrent EINet config with configurable inhibitory_mode."""
    return OmegaConf.create(
        {
            "architecture": {
                "excitatory_layer_sizes": [6, 4],
                "inhibitory_layer_sizes": [],
                "excitatory_branch_factors": [2],
                "inhibitory_branch_factors": [1],
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [3],
                "ie_synapses_per_branch_per_layer": [2],
                "rec_ee_synapses_per_branch": [2],
                "rec_ie_synapses_per_branch": [1],
            },
            "transfer": {
                "input_mode": 1,
                "independent_pathways": False,
                "inhibitory_mode": inhibitory_mode,
            },
            "morphology": {
                "use_shunting": True,
                "weight_transform": "exp",
                "somatic_synapses": True,
                "dbl_init_method": "analytical_expectation",
            },
            "sparsity": {
                "init_method": "xavier_normal",
                "type": "standard",
                "deepst": {},
            },
            "reactivation": {
                "enabled": True,
                "type": "param_tanh",
                "init_m": 1.0,
                "init_b": 0.5,
            },
            "blocklinear": {},
            "implementation": {},
            "recurrent_ei": {"dt": 1.0, "output_mode": "last"},
        }
    )


def test_inhibitory_mode_none_blocks_transfer():
    """With inhibitory_mode='none', no layer receives transfer inhibitory."""
    params = _make_2layer_recurrent_params(inhibitory_mode="none")
    net = get_architecture("EINet", params, input_dim=8)
    assert isinstance(net, EINetwork)
    # Both layers should have direct_ff_inhibitory_to_excitatory = False
    for layer in net.layers:
        assert layer.config.direct_ff_inhibitory_to_excitatory is False

    # Capture ff_inhibitory argument to each layer via hooks
    captured = {}

    def _hook(name):
        def fn(module, args):
            captured[name] = args[1]  # ff_inhibitory is second positional arg

        return fn

    net.layers[0].register_forward_pre_hook(_hook("layer_0"))
    net.layers[1].register_forward_pre_hook(_hook("layer_1"))
    net(torch.randn(2, 4, 8))
    # Both layers should receive None inhibitory
    assert captured["layer_0"] is None
    assert captured["layer_1"] is None


def test_inhibitory_mode_first_only_layer0():
    """With inhibitory_mode='first', only layer 0 gets transfer inhibitory."""
    params = _make_2layer_recurrent_params(inhibitory_mode="first")
    net = get_architecture("EINet", params, input_dim=8)
    assert isinstance(net, EINetwork)
    assert net.layers[0].config.direct_ff_inhibitory_to_excitatory is True
    assert net.layers[1].config.direct_ff_inhibitory_to_excitatory is False

    captured = {}

    def _hook(name):
        def fn(module, args):
            captured[name] = args[1]

        return fn

    net.layers[0].register_forward_pre_hook(_hook("layer_0"))
    net.layers[1].register_forward_pre_hook(_hook("layer_1"))
    net(torch.randn(2, 4, 8))
    assert captured["layer_0"] is not None
    assert captured["layer_1"] is None


def test_inhibitory_mode_all_every_layer():
    """With inhibitory_mode='all', every layer gets transfer inhibitory."""
    params = _make_2layer_recurrent_params(inhibitory_mode="all")
    net = get_architecture("EINet", params, input_dim=8)
    assert isinstance(net, EINetwork)
    for layer in net.layers:
        assert layer.config.direct_ff_inhibitory_to_excitatory is True

    captured = {}

    def _hook(name):
        def fn(module, args):
            captured[name] = args[1]

        return fn

    net.layers[0].register_forward_pre_hook(_hook("layer_0"))
    net.layers[1].register_forward_pre_hook(_hook("layer_1"))
    net(torch.randn(2, 4, 8))
    assert captured["layer_0"] is not None
    assert captured["layer_1"] is not None


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


# ---------------------------------------------------------------------------
# Learnable tau tests
# ---------------------------------------------------------------------------


def test_learnable_tau_creates_parameter():
    """learnable_tau=True stores log_tau as nn.Parameter, not buffer."""
    pop = PopulationConfig(
        n_neurons=4,
        branch_factors=[3, 2],
        ff_excitatory_synapses=4,
        tau_base=10.0,
        tau_ratio=3.0,
        learnable_tau=True,
    )
    net = StatefulDendriNet(pop_config=pop, excitatory_input_dim=8, dt=1.0)
    assert net.learnable_tau is True
    assert net.log_tau is not None
    assert isinstance(net.log_tau, torch.nn.Parameter)
    assert net.log_tau.requires_grad is True
    assert net.log_tau.shape == (3,)  # 3 levels


def test_learnable_tau_initial_values_match_fixed():
    """Learnable tau initial values should match fixed-mode geometric spacing."""
    import math

    for learnable in [True, False]:
        pop = PopulationConfig(
            n_neurons=4,
            branch_factors=[3, 2],
            ff_excitatory_synapses=4,
            tau_base=10.0,
            tau_ratio=3.0,
            learnable_tau=learnable,
        )
        net = StatefulDendriNet(pop_config=pop, excitatory_input_dim=8, dt=1.0)
        decays = net.get_decays()
        expected = [math.exp(-1.0 / 90), math.exp(-1.0 / 30), math.exp(-1.0 / 10)]
        for i, exp_val in enumerate(expected):
            assert (
                abs(decays[i].item() - exp_val) < 1e-5
            ), f"Level {i}: learnable={learnable}, expected {exp_val}, got {decays[i].item()}"


def test_learnable_tau_gradient_flow():
    """Gradients must flow through log_tau during backprop."""
    pop = PopulationConfig(
        n_neurons=4,
        branch_factors=[2],
        ff_excitatory_synapses=4,
        rec_excitatory_synapses=3,
        tau_base=5.0,
        tau_ratio=5.0,
        learnable_tau=True,
    )
    net = StatefulDendriNet(
        pop_config=pop,
        excitatory_input_dim=8,
        recurrent_excitatory_input_dim=4,
        dt=1.0,
    )
    x = torch.randn(2, 8)
    state = None
    # Run 5 recurrent steps to accumulate tau-dependent dynamics
    for _ in range(5):
        out, state = net.step(x, state=state)
    loss = out.sum()
    loss.backward()
    assert net.log_tau.grad is not None, "log_tau received no gradient"
    assert (net.log_tau.grad.abs() > 0).any(), "log_tau gradient is all zeros"


def test_learnable_tau_current_taus_property():
    """current_taus property returns correct tau values."""
    pop = PopulationConfig(
        n_neurons=4,
        branch_factors=[2],
        ff_excitatory_synapses=4,
        tau_base=10.0,
        tau_ratio=3.0,
        learnable_tau=True,
    )
    net = StatefulDendriNet(pop_config=pop, excitatory_input_dim=8, dt=1.0)
    taus = net.current_taus
    assert taus.shape == (2,)  # 2 levels for branch_factors=[2]
    # distal = 10*3 = 30, soma = 10
    assert abs(taus[0].item() - 30.0) < 0.1
    assert abs(taus[1].item() - 10.0) < 0.1


def test_fixed_tau_has_no_log_tau_parameter():
    """learnable_tau=False should NOT create log_tau parameter."""
    pop = PopulationConfig(
        n_neurons=4,
        branch_factors=[2],
        ff_excitatory_synapses=4,
        tau_base=10.0,
        tau_ratio=3.0,
        learnable_tau=False,
    )
    net = StatefulDendriNet(pop_config=pop, excitatory_input_dim=8, dt=1.0)
    assert net.log_tau is None
    # No tau-related parameters should be in the parameter list
    param_names = [n for n, _ in net.named_parameters()]
    assert "log_tau" not in param_names


# ---------------------------------------------------------------------------
# Tau stability tests
# ---------------------------------------------------------------------------


def test_tau_stability_extreme_ratios():
    """Network should handle extreme dt/tau ratios without NaN."""
    for tau_base, tau_ratio in [(0.1, 100.0), (1000.0, 1.0), (1.0, 0.01)]:
        pop = PopulationConfig(
            n_neurons=4,
            branch_factors=[2],
            ff_excitatory_synapses=4,
            rec_excitatory_synapses=3,
            tau_base=tau_base,
            tau_ratio=tau_ratio,
        )
        net = StatefulDendriNet(
            pop_config=pop,
            excitatory_input_dim=8,
            recurrent_excitatory_input_dim=4,
            dt=1.0,
        )
        x = torch.randn(2, 8)
        state = None
        for _ in range(10):
            out, state = net.step(x, state=state)
        assert not torch.isnan(
            out
        ).any(), f"NaN with tau_base={tau_base}, tau_ratio={tau_ratio}"
        assert not torch.isinf(
            out
        ).any(), f"Inf with tau_base={tau_base}, tau_ratio={tau_ratio}"


def test_learnable_tau_stability_after_optimization():
    """Learnable taus should stay positive after gradient updates."""
    pop = PopulationConfig(
        n_neurons=4,
        branch_factors=[2],
        ff_excitatory_synapses=4,
        rec_excitatory_synapses=3,
        tau_base=5.0,
        tau_ratio=5.0,
        learnable_tau=True,
    )
    net = StatefulDendriNet(
        pop_config=pop,
        excitatory_input_dim=8,
        recurrent_excitatory_input_dim=4,
        dt=1.0,
    )
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    x = torch.randn(2, 8)
    for _ in range(20):
        opt.zero_grad()
        state = None
        for _ in range(5):
            out, state = net.step(x, state=state)
        loss = out.sum()
        loss.backward()
        opt.step()
    # Taus must remain positive (log-space guarantees this)
    taus = net.current_taus
    assert (taus > 0).all(), f"Negative taus after optimization: {taus}"
    assert not torch.isnan(taus).any(), f"NaN taus: {taus}"


# ---------------------------------------------------------------------------
# Equivalence test: recurrent-off path vs feedforward
# ---------------------------------------------------------------------------


def test_recurrent_off_matches_feedforward():
    """EINetwork with rec_*=0 should produce same output as feedforward path."""
    pop = PopulationConfig(
        n_neurons=8,
        branch_factors=[2],
        ff_excitatory_synapses=6,
        ff_inhibitory_synapses=3,
        rec_excitatory_synapses=0,
        rec_inhibitory_synapses=0,
    )
    cfg = EINetworkConfig(
        layers=[EILayerConfig(excitatory=pop, recurrent=False)],
        input_dim=16,
        use_transfer=False,
        output_mode="last",
    )
    net = EINetwork(cfg)
    torch.manual_seed(0)
    x_single = torch.randn(2, 16)  # single step
    x_seq = x_single.unsqueeze(1)  # [2, 1, 16] — 1-step sequence

    # FF path
    y_ff = net._forward_feedforward(x_single)
    # Recurrent path with 1 step returns (output, hidden)
    y_rec, _ = net._forward_recurrent(x_seq)

    assert torch.allclose(
        y_ff, y_rec, atol=1e-5
    ), f"FF and recurrent paths diverge: max diff = {(y_ff - y_rec).abs().max().item()}"


# ---------------------------------------------------------------------------
# LR schedule behavior test
# ---------------------------------------------------------------------------


def test_cosine_lr_schedule_values():
    """Cosine LR schedule should follow expected warmup + cosine curve."""

    from dendritic_modeling.training.strategies.recurrent_trainer import (
        RecurrentTrainer,
    )

    # Create a minimal trainer with cosine schedule
    # We need a dummy optimizer for the scheduler
    dummy_model = torch.nn.Linear(4, 2)
    dummy_opt = torch.optim.Adam(dummy_model.parameters(), lr=0.01)

    trainer = RecurrentTrainer(
        task="classification",
        grad_clip=1.0,
        lr_schedule="cosine",
        lr_warmup_epochs=5,
        optimizer=dummy_opt,
        epochs=20,
        batch_size=32,
        device="cpu",
        learning_rate=0.01,
    )
    trainer._build_scheduler(n_epochs=20)
    assert trainer.scheduler is not None

    # Check warmup phase: LR should increase linearly
    lrs = []
    torch_opt = trainer._get_torch_optimizer()
    for _epoch in range(20):
        lrs.append(torch_opt.param_groups[0]["lr"])
        trainer.scheduler.step()

    # Warmup: epochs 0-4, LR ramps from 0.002 to 0.01
    for i in range(5):
        expected_mult = (i + 1) / 5
        assert (
            abs(lrs[i] / 0.01 - expected_mult) < 0.01
        ), f"Warmup epoch {i}: expected mult={expected_mult}, got {lrs[i]/0.01}"

    # Post-warmup: cosine decay
    assert lrs[5] == pytest.approx(0.01, abs=1e-4)  # peak after warmup
    assert lrs[-1] < lrs[5]  # final LR < peak


# ---------------------------------------------------------------------------
# Factory learnable_tau passthrough test
# ---------------------------------------------------------------------------


def test_factory_passes_learnable_tau():
    """Factory should pass learnable_tau to PopulationConfig."""
    params = {
        "architecture": {
            "excitatory_layer_sizes": [32],
            "inhibitory_layer_sizes": [],
            "excitatory_branch_factors": [2],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [8],
            "rec_ee_synapses_per_branch": [4],
        },
        "transfer": {"input_mode": 0},
        "morphology": {"use_shunting": True, "weight_transform": "softplus"},
        "sparsity": {"init_method": "xavier_normal"},
        "recurrent_ei": {
            "tau_base": 5.0,
            "tau_ratio": 5.0,
            "learnable_tau": True,
            "output_mode": "last",
        },
    }
    net = get_architecture("EINet", params, input_dim=16)
    assert isinstance(net, EINetwork)
    e_pop = net.layers[0].e_population
    assert e_pop.learnable_tau is True
    assert e_pop.log_tau is not None
    assert e_pop.log_tau.requires_grad is True

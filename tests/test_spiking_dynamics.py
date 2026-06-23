"""Tests for opt-in spiking soma dynamics."""

import pytest
import torch

from dendritic_modeling.networks.architectures.recurrent import (
    LIFSoma,
    PopulationConfig,
    StatefulDendriNet,
    SurrogateSpike,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    _build_stateful_spiking_soma,
    _resolve_stateful_dendritic_spike_setup,
)


def test_surrogate_spike_has_finite_gradient():
    overdrive = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)

    spikes = SurrogateSpike.apply(overdrive, 10.0)
    spikes.sum().backward()

    assert torch.equal(spikes.detach(), torch.tensor([0.0, 1.0, 1.0]))
    assert torch.isfinite(overdrive.grad).all()
    assert torch.count_nonzero(overdrive.grad) == 3


def test_lif_soma_emits_spike_and_enforces_refractory():
    soma = LIFSoma(
        threshold=0.2,
        reset=0.0,
        tau=1.0,
        dt=1.0,
        refractory_steps=2,
        output_mode="spikes",
    )
    x = torch.ones(2, 3)

    y0, v0, r0, readout0, spikes0 = soma(x, None, None, None)
    y1, v1, r1, _, spikes1 = soma(x, v0, r0, readout0)

    assert torch.equal(y0, torch.ones_like(y0))
    assert torch.equal(spikes0, torch.ones_like(spikes0))
    assert torch.equal(y1, torch.zeros_like(y1))
    assert torch.equal(spikes1, torch.zeros_like(spikes1))
    assert torch.equal(v1, torch.zeros_like(v1))
    assert torch.equal(r1, torch.ones_like(r1))


def _spiking_population_config(readout="spikes"):
    return PopulationConfig(
        n_neurons=4,
        branch_factors=[1],
        ff_excitatory_synapses=2,
        ff_inhibitory_synapses=0,
        rec_excitatory_synapses=1,
        rec_inhibitory_synapses=0,
        reactivate=False,
        somatic_synapses=True,
        dbl_init_method="default",
        weight_transform="exp",
        dynamics_mode="spike",
        spike_threshold=-0.1,
        spike_tau=2.0,
        spike_readout=readout,
        spike_readout_tau=3.0,
    )


def test_build_stateful_spiking_soma_preserves_config_fields():
    cfg = _spiking_population_config(readout="rate")

    soma = _build_stateful_spiking_soma(cfg, dynamics_mode="spike", dt=0.5)
    disabled = _build_stateful_spiking_soma(cfg, dynamics_mode="conductance", dt=0.5)

    assert isinstance(soma, LIFSoma)
    assert soma.threshold == cfg.spike_threshold
    assert soma.reset == cfg.spike_reset
    assert soma.tau == cfg.spike_tau
    assert soma.dt == 0.5
    assert soma.refractory_steps == cfg.spike_refractory_steps
    assert soma.surrogate_beta == cfg.spike_surrogate_beta
    assert soma.output_mode == "rate"
    assert soma.readout_tau == cfg.spike_readout_tau
    assert disabled is None


def test_stateful_dendrinet_spike_mode_outputs_binary_spikes_and_gradients():
    torch.manual_seed(0)
    net = StatefulDendriNet(
        pop_config=_spiking_population_config(),
        excitatory_input_dim=5,
        recurrent_excitatory_input_dim=4,
        dt=1.0,
    )
    x = torch.rand(3, 5)
    recurrent = torch.rand(3, 4)

    y, state = net.step(x, recurrent_input=recurrent)
    y.sum().backward()

    assert y.shape == (3, 4)
    assert torch.equal(y, torch.ones_like(y))
    assert state.v_soma is not None
    assert state.refractory_counter is not None
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in net.parameters()
    )


def test_stateful_dendrinet_rate_decoded_spike_readout_accumulates():
    torch.manual_seed(0)
    net = StatefulDendriNet(
        pop_config=_spiking_population_config(readout="rate"),
        excitatory_input_dim=5,
        recurrent_excitatory_input_dim=4,
        dt=1.0,
    )
    x = torch.rand(2, 5)
    recurrent = torch.rand(2, 4)

    y0, state = net.step(x, recurrent_input=recurrent)
    y1, state = net.step(x, recurrent_input=recurrent, state=state)

    assert y0.shape == (2, 4)
    assert torch.isfinite(y1).all()
    assert (y1 >= y0).all()
    assert state.spike_readout is not None


def _dendritic_spike_population_config(**overrides):
    params = {
        "n_neurons": 4,
        "branch_factors": [1],
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 0,
        "rec_excitatory_synapses": 0,
        "rec_inhibitory_synapses": 0,
        "reactivate": False,
        "somatic_synapses": True,
        "dbl_init_method": "default",
        "weight_transform": "exp",
        "tau_base": 2.0,
        "tau_ratio": 1.0,
        "dendritic_spikes_enabled": True,
        "dendritic_spike_levels": "distal",
        "dendritic_spike_threshold": -1.0,
        "dendritic_spike_plateau_amplitude": 1.0,
        "dendritic_spike_plateau_tau": 2.0,
        "dendritic_spike_propagation": "additive",
    }
    params.update(overrides)
    return PopulationConfig(**params)


def test_resolve_stateful_dendritic_spike_setup_preserves_config_fields():
    cfg = _dendritic_spike_population_config(
        dendritic_spike_levels="proximal",
        dendritic_spike_mode="threshold",
        dendritic_spike_threshold=0.25,
        dendritic_spike_plateau_amplitude=1.5,
        dendritic_spike_plateau_tau=4.0,
        dendritic_spike_refractory_steps=2,
        dendritic_spike_surrogate_beta=7.0,
        dendritic_spike_propagation="replace",
    )
    disabled_cfg = _dendritic_spike_population_config(dendritic_spikes_enabled=False)

    setup = _resolve_stateful_dendritic_spike_setup(cfg, n_levels=3, dt=0.25)
    disabled_setup = _resolve_stateful_dendritic_spike_setup(
        disabled_cfg,
        n_levels=3,
        dt=0.25,
    )

    assert setup.enabled is True
    assert setup.mode == "threshold"
    assert setup.threshold == 0.25
    assert setup.plateau_amplitude == 1.5
    assert setup.plateau_tau == 4.0
    assert setup.refractory_steps == 2
    assert setup.surrogate_beta == 7.0
    assert setup.propagation == "replace"
    assert setup.level_indices == {1}
    assert setup.dynamics.enabled is True
    assert setup.dynamics.level_indices == {1}
    assert setup.dynamics.dt == 0.25
    assert disabled_setup.enabled is False
    assert disabled_setup.level_indices == set()
    assert disabled_setup.dynamics.enabled is False


def test_stateful_dendrinet_dendritic_plateau_changes_branch_output():
    torch.manual_seed(2)
    control_cfg = _dendritic_spike_population_config(dendritic_spikes_enabled=False)
    plateau_cfg = _dendritic_spike_population_config()
    control = StatefulDendriNet(control_cfg, excitatory_input_dim=5, dt=1.0)
    plateau = StatefulDendriNet(plateau_cfg, excitatory_input_dim=5, dt=1.0)
    plateau.load_state_dict(control.state_dict())
    x = torch.rand(3, 5)

    y_control, state_control = control.step(x)
    y_plateau, state_plateau = plateau.step(x)

    assert y_plateau.shape == y_control.shape == (3, 4)
    assert not torch.allclose(y_plateau, y_control)
    assert state_control.dendritic_spike_events is None
    assert state_plateau.branch_voltage is not None
    assert state_plateau.dendritic_spike_events is not None
    assert torch.equal(
        state_plateau.dendritic_spike_events[0],
        torch.ones_like(state_plateau.dendritic_spike_events[0]),
    )
    assert torch.equal(
        state_plateau.dendritic_spike_events[-1],
        torch.zeros_like(state_plateau.dendritic_spike_events[-1]),
    )


def test_dendritic_spike_refractory_blocks_next_local_event():
    torch.manual_seed(3)
    cfg = _dendritic_spike_population_config(
        dendritic_spike_mode="threshold",
        dendritic_spike_refractory_steps=2,
    )
    net = StatefulDendriNet(cfg, excitatory_input_dim=5, dt=1.0)
    x = torch.rand(2, 5)

    _, state = net.step(x)
    _, state = net.step(x, state=state)

    assert state.dendritic_spike_events is not None
    assert torch.equal(
        state.dendritic_spike_events[0],
        torch.zeros_like(state.dendritic_spike_events[0]),
    )


def test_dendritic_spike_levels_reject_soma_index():
    cfg = _dendritic_spike_population_config(dendritic_spike_levels=[1])

    with pytest.raises(ValueError, match="non-soma"):
        StatefulDendriNet(cfg, excitatory_input_dim=5, dt=1.0)

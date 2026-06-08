"""Tests for recurrent synapse-type kinetics."""

import pytest
import torch

from dendritic_modeling.networks.architectures.recurrent import (
    PopulationConfig,
    StatefulDendriNet,
    SynapseTypeConfig,
    build_synapse_type_set,
)


def _population_config(
    synapse_types=None, *, ff_inhibitory_synapses=1, use_shunting=True
):
    return PopulationConfig(
        n_neurons=4,
        branch_factors=[1],
        ff_excitatory_synapses=2,
        ff_inhibitory_synapses=ff_inhibitory_synapses,
        rec_excitatory_synapses=1,
        rec_inhibitory_synapses=1,
        use_shunting=use_shunting,
        reactivate=False,
        somatic_synapses=True,
        dbl_init_method="default",
        weight_transform="exp",
        tau_base=10.0,
        tau_ratio=1.0,
        synapse_types=synapse_types or {},
    )


def _network(config):
    return StatefulDendriNet(
        pop_config=config,
        excitatory_input_dim=5,
        inhibitory_input_dim=3,
        recurrent_excitatory_input_dim=4,
        recurrent_inhibitory_input_dim=2,
        dt=1.0,
    )


def test_synapse_type_set_normalizes_fractions_and_aliases():
    synapses = build_synapse_type_set(
        {
            "enabled": True,
            "excitatory": {
                "AMPA": {"weight": 4.0, "tau": 2.0},
                "NMDA": {"weight": 1.0, "tau": 70.0},
            },
            "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
        }
    )

    assert synapses.enabled
    assert [syn.name for syn in synapses.excitatory.types] == ["AMPA", "NMDA"]
    assert synapses.excitatory.types[0].fraction == 0.8
    assert synapses.excitatory.types[1].fraction == 0.2
    assert synapses.inhibitory.types[0].name == "GABA_A"


def test_synapse_type_set_accepts_biophysical_aliases():
    synapses = build_synapse_type_set(
        {
            "enabled": True,
            "excitatory": [
                {
                    "name": "NMDA",
                    "tau_r": 2.0,
                    "tau_d": 80.0,
                    "mg_block": True,
                    "short_term_plasticity": {
                        "enabled": True,
                        "u0": 0.25,
                        "depression": 0.5,
                        "differentiable_state": True,
                    },
                }
            ],
            "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
        }
    )

    nmda = synapses.excitatory.types[0]
    assert nmda.tau_rise == 2.0
    assert nmda.tau_decay == 80.0
    assert nmda.magnesium_block
    assert nmda.voltage_gate == "nmda_magnesium"
    assert nmda.stp_enabled
    assert nmda.stp_u0 == 0.25
    assert nmda.stp_depression == 0.5
    assert nmda.stp_differentiable_state


def test_synapse_types_enabled_requires_explicit_polarity_lists():
    with pytest.raises(ValueError, match="requires explicit"):
        build_synapse_type_set({"enabled": True})
    with pytest.raises(ValueError, match="inhibitory"):
        build_synapse_type_set(
            {
                "enabled": True,
                "excitatory": [{"name": "AMPA", "fraction": 1.0, "tau": None}],
            }
        )


def test_synapse_types_enabled_rejects_all_disabled_entries():
    with pytest.raises(ValueError, match="enabled entry"):
        build_synapse_type_set(
            {
                "enabled": True,
                "excitatory": [
                    {"name": "AMPA", "fraction": 1.0, "tau": None, "enabled": False}
                ],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
            }
        )


def test_synapse_type_reversal_potential_is_bounded():
    with pytest.raises(ValueError, match="reversal_potential"):
        SynapseTypeConfig(name="too_large", reversal_potential=2.1)
    with pytest.raises(ValueError, match="reversal_potential"):
        SynapseTypeConfig(name="too_negative", reversal_potential=-1.1)


def test_fractional_conductance_power_has_finite_zero_gradient():
    net = _network(_population_config())
    raw = torch.zeros(2, 4, requires_grad=True)
    synapse_type = SynapseTypeConfig(name="fractional", conductance_power=0.5)

    out = net._apply_synapse_nonlinearity(raw, synapse_type)
    assert torch.isfinite(out).all()
    out.sum().backward()

    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_legacy_equivalent_synapse_types_match_legacy_step():
    torch.manual_seed(0)
    legacy = _network(_population_config())
    typed = _network(
        _population_config(
            {
                "enabled": True,
                "excitatory": [{"name": "AMPA", "fraction": 1.0, "tau": None}],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
            }
        )
    )
    typed.load_state_dict(legacy.state_dict())

    state_legacy = None
    state_typed = None
    max_abs_diff = 0.0
    for _ in range(100):
        x = torch.rand(3, 5)
        inhibitory = torch.rand(3, 3)
        rec_exc = torch.rand(3, 4)
        rec_inh = torch.rand(3, 2)
        y_legacy, state_legacy = legacy.step(
            x,
            inhibitory_input=inhibitory,
            recurrent_input=rec_exc,
            rec_inhibitory_input=rec_inh,
            state=state_legacy,
        )
        y_typed, state_typed = typed.step(
            x,
            inhibitory_input=inhibitory,
            recurrent_input=rec_exc,
            rec_inhibitory_input=rec_inh,
            state=state_typed,
        )
        max_abs_diff = max(max_abs_diff, (y_typed - y_legacy).abs().max().item())

    assert max_abs_diff < 1e-5
    assert torch.allclose(state_typed.trace_E[0], state_legacy.trace_E[0], atol=1e-6)
    assert torch.allclose(state_typed.trace_I[0], state_legacy.trace_I[0], atol=1e-6)
    assert state_typed.typed_traces is not None
    assert state_legacy.branch_voltage is None
    assert state_typed.branch_voltage is not None


def test_additive_legacy_equivalent_synapse_types_match_legacy_step():
    torch.manual_seed(1)
    legacy = _network(_population_config(use_shunting=False))
    typed = _network(
        _population_config(
            {
                "enabled": True,
                "excitatory": [
                    {
                        "name": "AMPA",
                        "fraction": 1.0,
                        "tau": None,
                        "reversal_potential": 1.0,
                    }
                ],
                "inhibitory": [
                    {
                        "name": "GABA_A",
                        "fraction": 1.0,
                        "tau": None,
                        "reversal_potential": 0.0,
                    }
                ],
            },
            use_shunting=False,
        )
    )
    typed.load_state_dict(legacy.state_dict())

    state_legacy = None
    state_typed = None
    max_abs_diff = 0.0
    for _ in range(100):
        x = torch.rand(3, 5)
        inhibitory = torch.rand(3, 3)
        rec_exc = torch.rand(3, 4)
        rec_inh = torch.rand(3, 2)
        y_legacy, state_legacy = legacy.step(
            x,
            inhibitory_input=inhibitory,
            recurrent_input=rec_exc,
            rec_inhibitory_input=rec_inh,
            state=state_legacy,
        )
        y_typed, state_typed = typed.step(
            x,
            inhibitory_input=inhibitory,
            recurrent_input=rec_exc,
            rec_inhibitory_input=rec_inh,
            state=state_typed,
        )
        max_abs_diff = max(max_abs_diff, (y_typed - y_legacy).abs().max().item())

    assert max_abs_diff < 1e-5
    assert torch.allclose(state_typed.trace_E[0], state_legacy.trace_E[0], atol=1e-6)
    assert torch.allclose(state_typed.trace_I[0], state_legacy.trace_I[0], atol=1e-6)
    assert state_typed.typed_traces is not None


def test_legacy_state_does_not_allocate_branch_voltage_cache():
    legacy = _network(_population_config())
    typed = _network(
        _population_config(
            {
                "enabled": True,
                "excitatory": [{"name": "AMPA", "fraction": 1.0, "tau": None}],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
            }
        )
    )

    legacy_state = legacy.init_state(2, torch.device("cpu"))
    typed_state = typed.init_state(2, torch.device("cpu"))

    assert legacy_state.branch_voltage is None
    assert typed_state.branch_voltage is not None


def test_synapse_type_tau_controls_trace_rise():
    net = _network(
        _population_config(
            {
                "enabled": True,
                "excitatory": [
                    {"name": "fast", "fraction": 1.0, "tau": 1.0},
                    {"name": "slow", "fraction": 1.0, "tau": 100.0},
                ],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
            },
            ff_inhibitory_synapses=0,
        )
    )

    _, state = net.step(torch.ones(2, 5))

    assert state.typed_traces is not None
    fast = state.typed_traces["ff_exc:fast"][0]
    slow = state.typed_traces["ff_exc:slow"][0]
    assert fast.mean() > slow.mean()


def test_double_exponential_synapse_keeps_rise_and_decay_traces():
    torch.manual_seed(2)
    net = _network(
        _population_config(
            {
                "enabled": True,
                "excitatory": [
                    {
                        "name": "AMPA",
                        "fraction": 1.0,
                        "tau_rise": 1.0,
                        "tau_decay": 20.0,
                    }
                ],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
            },
            ff_inhibitory_synapses=0,
        )
    )

    _, state = net.step(torch.ones(2, 5))
    _, state = net.step(torch.zeros(2, 5), state=state)

    assert state.typed_traces is not None
    assert "ff_exc:AMPA:rise" in state.typed_traces
    assert "ff_exc:AMPA:decay" in state.typed_traces
    assert "ff_exc:AMPA" in state.typed_traces
    assert state.typed_traces["ff_exc:AMPA"][0].mean() > 0


def test_nmda_magnesium_block_is_voltage_dependent():
    torch.manual_seed(3)
    net = _network(
        _population_config(
            {
                "enabled": True,
                "excitatory": [
                    {
                        "name": "NMDA",
                        "fraction": 1.0,
                        "tau": None,
                        "magnesium_block": True,
                    }
                ],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
            },
            ff_inhibitory_synapses=0,
        )
    )
    low_state = net.init_state(2, torch.device("cpu"))
    high_state = net.init_state(2, torch.device("cpu"))
    assert low_state.branch_voltage is not None
    assert high_state.branch_voltage is not None
    for trace in low_state.branch_voltage:
        trace.fill_(-10.0)
    for trace in high_state.branch_voltage:
        trace.fill_(10.0)

    _, low_state = net.step(torch.ones(2, 5), state=low_state)
    _, high_state = net.step(torch.ones(2, 5), state=high_state)

    low_nmda = low_state.typed_traces["ff_exc:NMDA"][0]
    high_nmda = high_state.typed_traces["ff_exc:NMDA"][0]
    assert high_nmda.mean() > low_nmda.mean()


def test_nmda_magnesium_block_is_sensitive_on_unit_voltage_range():
    net = _network(_population_config())
    synapse_type = SynapseTypeConfig(name="NMDA", magnesium_block=True)
    trace = torch.ones(2, 4)

    low = net._apply_voltage_gate(trace, synapse_type, torch.zeros_like(trace))
    high = net._apply_voltage_gate(trace, synapse_type, torch.ones_like(trace))

    assert (high - low).mean() > 0.15


def test_additive_synapse_types_reject_non_normalized_reversal_potentials():
    with pytest.raises(ValueError, match="use_shunting=true"):
        _network(
            PopulationConfig(
                n_neurons=4,
                branch_factors=[1],
                ff_excitatory_synapses=2,
                ff_inhibitory_synapses=1,
                rec_excitatory_synapses=0,
                rec_inhibitory_synapses=0,
                use_shunting=False,
                reactivate=False,
                somatic_synapses=True,
                dbl_init_method="default",
                synapse_types={
                    "enabled": True,
                    "excitatory": [
                        {
                            "name": "slow_E",
                            "fraction": 1.0,
                            "tau": None,
                            "reversal_potential": 0.6,
                        }
                    ],
                    "inhibitory": [
                        {
                            "name": "GABA_B",
                            "fraction": 1.0,
                            "tau": None,
                            "reversal_potential": -0.25,
                        }
                    ],
                },
            )
        )


def test_short_term_plasticity_tracks_utilization_and_resources():
    torch.manual_seed(4)
    net = _network(
        _population_config(
            {
                "enabled": True,
                "excitatory": [
                    {
                        "name": "STP",
                        "fraction": 1.0,
                        "tau": None,
                        "stp": {
                            "enabled": True,
                            "u0": 0.3,
                            "facilitation": 0.2,
                            "depression": 0.5,
                        },
                    }
                ],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
            },
            ff_inhibitory_synapses=0,
        )
    )

    _, state = net.step(torch.ones(2, 5))

    assert state.typed_traces is not None
    assert "ff_exc:STP:stp_u" in state.typed_traces
    assert "ff_exc:STP:stp_x" in state.typed_traces
    utilization = state.typed_traces["ff_exc:STP:stp_u"][0]
    resources = state.typed_traces["ff_exc:STP:stp_x"][0]
    assert utilization.min() >= 0
    assert utilization.max() <= 1
    assert resources.min() >= 0
    assert resources.max() <= 1


def test_short_term_plasticity_differentiable_state_controls_history_gradient():
    net = _network(_population_config())
    raw_first = torch.ones(2, 4, requires_grad=True)
    raw_second = torch.ones(2, 4, requires_grad=True)

    fixed_state_type = SynapseTypeConfig(
        name="STP",
        stp_enabled=True,
        stp_u0=0.3,
        stp_facilitation=0.4,
        stp_depression=0.5,
        stp_differentiable_state=False,
    )
    differentiable_type = SynapseTypeConfig(
        name="STP",
        stp_enabled=True,
        stp_u0=0.3,
        stp_facilitation=0.4,
        stp_depression=0.5,
        stp_differentiable_state=True,
    )

    fixed_traces = {}
    net._apply_short_term_plasticity(
        raw=raw_first,
        prev_typed_traces=None,
        new_typed_traces=fixed_traces,
        key="ff_exc:STP",
        level_idx=0,
        synapse_type=fixed_state_type,
    )
    fixed_next = {}
    fixed_output = net._apply_short_term_plasticity(
        raw=raw_second,
        prev_typed_traces=fixed_traces,
        new_typed_traces=fixed_next,
        key="ff_exc:STP",
        level_idx=0,
        synapse_type=fixed_state_type,
    )
    fixed_output.sum().backward()
    assert raw_first.grad is None

    raw_first.grad = None
    raw_second.grad = None
    differentiable_traces = {}
    net._apply_short_term_plasticity(
        raw=raw_first,
        prev_typed_traces=None,
        new_typed_traces=differentiable_traces,
        key="ff_exc:STP",
        level_idx=0,
        synapse_type=differentiable_type,
    )
    differentiable_next = {}
    differentiable_output = net._apply_short_term_plasticity(
        raw=raw_second,
        prev_typed_traces=differentiable_traces,
        new_typed_traces=differentiable_next,
        key="ff_exc:STP",
        level_idx=0,
        synapse_type=differentiable_type,
    )
    differentiable_output.sum().backward()
    assert raw_first.grad is not None
    assert raw_first.grad.abs().sum() > 0


def test_inhibitory_reversal_potential_changes_shunting_output():
    torch.manual_seed(1)
    base_types = {
        "enabled": True,
        "excitatory": [{"name": "AMPA", "fraction": 1.0, "tau": None}],
        "inhibitory": [
            {
                "name": "GABA_A",
                "fraction": 1.0,
                "tau": None,
                "reversal_potential": 0.0,
            }
        ],
    }
    hyperpolarizing_types = {
        **base_types,
        "inhibitory": [
            {
                "name": "GABA_A",
                "fraction": 1.0,
                "tau": None,
                "reversal_potential": -1.0,
            }
        ],
    }
    base = _network(_population_config(base_types))
    hyperpolarizing = _network(_population_config(hyperpolarizing_types))
    hyperpolarizing.load_state_dict(base.state_dict())

    x = torch.ones(2, 5)
    inhibitory = torch.ones(2, 3)
    y_base, _ = base.step(x, inhibitory_input=inhibitory)
    y_hyperpolarizing, _ = hyperpolarizing.step(x, inhibitory_input=inhibitory)

    assert (y_hyperpolarizing < y_base).all()

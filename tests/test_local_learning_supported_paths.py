"""LocalCA labels populations from the model and rejects paths it cannot model."""

from __future__ import annotations

import pytest
import torch
from test_biological_constraints import _einet, _population
from test_trainer_configuration import IN_DIM, _data, _make_trainer, _state, _wrap

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.structure import inhibitory_branch_layer_ids


def _model(core_type, core):
    torch.manual_seed(0)
    return _wrap(
        get_architecture(
            core_type, {**core, "biological_neuron": False}, input_dim=IN_DIM
        )
    )


def _inhibitory_names(model):
    ids = inhibitory_branch_layer_ids(model)
    return {name for name, module in model.named_modules() if id(module) in ids}


@pytest.mark.parametrize(
    "core_type, core, prefix",
    [
        ("einet", _einet(), "core_network.layers.0.inhibitory_cells."),
        ("population_network", _population(), "core_network.layers.0.populations.i."),
    ],
)
def test_inhibitory_populations_are_found_from_the_model(core_type, core, prefix):
    model = _model(core_type, core)
    names = _inhibitory_names(model)
    branch_layers = {
        name
        for name, module in model.named_modules()
        if isinstance(module, DendriticBranchLayer)
    }
    assert names and names == {
        name for name in branch_layers if name.startswith(prefix)
    }


def test_freeze_keeps_the_population_network_inhibitory_population_fixed():
    model = _model("population_network", _population())
    initial = _state(model)
    trainer, _ = _make_trainer(
        "local_ca",
        model,
        epochs=1,
        local_rule_config={"explicit_inhibitory_update_mode": "freeze"},
    )
    trainer.train(model, _data(), _data(seed=1))

    final = _state(model)
    inhibitory = [name for name in initial if ".populations.i." in name]
    excitatory = [name for name in initial if ".populations.e." in name]
    assert inhibitory and excitatory
    assert all(torch.equal(initial[name], final[name]) for name in inhibitory)
    assert any(not torch.equal(initial[name], final[name]) for name in excitatory)


def test_freeze_without_an_inhibitory_population_raises():
    core = _population(
        connections=[{"source": "input", "target": "e"}],
    )
    core["population_network"]["layers"][0]["populations"].pop()
    model = _model("population_network", core)
    trainer, _ = _make_trainer(
        "local_ca",
        model,
        local_rule_config={"explicit_inhibitory_update_mode": "freeze"},
    )
    with pytest.raises(ValueError, match="no inhibitory population to freeze"):
        trainer.train(model, _data(), _data(seed=1))


@pytest.mark.parametrize(
    "defaults, feature",
    [
        ({"soma_feedback_enabled": True}, "soma feedback"),
        ({"cross_level_mode": "parallel_readout"}, "parallel_readout"),
        ({"cross_level_mode": "all_active_star"}, "all_active_star"),
    ],
)
def test_forward_paths_the_local_rules_do_not_model_raise(defaults, feature):
    model = _model("population_network", _population(defaults=defaults))
    trainer, _ = _make_trainer("local_ca", model, local_rule_config={})
    with pytest.raises(ValueError, match=feature):
        trainer.train(model, _data(), _data(seed=1))


def test_unmodeled_forward_features_can_be_allowed_as_a_rule_variant(caplog):
    defaults = {"cross_level_mode": "parallel_readout"}
    model = _model("population_network", _population(defaults=defaults))
    trainer, _ = _make_trainer(
        "local_ca",
        model,
        epochs=1,
        local_rule_config={"allow_unmodeled_forward_features": True},
    )
    initial = _state(model)
    with caplog.at_level("WARNING"):
        trainer.train(model, _data(), _data(seed=1))
    assert "allow_unmodeled_forward_features" in caplog.text
    assert any(not torch.equal(initial[k], v) for k, v in _state(model).items())


def test_parameters_the_local_rules_never_update_are_reported(caplog):
    import logging

    model = _model("population_network", _population())
    model.unused_gain = torch.nn.Parameter(torch.ones(2))
    trainer, _ = _make_trainer("local_ca", model, epochs=1, local_rule_config={})
    with caplog.at_level(logging.WARNING):
        trainer.train(model, _data(), _data(seed=1))
    assert "unused_gain" in trainer.uncovered_parameters
    assert "receive no gradient from the local rules" in caplog.text


def _equal_width_einet():
    """E and I somas of equal width, where width matching misrouted errors."""
    core = _einet()
    core["architecture"]["inhibitory_layer_sizes"] = [6]
    return _model("einet", core)


def test_only_the_readout_tree_receives_per_soma_errors():
    from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_mixin import (
        _records_by_tree,
    )

    model = _equal_width_einet()
    trainer, _ = _make_trainer("local_ca", model, local_rule_config={})
    records, handles = trainer._attach_local_recorders(model)
    try:
        model(torch.rand(4, IN_DIM))
    finally:
        trainer.remove_forward_hooks(handles)
    groups, readout = _records_by_tree(model, records)
    readout_names = {records[index]["module_name"] for index in groups[readout]}
    assert readout_names and all(".excitatory_cells." in name for name in readout_names)
    assert any(".inhibitory_cells." in rec["module_name"] for rec in records)


@pytest.mark.parametrize("mode", ["per_soma_shared", "path_transport"])
def test_per_soma_and_path_modes_train_models_with_inhibitory_cells(mode):
    model = _equal_width_einet()
    initial = _state(model)
    trainer, _ = _make_trainer(
        "local_ca", model, epochs=1, local_rule_config={"error_broadcast_mode": mode}
    )
    trainer.train(model, _data(), _data(seed=1))
    final = _state(model)
    assert any(not torch.equal(initial[name], final[name]) for name in initial)


def test_row_weight_normalization_is_rejected():
    core = _einet()
    core["sparsity"]["weight_norm_order"] = 2
    model = _model("einet", core)
    trainer, _ = _make_trainer("local_ca", model, local_rule_config={})
    with pytest.raises(ValueError, match="weight_norm_order"):
        trainer.train(model, _data(), _data(seed=1))


def test_stdp_on_a_compact_indexed_bank_is_rejected():
    core = _einet()
    core["sparsity"]["type"] = "indexed"
    model = _model("einet", core)
    trainer, _ = _make_trainer(
        "local_ca", model, local_rule_config={"rule_variant": "stdp"}
    )
    with pytest.raises(ValueError, match="STDP updates TopK synapse banks"):
        trainer.train(model, _data(), _data(seed=1))


@pytest.mark.parametrize("reversals", [{"e_rev_exc": 1.5}, {"e_rev_inh": -0.2}])
def test_reversal_potentials_other_than_the_models_are_rejected(reversals):
    from dendritic_modeling.config.training import ThreeFactorConfig

    with pytest.raises(ValueError, match="reversal potentials"):
        ThreeFactorConfig(**reversals)


def test_a_forced_dynamics_mode_must_match_every_layer():
    model = _model("einet", _einet())
    trainer, _ = _make_trainer(
        "local_ca",
        model,
        local_rule_config={"three_factor": {"dynamics_mode": "additive"}},
    )
    with pytest.raises(ValueError, match="integrates by shunting"):
        trainer.train(model, _data(), _data(seed=1))


@pytest.mark.parametrize(
    "core_type, core",
    [("einet", _einet()), ("population_network", _population())],
)
def test_total_conductance_is_the_forward_denominator(core_type, core):
    _assert_total_conductance_matches_forward(
        _model(core_type, core), torch.rand(4, IN_DIM)
    )


def test_total_conductance_keeps_negative_conductance_terms():
    """Signed presynaptic activity gives negative terms the forward keeps."""
    torch.manual_seed(0)
    layer = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=3,
        excitatory_synapses_per_branch=2,
        inhibitory_input_dim=3,
        inhibitory_synapses_per_branch=2,
        use_shunting=True,
        weight_transform="softplus",
    )
    x = torch.tensor([[1.0, -2.0, 0.5], [-1.5, 0.3, -0.2]])
    records = _assert_total_conductance_matches_forward(
        layer, x, inhibitory_input=x.flip(0)
    )
    assert (records[0]["exc_out"] < 0).any()


def _assert_total_conductance_matches_forward(model, x, **forward_kwargs):
    from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast import (
        compute_layer_total_conductance,
    )

    trainer, _ = _make_trainer("local_ca", model, local_rule_config={})
    for module in model.modules():
        if isinstance(module, DendriticBranchLayer):
            module._store_diagnostics = True
    records, handles = trainer._attach_local_recorders(model)
    try:
        model(x, **forward_kwargs)
    finally:
        trainer.remove_forward_hooks(handles)
    for rec in records:
        g_tot = compute_layer_total_conductance(rec, rec["v_n"])
        torch.testing.assert_close(g_tot, rec["layer"]._diag_g_tot.expand_as(g_tot))
    return records


def test_a_mismatched_dynamics_mode_is_an_explicit_rule_variant():
    model = _model("einet", _einet())
    trainer, _ = _make_trainer(
        "local_ca",
        model,
        epochs=1,
        local_rule_config={
            "three_factor": {
                "dynamics_mode": "additive",
                "allow_mismatched_dynamics_mode": True,
            }
        },
    )
    initial = _state(model)
    trainer.train(model, _data(), _data(seed=1))
    assert trainer._resolve_layer_dynamics_mode({"layer": None}) == "additive"
    assert any(not torch.equal(initial[k], v) for k, v in _state(model).items())


def test_allowing_a_mismatch_needs_a_forced_mode():
    from dendritic_modeling.config.training import ThreeFactorConfig

    with pytest.raises(ValueError, match="needs dynamics_mode"):
        ThreeFactorConfig(allow_mismatched_dynamics_mode=True)

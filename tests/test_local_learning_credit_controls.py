"""LocalCA path-transport projections and the frozen-core update mode."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from test_biological_constraints import _einet
from test_local_learning_supported_paths import _model
from test_trainer_configuration import (
    IN_DIM,
    _data,
    _einet_model,
    _make_trainer,
    _optimizer,
    _trainer_kwargs_from_typed,
    _typed_config,
)

from dendritic_modeling.config.training import LocalRuleConfig
from dendritic_modeling.training.factory import get_trainer
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_mixin import (
    _records_by_tree,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_transport_mixin import (
    _project_transported_errors,
)

PATH = {"error_broadcast_mode": "path_transport"}


def _records(widths, block_sizes):
    """Readout-tree records from distal to soma with the given block sizes."""
    records = [{"v_n": torch.zeros(1, widths[0])}]
    for width, block_size in zip(widths[1:], block_sizes):
        blk_module = SimpleNamespace(block_size=block_size)
        records.append({"v_n": torch.zeros(1, width), "blk_module": blk_module})
    return records


def _field(widths, batch=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, width, generator=generator) for width in widths]


# ----------------------------------------------------------------------------
# The projection
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("projection", ["arbor_mean", "subtree_mean"])
def test_projection_replaces_errors_by_group_means(projection):
    # Two neurons, three children per soma and two per proximal compartment.
    widths = [12, 6, 2]
    exact = _field(widths)
    projected = _project_transported_errors(_records(widths, [2, 3]), exact, projection)

    distal, proximal, soma = exact
    if projection == "arbor_mean":
        groups = [distal.reshape(5, 2, 6), proximal.reshape(5, 2, 3)]
        mean = torch.cat(groups, dim=-1).mean(dim=-1)
        expected = [mean.repeat_interleave(6, dim=1), mean.repeat_interleave(3, dim=1)]
    else:
        groups = [distal.reshape(5, 6, 2), proximal.reshape(5, 6, 1)]
        mean = torch.cat(groups, dim=-1).mean(dim=-1)
        expected = [mean.repeat_interleave(2, dim=1), mean]
    torch.testing.assert_close(projected[0], expected[0])
    torch.testing.assert_close(projected[1], expected[1])
    assert projected[2] is soma


@pytest.mark.parametrize("projection", ["arbor_mean", "subtree_mean"])
def test_projection_keeps_each_neurons_summed_error(projection):
    widths = [12, 6, 2]
    exact = _field(widths, seed=1)
    projected = _project_transported_errors(_records(widths, [2, 3]), exact, projection)

    def per_neuron(field):
        return field[0].reshape(5, 2, 6).sum(-1) + field[1].reshape(5, 2, 3).sum(-1)

    torch.testing.assert_close(per_neuron(projected), per_neuron(exact))


def _three_column_dictionary(distal, proximal, soma):
    b, n = soma.shape
    coefficients = (proximal.reshape(b, n, 3) + distal.reshape(b, n, 3, 3).sum(-1)) / 4
    return [
        coefficients.unsqueeze(-1).expand(-1, -1, -1, 3).reshape_as(distal),
        coefficients.reshape_as(proximal),
        soma,
    ]


def _one_column_dictionary(distal, proximal, soma):
    b, n = soma.shape
    coefficient = (
        distal.reshape(b, n, 9).sum(-1) + proximal.reshape(b, n, 3).sum(-1)
    ) / 12
    return [
        coefficient.unsqueeze(-1).expand(-1, -1, 9).reshape_as(distal),
        coefficient.unsqueeze(-1).expand(-1, -1, 3).reshape_as(proximal),
        soma,
    ]


@pytest.mark.parametrize(
    "projection, dictionary",
    [
        ("subtree_mean", _three_column_dictionary),
        ("arbor_mean", _one_column_dictionary),
    ],
)
def test_projection_equals_the_written_out_dictionaries_of_a_three_by_three_tree(
    projection, dictionary
):
    widths = [9 * 7, 3 * 7, 7]
    exact = _field(widths, batch=11, seed=2)
    projected = _project_transported_errors(_records(widths, [3, 3]), exact, projection)
    for value, expected in zip(projected, dictionary(*exact)):
        assert torch.equal(value, expected)


def test_projection_needs_a_transported_error_at_every_compartment():
    widths = [12, 6, 2]
    exact = _field(widths)
    exact[0] = None
    with pytest.raises(ValueError, match="exact transported error"):
        _project_transported_errors(_records(widths, [2, 3]), exact, "arbor_mean")


def test_projection_rejects_layers_their_blocks_do_not_tile():
    widths = [10, 6, 2]
    with pytest.raises(ValueError, match="children per parent"):
        _project_transported_errors(
            _records(widths, [2, 3]), _field(widths), "subtree_mean"
        )


def test_projection_needs_dendritic_compartments():
    with pytest.raises(ValueError, match="records only its soma"):
        _project_transported_errors(_records([4], []), _field([4]), "arbor_mean")


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


def test_defaults_keep_the_exact_field_and_local_core_updates():
    config = LocalRuleConfig()
    assert config.path_transport_projection == "none"
    assert config.core_update_mode == "local"


def test_projection_requires_path_transport():
    with pytest.raises(ValueError, match="error_broadcast_mode: path_transport"):
        LocalRuleConfig(path_transport_projection="subtree_mean")
    config = LocalRuleConfig(**PATH, path_transport_projection="Subtree-Mean")
    assert config.path_transport_projection == "subtree_mean"


@pytest.mark.parametrize("field", ["path_transport_projection", "core_update_mode"])
def test_unknown_values_raise(field):
    with pytest.raises(ValueError, match=field):
        LocalRuleConfig(**PATH, **{field: "median"})


def test_a_local_run_that_trains_nothing_raises():
    with pytest.raises(ValueError, match="would train no parameter"):
        LocalRuleConfig(core_update_mode="none", decoder_update_mode="none")
    LocalRuleConfig(
        core_update_mode="none",
        decoder_update_mode="none",
        training_schedule=[{"start_epoch": 2, "decoder_update_mode": "local"}],
    )


@pytest.mark.parametrize("strategy", ["local_ca", "multi_stage"])
def test_the_options_reach_the_local_trainer_from_a_typed_config(tmp_path, strategy):
    from dendritic_modeling.training.strategies import multi_stage

    block = {
        **PATH,
        "path_transport_projection": "arbor_mean",
        "core_update_mode": "none",
    }
    main = {"strategy": strategy, "learning_strategy_config": block}
    if strategy == "multi_stage":
        stages = [{"learning_strategy": "local_ca", "epochs": 1}]
        main["strategies"] = {"multi_stage": {"stages": stages}}
    config = _typed_config(tmp_path, main)
    loaded = config.training.main.learning_strategy_config
    assert loaded.path_transport_projection == "arbor_mean"
    assert loaded.core_update_mode == "none"

    model = _einet_model()
    kwargs = _trainer_kwargs_from_typed(config, tmp_path)
    trainer = get_trainer(strategy, _optimizer(model), kwargs)
    if strategy == "multi_stage":
        stage = trainer.multi_stage_config.stages[0]
        stage_config = multi_stage._build_stage_trainer_config(
            trainer.base_trainer_config, trainer.multi_stage_config, stage
        )
        local = stage_config["local_rule_config"]
        assert local["path_transport_projection"] == "arbor_mean"
        assert local["core_update_mode"] == "none"
    else:
        assert trainer.local_cfg.path_transport_projection == "arbor_mean"
        assert trainer.local_cfg.core_update_mode == "none"


# ----------------------------------------------------------------------------
# Training a two-level tree
# ----------------------------------------------------------------------------


def _tree_model():
    return _model("einet", _einet(**{"architecture.excitatory_branch_factors": [2, 2]}))


def _parameters(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def _readout_fields(trainer, model, x, delta):
    records, handles = trainer._attach_local_recorders(model)
    try:
        model(x)
    finally:
        trainer.remove_forward_hooks(handles)
    state = trainer._prepare_local_rule_broadcast_state(
        model=model,
        layer_records=records,
        delta=delta,
        delta_scalar=trainer._reduce_error_to_scalar(delta),
    )
    groups, readout = _records_by_tree(model, records)
    indices = groups[readout]
    return [records[i] for i in indices], [state.transported_errors[i] for i in indices]


@pytest.mark.parametrize("projection", ["arbor_mean", "subtree_mean"])
def test_the_trainer_delivers_the_projected_exact_field(projection):
    model = _tree_model()
    exact_trainer, _ = _make_trainer("local_ca", model, local_rule_config=PATH)
    projected_trainer, _ = _make_trainer(
        "local_ca",
        model,
        local_rule_config={**PATH, "path_transport_projection": projection},
    )
    generator = torch.Generator().manual_seed(3)
    x = torch.rand(4, IN_DIM, generator=generator)
    delta = torch.randn(4, 6, generator=generator)

    records, exact = _readout_fields(exact_trainer, model, x, delta)
    _, projected = _readout_fields(projected_trainer, model, x, delta)

    assert len(records) == 3
    expected = _project_transported_errors(records, exact, projection)
    for value, reference in zip(projected, expected):
        torch.testing.assert_close(value, reference)
    assert not torch.equal(projected[0], exact[0])


def _train(local_rule_config):
    model = _tree_model()
    trainer, _ = _make_trainer(
        "local_ca", model, epochs=1, local_rule_config=local_rule_config
    )
    trainer.train(model, _data(), _data(seed=1))
    return model


def test_explicit_defaults_train_identically():
    implicit = _train(PATH).state_dict()
    explicit = _train(
        {**PATH, "path_transport_projection": "none", "core_update_mode": "local"}
    ).state_dict()
    assert implicit.keys() == explicit.keys()
    assert all(torch.equal(implicit[name], explicit[name]) for name in implicit)


def test_a_projection_changes_the_trained_core():
    exact = _parameters(_train(PATH))
    projected = _parameters(
        _train({**PATH, "path_transport_projection": "subtree_mean"})
    )
    core = [name for name in exact if name.startswith("core_network.")]
    assert core
    assert any(not torch.equal(exact[name], projected[name]) for name in core)


@pytest.mark.parametrize("decoder_update_mode", ["backprop", "local"])
def test_core_update_mode_none_trains_only_the_decoder(decoder_update_mode):
    model = _tree_model()
    initial = _parameters(model)
    trainer, _ = _make_trainer(
        "local_ca",
        model,
        epochs=1,
        local_rule_config={
            "core_update_mode": "none",
            "decoder_update_mode": decoder_update_mode,
        },
    )
    trainer.train(model, _data(), _data(seed=1))
    final = _parameters(model)

    core = [name for name in initial if name.startswith("core_network.")]
    decoder = [name for name in initial if name.startswith("decoder_network.")]
    assert core and decoder
    assert all(torch.equal(initial[name], final[name]) for name in core)
    assert any(not torch.equal(initial[name], final[name]) for name in decoder)
    assert set(core) <= set(trainer.uncovered_parameters)

"""LocalCA and the across-output z-score of use_additive_normalization."""

from __future__ import annotations

import logging

import pytest
import torch
from test_biological_constraints import _einet
from test_local_learning_additive_fairness import (
    _capture_record,
    _make_trainer as _layer_trainer,
    _run_local_update,
    _SingleBranchModel,
)
from test_local_learning_supported_paths import _model
from test_trainer_configuration import (
    IN_DIM,
    _make_trainer,
    _trainer_kwargs_from_typed,
    _typed_config,
)

from dendritic_modeling.training.factory import get_trainer
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast import (
    additive_zscore_vjp,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_mixin import (
    _records_by_tree,
)

EXACT = "normalization_jacobian"


def _zscore(voltage, epsilon):
    centered = voltage - voltage.mean(dim=-1, keepdim=True)
    std = voltage.std(dim=-1, keepdim=True, unbiased=False)
    return centered / std.clamp_min(epsilon), std


def test_the_vector_jacobian_product_is_the_z_score_derivative():
    generator = torch.Generator().manual_seed(0)
    voltage = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    voltage[3] = 0.7  # constant row: the scale is clamped
    voltage.requires_grad_(True)
    error = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    z, std = _zscore(voltage, 1e-6)
    (reference,) = torch.autograd.grad((z * error).sum(), voltage)
    pulled = additive_zscore_vjp(error, z.detach(), std.detach(), 1e-6)
    # The clamped row is divided by epsilon, so compare relative errors.
    torch.testing.assert_close(pulled, reference, rtol=1e-12, atol=1e-12)


def _layer_update(normalized, mode):
    torch.manual_seed(18)
    model = _SingleBranchModel(use_shunting=False).double()
    model.branch.use_additive_normalization = normalized
    trainer = _layer_trainer(dynamics_mode="auto")
    trainer.local_cfg.update_inactive_weights = False
    trainer.local_cfg.three_factor.additive_gain_mode = mode
    x = torch.tensor(
        [[0.2, 0.5, 0.9, 0.1], [0.7, 0.1, 0.3, 0.6], [0.4, 0.8, 0.6, 0.2]],
        dtype=torch.float64,
    )
    delta = torch.tensor([[0.1, -0.2], [-0.4, 0.3], [0.2, 0.1]], dtype=torch.float64)
    rec = _capture_record(trainer, model, x)
    branch = model.branch
    params = (
        branch.branch_excitation.pre_w,
        branch.branch_inhibition.pre_w,
        branch.branches_to_output.log_weight,
    )
    reference = torch.autograd.grad((model(x) * delta).sum() / len(x), params)
    actual = _run_local_update(trainer, model, rec, delta)
    return max(float((a - b).abs().max()) for a, b in zip(actual, reference)), rec


def test_the_exact_mode_gives_the_gradient_of_a_z_scored_layer():
    error, rec = _layer_update(True, EXACT)
    assert error < 1e-12
    assert "additive_scale" in rec


def test_omitting_the_jacobian_is_an_approximation_on_a_z_scored_layer():
    error, rec = _layer_update(True, "none")
    assert error > 1e-2
    assert "additive_scale" not in rec


@pytest.mark.parametrize("mode", ["none", EXACT])
def test_raw_additive_layers_are_exact_in_either_mode(mode):
    error, rec = _layer_update(False, mode)
    assert error < 1e-12
    assert "additive_scale" not in rec


def _tree(normalized):
    core = _einet(
        **{
            "architecture.excitatory_branch_factors": [2, 2],
            "morphology.use_shunting": False,
        }
    )
    core["morphology"]["use_additive_normalization"] = normalized
    return _model("einet", core).double()


def _readout_tree_error(normalized, mode):
    model = _tree(normalized)
    trainer, _ = _make_trainer(
        "local_ca",
        model,
        local_rule_config={
            "error_broadcast_mode": "path_transport",
            "update_inactive_weights": False,
            "three_factor": {"additive_gain_mode": mode},
        },
    )
    generator = torch.Generator().manual_seed(4)
    x = torch.rand(5, IN_DIM, generator=generator, dtype=torch.float64)
    records, handles = trainer._attach_local_recorders(model)
    try:
        soma = model.core_network(x)
    finally:
        trainer.remove_forward_hooks(handles)
    delta = torch.randn(soma.shape, generator=generator, dtype=torch.float64)
    groups, readout = _records_by_tree(model, records)
    assert len(groups[readout]) == 3
    model.zero_grad(set_to_none=True)
    trainer._apply_local_rule_gradients(
        model,
        records,
        v0=torch.zeros_like(delta),
        delta=delta,
        y_target=None,
        update_reactivation=True,
    )
    local = {name: value.grad for name, value in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    (model.core_network(x) * delta).sum().div(len(x)).backward()
    errors = [
        float((local[name] - value.grad).abs().max())
        for name, value in model.named_parameters()
        if ".excitatory_cells." in name and local[name] is not None
    ]
    assert len(errors) == 14
    return max(errors)


def test_path_transport_through_z_scored_parents_is_exact_in_the_exact_mode():
    assert _readout_tree_error(True, EXACT) < 1e-12
    assert _readout_tree_error(True, "none") > 1e-3
    assert _readout_tree_error(False, EXACT) < 1e-12


def test_recording_the_scale_leaves_the_forward_unchanged():
    model = _tree(True)
    x = torch.rand(4, IN_DIM, dtype=torch.float64)
    plain = model(x)
    layers = [
        m for m in model.modules() if getattr(m, "use_additive_normalization", False)
    ]
    assert layers
    for layer in layers:
        layer._record_additive_scale = True
    recorded = model(x)
    assert torch.equal(plain, recorded)
    assert all(hasattr(layer, "_last_additive_scale") for layer in layers)


def test_the_default_mode_records_nothing():
    model = _tree(True)
    trainer, _ = _make_trainer("local_ca", model, local_rule_config={})
    records, handles = trainer._attach_local_recorders(model)
    try:
        model(torch.rand(4, IN_DIM, dtype=torch.float64))
    finally:
        trainer.remove_forward_hooks(handles)
    assert not any("additive_scale" in rec for rec in records)
    assert not any(
        hasattr(m, "_last_additive_scale") or hasattr(m, "_record_additive_scale")
        for m in model.modules()
    )


@pytest.mark.parametrize("mode, warns", [("none", True), (EXACT, False)])
def test_training_z_scored_layers_without_the_jacobian_warns(caplog, mode, warns):
    from test_trainer_configuration import _data

    model = _tree(True).float()
    trainer, _ = _make_trainer(
        "local_ca",
        model,
        epochs=1,
        local_rule_config={"three_factor": {"additive_gain_mode": mode}},
    )
    with caplog.at_level(logging.WARNING):
        trainer.train(model, _data(), _data(seed=1))
    assert (
        "omits the derivative of use_additive_normalization" in caplog.text
    ) is warns


@pytest.mark.parametrize("strategy", ["local_ca", "multi_stage"])
def test_the_exact_mode_reaches_the_local_trainer_from_a_typed_config(
    tmp_path, strategy
):
    from test_trainer_configuration import _einet_model, _optimizer

    from dendritic_modeling.training.strategies import multi_stage

    block = {"three_factor": {"additive_gain_mode": EXACT}}
    main = {"strategy": strategy, "learning_strategy_config": block}
    if strategy == "multi_stage":
        stages = [{"learning_strategy": "local_ca", "epochs": 1}]
        main["strategies"] = {"multi_stage": {"stages": stages}}
    config = _typed_config(tmp_path, main)
    assert (
        config.training.main.learning_strategy_config.three_factor.additive_gain_mode
        == EXACT
    )
    model = _einet_model()
    trainer = get_trainer(
        strategy, _optimizer(model), _trainer_kwargs_from_typed(config, tmp_path)
    )
    if strategy == "multi_stage":
        stage = trainer.multi_stage_config.stages[0]
        stage_config = multi_stage._build_stage_trainer_config(
            trainer.base_trainer_config, trainer.multi_stage_config, stage
        )
        assert (
            stage_config["local_rule_config"]["three_factor"]["additive_gain_mode"]
            == EXACT
        )
    else:
        assert trainer.local_cfg.three_factor.additive_gain_mode == EXACT

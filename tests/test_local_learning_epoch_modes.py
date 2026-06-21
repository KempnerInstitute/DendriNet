from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning import (
    _accumulate_backprop_gradients,
    _apply_decoder_local_gradients,
    _apply_path_propagation_factor,
    _clear_topk_mask_cache,
    _clip_model_gradients,
    _collect_backprop_param_groups,
    _collect_topk_modules,
    _enable_topk_mask_cache,
    _forward_with_local_recorders,
    _has_dendritic_branch_layers,
    _prepare_hsic_target,
    _resolve_epoch_update_modes,
    _resolve_stdp_error_signals,
    _supports_local_learning_train,
    _zero_optimizer_gradients,
)
from dendritic_modeling.training.strategies.local_learning_gradients_mixin import (
    LocalLearningGradientMixin,
    _accumulate_parameter_grad,
    _apply_optional_alignment,
    _factor_pre_weight_grad,
    _reactivation_output_slope,
    _scale_reactivation_error,
)
from dendritic_modeling.training.strategies.local_learning_state import (
    _LayerModulators,
    _PostFactors,
)


def _local_cfg(**overrides):
    values = {
        "update_reactivation": True,
        "encoder_update_mode": "backprop",
        "decoder_update_mode": "local",
        "freeze_encoder_epochs": 0,
        "freeze_reactivation_epochs": 0,
        "freeze_decoder_epochs": 0,
        "training_schedule": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _hsic_cfg(*, enabled=True, target_source="labels"):
    return SimpleNamespace(
        hsic=SimpleNamespace(enabled=enabled, target_source=target_source)
    )


def _path_cfg(*, use_path_propagation=True):
    return SimpleNamespace(
        morphology_aware=SimpleNamespace(
            use_path_propagation=use_path_propagation,
        )
    )


def test_accumulate_parameter_grad_initializes_with_parameter_dtype():
    param = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    grad = torch.tensor([1.25, -0.5], dtype=torch.float32)

    _accumulate_parameter_grad(param, grad)

    assert param.grad is not None
    assert param.grad.dtype == torch.float64
    torch.testing.assert_close(
        param.grad,
        torch.tensor([1.25, -0.5], dtype=torch.float64),
    )


def test_accumulate_parameter_grad_adds_to_existing_gradient():
    param = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    param.grad = torch.tensor([1.0, 2.0], dtype=torch.float32)

    _accumulate_parameter_grad(param, torch.tensor([0.5, -1.0], dtype=torch.float32))

    torch.testing.assert_close(param.grad, torch.tensor([1.5, 1.0]))


def test_scale_reactivation_error_applies_scalar_modulators_in_order():
    error_signal = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

    scaled = _scale_reactivation_error(
        error_signal,
        rho=2.0,
        phi=0.25,
        branch_scale=-3.0,
    )

    expected = ((error_signal * 2.0) * 0.25) * -3.0
    torch.testing.assert_close(scaled, expected)


def test_scale_reactivation_error_broadcasts_tensor_modulators():
    error_signal = torch.tensor([[1.0, -2.0, 0.5], [0.5, 3.0, -1.0]])
    rho = torch.tensor([2.0, 0.5, -1.0])
    branch_scale = torch.tensor([1.5, -2.0, 0.25])

    scaled = _scale_reactivation_error(
        error_signal,
        rho=rho,
        phi=0.25,
        branch_scale=branch_scale,
    )

    expected = ((error_signal * rho.unsqueeze(0)) * 0.25) * branch_scale.unsqueeze(0)
    torch.testing.assert_close(scaled, expected)


def test_reactivation_output_slope_matches_existing_formula():
    output = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])

    slope = _reactivation_output_slope(output)

    torch.testing.assert_close(slope, 1 - (2 * output - 1) ** 2)


def test_apply_optional_alignment_matches_existing_formula():
    grad = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
    alignment = torch.tensor([[0.5, -1.0], [2.0, 0.25]], dtype=torch.float64)

    aligned = _apply_optional_alignment(grad, alignment)

    torch.testing.assert_close(aligned, grad * alignment.to(dtype=grad.dtype))


def test_apply_optional_alignment_ignores_missing_or_mismatched_alignment():
    grad = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)

    assert _apply_optional_alignment(grad, None) is grad
    assert _apply_optional_alignment(grad, torch.ones(2, 1)) is grad


def test_factor_pre_weight_grad_matches_existing_formula():
    factor = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]])
    x_pre = torch.tensor([[2.0, -1.0, 0.5], [1.5, 0.0, -2.0], [0.25, 3.0, 1.0]])

    normalized = _factor_pre_weight_grad(
        factor,
        x_pre,
        normalize_by_batch=True,
        batch_size=3,
    )
    unnormalized = _factor_pre_weight_grad(
        factor,
        x_pre,
        normalize_by_batch=False,
        batch_size=3,
    )

    expected = factor.t() @ x_pre
    torch.testing.assert_close(normalized, expected / 3.0)
    torch.testing.assert_close(unnormalized, expected)


def test_apply_layer_local_gradients_dispatches_topk_paths_in_order():
    class _Probe(LocalLearningGradientMixin):
        def __init__(self):
            self.calls = []

        def _apply_excitatory_topk_local_grad(self, **kwargs):
            self.calls.append(
                (
                    "exc",
                    kwargs["module_key"],
                    kwargs["input_key"],
                    kwargs["mask_key"],
                )
            )
            if kwargs["module_key"] == "exc_module":
                return "feedforward-exc"
            return "recurrent-exc"

        def _apply_inhibitory_topk_local_grad(self, **kwargs):
            self.calls.append(
                (
                    "inh",
                    kwargs["module_key"],
                    kwargs["input_key"],
                    kwargs["mask_key"],
                )
            )

        def _apply_dendritic_block_local_grad(self, **_kwargs):
            self.calls.append(("block",))

        def _apply_reactivation_local_grad(self, **_kwargs):
            self.calls.append(("reactivation",))

        def _stdp_enabled(self):
            return True

        def _apply_stdp_pathways(self, **_kwargs):
            self.calls.append(("stdp",))

        def _apply_hsic_auxiliary_gradients(self, **kwargs):
            self.calls.append(("hsic", kwargs["exc_layer"]))

    trainer = _Probe()
    tensor = torch.ones(2, 3)
    modulators = _LayerModulators(
        rho=1.0,
        phi=1.0,
        branch_scale=1.0,
        dendritic_branch_scale=1.0,
        role_synaptic_alignment=None,
        role_block_alignment=None,
    )
    post_factors = _PostFactors(
        e_v=tensor,
        post_factor=tensor,
        post_factor_raw=tensor,
    )

    trainer._apply_layer_local_gradients(
        rec={},
        layer_idx=0,
        num_layers=1,
        y_target=None,
        batch_size=2,
        e_n=tensor,
        stdp_error_signal=tensor,
        v_n=tensor,
        r_tot=1.0,
        layer_dynamics_mode="conductance",
        modulators=modulators,
        post_factors=post_factors,
        update_reactivation=True,
    )

    assert trainer.calls == [
        ("exc", "exc_module", "x_exc", "exc_mask"),
        ("exc", "rec_exc_module", "x_rec_exc", "rec_exc_mask"),
        ("inh", "inh_module", "x_inh", "inh_mask"),
        ("inh", "rec_inh_module", "x_rec_inh", "rec_inh_mask"),
        ("block",),
        ("reactivation",),
        ("stdp",),
        ("hsic", "feedforward-exc"),
    ]


def test_resolve_epoch_update_modes_uses_base_modes_after_freeze_window():
    cfg = _local_cfg(
        freeze_encoder_epochs=2,
        freeze_reactivation_epochs=2,
        freeze_decoder_epochs=2,
    )

    assert _resolve_epoch_update_modes(cfg, epoch_counter=3) == (
        True,
        "backprop",
        "local",
    )


def test_resolve_epoch_update_modes_preserves_inclusive_freeze_boundary():
    cfg = _local_cfg(
        freeze_encoder_epochs=2,
        freeze_reactivation_epochs=2,
        freeze_decoder_epochs=2,
    )

    assert _resolve_epoch_update_modes(cfg, epoch_counter=2) == (
        False,
        "none",
        "none",
    )


def test_resolve_epoch_update_modes_training_schedule_overrides_freeze_modes():
    cfg = _local_cfg(
        freeze_encoder_epochs=10,
        freeze_reactivation_epochs=10,
        freeze_decoder_epochs=10,
        training_schedule=[
            {
                "start_epoch": 3,
                "end_epoch": 5,
                "update_reactivation": True,
                "encoder_update_mode": "BackProp",
                "decoder_update_mode": "LOCAL",
            }
        ],
    )

    assert _resolve_epoch_update_modes(cfg, epoch_counter=4) == (
        True,
        "backprop",
        "local",
    )


def test_resolve_epoch_update_modes_uses_first_matching_schedule_entry():
    cfg = _local_cfg(
        training_schedule=[
            {"start_epoch": 1, "end_epoch": 3, "encoder_update_mode": "none"},
            {"start_epoch": 1, "end_epoch": 3, "encoder_update_mode": "backprop"},
        ],
    )

    assert _resolve_epoch_update_modes(cfg, epoch_counter=2) == (
        True,
        "none",
        "local",
    )


def test_resolve_epoch_update_modes_bad_schedule_range_matches_all_epochs():
    cfg = _local_cfg(
        training_schedule=[
            {
                "start_epoch": object(),
                "end_epoch": object(),
                "update_reactivation": False,
                "decoder_update_mode": "BackProp",
            }
        ],
    )

    assert _resolve_epoch_update_modes(cfg, epoch_counter=99) == (
        False,
        "backprop",
        "backprop",
    )


def test_has_dendritic_branch_layers_returns_false_when_model_iteration_fails():
    class _BrokenModules(nn.Module):
        def modules(self):
            raise RuntimeError("broken module traversal")

    assert _has_dendritic_branch_layers(_BrokenModules()) is False


@pytest.mark.parametrize("model", [nn.Linear(2, 1), SimpleNamespace(core_network=None)])
def test_supports_local_learning_train_rejects_plain_models(model):
    assert _supports_local_learning_train(model) is False


def test_collect_topk_modules_finds_nested_topk_modules_in_module_order():
    topk_a = TopKLinear(in_features=4, out_features=2, K=2)
    topk_b = TopKLinear(in_features=3, out_features=1, K=1)
    model = nn.Sequential(nn.Linear(4, 4), nn.Sequential(topk_a, nn.ReLU(), topk_b))

    assert _collect_topk_modules(model) == [topk_a, topk_b]


def test_topk_mask_cache_helpers_toggle_flags_and_clear_last_masks():
    topk_a = TopKLinear(in_features=4, out_features=2, K=2)
    topk_b = TopKLinear(in_features=3, out_features=1, K=1)
    topk_modules = [topk_a, topk_b]

    _enable_topk_mask_cache(topk_modules)
    assert [module.cache_mask for module in topk_modules] == [True, True]

    topk_a._last_forward_weight_mask = torch.ones_like(topk_a.pre_w, dtype=torch.bool)
    topk_b._last_forward_weight_mask = torch.ones_like(topk_b.pre_w, dtype=torch.bool)

    _clear_topk_mask_cache(topk_modules)

    assert [module.cache_mask for module in topk_modules] == [False, False]
    assert topk_a._last_forward_weight_mask is None
    assert topk_b._last_forward_weight_mask is None


def test_topk_mask_cache_helpers_do_not_mutate_topk_parameters():
    topk = TopKLinear(in_features=4, out_features=2, K=2)
    before = topk.pre_w.detach().clone()

    _enable_topk_mask_cache([topk])
    _clear_topk_mask_cache([topk])

    assert torch.equal(topk.pre_w, before)


def test_prepare_hsic_target_returns_none_when_disabled():
    y_batch = torch.tensor([0, 1])
    y_hat = torch.zeros(2, 3)

    assert _prepare_hsic_target(_hsic_cfg(enabled=False), y_batch, y_hat) is None


def test_prepare_hsic_target_one_hots_1d_integer_labels():
    y_batch = torch.tensor([2, 0])
    y_hat = torch.zeros(2, 3, dtype=torch.float64)

    target = _prepare_hsic_target(_hsic_cfg(), y_batch, y_hat)

    assert target.dtype == torch.float64
    assert torch.equal(
        target,
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
    )


def test_prepare_hsic_target_one_hots_column_integer_labels():
    y_batch = torch.tensor([[1], [0]])
    y_hat = torch.zeros(2, 2, dtype=torch.float32)

    target = _prepare_hsic_target(_hsic_cfg(), y_batch, y_hat)

    assert torch.equal(target, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_prepare_hsic_target_casts_soft_label_targets_to_prediction_dtype():
    y_batch = torch.tensor([[0.25, 0.75], [1.0, 0.0]], dtype=torch.float64)
    y_hat = torch.zeros(2, 2, dtype=torch.float32)

    target = _prepare_hsic_target(_hsic_cfg(), y_batch, y_hat)

    assert target.dtype == torch.float32
    assert torch.equal(target, y_batch.to(torch.float32))


def test_prepare_hsic_target_uses_detached_predictions_for_non_label_source():
    y_batch = torch.tensor([0, 1])
    y_hat = torch.randn(2, 3, requires_grad=True)

    target = _prepare_hsic_target(
        _hsic_cfg(target_source="predictions"),
        y_batch,
        y_hat,
    )

    assert target is not y_hat
    assert target.requires_grad is False
    assert torch.equal(target, y_hat.detach())


def test_zero_optimizer_gradients_falls_back_for_legacy_optimizers():
    class _LegacyOptimizer:
        def __init__(self):
            self.calls = []

        def zero_grad(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            if kwargs:
                raise TypeError("set_to_none is unsupported")

    optimizer = _LegacyOptimizer()

    _zero_optimizer_gradients(optimizer)

    assert optimizer.calls == [
        ((), {"set_to_none": True}),
        ((), {}),
    ]


def test_collect_backprop_param_groups_preserves_encoder_then_decoder_order():
    encoder = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    decoder = nn.Sequential(nn.Linear(2, 5), nn.Linear(5, 1))
    encoder[1].weight.requires_grad_(False)
    decoder[0].bias.requires_grad_(False)
    model = SimpleNamespace(encoder_network=encoder, decoder_network=decoder)

    groups = _collect_backprop_param_groups(
        model,
        encoder_update_mode="backprop",
        decoder_update_mode="backprop",
    )

    assert groups == [
        [p for p in encoder.parameters() if p.requires_grad],
        [p for p in decoder.parameters() if p.requires_grad],
    ]


def test_collect_backprop_param_groups_skips_disabled_modes_and_empty_groups():
    encoder = nn.Linear(3, 2)
    for param in encoder.parameters():
        param.requires_grad_(False)
    decoder = nn.Linear(2, 1)
    model = SimpleNamespace(encoder_network=encoder, decoder_network=decoder)

    assert (
        _collect_backprop_param_groups(
            model,
            encoder_update_mode="backprop",
            decoder_update_mode="local",
        )
        == []
    )


def test_accumulate_backprop_gradients_preserves_old_accumulation_semantics():
    p1 = nn.Parameter(torch.tensor(2.0))
    p2 = nn.Parameter(torch.tensor(3.0))
    unused = nn.Parameter(torch.tensor(5.0))
    p2.grad = torch.tensor(10.0)
    loss = p1 * p2 + p1.square()

    _accumulate_backprop_gradients(loss, [[p1, unused], [p2]])

    assert torch.equal(p1.grad, torch.tensor(7.0))
    assert p1.grad.requires_grad is False
    assert unused.grad is None
    assert torch.equal(p2.grad, torch.tensor(12.0))


def test_apply_decoder_local_gradients_preserves_exact_accumulation_semantics():
    lin = nn.Linear(2, 2, bias=True)
    lin.weight.grad = torch.full_like(lin.weight, 10.0)
    lin.bias.grad = torch.full_like(lin.bias, -1.0)
    h_in = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    delta_out = torch.tensor([[0.5, 1.0], [1.5, -0.5]], requires_grad=True)

    _apply_decoder_local_gradients(
        lin_mod=lin,
        h_in=h_in,
        delta_out=delta_out,
        normalize_by_batch=True,
    )

    expected_grad_w = 0.5 * (delta_out.detach().t() @ h_in.detach())
    expected_grad_b = 0.5 * delta_out.detach().sum(dim=0)
    assert torch.equal(
        lin.weight.grad, torch.full_like(lin.weight, 10.0) + expected_grad_w
    )
    assert torch.equal(lin.bias.grad, torch.full_like(lin.bias, -1.0) + expected_grad_b)
    assert lin.weight.grad.requires_grad is False
    assert lin.bias.grad.requires_grad is False


def test_apply_decoder_local_gradients_is_noop_for_invalid_cache_entries():
    lin = nn.Linear(2, 2, bias=True)

    _apply_decoder_local_gradients(
        lin_mod=lin,
        h_in=None,
        delta_out=torch.ones(1, 2),
        normalize_by_batch=False,
    )

    assert lin.weight.grad is None
    assert lin.bias.grad is None


def test_clip_model_gradients_clamps_existing_gradients():
    model = nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[2.0, -3.0]])
    model.bias.grad = torch.tensor([0.1])

    _clip_model_gradients(model, clip_grad_value=0.25)

    assert torch.equal(model.weight.grad, torch.tensor([[0.25, -0.25]]))
    assert torch.equal(model.bias.grad, torch.tensor([0.1]))


@pytest.mark.parametrize("clip_grad_value", [None, 0, -0.25])
def test_clip_model_gradients_preserves_disabled_guard_behavior(clip_grad_value):
    model = nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[2.0, -3.0]])
    before = model.weight.grad.clone()

    _clip_model_gradients(model, clip_grad_value=clip_grad_value)

    assert torch.equal(model.weight.grad, before)


def test_forward_with_local_recorders_cleans_hooks_and_topk_cache_on_success():
    class _Handle:
        def __init__(self):
            self.removed = False

        def remove(self):
            self.removed = True

    class _Trainer:
        def __init__(self):
            self.handle = _Handle()
            self.records = [{"v_n": torch.ones(1, 2)}]

        def _attach_local_recorders(self, model):
            return self.records, [self.handle]

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.topk = TopKLinear(in_features=2, out_features=1, K=1)

        def forward(self, x):
            assert self.topk.cache_mask is True
            self.topk._last_forward_weight_mask = torch.ones_like(
                self.topk.pre_w,
                dtype=torch.bool,
            )
            return x + 1.0

    trainer = _Trainer()
    model = _Model()
    x_batch = torch.zeros(1, 2)

    y_hat, records = _forward_with_local_recorders(trainer, model, x_batch)

    assert torch.equal(y_hat, torch.ones(1, 2))
    assert records is trainer.records
    assert trainer.handle.removed is True
    assert model.topk.cache_mask is False
    assert model.topk._last_forward_weight_mask is None


def test_forward_with_local_recorders_cleans_hooks_and_topk_cache_on_exception():
    class _Handle:
        def __init__(self):
            self.removed = False

        def remove(self):
            self.removed = True

    class _Trainer:
        def __init__(self):
            self.handle = _Handle()

        def _attach_local_recorders(self, model):
            return [], [self.handle]

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.topk = TopKLinear(in_features=2, out_features=1, K=1)

        def forward(self, x):
            assert self.topk.cache_mask is True
            self.topk._last_forward_weight_mask = torch.ones_like(
                self.topk.pre_w,
                dtype=torch.bool,
            )
            raise RuntimeError("forward failed")

    trainer = _Trainer()
    model = _Model()

    with pytest.raises(RuntimeError, match="forward failed"):
        _forward_with_local_recorders(trainer, model, torch.zeros(1, 2))

    assert trainer.handle.removed is True
    assert model.topk.cache_mask is False
    assert model.topk._last_forward_weight_mask is None


def test_resolve_stdp_error_signals_keeps_error_for_non_stdp_rules():
    e_n = torch.tensor([[1.0, -2.0]])

    resolved_e_n, stdp_error_signal = _resolve_stdp_error_signals(
        SimpleNamespace(rule_variant="hebbian"),
        e_n,
    )

    assert resolved_e_n is e_n
    assert stdp_error_signal is e_n


def test_resolve_stdp_error_signals_zeroes_local_error_for_stdp_rules():
    e_n = torch.tensor([[1.0, -2.0]])

    resolved_e_n, stdp_error_signal = _resolve_stdp_error_signals(
        SimpleNamespace(rule_variant="STDP"),
        e_n,
    )

    assert torch.equal(resolved_e_n, torch.zeros_like(e_n))
    assert resolved_e_n is not e_n
    assert stdp_error_signal is e_n


def test_apply_path_propagation_factor_multiplies_enabled_tensor_factor():
    e_n = torch.tensor([[2.0, -3.0]])
    path_factor = torch.tensor([[0.5, 2.0]])

    resolved_e_n = _apply_path_propagation_factor(
        local_cfg=_path_cfg(use_path_propagation=True),
        broadcast_state=SimpleNamespace(mode="broadcast"),
        rec={"path_factor": path_factor},
        e_n=e_n,
    )

    assert torch.equal(resolved_e_n, e_n * path_factor)


@pytest.mark.parametrize(
    ("use_path_propagation", "mode", "path_factor"),
    [
        (False, "broadcast", torch.tensor([[0.5, 2.0]])),
        (True, "path_transport", torch.tensor([[0.5, 2.0]])),
        (True, "broadcast", 0.5),
    ],
)
def test_apply_path_propagation_factor_preserves_noop_cases(
    use_path_propagation,
    mode,
    path_factor,
):
    e_n = torch.tensor([[2.0, -3.0]])

    resolved_e_n = _apply_path_propagation_factor(
        local_cfg=_path_cfg(use_path_propagation=use_path_propagation),
        broadcast_state=SimpleNamespace(mode=mode),
        rec={"path_factor": path_factor},
        e_n=e_n,
    )

    assert resolved_e_n is e_n

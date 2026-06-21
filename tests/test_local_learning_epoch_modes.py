from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning import (
    _accumulate_backprop_gradients,
    _clear_topk_mask_cache,
    _collect_backprop_param_groups,
    _collect_topk_modules,
    _enable_topk_mask_cache,
    _has_dendritic_branch_layers,
    _prepare_hsic_target,
    _resolve_epoch_update_modes,
    _supports_local_learning_train,
    _zero_optimizer_gradients,
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

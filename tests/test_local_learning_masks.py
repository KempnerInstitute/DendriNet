"""Tests for TopK forward-mask caching and local-learning mask usage."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.config.training import LocalRuleConfig
from dendritic_modeling.networks import DendriticBranchLayer, TopKLinear
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


class _SingleBranchModel(nn.Module):
    """Minimal model exposing one dendritic branch layer."""

    def __init__(self):
        super().__init__()
        self.branch = DendriticBranchLayer(
            output_dim=4,
            excitatory_input_dim=8,
            excitatory_synapses_per_branch=2,
            topk_noise_level=0.5,
            reactivate=False,
        )

    def forward(self, x):
        return self.branch(x)


class _DualSynapseBranchModel(nn.Module):
    """Branch layer with E and I TopK paths for partial-registration checks."""

    def __init__(self):
        super().__init__()
        self.branch = DendriticBranchLayer(
            output_dim=4,
            excitatory_input_dim=8,
            excitatory_synapses_per_branch=2,
            inhibitory_input_dim=8,
            inhibitory_synapses_per_branch=2,
            topk_noise_level=0.5,
            reactivate=False,
        )

    def forward(self, x, inhibitory_x):
        return self.branch(x, inhibitory_x)


class _IndexedBranchModel(nn.Module):
    """Minimal branch with compact fixed synaptic indices."""

    def __init__(self):
        super().__init__()
        self.branch = DendriticBranchLayer(
            output_dim=4,
            excitatory_input_dim=8,
            excitatory_synapses_per_branch=2,
            excitatory_connection_indices=torch.tensor(
                [[0, 1], [2, 3], [4, 5], [6, 7]]
            ),
            topk_type="indexed",
            reactivate=False,
        )

    def forward(self, x):
        return self.branch(x)


class _DecoderOnlyModel(nn.Module):
    """Model with a decoder network for decoder-recorder cleanup checks."""

    def __init__(self):
        super().__init__()
        self.decoder_network = nn.Sequential(nn.Linear(3, 2))

    def forward(self, x):
        return self.decoder_network(x)


class _DDPStyleWrapper(nn.Module):
    """Minimal wrapper exposing the same ``module`` boundary as DDP."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _count_forward_and_pre_hooks(model: nn.Module) -> int:
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in model.modules()
    )


def test_topk_forward_does_not_cache_by_default():
    """Forward pass should NOT cache the mask when cache_mask is False."""
    torch.manual_seed(7)
    layer = TopKLinear(in_features=12, out_features=5, K=3, noise_level=0.5)
    layer.initialize()

    x = torch.rand(4, 12)
    _ = layer(x)

    assert layer._last_forward_weight_mask is None


def test_topk_forward_caches_exact_mask_when_enabled():
    """Forward pass should persist the exact sampled TopK mask when cache_mask=True."""
    torch.manual_seed(7)
    layer = TopKLinear(in_features=12, out_features=5, K=3, noise_level=0.5)
    layer.initialize()
    layer.cache_mask = True

    x = torch.rand(4, 12)
    out = layer(x)

    cached_mask = layer._last_forward_weight_mask
    assert isinstance(cached_mask, torch.Tensor)
    assert cached_mask.shape == (5, 12)
    assert torch.all(cached_mask.sum(dim=1) == 3)

    expected = x @ (cached_mask * layer.weight()).t()
    torch.testing.assert_close(out, expected)


def test_local_recorders_use_cached_topk_mask_without_extra_sampling():
    """Local-learning hooks should not trigger a second noisy TopK sampling."""
    torch.manual_seed(11)
    model = _SingleBranchModel()
    topk = model.branch.branch_excitation
    assert isinstance(topk, TopKLinear)

    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}

    records, handles = trainer._attach_local_recorders(model)
    topk.cache_mask = True
    x = torch.rand(6, 8)

    try:
        with patch.object(topk, "weight_mask", wraps=topk.weight_mask) as wrapped_mask:
            _ = model(x)
            assert wrapped_mask.call_count == 1
    finally:
        for handle in handles:
            handle.remove()

    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec.get("exc_mask"), torch.Tensor)
    torch.testing.assert_close(rec["exc_mask"], topk._last_forward_weight_mask)


def test_local_recorders_do_not_materialize_indexed_dense_mask(monkeypatch):
    model = _IndexedBranchModel()
    synapse = model.branch.branch_excitation
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}
    records, handles = trainer._attach_local_recorders(model)

    def reject_dense_mask():
        raise AssertionError("compact local recording must not build a dense mask")

    monkeypatch.setattr(synapse, "weight_mask", reject_dense_mask)
    try:
        _ = model(torch.rand(3, 8))
    finally:
        for handle in handles:
            handle.remove()

    assert records[0]["exc_module"] is synapse
    assert records[0]["exc_mask"] is None


def test_local_recorders_remove_hooks_after_failed_registration(monkeypatch):
    model = _SingleBranchModel()
    topk = model.branch.branch_excitation
    assert isinstance(topk, TopKLinear)

    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}

    def _failing_register(*args, **kwargs):
        raise RuntimeError("local recorder hook registration failed")

    monkeypatch.setattr(topk, "register_forward_hook", _failing_register)

    with pytest.raises(RuntimeError, match="local recorder hook registration failed"):
        trainer._attach_local_recorders(model)

    assert sum(len(module._forward_hooks) for module in model.modules()) == 0


def test_local_recorders_remove_decoder_pre_hook_after_linear_registration_failure(
    monkeypatch,
):
    model = _DecoderOnlyModel()
    decoder_linear = model.decoder_network[0]

    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}

    def _failing_register(*args, **kwargs):
        raise RuntimeError("decoder recorder hook registration failed")

    monkeypatch.setattr(decoder_linear, "register_forward_hook", _failing_register)

    with pytest.raises(RuntimeError, match="decoder recorder hook registration failed"):
        trainer._attach_local_recorders(model)

    assert _count_forward_and_pre_hooks(model) == 0


def test_local_recorders_capture_decoder_through_ddp_style_wrapper():
    model = _DecoderOnlyModel()
    wrapped = _DDPStyleWrapper(model)
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}

    _records, handles = trainer._attach_local_recorders(wrapped)
    x = torch.randn(4, 3)
    try:
        _ = wrapped(x)
    finally:
        for handle in handles:
            handle.remove()

    assert trainer._decoder_cache["module"] is model.decoder_network[0]
    torch.testing.assert_close(trainer._decoder_cache["input"], x)
    assert _count_forward_and_pre_hooks(wrapped) == 0


def test_local_recorders_remove_partial_topk_hooks_after_failed_registration(
    monkeypatch,
):
    model = _DualSynapseBranchModel()
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}

    def _failing_register(*args, **kwargs):
        raise RuntimeError("local recorder inhibitory hook registration failed")

    monkeypatch.setattr(
        model.branch.branch_inhibition,
        "register_forward_hook",
        _failing_register,
    )

    with pytest.raises(
        RuntimeError,
        match="local recorder inhibitory hook registration failed",
    ):
        trainer._attach_local_recorders(model)

    assert sum(len(module._forward_hooks) for module in model.modules()) == 0


def test_explicit_inhibitory_update_mode_only_gates_i_cell_records():
    """Freeze mode skips explicit I-cell records but not E-cell I synapses."""
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)

    trainer.local_cfg = LocalRuleConfig(explicit_inhibitory_update_mode="freeze")
    assert not trainer._should_update_local_record(
        {"population": "explicit_inhibitory"}
    )
    assert trainer._should_update_local_record({"population": "excitatory"})

    trainer.local_cfg = LocalRuleConfig(explicit_inhibitory_update_mode="local_ca")
    assert trainer._should_update_local_record({"population": "explicit_inhibitory"})


def test_deprecated_inhibitory_cell_update_mode_alias_still_loads():
    """The old shorter YAML key should map to the canonical explicit-I policy."""
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)

    local_cfg = trainer._build_local_rule_config(
        {"inhibitory_cell_update_mode": "freeze"}
    )

    assert local_cfg.explicit_inhibitory_update_mode == "freeze"
    assert not local_cfg.update_explicit_inhibitory_cells

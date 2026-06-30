"""Tests for local-learning record-level rule orchestration."""

from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_rule_mixin import (
    LocalLearningRuleMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerModulators,
    _PostFactors,
)


def _modulators() -> _LayerModulators:
    return _LayerModulators(
        rho=1.0,
        phi=1.0,
        branch_scale=1.0,
        dendritic_branch_scale=1.0,
        role_synaptic_alignment=None,
        role_block_alignment=None,
    )


class _RuleProbe(LocalLearningRuleMixin):
    """Minimal object exposing the local-rule mixin contract."""

    def __init__(self, *, rule_variant: str = "4f", use_path_propagation: bool = True):
        self.local_cfg = SimpleNamespace(
            rule_variant=rule_variant,
            morphology_aware=SimpleNamespace(
                use_path_propagation=use_path_propagation,
            ),
        )
        self.calls: list[str] = []
        self.applied: list[dict] = []

    def _should_update_local_record(self, rec):
        self.calls.append("should_update")
        return rec.get("should_update", True)

    def _compute_local_broadcast_error(
        self,
        *,
        rec,
        layer_idx,
        out_features,
        v_n,
        delta,
        delta_scalar,
        broadcast_state,
    ):
        self.calls.append("broadcast")
        assert layer_idx == rec["layer_idx"]
        assert out_features == v_n.size(1)
        assert broadcast_state is rec["broadcast_state"]
        return rec["e_n"]

    def _resolve_layer_dynamics_mode(self, rec):
        self.calls.append("mode")
        return rec["layer_dynamics_mode"]

    def _compute_local_input_resistance(self, *, rec, v_n, layer_dynamics_mode):
        self.calls.append("resistance")
        assert layer_dynamics_mode == rec["layer_dynamics_mode"]
        return rec["r_tot"]

    def _compute_local_layer_modulators(self, *, rec, v0, layer_depth):
        self.calls.append("modulators")
        assert layer_depth == rec["layer_idx"] + 1
        assert v0 is rec["v0"]
        return rec["modulators"]

    def _compute_local_post_factors(
        self,
        *,
        rec,
        e_n,
        v_n,
        r_tot,
        layer_dynamics_mode,
        layer_idx,
        modulators,
    ):
        self.calls.append("post")
        assert layer_idx == rec["layer_idx"]
        assert layer_dynamics_mode == rec["layer_dynamics_mode"]
        assert r_tot is rec["r_tot"]
        assert modulators is rec["modulators"]
        return _PostFactors(e_v=e_n, post_factor=e_n + 1.0, post_factor_raw=e_n + 2.0)

    def _apply_layer_local_gradients(self, **kwargs):
        self.calls.append("apply")
        self.applied.append(kwargs)


def _record(
    *,
    v0: torch.Tensor,
    broadcast_state,
    path_factor: torch.Tensor | None = None,
) -> dict:
    rec = {
        "layer_idx": 1,
        "v_n": torch.ones(2, 3),
        "e_n": torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]),
        "r_tot": torch.tensor([[2.0, 3.0, 4.0]]),
        "layer_dynamics_mode": "conductance",
        "modulators": _modulators(),
        "v0": v0,
        "broadcast_state": broadcast_state,
    }
    if path_factor is not None:
        rec["path_factor"] = path_factor
    return rec


def test_prepare_local_rule_record_state_preserves_error_pipeline():
    broadcast_state = SimpleNamespace(mode="soma")
    v0 = torch.zeros(2, 4)
    path_factor = torch.tensor([[2.0, 0.5, 1.0], [1.0, 2.0, 0.25]])
    rec = _record(v0=v0, broadcast_state=broadcast_state, path_factor=path_factor)
    probe = _RuleProbe(rule_variant="4f", use_path_propagation=True)

    state = probe._prepare_local_rule_record_state(
        rec=rec,
        layer_idx=rec["layer_idx"],
        v0=v0,
        delta=torch.ones(2, 3),
        delta_scalar=torch.ones(2, 1),
        v_n=rec["v_n"],
        broadcast_state=broadcast_state,
    )

    expected_error = rec["e_n"] * path_factor
    torch.testing.assert_close(state.e_n, expected_error)
    torch.testing.assert_close(state.stdp_error_signal, rec["e_n"])
    assert state.v_n is rec["v_n"]
    assert state.r_tot is rec["r_tot"]
    assert state.layer_dynamics_mode == "conductance"
    assert state.modulators is rec["modulators"]
    torch.testing.assert_close(state.post_factors.e_v, expected_error)
    assert probe.calls == ["broadcast", "mode", "resistance", "modulators", "post"]


def test_apply_local_rule_record_gradients_dispatches_prepared_state():
    broadcast_state = SimpleNamespace(mode="soma")
    v0 = torch.zeros(2, 4)
    rec = _record(
        v0=v0,
        broadcast_state=broadcast_state,
        path_factor=torch.full((2, 3), 0.5),
    )
    probe = _RuleProbe(rule_variant="4f", use_path_propagation=True)
    y_target = torch.tensor([0, 1])

    probe._apply_local_rule_record_gradients(
        rec=rec,
        layer_idx=rec["layer_idx"],
        num_layers=3,
        v0=v0,
        delta=torch.ones(2, 3),
        delta_scalar=torch.ones(2, 1),
        batch_size=2,
        broadcast_state=broadcast_state,
        y_target=y_target,
        update_reactivation=False,
    )

    assert probe.calls == [
        "should_update",
        "broadcast",
        "mode",
        "resistance",
        "modulators",
        "post",
        "apply",
    ]
    applied = probe.applied[0]
    expected_error = rec["e_n"] * rec["path_factor"]
    torch.testing.assert_close(applied["e_n"], expected_error)
    assert applied["stdp_error_signal"] is rec["e_n"]
    assert applied["v_n"] is rec["v_n"]
    assert applied["r_tot"] is rec["r_tot"]
    assert applied["modulators"] is rec["modulators"]
    assert applied["y_target"] is y_target
    assert applied["batch_size"] == 2
    assert applied["num_layers"] == 3
    assert applied["update_reactivation"] is False


def test_apply_local_rule_record_gradients_preserves_skip_cases():
    broadcast_state = SimpleNamespace(mode="soma")
    v0 = torch.zeros(2, 4)
    frozen = _record(v0=v0, broadcast_state=broadcast_state)
    frozen["should_update"] = False
    missing = _record(v0=v0, broadcast_state=broadcast_state)
    missing.pop("v_n")
    probe = _RuleProbe()

    kwargs = {
        "layer_idx": 1,
        "num_layers": 2,
        "v0": v0,
        "delta": torch.ones(2, 3),
        "delta_scalar": torch.ones(2, 1),
        "batch_size": 2,
        "broadcast_state": broadcast_state,
        "y_target": None,
        "update_reactivation": True,
    }
    probe._apply_local_rule_record_gradients(rec=frozen, **kwargs)
    probe._apply_local_rule_record_gradients(rec=missing, **kwargs)

    assert probe.calls == ["should_update", "should_update"]
    assert probe.applied == []

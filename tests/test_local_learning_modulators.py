"""Tests for local-learning modulator orchestration helpers."""

from types import SimpleNamespace

import pytest
import torch

from dendritic_modeling.training.strategies.local_learning_modulators import (
    LocalLearningModulatorMixin,
    _apply_layerwise_scale,
)


def _cfg(
    *,
    rule_variant: str = "3f",
    rho_scale: float = 1.0,
    phi_scale: float = 1.0,
    morphology_mode: str = "none",
    use_branch_role_rules: bool = False,
    use_branch_type_rules: bool = False,
    use_branch_length_modulation: bool = False,
    phi_mode: str = "conditional",
    phi_estimator: str = "conditional_ema",
):
    return SimpleNamespace(
        rule_variant=rule_variant,
        four_factor=SimpleNamespace(layer_wise_rho_scale=rho_scale),
        five_factor=SimpleNamespace(
            phi_mode=phi_mode,
            phi_estimator=phi_estimator,
            layer_wise_phi_scale=phi_scale,
        ),
        morphology_aware=SimpleNamespace(
            morphology_modulator_mode=morphology_mode,
            use_branch_role_rules=use_branch_role_rules,
            use_branch_type_rules=use_branch_type_rules,
            use_branch_length_modulation=use_branch_length_modulation,
        ),
    )


class _ModulatorProbe(LocalLearningModulatorMixin):
    """Minimal object exposing the modulator mixin contract."""

    def __init__(self, local_cfg):
        self.local_cfg = local_cfg
        self.calls: list[str] = []

    def _compute_layer_rho(self, rec, v0):
        self.calls.append("rho")
        return rec.get("rho", 1.0)

    def _compute_branch_depth_modulator(self, rec, rho):
        self.calls.append("depth")
        return rho * 10.0

    def _compute_branch_centrality_modulator(self, rec, rho):
        self.calls.append("centrality")
        return rho * 20.0

    def _compute_layer_phi(self, rec):
        self.calls.append("phi")
        return rec.get("phi", 1.0)

    def _compute_layer_phi_conditional(self, rec):
        self.calls.append("phi_conditional")
        return rec.get("phi_conditional", 1.0)

    def _get_branch_role_scales(self, rec):
        self.calls.append("role_scales")
        return rec["branch_scale"], rec["dendritic_branch_scale"]

    def _get_branch_role_alignment_factors(self, rec):
        self.calls.append("role_alignment")
        return rec["syn_alignment"], rec["block_alignment"]

    def _get_branch_type_scale(self, rec):
        self.calls.append("branch_type")
        return rec["branch_type_scale"]


def test_apply_layerwise_scale_preserves_positive_and_disabled_modes():
    assert _apply_layerwise_scale(8.0, layer_depth=4, scale=2.0) == pytest.approx(2.0)
    assert _apply_layerwise_scale(8.0, layer_depth=4, scale=0.0) == 8.0


def test_3f_modulators_are_neutral_without_branch_rules():
    probe = _ModulatorProbe(_cfg(rule_variant="3f"))
    result = probe._compute_local_layer_modulators(
        rec={},
        v0=torch.ones(2, 3),
        layer_depth=4,
    )

    assert result.rho == 1.0
    assert result.phi == 1.0
    assert result.branch_scale == 1.0
    assert result.dendritic_branch_scale == 1.0
    assert result.role_synaptic_alignment is None
    assert result.role_block_alignment is None
    assert probe.calls == []


def test_rho_modulator_preserves_layer_scale_then_depth_mode_order():
    probe = _ModulatorProbe(
        _cfg(rule_variant="4f", rho_scale=2.0, morphology_mode="depth")
    )
    result = probe._compute_local_layer_modulators(
        rec={"rho": 8.0},
        v0=torch.ones(2, 3),
        layer_depth=4,
    )

    assert result.rho == pytest.approx(20.0)
    assert result.phi == 1.0
    assert probe.calls == ["rho", "depth"]


def test_rho_modulator_preserves_centrality_mode():
    probe = _ModulatorProbe(
        _cfg(rule_variant="4f", rho_scale=1.0, morphology_mode="centrality")
    )
    result = probe._compute_local_layer_modulators(
        rec={"rho": 2.0},
        v0=torch.ones(2, 3),
        layer_depth=4,
    )

    assert result.rho == pytest.approx(20.0)
    assert probe.calls == ["rho", "centrality"]


def test_rho_morphology_dispatch_preserves_mode_selection():
    depth_probe = _ModulatorProbe(_cfg(rule_variant="4f", morphology_mode="depth"))
    depth = depth_probe._apply_local_rho_morphology_modulator({"rho": 8.0}, 2.0)
    assert depth == pytest.approx(20.0)
    assert depth_probe.calls == ["depth"]

    centrality_probe = _ModulatorProbe(
        _cfg(rule_variant="4f", morphology_mode="centrality")
    )
    centrality = centrality_probe._apply_local_rho_morphology_modulator(
        {"rho": 8.0},
        2.0,
    )
    assert centrality == pytest.approx(40.0)
    assert centrality_probe.calls == ["centrality"]

    neutral_probe = _ModulatorProbe(_cfg(rule_variant="4f", morphology_mode="none"))
    assert neutral_probe._apply_local_rho_morphology_modulator({"rho": 8.0}, 2.0) == 2.0
    assert neutral_probe.calls == []


def test_phi_modulator_preserves_conditional_and_variance_dispatch():
    conditional_probe = _ModulatorProbe(
        _cfg(
            rule_variant="5f",
            phi_scale=2.0,
            phi_mode="conditional",
            phi_estimator="conditional_ema",
        )
    )
    conditional = conditional_probe._compute_local_phi_modulator(
        rec={"phi_conditional": 8.0},
        layer_depth=4,
    )
    assert conditional == pytest.approx(2.0)
    assert conditional_probe.calls == ["phi_conditional"]

    variance_probe = _ModulatorProbe(
        _cfg(
            rule_variant="5f",
            phi_scale=2.0,
            phi_mode="conditional",
            phi_estimator="variance_ema",
        )
    )
    variance = variance_probe._compute_local_phi_modulator(
        rec={"phi": 8.0},
        layer_depth=4,
    )
    assert variance == pytest.approx(2.0)
    assert variance_probe.calls == ["phi"]


def test_phi_value_dispatch_preserves_estimator_selection_without_scaling():
    conditional_probe = _ModulatorProbe(
        _cfg(
            rule_variant="5f",
            phi_scale=2.0,
            phi_mode="conditional",
            phi_estimator="conditional_ema",
        )
    )
    conditional = conditional_probe._compute_local_phi_value(
        rec={"phi_conditional": 8.0, "phi": 2.0}
    )
    assert conditional == 8.0
    assert conditional_probe.calls == ["phi_conditional"]

    variance_probe = _ModulatorProbe(
        _cfg(
            rule_variant="5f",
            phi_scale=2.0,
            phi_mode="conditional",
            phi_estimator="variance_ema",
        )
    )
    variance = variance_probe._compute_local_phi_value(
        rec={"phi_conditional": 8.0, "phi": 2.0}
    )
    assert variance == 2.0
    assert variance_probe.calls == ["phi"]

    direct_probe = _ModulatorProbe(_cfg(rule_variant="5f", phi_mode="variance"))
    direct = direct_probe._compute_local_phi_value(
        rec={"phi_conditional": 8.0, "phi": 3.0}
    )
    assert direct == 3.0
    assert direct_probe.calls == ["phi"]


def test_phi_modulator_preserves_nonconditional_dispatch():
    probe = _ModulatorProbe(_cfg(rule_variant="5f", phi_scale=1.0, phi_mode="variance"))
    phi = probe._compute_local_phi_modulator(
        rec={"phi": 6.0},
        layer_depth=9,
    )

    assert phi == pytest.approx(2.0)
    assert probe.calls == ["phi"]


def test_branch_scales_preserve_role_rule_priority():
    syn_alignment = torch.tensor([1.0, 2.0])
    block_alignment = torch.tensor([[3.0, 4.0]])
    rec = {
        "branch_scale": torch.tensor([0.5, 1.5]),
        "dendritic_branch_scale": torch.tensor([[0.25], [0.75]]),
        "syn_alignment": syn_alignment,
        "block_alignment": block_alignment,
        "branch_type_scale": torch.tensor([9.0, 10.0]),
    }
    probe = _ModulatorProbe(
        _cfg(use_branch_role_rules=True, use_branch_type_rules=True)
    )
    result = probe._compute_local_layer_modulators(
        rec=rec,
        v0=torch.ones(2, 3),
        layer_depth=1,
    )

    torch.testing.assert_close(result.branch_scale, rec["branch_scale"])
    torch.testing.assert_close(
        result.dendritic_branch_scale,
        rec["dendritic_branch_scale"],
    )
    assert result.role_synaptic_alignment is syn_alignment
    assert result.role_block_alignment is block_alignment
    assert probe.calls == ["role_scales", "role_alignment"]


def test_branch_role_scale_helper_preserves_alignment_outputs():
    syn_alignment = torch.tensor([1.0, 2.0])
    block_alignment = torch.tensor([[3.0, 4.0]])
    rec = {
        "branch_scale": torch.tensor([0.5, 1.5]),
        "dendritic_branch_scale": torch.tensor([[0.25], [0.75]]),
        "syn_alignment": syn_alignment,
        "block_alignment": block_alignment,
    }
    probe = _ModulatorProbe(_cfg(use_branch_role_rules=True))

    (
        branch_scale,
        dendritic_branch_scale,
        role_synaptic_alignment,
        role_block_alignment,
    ) = probe._compute_local_branch_role_scales(rec)

    torch.testing.assert_close(branch_scale, rec["branch_scale"])
    torch.testing.assert_close(dendritic_branch_scale, rec["dendritic_branch_scale"])
    assert role_synaptic_alignment is syn_alignment
    assert role_block_alignment is block_alignment
    assert probe.calls == ["role_scales", "role_alignment"]


def test_branch_scales_preserve_branch_type_fallback():
    branch_type_scale = torch.tensor([0.75, 1.25])
    probe = _ModulatorProbe(_cfg(use_branch_type_rules=True))
    result = probe._compute_local_layer_modulators(
        rec={"branch_type_scale": branch_type_scale},
        v0=torch.ones(2, 3),
        layer_depth=1,
    )

    assert result.branch_scale is branch_type_scale
    assert result.dendritic_branch_scale is branch_type_scale
    assert result.role_synaptic_alignment is None
    assert result.role_block_alignment is None
    assert probe.calls == ["branch_type"]


def test_branch_type_scale_helper_preserves_shared_scale_output():
    branch_type_scale = torch.tensor([0.75, 1.25])
    probe = _ModulatorProbe(_cfg(use_branch_type_rules=True))

    (
        branch_scale,
        dendritic_branch_scale,
        role_synaptic_alignment,
        role_block_alignment,
    ) = probe._compute_local_branch_type_scales(
        {"branch_type_scale": branch_type_scale}
    )

    assert branch_scale is branch_type_scale
    assert dendritic_branch_scale is branch_type_scale
    assert role_synaptic_alignment is None
    assert role_block_alignment is None
    assert probe.calls == ["branch_type"]


def test_branch_length_scale_reuses_branch_type_fallback():
    branch_type_scale = torch.tensor([0.25, 1.75])
    probe = _ModulatorProbe(_cfg(use_branch_length_modulation=True))

    result = probe._compute_local_layer_modulators(
        rec={"branch_type_scale": branch_type_scale},
        v0=torch.ones(2, 3),
        layer_depth=1,
    )

    assert result.branch_scale is branch_type_scale
    assert result.dendritic_branch_scale is branch_type_scale
    assert result.role_synaptic_alignment is None
    assert result.role_block_alignment is None
    assert probe.calls == ["branch_type"]

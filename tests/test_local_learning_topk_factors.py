"""Tests for inhibitory TopK local-learning factor helpers."""

from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_topk_mixin import (
    LocalLearningTopKGradientMixin,
    _additive_inhibitory_topk_factor,
    _apply_inhibitory_topk_modulators,
    _conductance_inhibitory_topk_factor,
    _dispatch_topk_gradient_paths,
)


class _TopKFactorProbe(LocalLearningTopKGradientMixin):
    """Minimal object exposing the inhibitory TopK factor helpers."""

    def __init__(
        self,
        *,
        rule_variant: str = "3f",
        additive_mode: str = "none",
        use_driving_force: bool = True,
    ):
        self.local_cfg = SimpleNamespace(
            rule_variant=rule_variant,
            three_factor=SimpleNamespace(
                additive_gain_mode=additive_mode,
                use_driving_force=use_driving_force,
                e_rev_inh=-0.5,
                theta=0.25,
            ),
        )

    def _compute_additive_pseudo_signals(self, rec, v_n):
        return rec["pseudo_R"], rec["pseudo_drive"]


def test_apply_inhibitory_topk_modulators_preserves_scalar_rule_variants():
    factor = torch.tensor([[1.0, -2.0], [3.0, 4.0]])

    three_factor = _apply_inhibitory_topk_modulators(
        factor,
        rho=2.0,
        phi=3.0,
        branch_scale=5.0,
        rule_variant="3f",
    )
    four_factor = _apply_inhibitory_topk_modulators(
        factor,
        rho=2.0,
        phi=3.0,
        branch_scale=5.0,
        rule_variant="4f",
    )
    five_factor = _apply_inhibitory_topk_modulators(
        factor,
        rho=2.0,
        phi=3.0,
        branch_scale=5.0,
        rule_variant="5f",
    )

    torch.testing.assert_close(three_factor, factor * 5.0)
    torch.testing.assert_close(four_factor, factor * 2.0 * 5.0)
    torch.testing.assert_close(five_factor, factor * 2.0 * 3.0 * 5.0)


def test_apply_inhibitory_topk_modulators_preserves_tensor_rho_for_3f():
    factor = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    rho = torch.tensor([2.0, -0.5])
    branch_scale = torch.tensor([0.25, 3.0])

    actual = _apply_inhibitory_topk_modulators(
        factor,
        rho=rho,
        phi=7.0,
        branch_scale=branch_scale,
        rule_variant="3f",
    )

    expected = factor * rho.unsqueeze(0) * branch_scale.unsqueeze(0)
    torch.testing.assert_close(actual, expected)


def test_dispatch_topk_gradient_paths_preserves_order_and_first_return():
    seen = []

    def apply_path(path):
        seen.append(path.pathway)
        return f"{path.pathway}-layer"

    actual = _dispatch_topk_gradient_paths(
        (
            SimpleNamespace(pathway="exc"),
            SimpleNamespace(pathway="rec_exc"),
            SimpleNamespace(pathway="rec_inh"),
        ),
        apply_path,
    )

    assert seen == ["exc", "rec_exc", "rec_inh"]
    assert actual == "exc-layer"


def test_conductance_inhibitory_topk_factor_preserves_formula():
    e_v = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    r_tot = torch.tensor([[2.0, 3.0]])

    driving = _conductance_inhibitory_topk_factor(
        e_v=e_v,
        v_n=v_n,
        r_tot=r_tot,
        e_rev_inh=-0.5,
        theta=0.25,
        use_driving_force=True,
    )
    non_driving = _conductance_inhibitory_topk_factor(
        e_v=e_v,
        v_n=v_n,
        r_tot=r_tot,
        e_rev_inh=-0.5,
        theta=0.25,
        use_driving_force=False,
    )

    torch.testing.assert_close(driving, e_v * r_tot * (-0.5 - v_n))
    torch.testing.assert_close(non_driving, e_v * r_tot * (0.25 - v_n))


def test_additive_inhibitory_topk_factor_preserves_modes():
    e_v = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    post_factor_raw = torch.tensor([[9.0, 8.0]])
    pseudo_R = torch.tensor([[4.0, 2.0]])

    input_dependent = _additive_inhibitory_topk_factor(
        e_v=e_v,
        v_n=v_n,
        post_factor_raw=post_factor_raw,
        additive_mode="input_dependent",
        theta=0.25,
        pseudo_R=pseudo_R,
    )
    learned = _additive_inhibitory_topk_factor(
        e_v=e_v,
        v_n=v_n,
        post_factor_raw=post_factor_raw,
        additive_mode="learned_gain",
        theta=0.25,
        pseudo_R=None,
    )
    running = _additive_inhibitory_topk_factor(
        e_v=e_v,
        v_n=v_n,
        post_factor_raw=post_factor_raw,
        additive_mode="running_stats",
        theta=0.25,
        pseudo_R=None,
    )
    neutral = _additive_inhibitory_topk_factor(
        e_v=e_v,
        v_n=v_n,
        post_factor_raw=post_factor_raw,
        additive_mode="none",
        theta=0.25,
        pseudo_R=None,
    )

    torch.testing.assert_close(input_dependent, e_v * pseudo_R * (0.25 - v_n))
    torch.testing.assert_close(learned, -post_factor_raw)
    torch.testing.assert_close(running, -post_factor_raw)
    torch.testing.assert_close(neutral, -e_v)


def test_base_inhibitory_topk_factor_preserves_conductance_modes():
    e_v = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    r_tot = torch.tensor([[2.0, 3.0]])
    post_factor_raw = torch.tensor([[9.0, 8.0]])

    driving = _TopKFactorProbe(use_driving_force=True)
    non_driving = _TopKFactorProbe(use_driving_force=False)

    driving_factor = driving._compute_base_inhibitory_topk_factor(
        rec={},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=r_tot,
        layer_dynamics_mode="conductance",
    )
    non_driving_factor = non_driving._compute_base_inhibitory_topk_factor(
        rec={},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=r_tot,
        layer_dynamics_mode="conductance",
    )

    torch.testing.assert_close(driving_factor, e_v * r_tot * (-0.5 - v_n))
    torch.testing.assert_close(non_driving_factor, e_v * r_tot * (0.25 - v_n))


def test_base_inhibitory_topk_factor_preserves_additive_modes():
    e_v = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    r_tot = torch.tensor([[2.0, 3.0]])
    post_factor_raw = torch.tensor([[9.0, 8.0]])

    input_dependent = _TopKFactorProbe(additive_mode="input_dependent")
    pseudo_R = torch.tensor([[4.0, 2.0]])
    input_factor = input_dependent._compute_base_inhibitory_topk_factor(
        rec={"pseudo_R": pseudo_R, "pseudo_drive": torch.ones_like(pseudo_R)},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=r_tot,
        layer_dynamics_mode="additive",
    )
    torch.testing.assert_close(input_factor, e_v * pseudo_R * (0.25 - v_n))

    learned = _TopKFactorProbe(additive_mode="learned_gain")
    learned_factor = learned._compute_base_inhibitory_topk_factor(
        rec={},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=r_tot,
        layer_dynamics_mode="additive",
    )
    torch.testing.assert_close(learned_factor, -post_factor_raw)

    running = _TopKFactorProbe(additive_mode="running_stats")
    running_factor = running._compute_base_inhibitory_topk_factor(
        rec={},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=r_tot,
        layer_dynamics_mode="additive",
    )
    torch.testing.assert_close(running_factor, -post_factor_raw)

    neutral = _TopKFactorProbe(additive_mode="none")
    neutral_factor = neutral._compute_base_inhibitory_topk_factor(
        rec={},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=r_tot,
        layer_dynamics_mode="additive",
    )
    torch.testing.assert_close(neutral_factor, -e_v)


def test_compute_inhibitory_topk_factor_combines_base_and_modulators():
    probe = _TopKFactorProbe(rule_variant="5f", use_driving_force=True)
    e_v = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    r_tot = torch.tensor([[2.0, 3.0]])
    post_factor_raw = torch.tensor([[9.0, 8.0]])
    branch_scale = torch.tensor([0.25, 3.0])

    actual = probe._compute_inhibitory_topk_factor(
        rec={},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=r_tot,
        rho=2.0,
        phi=3.0,
        branch_scale=branch_scale,
        layer_dynamics_mode="conductance",
    )

    base = e_v * r_tot * (-0.5 - v_n)
    expected = base * 2.0 * 3.0 * branch_scale.unsqueeze(0)
    torch.testing.assert_close(actual, expected)

"""Tests for dendritic-block local-learning factor helpers."""

from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_dendritic_block_mixin import (
    LocalLearningDendriticBlockGradientMixin,
    _additive_dendritic_block_factor,
    _apply_dendritic_block_modulators,
    _conductance_dendritic_block_factor,
    _dendritic_block_homeostasis_factor,
    _expand_resistance_for_blocks,
)


class _DendriticBlockFactorProbe(LocalLearningDendriticBlockGradientMixin):
    """Minimal object exposing dendritic-block factor helpers."""

    def __init__(
        self,
        *,
        rule_variant: str = "3f",
        additive_mode: str = "none",
    ):
        self.local_cfg = SimpleNamespace(
            rule_variant=rule_variant,
            three_factor=SimpleNamespace(additive_gain_mode=additive_mode),
        )

    def _compute_additive_pseudo_signals(self, rec, v_n):
        return rec["pseudo_R"], rec["pseudo_drive"]


def test_expand_resistance_for_blocks_preserves_tensor_and_scalar_modes():
    v_n = torch.tensor([[0.5, 0.75]])
    r_tot = torch.tensor([[2.0, 3.0]])

    tensor_result = _expand_resistance_for_blocks(r_tot, v_n)
    scalar_result = _expand_resistance_for_blocks(4.0, v_n)

    torch.testing.assert_close(tensor_result, r_tot.unsqueeze(-1))
    torch.testing.assert_close(scalar_result, torch.ones_like(v_n).unsqueeze(-1) * 4.0)


def test_apply_dendritic_block_modulators_preserves_scalar_rule_variants():
    factor = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])

    three_factor = _apply_dendritic_block_modulators(
        factor,
        rho=2.0,
        phi=3.0,
        dendritic_branch_scale=5.0,
        rule_variant="3f",
    )
    four_factor = _apply_dendritic_block_modulators(
        factor,
        rho=2.0,
        phi=3.0,
        dendritic_branch_scale=5.0,
        rule_variant="4f",
    )
    five_factor = _apply_dendritic_block_modulators(
        factor,
        rho=2.0,
        phi=3.0,
        dendritic_branch_scale=5.0,
        rule_variant="5f",
    )

    torch.testing.assert_close(three_factor, factor * 5.0)
    torch.testing.assert_close(four_factor, factor * 2.0 * 5.0)
    torch.testing.assert_close(five_factor, factor * 2.0 * 3.0 * 5.0)


def test_apply_dendritic_block_modulators_preserves_tensor_branch_shapes():
    factor = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    rho = torch.tensor([2.0, -0.5])
    per_neuron_scale = torch.tensor([0.25, 3.0])
    per_block_scale = torch.tensor([[0.25, 3.0], [2.0, -1.0]])

    per_neuron = _apply_dendritic_block_modulators(
        factor,
        rho=rho,
        phi=3.0,
        dendritic_branch_scale=per_neuron_scale,
        rule_variant="4f",
    )
    per_block = _apply_dendritic_block_modulators(
        factor,
        rho=rho,
        phi=3.0,
        dendritic_branch_scale=per_block_scale,
        rule_variant="5f",
    )
    three_factor_tensor_rho = _apply_dendritic_block_modulators(
        factor,
        rho=rho,
        phi=3.0,
        dendritic_branch_scale=per_neuron_scale,
        rule_variant="3f",
    )

    torch.testing.assert_close(
        per_neuron,
        factor * rho.unsqueeze(0).unsqueeze(-1) * per_neuron_scale[None, :, None],
    )
    torch.testing.assert_close(
        per_block,
        factor * rho.unsqueeze(0).unsqueeze(-1) * 3.0 * per_block_scale.unsqueeze(0),
    )
    torch.testing.assert_close(
        three_factor_tensor_rho,
        factor * per_neuron_scale[None, :, None],
    )


def test_conductance_dendritic_block_factor_preserves_formula():
    e_v = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    r_tot = torch.tensor([[2.0, 3.0]])
    x_blk = torch.tensor([[[1.0, 0.25], [2.0, 0.5]]])

    actual = _conductance_dendritic_block_factor(
        e_v=e_v,
        v_n=v_n,
        r_tot=r_tot,
        x_blk=x_blk,
    )

    expected = e_v.unsqueeze(-1) * r_tot.unsqueeze(-1) * (x_blk - v_n.unsqueeze(-1))
    torch.testing.assert_close(actual, expected)


def test_additive_dendritic_block_factor_preserves_modes():
    e_v = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    x_blk = torch.tensor([[[1.0, 0.25], [2.0, 0.5]]])
    post_factor_raw = torch.tensor([[9.0, 8.0]])
    pseudo_R = torch.tensor([[4.0, 2.0]])

    input_dependent = _additive_dendritic_block_factor(
        e_v=e_v,
        v_n=v_n,
        post_factor_raw=post_factor_raw,
        x_blk=x_blk,
        additive_mode="input_dependent",
        pseudo_R=pseudo_R,
    )
    running = _additive_dendritic_block_factor(
        e_v=e_v,
        v_n=v_n,
        post_factor_raw=post_factor_raw,
        x_blk=x_blk,
        additive_mode="running_stats",
        pseudo_R=None,
    )
    neutral = _additive_dendritic_block_factor(
        e_v=e_v,
        v_n=v_n,
        post_factor_raw=post_factor_raw,
        x_blk=x_blk,
        additive_mode="none",
        pseudo_R=None,
    )

    torch.testing.assert_close(
        input_dependent,
        e_v.unsqueeze(-1) * pseudo_R.unsqueeze(-1) * (x_blk - v_n.unsqueeze(-1)),
    )
    torch.testing.assert_close(running, post_factor_raw.unsqueeze(-1) * x_blk)
    torch.testing.assert_close(neutral, e_v.unsqueeze(-1) * x_blk)


def test_base_dendritic_block_factor_preserves_conductance_mode():
    probe = _DendriticBlockFactorProbe()
    e_v = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    r_tot = torch.tensor([[2.0, 3.0]])
    x_blk = torch.tensor([[[1.0, 0.25], [2.0, 0.5]]])
    post_factor_raw = torch.tensor([[9.0, 8.0]])

    actual = probe._compute_base_dendritic_block_factor(
        rec={},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=r_tot,
        x_blk=x_blk,
        layer_dynamics_mode="conductance",
    )

    expected = e_v.unsqueeze(-1) * r_tot.unsqueeze(-1) * (x_blk - v_n.unsqueeze(-1))
    torch.testing.assert_close(actual, expected)


def test_base_dendritic_block_factor_preserves_additive_modes():
    e_v = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    x_blk = torch.tensor([[[1.0, 0.25], [2.0, 0.5]]])
    post_factor_raw = torch.tensor([[9.0, 8.0]])

    input_dependent = _DendriticBlockFactorProbe(additive_mode="input_dependent")
    pseudo_R = torch.tensor([[4.0, 2.0]])
    input_factor = input_dependent._compute_base_dendritic_block_factor(
        rec={"pseudo_R": pseudo_R, "pseudo_drive": torch.ones_like(pseudo_R)},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=1.0,
        x_blk=x_blk,
        layer_dynamics_mode="additive",
    )
    torch.testing.assert_close(
        input_factor,
        e_v.unsqueeze(-1) * pseudo_R.unsqueeze(-1) * (x_blk - v_n.unsqueeze(-1)),
    )

    running = _DendriticBlockFactorProbe(additive_mode="running_stats")
    running_factor = running._compute_base_dendritic_block_factor(
        rec={},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=1.0,
        x_blk=x_blk,
        layer_dynamics_mode="additive",
    )
    torch.testing.assert_close(running_factor, post_factor_raw.unsqueeze(-1) * x_blk)

    neutral = _DendriticBlockFactorProbe(additive_mode="none")
    neutral_factor = neutral._compute_base_dendritic_block_factor(
        rec={},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=1.0,
        x_blk=x_blk,
        layer_dynamics_mode="additive",
    )
    torch.testing.assert_close(neutral_factor, e_v.unsqueeze(-1) * x_blk)


def test_compute_dendritic_block_factor_combines_base_and_modulators():
    probe = _DendriticBlockFactorProbe(rule_variant="5f")
    e_v = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    r_tot = torch.tensor([[2.0, 3.0]])
    x_blk = torch.tensor([[[1.0, 0.25], [2.0, 0.5]]])
    post_factor_raw = torch.tensor([[9.0, 8.0]])
    branch_scale = torch.tensor([[0.25, 3.0], [2.0, -1.0]])

    actual = probe._compute_dendritic_block_factor(
        rec={},
        e_v=e_v,
        post_factor_raw=post_factor_raw,
        v_n=v_n,
        r_tot=r_tot,
        rho=2.0,
        phi=3.0,
        dendritic_branch_scale=branch_scale,
        x_blk=x_blk,
        layer_dynamics_mode="conductance",
    )

    base = e_v.unsqueeze(-1) * r_tot.unsqueeze(-1) * (x_blk - v_n.unsqueeze(-1))
    expected = base * 2.0 * 3.0 * branch_scale.unsqueeze(0)
    torch.testing.assert_close(actual, expected)


def test_dendritic_block_homeostasis_factor_preserves_resistance_modes():
    probe = _DendriticBlockFactorProbe()
    voltage_homeo_error = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    r_tot = torch.tensor([[2.0, 3.0]])
    x_blk = torch.tensor([[[1.0, 0.25], [2.0, 0.5]]])

    conductance = probe._compute_dendritic_block_homeostasis_factor(
        voltage_homeo_error=voltage_homeo_error,
        v_n=v_n,
        r_tot=r_tot,
        x_blk=x_blk,
        layer_dynamics_mode="conductance",
    )
    additive = probe._compute_dendritic_block_homeostasis_factor(
        voltage_homeo_error=voltage_homeo_error,
        v_n=v_n,
        r_tot=1.0,
        x_blk=x_blk,
        layer_dynamics_mode="additive",
    )

    torch.testing.assert_close(
        conductance,
        voltage_homeo_error.unsqueeze(-1)
        * r_tot.unsqueeze(-1)
        * (x_blk - v_n.unsqueeze(-1)),
    )
    torch.testing.assert_close(additive, voltage_homeo_error.unsqueeze(-1) * x_blk)


def test_dendritic_block_homeostasis_helper_preserves_resistance_modes():
    voltage_homeo_error = torch.tensor([[0.2, -0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    r_tot = torch.tensor([[2.0, 3.0]])
    x_blk = torch.tensor([[[1.0, 0.25], [2.0, 0.5]]])

    conductance = _dendritic_block_homeostasis_factor(
        voltage_homeo_error=voltage_homeo_error,
        v_n=v_n,
        r_tot=r_tot,
        x_blk=x_blk,
        layer_dynamics_mode="conductance",
    )
    additive = _dendritic_block_homeostasis_factor(
        voltage_homeo_error=voltage_homeo_error,
        v_n=v_n,
        r_tot=1.0,
        x_blk=x_blk,
        layer_dynamics_mode="additive",
    )

    torch.testing.assert_close(
        conductance,
        voltage_homeo_error.unsqueeze(-1)
        * r_tot.unsqueeze(-1)
        * (x_blk - v_n.unsqueeze(-1)),
    )
    torch.testing.assert_close(additive, voltage_homeo_error.unsqueeze(-1) * x_blk)

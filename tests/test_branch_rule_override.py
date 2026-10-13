"""Analysis-only branch rule overrides replace only the shunting voltage algebra."""

from __future__ import annotations

import pytest
import torch
from test_additive_comparison_controls import _components, _configured_layer, _inputs

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    BRANCH_RULE_OVERRIDES,
    BranchRuleOverride,
    override_branch_rules,
)


def _shunting():
    layer = _configured_layer(use_shunting=True)
    inputs = _inputs()
    E, inhibitory, C, G = _components(layer, *inputs)
    return layer, inputs, E, inhibitory, C, E + C, E + inhibitory + G


def _cases(layer, E, inhibitory, C, N, T):
    """Fields of each rule and the voltage it should give."""
    eps = layer.epsilon
    n0, t0 = N.mean(dim=0), T.mean(dim=0)
    d0 = 1 + t0 + eps
    v0 = n0 / d0
    rolled = torch.roll(T, shifts=1, dims=0)
    raw = E - inhibitory + C
    external = torch.tensor([0.3, 0.9])
    permutation = torch.tensor([1, 0])
    moments = {
        "raw_mean": raw.mean(dim=0),
        "raw_std": raw.std(dim=0),
        "voltage_mean": torch.tensor([0.1, -0.2]),
        "voltage_std": torch.tensor([0.5, 2.0]),
    }
    anchors = {"anchor_numerator": n0, "anchor_conductance": t0}
    return {
        "anchor_tangent": (anchors, v0 + ((N - n0) - v0 * (T - t0)) / d0),
        "sample_rolled_tangent": (anchors, v0 + ((N - n0) - v0 * (rolled - t0)) / d0),
        "frozen_denominator": ({"anchor_conductance": t0}, N / d0),
        "sample_rolled_denominator": ({}, N / (1 + rolled + eps)),
        "external_denominator": (
            {"external_conductance": external},
            N / (1 + external + eps),
        ),
        "permuted_current_denominator": (
            {"denominator_permutation": permutation},
            N / (1 + T.index_select(0, permutation) + eps),
        ),
        "raw_additive": ({}, raw),
        "raw_additive_moment_matched": (
            moments,
            (raw - moments["raw_mean"]) * (moments["voltage_std"] / moments["raw_std"])
            + moments["voltage_mean"],
        ),
    }


def test_every_rule_is_covered():
    layer, _, E, inhibitory, C, N, T = _shunting()
    assert set(_cases(layer, E, inhibitory, C, N, T)) == BRANCH_RULE_OVERRIDES


@pytest.mark.parametrize("rule", sorted(BRANCH_RULE_OVERRIDES))
def test_each_rule_replaces_only_the_branch_algebra(rule):
    layer, inputs, E, inhibitory, C, N, T = _shunting()
    fields, expected = _cases(layer, E, inhibitory, C, N, T)[rule]
    trained = layer(*inputs)
    state = {key: value.clone() for key, value in layer.state_dict().items()}

    with layer.override_branch_rule(BranchRuleOverride(rule, **fields)):
        observed = layer(*inputs)
        assert layer.state_dict().keys() == state.keys()

    torch.testing.assert_close(observed, expected)
    # The trained rule and state are back after the block.
    assert torch.equal(layer(*inputs), trained)
    assert all(torch.equal(layer.state_dict()[key], state[key]) for key in state)


def test_each_layer_takes_its_own_override():
    first, inputs, *_ = _shunting()
    second, *_ = _shunting()
    trained = first(*inputs)
    with override_branch_rules(
        {
            first: BranchRuleOverride("raw_additive"),
            second: BranchRuleOverride(
                "external_denominator", external_conductance=torch.tensor([0.5, 0.5])
            ),
        }
    ):
        raw = first(*inputs)
        external = second(*inputs)
    assert not torch.allclose(raw, trained)
    assert not torch.allclose(external, raw)


def test_a_permutation_leaves_a_conductance_without_batch_dimension_unchanged():
    layer = _configured_layer(use_shunting=True)
    numerator = torch.tensor([[0.5, 0.8], [1.0, 1.4]])
    constant = torch.tensor([0.2, 0.6])
    override = BranchRuleOverride(
        "permuted_current_denominator", denominator_permutation=torch.tensor([1, 0])
    )
    observed = override.voltage(
        layer, numerator, constant, excitation=numerator, inhibition=0, child_current=0
    )
    torch.testing.assert_close(observed, numerator / (1 + constant + layer.epsilon))


def test_a_permutation_must_cover_the_batch():
    layer, inputs, *_ = _shunting()
    override = BranchRuleOverride(
        "permuted_current_denominator", denominator_permutation=torch.tensor([0])
    )
    with layer.override_branch_rule(override):
        with pytest.raises(ValueError, match="1 indices for a batch of 2"):
            layer(*inputs)


@pytest.mark.parametrize(
    "rule, fields, error, match",
    [
        ("learned_tangent", {}, ValueError, "unknown branch rule override"),
        ("anchor_tangent", {"anchor_conductance": torch.ones(2)}, ValueError, "needs"),
        (
            "sample_rolled_denominator",
            {"anchor_conductance": torch.ones(2)},
            ValueError,
            "does not use anchor_conductance",
        ),
        ("frozen_denominator", {"anchor_conductance": [1.0]}, TypeError, "tensor"),
        (
            "permuted_current_denominator",
            {"denominator_permutation": torch.tensor([1.0, 0.0])},
            ValueError,
            "integer",
        ),
    ],
)
def test_rule_fields_are_checked_when_the_override_is_built(rule, fields, error, match):
    with pytest.raises(error, match=match):
        BranchRuleOverride(rule, **fields)


def test_overrides_are_refused_where_they_would_be_ignored():
    with pytest.raises(ValueError, match="additive layer would ignore them"):
        with _configured_layer().override_branch_rule(
            BranchRuleOverride("raw_additive")
        ):
            pass
    leaky = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=3,
        excitatory_synapses_per_branch=2,
        use_shunting=True,
        leak_conductance=2.0,
        reactivate=False,
    )
    with pytest.raises(ValueError, match="fixed unit leak"):
        with leaky.override_branch_rule(BranchRuleOverride("raw_additive")):
            pass

    shunting = _configured_layer(use_shunting=True)
    current = torch.full((1, 2), 0.3)
    with shunting.override_branch_rule(BranchRuleOverride("raw_additive")):
        with pytest.raises(ValueError, match="voltages computed from currents"):
            shunting.voltage_from_currents(current, 0, current)


def test_the_trained_rule_returns_after_an_exception():
    layer, inputs, *_ = _shunting()
    trained = layer(*inputs)
    with pytest.raises(RuntimeError, match="stop"):
        with layer.override_branch_rule(BranchRuleOverride("raw_additive")):
            raise RuntimeError("stop")
    assert torch.equal(layer(*inputs), trained)


def test_a_rule_written_on_the_layer_raises_instead_of_being_ignored():
    layer, inputs, *_ = _shunting()
    layer._analysis_rule_override = "frozen_denominator"
    with pytest.raises(TypeError, match="override_branch_rules"):
        layer(*inputs)

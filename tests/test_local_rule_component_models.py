"""The local-rule component analysis follows the LocalCA trainer on built models."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from test_biological_constraints import _einet, _population
from test_trainer_configuration import IN_DIM, N_CLASSES, _wrap
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.local_rule_components import (
    LocalRuleComponentAnalyzer,
    _activation_derivative_for_record,
    _delta_out_for_outputs,
)
from dendritic_modeling.config.analysis import LocalRuleComponentAnalysisParams
from dendritic_modeling.networks.architectures.factory import get_architecture


def _model(core_type, core):
    torch.manual_seed(0)
    return _wrap(get_architecture(core_type, core, input_dim=IN_DIM))


def _dataset(n=8):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(n, IN_DIM, generator=generator)
    y = torch.randint(0, N_CLASSES, (n,), generator=generator)
    return TensorDataset(x, y)


def _rows(model, **params):
    analyzer = LocalRuleComponentAnalyzer(LocalRuleComponentAnalysisParams(**params))
    return analyzer.analyze(model=model, test_dataset=_dataset())["rows"]


def test_inhibitory_population_rows_are_labelled_from_the_model():
    rows = _rows(_model("population_network", _population()))
    labels = {
        (".populations.i." in row["module_name"], row["population"]) for row in rows
    }
    assert labels == {(True, "explicit_inhibitory"), (False, "excitatory")}


def test_layers_the_local_rules_do_not_model_are_rejected():
    model = _model("einet", _einet(**{"morphology.leak_conductance": 2.0}))
    with pytest.raises(ValueError, match="configurable leak"):
        _rows(model)


def test_gate_slopes_without_a_closed_form_are_exact():
    layer = nn.Module()
    layer.reactivation = nn.SiLU()
    v_n = torch.linspace(-3.0, 3.0, 7).reshape(1, -1)
    expected = torch.sigmoid(v_n) * (1 + v_n * (1 - torch.sigmoid(v_n)))
    derivative = _activation_derivative_for_record({"layer": layer, "v_n": v_n})
    torch.testing.assert_close(derivative, expected)


def test_indexed_synapse_banks_get_synapse_rows():
    rows = _rows(_model("einet", _einet(**{"sparsity.type": "indexed"})))
    synapse_types = {row["synapse_type"] for row in rows}
    assert {"excitatory", "inhibitory"} <= synapse_types


def test_a_frozen_model_still_maps_its_error_through_the_decoder():
    model = _model("einet", _einet())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rows = _rows(model, broadcast_mode="per_soma")
    assert any(row["component"] == "broadcast_error" for row in rows)


def test_bce_error_is_sigmoid_minus_target():
    y_hat = torch.tensor([[0.3, -1.2]])
    y = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(
        _delta_out_for_outputs(y_hat, y, "bce"), torch.sigmoid(y_hat) - y
    )


@pytest.mark.parametrize(
    "params, match",
    [
        ({"broadcast_mode": "low_rank"}, "broadcast_mode"),
        ({"broadcast_mode": "neuron_wise"}, "broadcast_mode"),
        ({"error_mode": "hinge"}, "error_mode"),
        ({"e_rev_exc": 0.8}, "reversal potentials"),
        ({"e_rev_inh": -0.1}, "reversal potentials"),
    ],
)
def test_unsupported_analysis_settings_raise(params, match):
    with pytest.raises(ValueError, match=match):
        LocalRuleComponentAnalysisParams(**params)


def test_a_recurrent_unroll_is_rejected():
    model = _model("einet", _einet())
    branch = next(
        module
        for module in model.modules()
        if type(module).__name__ == "DendriticBranchLayer"
    )
    original_forward = branch.forward

    def _twice(*args, **kwargs):
        original_forward(*args, **kwargs)
        return original_forward(*args, **kwargs)

    branch.forward = _twice
    with pytest.raises(ValueError, match="more than once"):
        _rows(model, include_broadcast=False)

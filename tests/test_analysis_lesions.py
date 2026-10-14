"""Lesions switch synapses off whatever their weight transform."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from test_biological_constraints import _einet

from dendritic_modeling.analysis.metrics.layer_contribution import (
    SingleLayerContributionAnalyzer,
)
from dendritic_modeling.analysis.utils.lesions import lesion_raw_value
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import DendriticBranchLayer, Identity
from dendritic_modeling.networks.architectures.factory import get_architecture


@pytest.mark.parametrize(
    "transform, value",
    [("identity", 0.0), ("relu", 0.0), ("exp", -1000.0), ("softplus", -1000.0)],
)
def test_lesion_value_follows_the_banks_transform(transform, value):
    assert lesion_raw_value(SimpleNamespace(weight_transform=transform)) == value


def test_unknown_transform_cannot_be_lesioned():
    with pytest.raises(ValueError, match="cannot lesion"):
        lesion_raw_value(SimpleNamespace(weight_transform="tanh"))


def _signed_einet():
    torch.manual_seed(0)
    core = get_architecture("dendritic_signed", _einet(), input_dim=8)
    return Classifier(Identity(8), core, nn.Linear(core.output_dim, 3))


def _call(analyzer, model, module_name, module):
    state = {key: value.clone() for key, value in model.state_dict().items()}
    return analyzer.evaluate_layer_contribution(
        model=model,
        test_dataset=None,
        module=module,
        module_name=module_name,
        contribution_type="both",
        results_dict={"both": {}},
        state_dict=state,
    )


def test_layer_contribution_silences_signed_layers_and_restores_the_model():
    model = _signed_einet()
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DendriticBranchLayer)
        and module.branch_excitation is not None
        and module.branch_inhibition is not None
    ]
    module_name, module = layers[0]
    others = [layer for name, layer in layers if name != module_name]
    assert others
    analyzer = SingleLayerContributionAnalyzer(SimpleNamespace(accuracy=True))
    before = {key: value.clone() for key, value in model.state_dict().items()}

    def _check(*_args, **_kwargs):
        for layer in others:
            assert torch.all(layer.branch_excitation.weight() == 0)
            assert torch.all(layer.branch_inhibition.weight() == 0)
        raise RuntimeError("evaluation failed")

    target = "dendritic_modeling.analysis.metrics.layer_contribution.evaluate_accuracy"
    with patch(target, side_effect=_check), pytest.raises(RuntimeError):
        _call(analyzer, model, module_name, module)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])

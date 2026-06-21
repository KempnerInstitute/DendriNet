"""Tests for local-rule component analysis helpers."""

import torch
import torch.nn as nn

from dendritic_modeling.analysis.tools.local_rule_components import (
    LocalRuleComponentAnalyzer,
)
from dendritic_modeling.config.analysis import LocalRuleComponentAnalysisParams
from dendritic_modeling.networks import DendriticBranchLayer, TopKLinear


class _FeedforwardBranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch = DendriticBranchLayer(
            output_dim=2,
            excitatory_input_dim=3,
            excitatory_synapses_per_branch=2,
            inhibitory_input_dim=3,
            inhibitory_synapses_per_branch=2,
            use_shunting=False,
            reactivate=False,
            topk_noise_level=0.0,
            weight_transform="identity",
        )

    def forward(self, x: torch.Tensor, inhibitory_input: torch.Tensor) -> torch.Tensor:
        return self.branch(x, inhibitory_input=inhibitory_input)


def test_local_rule_component_recorders_preserve_topk_keys() -> None:
    torch.manual_seed(41)
    model = _FeedforwardBranchModel()
    analyzer = LocalRuleComponentAnalyzer(
        LocalRuleComponentAnalysisParams(include_broadcast=False)
    )
    x = torch.tensor([[0.2, 0.6, 0.1], [0.4, 0.3, 0.7]], dtype=torch.float32)
    inhibitory_input = torch.tensor(
        [[0.5, 0.1, 0.4], [0.3, 0.9, 0.2]], dtype=torch.float32
    )

    records, handles, decoder_cache = analyzer._attach_recorders(model)
    try:
        _ = model(x, inhibitory_input)
    finally:
        for handle in handles:
            handle.remove()

    assert decoder_cache == {}
    assert len(records) == 1
    rec = records[0]
    assert rec["module_name"] == "branch"
    assert rec["population"] == "excitatory"
    assert rec["branch_module_index"] == -1
    assert torch.equal(rec["x_exc"], x)
    assert torch.equal(rec["x_inh"], inhibitory_input)
    assert isinstance(rec["exc_module"], TopKLinear)
    assert isinstance(rec["inh_module"], TopKLinear)
    assert rec["exc_out"].shape == (2, 2)
    assert rec["inh_out"].shape == (2, 2)
    assert rec["exc_mask"].shape == model.branch.branch_excitation.pre_w.shape
    assert rec["inh_mask"].shape == model.branch.branch_inhibition.pre_w.shape

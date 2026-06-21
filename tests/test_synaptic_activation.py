import torch

from dendritic_modeling.analysis.tools.synaptic_activation import (
    SynapticActivationAnalyzer,
)
from dendritic_modeling.config import SynapticActivationAnalysisParams


def test_synaptic_activation_summary_math_is_stable():
    analyzer = SynapticActivationAnalyzer(
        SynapticActivationAnalysisParams(
            network_summary=True,
            layer_summary=True,
            branch_summary=True,
            raw_summary=True,
            logspace=False,
        )
    )
    activation = torch.tensor(
        [
            [[0.0, 2.0], [4.0, 0.0]],
            [[6.0, 8.0], [0.0, 10.0]],
        ]
    )
    analyzer.data_dict = {"layer0": {"ee": activation.clone()}}

    results = analyzer._build_results_from_activations()

    positive_values = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    assert results["network_summary"]["ee"]["mean"] == positive_values.mean().item()
    assert (
        results["network_summary"]["ee"]["std"]
        == positive_values.std(unbiased=False).item()
    )
    assert results["layer_summary"]["layer0"]["ee"] == results["network_summary"]["ee"]
    assert (
        results["branch_summary"]["layer0"]["ee"]["mean"]
        == activation.mean(dim=0).tolist()
    )
    assert (
        results["branch_summary"]["layer0"]["ee"]["std"]
        == activation.std(dim=0, unbiased=False).tolist()
    )
    assert results["raw_summary"]["layer0"]["ee"] == activation.tolist()

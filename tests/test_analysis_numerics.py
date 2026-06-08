import torch

from dendritic_modeling.analysis.tools.compartment_snr import CompartmentSNRAnalyzer
from dendritic_modeling.config.analysis import CompartmentSNRAnalysisParams


def test_compartment_snr_multivariate_single_feature_is_finite():
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())
    data = torch.tensor([[0.0], [0.2], [1.0], [1.2]])
    labels = torch.tensor([0, 0, 1, 1])

    stats = analyzer._compute_discriminability_multivariate(
        data_tensor=data,
        labels=labels,
        prefix="single",
        stats_dict={},
    )

    assert torch.isfinite(torch.tensor(list(stats.values()))).all()
    assert stats["single_d_total"] > 0.0

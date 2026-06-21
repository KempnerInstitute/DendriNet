import torch
from torch import nn

from dendritic_modeling.analysis.tools.synaptic_activation import (
    SynapticActivationAnalyzer,
)
from dendritic_modeling.config import SynapticActivationAnalysisParams
from dendritic_modeling.networks import DendriticBranchLayer


def _count_registered_forward_hooks(model: torch.nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


def _branch_layer() -> DendriticBranchLayer:
    return DendriticBranchLayer(
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


class _SynapticActivationHookModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = _branch_layer()


def test_synaptic_activation_hooks_collect_topk_outputs():
    torch.manual_seed(7)
    model = _SynapticActivationHookModel()
    analyzer = SynapticActivationAnalyzer(SynapticActivationAnalysisParams())
    analyzer.data_dict = {}
    x_exc = torch.rand(4, 3)
    x_inh = torch.rand(4, 3)

    handles = analyzer.attach_forward_hooks(model)
    try:
        with torch.no_grad():
            model.layer(x_exc, x_inh, None)
    finally:
        analyzer.remove_forward_hooks(handles)

    assert len(handles) == 2
    assert _count_registered_forward_hooks(model) == 0
    assert set(analyzer.data_dict["layer"]) == {"exc", "inh"}
    assert len(analyzer.data_dict["layer"]["exc"]) == 1
    assert len(analyzer.data_dict["layer"]["inh"]) == 1

    with torch.no_grad():
        expected_exc = model.layer.branch_excitation(x_exc).detach().cpu()
        expected_inh = model.layer.branch_inhibition(x_inh).detach().cpu()
    torch.testing.assert_close(analyzer.data_dict["layer"]["exc"][0], expected_exc)
    torch.testing.assert_close(analyzer.data_dict["layer"]["inh"][0], expected_inh)


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

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.correlation_analysis import CorrelationAnalyzer
from dendritic_modeling.config.analysis import (
    CorrelationAnalysisParams,
    CorrelationComputeParams,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture


def _direct_inhibitory_model() -> Classifier:
    config = {
        "architecture": {
            "excitatory_layer_sizes": [4],
            "inhibitory_layer_sizes": [],
            "excitatory_branch_factors": [2],
            "inhibitory_branch_factors": [],
            "inhibitory_network_type": "dendritic",
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [2],
            "ei_synapses_per_branch_per_layer": [0],
            "ie_synapses_per_branch_per_layer": [2],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {
            "input_mode": 1,
            "independent_pathways": True,
            "excitatory_dim": 3,
            "inhibitory_dim": 3,
            "allow_direct_inhibitory_stream": True,
        },
        "morphology": {
            "use_shunting": True,
            "weight_transform": "softplus",
            "dbl_init_method": "analytical_expectation",
            "somatic_synapses": False,
        },
        "sparsity": {
            "init_method": "xavier_normal",
            "noise_level": 0.0,
            "type": "standard",
        },
        "reactivation": {
            "enabled": True,
            "type": "param_tanh",
            "init_m": 1.5,
            "init_b": 0.5,
        },
        "blocklinear": {"efficient": False},
        "implementation": {},
    }
    core = get_architecture("dendritic_shunting", config, input_dim=6)
    return Classifier(Identity(6), core, Identity(4))


def _toy_dataset() -> tuple[torch.Tensor, torch.Tensor, TensorDataset]:
    torch.manual_seed(3)
    x = torch.rand(8, 6)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return x, labels, TensorDataset(x, labels)


def _count_registered_forward_hooks(model: torch.nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


def test_correlation_branch_hooks_collect_expected_payloads():
    model = _direct_inhibitory_model()
    x, _labels, _dataset = _toy_dataset()
    analyzer = CorrelationAnalyzer(CorrelationAnalysisParams())

    handles = analyzer.attach_forward_hooks(model)
    try:
        with torch.no_grad():
            model(x)
    finally:
        analyzer.remove_forward_hooks(handles)

    assert len(handles) == 1
    layer_data = analyzer.data_dict[
        "core_network.layers.0.excitatory_cells.branch_layers.0"
    ]
    assert set(layer_data) == {
        "excitation",
        "inhibition",
        "output",
        "excitatory_input",
        "inhibitory_input",
    }
    assert layer_data["excitation"].shape == (8, 8)
    assert layer_data["inhibition"].shape == (8, 8)
    assert layer_data["output"].shape == (8, 8)
    assert layer_data["excitatory_input"].shape == (8, 3)
    assert layer_data["inhibitory_input"].shape == (8, 3)


def test_correlation_synaptic_hooks_collect_expected_payloads():
    model = _direct_inhibitory_model()
    x, _labels, _dataset = _toy_dataset()
    params = CorrelationAnalysisParams(computation_level="synaptic")
    analyzer = CorrelationAnalyzer(params)

    handles = analyzer.attach_forward_hooks(model)
    try:
        with torch.no_grad():
            model(x)
    finally:
        analyzer.remove_forward_hooks(handles)

    assert len(handles) == 4
    first_layer = analyzer.data_dict[
        "core_network.layers.0.excitatory_cells.branch_layers.0"
    ]
    assert set(first_layer) == {
        "E_synaptic",
        "E_indices",
        "E_weights",
        "E_branch_output",
        "I_synaptic",
        "I_indices",
        "I_weights",
        "I_branch_output",
        "output",
    }
    assert first_layer["E_synaptic"].shape == (8, 8, 2)
    assert first_layer["I_synaptic"].shape == (8, 8, 2)
    assert first_layer["E_indices"].shape == (8, 2)
    assert first_layer["I_indices"].shape == (8, 2)
    assert first_layer["E_weights"].shape == (8, 2)
    assert first_layer["I_weights"].shape == (8, 2)
    assert first_layer["E_branch_output"].shape == (8, 8)
    assert first_layer["I_branch_output"].shape == (8, 8)
    assert first_layer["output"].shape == (8, 8)

    second_layer = analyzer.data_dict[
        "core_network.layers.0.excitatory_cells.branch_layers.1"
    ]
    assert set(second_layer) == {"output"}
    assert second_layer["output"].shape == (8, 4)


def test_correlation_synaptic_hooks_remove_partial_registration(monkeypatch):
    model = _direct_inhibitory_model()
    params = CorrelationAnalysisParams(computation_level="synaptic")
    analyzer = CorrelationAnalyzer(params)
    first_layer = model.core_network.layers[0].excitatory_cells.branch_layers[0]
    original_register = first_layer.branch_inhibition.register_forward_hook

    def _failing_register(*args, **kwargs):
        raise RuntimeError("correlation hook registration failed")

    monkeypatch.setattr(
        first_layer.branch_inhibition,
        "register_forward_hook",
        _failing_register,
    )

    with pytest.raises(RuntimeError, match="correlation hook registration failed"):
        analyzer.attach_forward_hooks(model)

    monkeypatch.setattr(
        first_layer.branch_inhibition,
        "register_forward_hook",
        original_register,
    )
    assert _count_registered_forward_hooks(model) == 0


def test_correlation_nested_compute_config_overrides_legacy_flat_fields():
    params = CorrelationAnalysisParams(
        computation_level="synaptic",
        max_samples=17,
        compute=CorrelationComputeParams(
            computation_level="single_branch",
            max_samples=23,
        ),
    )

    analyzer = CorrelationAnalyzer(params)

    assert analyzer.computation_level == "single_branch"
    assert analyzer.max_samples == 23


def test_correlation_legacy_flat_config_is_used_when_nested_block_is_default():
    params = CorrelationAnalysisParams(
        computation_level="synaptic",
        max_samples=17,
        batch_process=False,
        per_layer_analysis=False,
        per_einet_analysis=False,
        per_neuron_analysis=True,
        n_synapse_pairs=23,
        sampling_strategy="random",
        stratified_categories=["within_branch"],
        max_pairs_per_category=5,
    )

    analyzer = CorrelationAnalyzer(params)

    assert analyzer.computation_level == "synaptic"
    assert analyzer.max_samples == 17
    assert analyzer.batch_process is False
    assert analyzer.per_layer_analysis is False
    assert analyzer.per_einet_analysis is False
    assert analyzer.per_neuron_analysis is True
    assert analyzer.n_synapse_pairs == 23
    assert analyzer.sampling_strategy == "random"
    assert analyzer.stratified_categories == ["within_branch"]
    assert analyzer.max_pairs_per_category == 5


def test_correlation_network_activity_arrays_flatten_and_concatenate_layers():
    analyzer = CorrelationAnalyzer(CorrelationAnalysisParams())
    analyzer.data_dict = {
        "layer0": {
            "excitation": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
            "inhibition": torch.arange(12, 24, dtype=torch.float32).reshape(2, 2, 3),
            "output": torch.arange(24, 36, dtype=torch.float32).reshape(2, 2, 3),
        },
        "layer1": {
            "excitation": torch.arange(36, 44, dtype=torch.float32).reshape(2, 4),
            "inhibition": torch.arange(44, 52, dtype=torch.float32).reshape(2, 4),
            "output": torch.arange(52, 60, dtype=torch.float32).reshape(2, 4),
        },
    }

    network_E, network_I, network_Vout = analyzer._network_activity_arrays()

    np.testing.assert_array_equal(
        network_E,
        np.hstack(
            [
                np.arange(12, dtype=np.float32).reshape(2, 6),
                np.arange(36, 44, dtype=np.float32).reshape(2, 4),
            ]
        ),
    )
    np.testing.assert_array_equal(
        network_I,
        np.hstack(
            [
                np.arange(12, 24, dtype=np.float32).reshape(2, 6),
                np.arange(44, 52, dtype=np.float32).reshape(2, 4),
            ]
        ),
    )
    np.testing.assert_array_equal(
        network_Vout,
        np.hstack(
            [
                np.arange(24, 36, dtype=np.float32).reshape(2, 6),
                np.arange(52, 60, dtype=np.float32).reshape(2, 4),
            ]
        ),
    )


def test_correlation_branch_result_includes_tuning_and_summary_stats():
    analyzer = CorrelationAnalyzer(CorrelationAnalysisParams())
    labels = np.array([0, 0, 1, 1])
    E = np.array([[0.0, 1.0], [0.2, 1.2], [1.0, 0.0], [1.2, 0.2]])
    Inh = np.array([[1.0, 0.0], [1.1, 0.1], [0.0, 1.0], [0.1, 1.1]])
    Vout = np.array([[0.1, 0.3], [0.2, 0.4], [0.7, 0.9], [0.8, 1.0]])

    result = analyzer._compute_branch_result(E, Inh, Vout, labels)

    assert "tuning" in result
    assert "summary_stats" in result
    assert result["tuning"]["n_classes"] == 2
    assert "EE_total_mean" in result["summary_stats"]

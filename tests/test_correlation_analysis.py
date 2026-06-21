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

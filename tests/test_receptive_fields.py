import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.core.branch_activation import BranchActivationAnalyzer
from dendritic_modeling.analysis.tools.compartment_snr import CompartmentSNRAnalyzer
from dendritic_modeling.analysis.tools.compartment_statistics import (
    CompartmentStatisticsAnalyzer,
)
from dendritic_modeling.analysis.tools.receptive_fields import ReceptiveFieldAnalyzer
from dendritic_modeling.analysis.tools.synaptic_activation import (
    SynapticActivationAnalyzer,
)
from dendritic_modeling.config.analysis import (
    BranchActivationAnalysisParams,
    CompartmentSNRAnalysisParams,
    CompartmentStatisticsAnalysisParams,
    ReceptiveFieldAnalysisParams,
    SynapticActivationAnalysisParams,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import DendriticBranchLayer, Identity
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


def _toy_dataset():
    torch.manual_seed(3)
    x = torch.rand(8, 6)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return x, labels, TensorDataset(x, labels)


def _count_registered_forward_hooks(model: torch.nn.Module) -> int:
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in model.modules()
    )


def _replace_forward_with_failure(model: torch.nn.Module, message: str) -> None:
    def _failing_forward(*args, **kwargs):
        raise RuntimeError(message)

    model.forward = _failing_forward


def test_compartment_snr_captures_direct_inhibitory_stream():
    model = _direct_inhibitory_model()
    x, labels, _dataset = _toy_dataset()
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())

    analyzer.collect_raw_data(model, x)

    assert hasattr(analyzer, "excitatory_inputs")
    assert hasattr(analyzer, "inhibitory_inputs")
    assert analyzer.excitatory_inputs.shape == (8, 3)
    assert analyzer.inhibitory_inputs.shape == (8, 3)

    stats = analyzer._input_analysis(labels)
    assert "exc_inputs_d_total" in stats
    assert "inh_inputs_d_total" in stats
    assert torch.isfinite(torch.tensor(list(stats.values()))).all()


def test_compartment_snr_collects_expected_branch_channels():
    model = _direct_inhibitory_model()
    x, _labels, _dataset = _toy_dataset()
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())

    analyzer.collect_raw_data(model, x)

    first_layer = analyzer.raw_dict["branch_layers.0"]
    second_layer = analyzer.raw_dict["branch_layers.1"]
    for branch_data in first_layer.values():
        assert set(branch_data) == {"depth", "exc", "inh", "vinf", "vout"}
        assert branch_data["exc"].shape == (x.shape[0],)
        assert branch_data["inh"].shape == (x.shape[0],)
        assert branch_data["vinf"].shape == (x.shape[0],)
        assert branch_data["vout"].shape == (x.shape[0],)

    for branch_data in second_layer.values():
        assert set(branch_data) == {"depth", "upstream", "vinf", "vout"}
        assert branch_data["upstream"].shape == (x.shape[0],)
        assert branch_data["vinf"].shape == (x.shape[0],)
        assert branch_data["vout"].shape == (x.shape[0],)


def test_compartment_snr_removes_hooks_after_failed_collection():
    model = _direct_inhibitory_model()
    x, _labels, _dataset = _toy_dataset()
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())
    assert _count_registered_forward_hooks(model) == 0

    _replace_forward_with_failure(model, "snr failed")
    with pytest.raises(RuntimeError, match="snr failed"):
        analyzer.collect_raw_data(model, x)

    assert _count_registered_forward_hooks(model) == 0


def test_compartment_snr_removes_hooks_after_failed_hook_registration(monkeypatch):
    model = _direct_inhibitory_model()
    x, _labels, _dataset = _toy_dataset()
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())

    def _failing_register(*_args, **_kwargs):
        raise RuntimeError("snr hook registration failed")

    monkeypatch.setattr(analyzer, "_register_activation_hook", _failing_register)

    with pytest.raises(RuntimeError, match="snr hook registration failed"):
        analyzer.collect_raw_data(model, x)

    assert _count_registered_forward_hooks(model) == 0


def test_compartment_statistics_collects_expected_branch_payloads():
    model = _direct_inhibitory_model()
    x, labels, _dataset = _toy_dataset()
    analyzer = CompartmentStatisticsAnalyzer(
        CompartmentStatisticsAnalysisParams(compute_entropy=False)
    )

    analyzer.collect_raw_data(model, x, labels)

    first_layer = analyzer.raw_dict[
        "core_network.layers.0.excitatory_cells.branch_layers.0"
    ]
    second_layer = analyzer.raw_dict[
        "core_network.layers.0.excitatory_cells.branch_layers.1"
    ]
    for branch_data in first_layer.values():
        assert set(branch_data) == {
            "depth",
            "exc_weights",
            "inh_weights",
            "exc_inputs",
            "inh_inputs",
            "exc_activations",
            "inh_activations",
        }
        assert branch_data["exc_weights"].shape == (2,)
        assert branch_data["inh_weights"].shape == (2,)
        assert branch_data["exc_inputs"].shape == (x.shape[0], 2)
        assert branch_data["inh_inputs"].shape == (x.shape[0], 2)
        assert branch_data["exc_activations"].shape == (x.shape[0],)
        assert branch_data["inh_activations"].shape == (x.shape[0],)

    for branch_data in second_layer.values():
        assert set(branch_data) == {"depth", "branch_weights"}
        assert branch_data["branch_weights"].shape == (2,)


def test_compartment_statistics_removes_hooks_after_failed_collection():
    model = _direct_inhibitory_model()
    x, labels, _dataset = _toy_dataset()
    analyzer = CompartmentStatisticsAnalyzer(
        CompartmentStatisticsAnalysisParams(compute_entropy=False)
    )
    assert _count_registered_forward_hooks(model) == 0

    _replace_forward_with_failure(model, "statistics failed")
    with pytest.raises(RuntimeError, match="statistics failed"):
        analyzer.collect_raw_data(model, x, labels)

    assert _count_registered_forward_hooks(model) == 0


def test_compartment_statistics_removes_hooks_after_failed_hook_registration(
    monkeypatch,
):
    model = _direct_inhibitory_model()
    x, labels, _dataset = _toy_dataset()
    analyzer = CompartmentStatisticsAnalyzer(
        CompartmentStatisticsAnalysisParams(compute_entropy=False)
    )
    original_register = analyzer._register_synapse_capture_hook
    call_count = 0

    def _register_once_then_fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_register(*args, **kwargs)
        raise RuntimeError("statistics hook registration failed")

    monkeypatch.setattr(
        analyzer,
        "_register_synapse_capture_hook",
        _register_once_then_fail,
    )

    with pytest.raises(RuntimeError, match="statistics hook registration failed"):
        analyzer.collect_raw_data(model, x, labels)

    assert call_count == 2
    assert _count_registered_forward_hooks(model) == 0


def test_synaptic_activation_removes_hooks_after_failed_hook_registration(
    monkeypatch,
):
    model = _direct_inhibitory_model()
    analyzer = SynapticActivationAnalyzer(SynapticActivationAnalysisParams())
    analyzer.data_dict = {}
    original_register = analyzer._register_synapse_hook
    call_count = 0

    def _register_once_then_fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_register(*args, **kwargs)
        raise RuntimeError("synaptic hook registration failed")

    monkeypatch.setattr(analyzer, "_register_synapse_hook", _register_once_then_fail)

    with pytest.raises(RuntimeError, match="synaptic hook registration failed"):
        analyzer.attach_forward_hooks(model)

    assert call_count == 2
    assert _count_registered_forward_hooks(model) == 0


def test_branch_activation_removes_hooks_after_failed_hook_registration(monkeypatch):
    model = _direct_inhibitory_model()
    analyzer = BranchActivationAnalyzer(BranchActivationAnalysisParams())
    analyzer.data_dict = {}
    original_register = DendriticBranchLayer.register_forward_hook
    call_count = 0

    def _register_once_then_fail(module, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_register(module, *args, **kwargs)
        raise RuntimeError("branch hook registration failed")

    monkeypatch.setattr(
        DendriticBranchLayer,
        "register_forward_hook",
        _register_once_then_fail,
    )

    with pytest.raises(RuntimeError, match="branch hook registration failed"):
        analyzer.attach_forward_hooks(model)

    assert call_count == 2
    assert _count_registered_forward_hooks(model) == 0


def test_receptive_field_removes_hooks_after_failed_hook_registration(monkeypatch):
    model = _direct_inhibitory_model()
    x, _labels, _dataset = _toy_dataset()
    analyzer = ReceptiveFieldAnalyzer(
        ReceptiveFieldAnalysisParams(compute_activation_tuning=True)
    )
    original_register = analyzer._register_activation_hook
    call_count = 0

    def _register_once_then_fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_register(*args, **kwargs)
        raise RuntimeError("hook registration failed")

    monkeypatch.setattr(analyzer, "_register_activation_hook", _register_once_then_fail)

    with pytest.raises(RuntimeError, match="hook registration failed"):
        analyzer._collect_activation_raw(model, x)

    assert call_count == 2
    assert _count_registered_forward_hooks(model) == 0


def test_receptive_field_checkpoint_and_activation_tuning(tmp_path):
    model = _direct_inhibitory_model()
    _x, _labels, dataset = _toy_dataset()
    params = ReceptiveFieldAnalysisParams(
        n_rows=1,
        n_cols=2,
        compute_activation_tuning=True,
    )
    analyzer = ReceptiveFieldAnalyzer(params)

    results = analyzer.analyze(
        model=model,
        test_dataset=dataset,
        device="cpu",
        save_path=str(tmp_path),
        filename="rf",
    )

    assert results is not None
    assert results["n_receptive_fields"] > 0
    assert "activation_tuning" in results
    first_layer = analyzer.activation_raw[
        "core_network.layers.0.excitatory_cells.branch_layers.0"
    ]
    second_layer = analyzer.activation_raw[
        "core_network.layers.0.excitatory_cells.branch_layers.1"
    ]
    for branch_data in first_layer.values():
        assert set(branch_data) == {"depth", "exc", "inh", "vout"}
        assert branch_data["exc"].shape == (_x.shape[0],)
        assert branch_data["inh"].shape == (_x.shape[0],)
        assert branch_data["vout"].shape == (_x.shape[0],)
    for branch_data in second_layer.values():
        assert set(branch_data) == {"depth", "vout"}
        assert branch_data["vout"].shape == (_x.shape[0],)
    assert (tmp_path / "rf_raw_receptive_fields.pt").exists()
    assert (tmp_path / "rf_activation_tuning.json").exists()

    loaded = ReceptiveFieldAnalyzer(
        ReceptiveFieldAnalysisParams(load_raw_data=True, n_rows=1, n_cols=2)
    )
    loaded_results = loaded.analyze(
        model=model,
        save_path=str(tmp_path),
        filename="rf",
    )

    assert loaded_results is not None
    assert loaded_results["raw_checkpoint_loaded"]
    assert loaded_results["n_receptive_fields"] == results["n_receptive_fields"]

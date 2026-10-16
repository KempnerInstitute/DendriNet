import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.analysis.core.branch_activation import BranchActivationAnalyzer
from dendritic_modeling.analysis.tools.compartment_snr import CompartmentSNRAnalyzer
from dendritic_modeling.analysis.tools.compartment_statistics import (
    CompartmentStatisticsAnalyzer,
)
from dendritic_modeling.analysis.tools.receptive_fields import ReceptiveFieldAnalyzer
from dendritic_modeling.analysis.tools.synaptic_activation import (
    SynapticActivationAnalyzer,
)
from dendritic_modeling.config import load_config
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
        "biological_neuron": False,
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
            "independent_pathways": True,
            "excitatory_dim": 3,
            "inhibitory_dim": 3,
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


def _direct_inhibitory_population_model() -> Classifier:
    config = {
        "biological_neuron": False,
        "population_network": {
            "use_transfer": True,
            "transfer_params": {
                "independent_pathways": True,
                "excitatory_dim": 3,
                "inhibitory_dim": 3,
            },
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e0",
                    "population_defaults": {
                        "ff_excitatory_synapses": 2,
                        "ff_inhibitory_synapses": 2,
                        "rec_excitatory_synapses": 0,
                        "rec_inhibitory_synapses": 0,
                        "somatic_synapses": False,
                        "use_shunting": True,
                        "reactivate": True,
                        "reactivation_type": "param_tanh",
                        "dbl_init_method": "default",
                        "weight_transform": "softplus",
                    },
                    "populations": [
                        {
                            "name": "e0",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [2],
                        }
                    ],
                    "connections": [
                        {"source": "input_e", "target": "e0"},
                        {"source": "input_i", "target": "e0"},
                    ],
                }
            ],
        },
    }
    core = get_architecture("population_network", config, input_dim=6)
    return Classifier(Identity(6), core, Identity(4))


class _BranchOnlyCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = DendriticBranchLayer(
            output_dim=4,
            excitatory_input_dim=6,
            excitatory_synapses_per_branch=2,
            inhibitory_input_dim=6,
            inhibitory_synapses_per_branch=2,
            use_shunting=True,
            reactivate=True,
            layer_idx=0,
        )
        self.output_dim = 4

    def forward(self, x):
        return self.branch(x, x)


def _branch_only_model() -> Classifier:
    return Classifier(Identity(6), _BranchOnlyCore(), Identity(4))


def _branch_encoder_model() -> Classifier:
    return Classifier(_BranchOnlyCore(), Identity(4), Identity(4))


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

    captured = analyzer._captured_inputs
    assert captured["excitatory_inputs"].shape == (8, 3)
    assert captured["inhibitory_inputs"].shape == (8, 3)

    stats = analyzer._input_analysis(labels)
    assert "exc_inputs_d_total" in stats
    assert "inh_inputs_d_total" in stats
    assert torch.isfinite(torch.tensor(list(stats.values()))).all()


def test_compartment_snr_input_analysis_uses_the_latest_collection():
    """A training-time call must not supply the final input analysis."""
    model = _direct_inhibitory_model()
    x, _labels, _dataset = _toy_dataset()
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())

    analyzer.collect_raw_data(model, x)  # e.g. an epoch call that skips inputs
    analyzer.collect_raw_data(model, 2.0 * x)
    fresh = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())
    fresh.collect_raw_data(model, 2.0 * x)

    for name in ("excitatory_inputs", "inhibitory_inputs"):
        torch.testing.assert_close(
            analyzer._captured_inputs[name], fresh._captured_inputs[name]
        )


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


def test_compartment_snr_runs_on_non_ei_branch_core(tmp_path):
    model = _branch_only_model()
    _x, _labels, dataset = _toy_dataset()
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())

    results = analyzer.analyze(
        model=model,
        test_dataset=dataset,
        device="cpu",
        save_path=str(tmp_path),
        filename="snr",
    )

    assert results is not None
    assert "layer_analysis" in results
    assert "branch_analysis" in results
    assert (tmp_path / "snr.json").exists()


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
        activation_sample_signals=["vout"],
        activation_sample_depths=[0],
        activation_sample_branch_indices=[0],
        activation_sample_branch_indices_by_depth={0: [1]},
        activation_samples_per_class=2,
        activation_sample_seed=17,
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
    assert "activation_samples" in results
    samples = results["activation_samples"]
    assert samples["selection_depends_on_activations"] is False
    assert samples["selection_depends_on_labels"] is True
    assert samples["signals"] == ["vout"]
    assert samples["depths"] == [0]
    assert samples["branch_indices"] == [0]
    assert samples["branch_indices_by_depth"] == {"0": [1]}
    assert samples["records"]
    assert {record["class_label"] for record in samples["records"]} == {0, 1}
    assert {record["depth"] for record in samples["records"]} == {0}
    assert {record["branch_index"] for record in samples["records"]} == {1}
    assert {record["signal"] for record in samples["records"]} == {"vout"}
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
        assert set(branch_data) == {"depth", "upstream", "vout"}
        assert branch_data["upstream"].shape == (_x.shape[0],)
        assert branch_data["vout"].shape == (_x.shape[0],)
    assert (tmp_path / "rf_raw_receptive_fields.pt").exists()
    assert (tmp_path / "rf_activation_tuning.json").exists()
    assert (tmp_path / "rf_activation_samples.json").exists()

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


def test_receptive_field_samples_upstream_and_depolarizing_drive():
    params = ReceptiveFieldAnalysisParams(
        activation_sample_signals=["upstream", "depolarizing_drive"],
        activation_sample_depths=[1],
        activation_samples_per_class=1,
        activation_sample_seed=3,
    )
    analyzer = ReceptiveFieldAnalyzer(params)
    analyzer.activation_raw = {
        "core.branch": {
            0: {
                "depth": 1,
                "exc": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "upstream": torch.tensor([0.5, 1.0, 1.5, 2.0]),
            }
        }
    }

    records = analyzer._sample_activation_tuning(torch.tensor([0, 0, 1, 1]))

    assert {record["signal"] for record in records} == {
        "upstream",
        "depolarizing_drive",
    }
    grouped = {
        (record["signal"], record["dataset_index"]): record["value"]
        for record in records
    }
    for dataset_index in {record["dataset_index"] for record in records}:
        assert grouped[("depolarizing_drive", dataset_index)] == pytest.approx(
            1.5 * (dataset_index + 1)
        )


def test_receptive_field_runs_on_non_ei_branch_core(tmp_path):
    model = _branch_only_model()
    analyzer = ReceptiveFieldAnalyzer(
        ReceptiveFieldAnalysisParams(n_rows=1, n_cols=2, rf_shape=(1, 6))
    )

    results = analyzer.analyze(
        model=model,
        save_path=str(tmp_path),
        filename="rf",
    )

    assert results is not None
    assert results["n_receptive_fields"] > 0
    assert (tmp_path / "rf_raw_receptive_fields.pt").exists()
    assert (tmp_path / "exc_receptive_fields.png").exists()
    assert (tmp_path / "inh_receptive_fields.png").exists()


def test_receptive_field_gate_checks_full_model(tmp_path):
    model = _branch_encoder_model()
    analyzer = ReceptiveFieldAnalyzer(
        ReceptiveFieldAnalysisParams(n_rows=1, n_cols=2, rf_shape=(1, 6))
    )

    results = analyzer.analyze(
        model=model,
        save_path=str(tmp_path),
        filename="rf",
    )

    assert results is not None
    assert results["n_receptive_fields"] > 0
    assert (tmp_path / "rf_raw_receptive_fields.pt").exists()


def test_receptive_field_rf_tuning_outputs(tmp_path):
    model = _direct_inhibitory_model()
    _x, _labels, dataset = _toy_dataset()
    analyzer = ReceptiveFieldAnalyzer(
        ReceptiveFieldAnalysisParams(
            rf_shape=(1, 3),
            compute_rf_tuning=True,
            plot_rf_tuning=False,
        )
    )

    results = analyzer.analyze(
        model=model,
        test_dataset=dataset,
        device="cpu",
        save_path=str(tmp_path),
        filename="rf",
    )

    assert results is not None
    assert "rf_tuning" in results
    rf_tuning = results["rf_tuning"]
    assert set(rf_tuning["agg_tuning"]) == {"exc", "inh"}
    assert rf_tuning["n_exc_receptive_fields"] > 0
    assert (tmp_path / "rf_rf_tuning.json").exists()
    assert (tmp_path / "rf.json").exists()


def test_receptive_field_component_rfs_outputs(tmp_path):
    model = _direct_inhibitory_model()
    _x, _labels, dataset = _toy_dataset()
    analyzer = ReceptiveFieldAnalyzer(
        ReceptiveFieldAnalysisParams(
            rf_shape=(1, 3),
            compute_component_rfs=True,
        )
    )

    results = analyzer.analyze(
        model=model,
        test_dataset=dataset,
        device="cpu",
        save_path=str(tmp_path),
        filename="rf",
    )

    assert results is not None
    assert "component_receptive_fields" in results
    summary = results["component_receptive_fields"]
    assert summary["n_soma"] == 4
    assert summary["map_shape"] == [1, 3]
    assert (tmp_path / "rf_component_receptive_fields.json").exists()
    assert (tmp_path / "rf_component_receptive_fields.npz").exists()
    assert (tmp_path / "rf_component_receptive_fields.png").exists()

    data = np.load(tmp_path / "rf_component_receptive_fields.npz")
    assert data["e_mean"].shape == (4, 1, 3)
    assert data["e_comp"].shape == (4, 1, 3)
    assert data["i_mean"].shape == (4, 1, 3)
    assert data["i_comp"].shape == (4, 1, 3)
    assert data["preferred"].shape == (4,)


def test_receptive_field_component_rfs_support_population_network(tmp_path):
    model = _direct_inhibitory_population_model()
    _x, _labels, dataset = _toy_dataset()
    analyzer = ReceptiveFieldAnalyzer(
        ReceptiveFieldAnalysisParams(
            rf_shape=(1, 3),
            inh_rf_shape=(1, 3),
            compute_component_rfs=True,
            plot_component_rfs=False,
        )
    )

    results = analyzer.analyze(
        model=model,
        test_dataset=dataset,
        device="cpu",
        save_path=str(tmp_path),
        filename="rf",
    )

    assert results is not None
    assert results["component_receptive_fields"]["n_soma"] == 4
    data = np.load(tmp_path / "rf_component_receptive_fields.npz")
    assert data["e_mean"].shape == (4, 1, 3)
    assert data["e_comp"].shape == (4, 1, 3)
    assert data["i_mean"].shape == (4, 1, 3)
    assert data["i_comp"].shape == (4, 1, 3)


def test_receptive_field_component_rfs_skip_recurrent_branches(tmp_path):
    model = _direct_inhibitory_model()
    model.core_network.layers[0].excitatory_cells.branch_layers[
        0
    ].input_recurrent = True
    _x, _labels, dataset = _toy_dataset()
    analyzer = ReceptiveFieldAnalyzer(
        ReceptiveFieldAnalysisParams(
            rf_shape=(1, 3),
            compute_component_rfs=True,
        )
    )

    results = analyzer.analyze(
        model=model,
        test_dataset=dataset,
        device="cpu",
        save_path=str(tmp_path),
        filename="rf",
    )

    assert results is not None
    assert "component_receptive_fields" not in results
    assert not (tmp_path / "rf_component_receptive_fields.json").exists()
    assert not (tmp_path / "rf_component_receptive_fields.npz").exists()
    assert not (tmp_path / "rf_component_receptive_fields.png").exists()


def test_manager_runs_snr_and_rf_from_loaded_analysis_aliases(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
analysis:
  performance:
    enabled: false
  information:
    enabled: false
  compartment_snr:
    enabled: true
    training: false
    params:
      input_analysis: true
      input_capture_mode: branch
      layer_analysis: true
      branch_analysis: true
      layer_aggregation: true
      branch_aggregation: true
  receptive_field:
    enabled: true
    training: false
    params:
      rf_shape: [1, 3]
      compute_component_rfs: true
      compute_rf_tuning: true
      plot_rf_tuning: false
""".lstrip()
    )
    config = load_config(str(config_path))
    model = _direct_inhibitory_model()
    _x, _labels, dataset = _toy_dataset()
    save_root = tmp_path / "analysis"
    manager = AnalysisManager(
        model=model,
        data={"train": dataset, "valid": dataset, "test": dataset},
        analysis_config=config.analysis,
        save_root=str(save_root),
        device="cpu",
    )

    manager.run_analysis(filename="final", training=False)

    assert (save_root / "compartment_snr_analysis" / "final.json").exists()
    rf_root = save_root / "receptive_field_analysis"
    assert (rf_root / "final_summary.json").exists()
    assert (rf_root / "final_component_receptive_fields.json").exists()
    assert (rf_root / "final_component_receptive_fields.npz").exists()
    assert (rf_root / "final_component_receptive_fields.png").exists()
    assert (rf_root / "final_rf_tuning.json").exists()
    assert (rf_root / "final.json").exists()


def test_receptive_field_rf_tuning_removes_hooks_after_failed_collection(
    monkeypatch,
):
    model = _direct_inhibitory_model()
    x, labels, _dataset = _toy_dataset()
    analyzer = ReceptiveFieldAnalyzer(
        ReceptiveFieldAnalysisParams(compute_rf_tuning=True)
    )
    layer = analyzer._get_rf_tuning_layer(model)
    assert layer is not None

    original_register = analyzer._rf_tuning_forward_hook
    call_count = 0

    def _fail_on_second_hook(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise RuntimeError("rf tuning hook failed")
        return original_register(*args, **kwargs)

    monkeypatch.setattr(analyzer, "_rf_tuning_forward_hook", _fail_on_second_hook)

    with pytest.raises(RuntimeError, match="rf tuning hook failed"):
        analyzer._collect_rf_tuning_activation_data(model, layer, x, labels)

    assert _count_registered_forward_hooks(model) == 0


def test_the_ei_ratio_summary_is_the_ratio_of_class_means():
    analyzer = ReceptiveFieldAnalyzer(ReceptiveFieldAnalysisParams())
    labels = torch.tensor([0, 0, 1, 1])
    excitation = torch.tensor([[1.0], [3.0], [2.0], [2.0]])
    # One sample with almost no inhibition: its own ratio is 3e6.
    inhibition = torch.tensor([[1.0], [1e-6], [-4.0], [4.0]])

    summary = analyzer._summarize_ei_ratio(excitation, inhibition, labels)

    eps = analyzer.epsilon
    assert summary["class_means"]["0"] == pytest.approx(2.0 / ((1 + 1e-6) / 2 + eps))
    assert summary["class_means"]["1"] == pytest.approx(2.0 / (4.0 + eps))
    assert summary["preferred_class"] == 0

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


def test_correlation_branch_hooks_remove_registration_failure(monkeypatch):
    model = _direct_inhibitory_model()
    analyzer = CorrelationAnalyzer(CorrelationAnalysisParams())
    branch_layer = model.core_network.layers[0].excitatory_cells.branch_layers[0]

    def _failing_register(*args, **kwargs):
        raise RuntimeError("correlation branch hook registration failed")

    monkeypatch.setattr(branch_layer, "register_forward_hook", _failing_register)

    with pytest.raises(
        RuntimeError,
        match="correlation branch hook registration failed",
    ):
        analyzer.attach_forward_hooks(model)

    assert _count_registered_forward_hooks(model) == 0


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


def test_correlation_contradicting_flat_and_nested_fields_raise():
    with pytest.raises(ValueError, match="conflicts with"):
        CorrelationAnalysisParams(
            computation_level="synaptic",
            compute=CorrelationComputeParams(computation_level="single_branch"),
        )

    params = CorrelationAnalysisParams(
        computation_level="single_branch",
        max_samples=23,
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


def _hooked_indices(module, x):
    analyzer = CorrelationAnalyzer(CorrelationAnalysisParams())
    analyzer.data_dict = {"layer": {}}
    hook = analyzer._make_synapse_hook("layer", "E")
    hook(module, (x,), module(x))
    return analyzer.data_dict["layer"]


def test_synaptic_correlation_uses_the_realized_synapses():
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
        IndexedSparseLinear,
    )
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
        TopKLinear,
    )

    x = torch.rand(3, 4)
    # Signed weights select by magnitude.
    signed = TopKLinear(in_features=4, out_features=2, K=2, weight_transform="identity")
    with torch.no_grad():
        signed.pre_w.copy_(
            torch.tensor([[-5.0, 1.0, 2.0, 0.5], [0.1, -0.2, 3.0, -4.0]])
        )
    record = _hooked_indices(signed, x)
    assert record["E_indices"].tolist() == [[0, 2], [2, 3]]

    # A compact indexed bank stores its connections as slots.
    indices = torch.tensor([[1, 3], [0, 2]])
    indexed = IndexedSparseLinear(4, 2, 2, connection_indices=indices)
    record = _hooked_indices(indexed, x)
    assert record["E_indices"].tolist() == indices.tolist()
    torch.testing.assert_close(
        record["E_synaptic"].sum(dim=-1), indexed(x), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(
    "params,path",
    [
        ({"compute": {"batch_process": False}}, "params.compute.batch_process"),
        ({"batch_process": True}, "params.batch_process"),
    ],
)
def test_retired_batch_process_loads_with_a_warning(tmp_path, caplog, params, path):
    """batch_process was stored but never read; it is retired, not reproduced."""
    import logging

    import yaml

    from dendritic_modeling.config import load_analysis_config

    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"analysis": {"correlation": {"enabled": True, "params": params}}}
        )
    )
    with caplog.at_level(logging.WARNING):
        config = load_analysis_config(str(config_path))
    assert f"Ignoring analysis.correlation.{path}: it was never read" in caplog.text
    assert not hasattr(config.correlation.params.compute, "batch_process")
    assert not hasattr(CorrelationAnalyzer(config.correlation.params), "batch_process")


def test_upgrade_config_removes_the_retired_batch_process():
    from dendritic_modeling.scripts.tools.upgrade_config import upgrade_mapping

    old = {
        "config_version": 1,
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist"},
        "analysis": {
            "correlation": {
                "enabled": True,
                "params": {
                    "batch_process": True,
                    "compute": {"batch_process": False, "max_samples": 7},
                },
            }
        },
    }
    upgraded, changes, unresolved = upgrade_mapping(old)
    assert unresolved == []
    params = upgraded["analysis"]["correlation"]["params"]
    assert "batch_process" not in params
    assert params["compute"] == {"max_samples": 7}
    assert sum("batch_process" in change for change in changes) == 2


def test_across_layer_is_gone_from_the_default_categories():
    from dendritic_modeling.config.analysis import CorrelationSamplingParams

    default = ["within_branch", "within_layer"]
    assert CorrelationSamplingParams().stratified_categories == default
    assert CorrelationAnalysisParams().stratified_categories == default
    analyzer = CorrelationAnalyzer(CorrelationAnalysisParams())
    assert analyzer.stratified_categories == default


def test_retiring_across_layer_keeps_synaptic_results_byte_identical(tmp_path):
    """across_layer was never read: sampling with it gives the same files."""
    import hashlib

    from dendritic_modeling.config.analysis import EvaluationRuntimeConfig

    _x, _labels, dataset = _toy_dataset()
    runtime = EvaluationRuntimeConfig(mode="materialize", num_workers=0)
    digests = []
    for name, categories in (
        ("default", None),
        ("with_across_layer", ["within_branch", "within_layer", "across_layer"]),
    ):
        torch.manual_seed(0)
        model = _direct_inhibitory_model()
        analyzer = CorrelationAnalyzer(
            CorrelationAnalysisParams(computation_level="synaptic")
        )
        if categories is not None:  # the list earlier versions sampled from
            analyzer.stratified_categories = categories
        np.random.seed(11)
        analyzer.analyze(
            model, dataset, save_path=str(tmp_path / name), runtime=runtime
        )
        digests.append(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((tmp_path / name).iterdir())
                if path.suffix != ".png"
            }
        )
    assert digests[0] and digests[0] == digests[1]


def test_a_written_across_layer_loads_with_a_warning(tmp_path, caplog):
    import logging

    import yaml

    from dendritic_modeling.config import keys as config_keys, load_analysis_config

    config_keys._WARNED_RETIRED_VALUES.clear()
    path = tmp_path / "analysis.yaml"
    old_default = ["within_branch", "within_layer", "across_layer"]
    path.write_text(
        yaml.safe_dump(
            {
                "analysis": {
                    "correlation": {
                        "enabled": True,
                        "params": {
                            # A config saved by an earlier version: the flat
                            # spelling at its old default, the grouped one set.
                            "stratified_categories": old_default,
                            "sampling": {
                                "stratified_categories": [
                                    "within_branch",
                                    "across_layer",
                                ]
                            },
                        },
                    }
                }
            }
        )
    )
    with caplog.at_level(logging.WARNING):
        config = load_analysis_config(str(path))
    params = config.correlation.params
    assert params.sampling.stratified_categories == ["within_branch"]
    assert params.stratified_categories == ["within_branch", "within_layer"]
    assert "entry 'across_layer': it was never implemented" in caplog.text


def test_an_unknown_stratified_category_raises():
    from dendritic_modeling.config.analysis import CorrelationSamplingParams

    with pytest.raises(ValueError, match="did you mean 'within_layer'"):
        CorrelationSamplingParams(stratified_categories=["within_layers"])
    with pytest.raises(ValueError, match="lists 'across'"):
        CorrelationAnalysisParams(stratified_categories=["across"])


def test_upgrade_config_removes_the_retired_across_layer_category():
    from dendritic_modeling.scripts.tools.upgrade_config import upgrade_mapping

    old_default = ["within_branch", "within_layer", "across_layer"]
    old = {
        "config_version": 1,
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist"},
        "analysis": {
            "correlation": {
                "enabled": True,
                "params": {
                    "stratified_categories": list(old_default),
                    "sampling": {"stratified_categories": list(old_default)},
                },
            }
        },
    }
    upgraded, changes, unresolved = upgrade_mapping(old)
    assert unresolved == []
    params = upgraded["analysis"]["correlation"]["params"]
    assert params["stratified_categories"] == ["within_branch", "within_layer"]
    assert params["sampling"]["stratified_categories"] == [
        "within_branch",
        "within_layer",
    ]
    assert sum("'across_layer'" in change for change in changes) == 2

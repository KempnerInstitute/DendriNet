import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis import (
    get_analyzer_spec,
    get_registered_analyzer_names,
    load_analyzer_class,
    register_analyzer,
    unregister_analyzer,
)
from dendritic_modeling.config import canonicalize_model_core_flags
from dendritic_modeling.config.training import OptimizerConfig
from dendritic_modeling.datasets import (
    get_registered_dataset_names,
    get_unified_datasets,
    register_dataset,
    unregister_dataset,
)
from dendritic_modeling.networks.architectures.factory import (
    get_architecture,
    get_available_architectures,
)
from dendritic_modeling.networks.architectures.registry import (
    register_architecture,
    unregister_architecture,
)
from dendritic_modeling.scripts.script_utils.config_utils import (
    canonicalize_model_core_flags as script_canonicalize_model_core_flags,
)
from dendritic_modeling.training.loss import (
    LossFunction,
    get_available_loss_functions,
    get_loss_function,
    register_loss_function,
    unregister_loss_function,
)
from dendritic_modeling.training.optimizers import (
    create_optimizer,
    get_available_optimizers,
    register_optimizer,
    unregister_optimizer,
    validate_optimizer_config,
)


def test_architecture_registry_adds_custom_type_without_shadowing_builtins():
    def build_linear(parameters, *, input_dim=None, suffix_input_dim=None):
        del suffix_input_dim
        return nn.Linear(int(input_dim), int(parameters["output_dim"]))

    register_architecture("unit_test_linear", build_linear)
    try:
        model = get_architecture("unit_test_linear", {"output_dim": 2}, input_dim=3)
        assert isinstance(model, nn.Linear)
        assert model.in_features == 3
        assert model.out_features == 2
        assert "unit_test_linear" in get_available_architectures()
        with pytest.raises(ValueError, match="built in"):
            register_architecture("mlp", build_linear)
    finally:
        unregister_architecture("unit_test_linear")


def test_optimizer_registry_adds_custom_optimizer_strategy():
    def build_sgd(parameters, config):
        return optim.SGD(parameters, lr=config.lr, weight_decay=config.weight_decay)

    register_optimizer("unit_test_optimizer", build_sgd)
    try:
        config = OptimizerConfig(name="unit_test_optimizer", lr=0.03)
        validate_optimizer_config(config)
        model = nn.Linear(2, 1)
        optimizer = create_optimizer(model.parameters(), config)
        assert isinstance(optimizer, optim.SGD)
        assert optimizer.defaults["lr"] == 0.03
        assert "unit_test_optimizer" in get_available_optimizers()
    finally:
        unregister_optimizer("unit_test_optimizer")


def test_loss_registry_adds_custom_loss_without_changing_case_sensitivity():
    class ZeroLoss(LossFunction):
        def forward(self, model, x, y=None):
            del model, y
            return x.sum() * 0.0

    register_loss_function("unit_test_zero", ZeroLoss)
    try:
        loss_fn = get_loss_function("unit_test_zero")
        assert isinstance(loss_fn, ZeroLoss)
        assert "unit_test_zero" in get_available_loss_functions()
        with pytest.raises(ValueError, match="Invalid loss function"):
            get_loss_function("MSE")
    finally:
        unregister_loss_function("unit_test_zero")


def test_dataset_registry_adds_custom_unified_dataset_loader():
    def build_dataset(task_cfg):
        assert task_cfg["dataset"] == "unit_test_dataset"
        x = torch.zeros(4, 2)
        y = torch.zeros(4, dtype=torch.long)
        ds = TensorDataset(x, y)
        return ds, ds, ds

    register_dataset("unit_test_dataset", build_dataset)
    try:
        train_ds, valid_ds, test_ds = get_unified_datasets(
            {"dataset": "unit_test_dataset", "parameters": {}}
        )
        assert len(train_ds) == len(valid_ds) == len(test_ds) == 4
        assert "unit_test_dataset" in get_registered_dataset_names()
    finally:
        unregister_dataset("unit_test_dataset")


def test_analyzer_registry_exposes_builtin_metadata_without_execution_change():
    names = get_registered_analyzer_names()
    assert "performance" in names
    spec = get_analyzer_spec("performance")
    assert spec is not None
    assert spec.config_field == "performance_analysis"
    assert spec.attribute_name == "performance_analyzer"
    assert spec.constructor_style == "params"
    assert load_analyzer_class("performance") is spec.load_class()


def test_analyzer_registry_rejects_unknown_constructor_style():
    with pytest.raises(ValueError, match="Unsupported analyzer constructor style"):
        register_analyzer(
            "unit_test_bad_analyzer",
            config_field="bad_analysis",
            class_path="dendritic_modeling.analysis.metrics.performance.PerformanceAnalyzer",
            attribute_name="bad_analyzer",
            save_subdir="bad_analysis",
            constructor_style="bad_style",
        )
    unregister_analyzer("unit_test_bad_analyzer")


def test_config_alias_canonicalization_is_config_level_and_script_compatible():
    payload = {
        "model": {
            "core": {
                "type": "dendritic_additive",
                "architecture": {},
                "connectivity": {},
                "morphology": {"use_shunting": True},
            }
        }
    }
    config_result = canonicalize_model_core_flags(payload)
    script_result = script_canonicalize_model_core_flags(payload)
    assert config_result == script_result
    morphology = config_result["model"]["core"]["morphology"]
    assert morphology["use_shunting"] is False
    # dendritic_additive is raw additive now (normalization is opt-in via
    # dendritic_normalized_additive).
    assert morphology["use_additive_normalization"] is False
    assert payload["model"]["core"]["morphology"]["use_shunting"] is True

import logging
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from dendritic_modeling.datasets import load_mnist_modulo10

# Import functions from their actual locations
from dendritic_modeling.training.utils.evaluation import accuracy_score
from dendritic_modeling.utils import Shaper
from dendritic_modeling.utils.gpu_estimation import (
    CIFAR10_FLAT_INPUT_SHAPE,
    IMAGENET_FLAT_INPUT_SHAPE,
    IMAGENET_IMAGE_INPUT_SHAPE,
    MNIST_FLAT_INPUT_SHAPE,
    GPUMemoryEstimator,
    MemoryBreakdown,
    _config_params,
    _core_params_from_config,
    _fallback_input_shape,
    _flat_input_dim,
    _resolve_batch_size,
    _resolve_n_classes,
    _resolve_task_config,
    _resolve_training_dtype,
)
from dendritic_modeling.utils.information.estimators import (
    EstimatorConfig,
    InformationEstimatorFactory,
)


def test_utils_reexports_hook_helpers():
    from dendritic_modeling import utils
    from dendritic_modeling.utils.hooks import (
        ForwardHookRemovalMixin,
        iter_child_modules_of_type,
        iter_modules_matching,
        iter_modules_of_type,
        iter_named_modules_matching,
        iter_named_modules_of_type,
        register_named_forward_hook_groups,
        register_named_forward_hooks,
    )

    assert utils.ForwardHookRemovalMixin is ForwardHookRemovalMixin
    assert utils.iter_child_modules_of_type is iter_child_modules_of_type
    assert utils.iter_modules_matching is iter_modules_matching
    assert utils.iter_modules_of_type is iter_modules_of_type
    assert utils.iter_named_modules_matching is iter_named_modules_matching
    assert utils.iter_named_modules_of_type is iter_named_modules_of_type
    assert (
        utils.register_named_forward_hook_groups is register_named_forward_hook_groups
    )
    assert utils.register_named_forward_hooks is register_named_forward_hooks


def test_iter_child_modules_of_type_preserves_child_only_order():
    from dendritic_modeling.utils.hooks import iter_child_modules_of_type

    nested = nn.Linear(3, 1)
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Sequential(nested))

    matches = list(iter_child_modules_of_type(model, nn.Linear))

    assert [module.out_features for module in matches] == [3]
    assert nested not in matches


def test_iter_child_modules_of_type_handles_plain_iterables():
    from dendritic_modeling.utils.hooks import iter_child_modules_of_type

    linear = nn.Linear(2, 3)
    modules = [nn.ReLU(), linear, nn.Sequential(nn.Linear(3, 1))]

    assert list(iter_child_modules_of_type(modules, nn.Linear)) == [linear]


def test_iter_modules_of_type_preserves_module_order():
    from dendritic_modeling.utils.hooks import iter_modules_of_type

    model = nn.Module()
    model.first = nn.Linear(2, 3)
    model.block = nn.Sequential(nn.ReLU(), nn.Linear(3, 1))

    matches = list(iter_modules_of_type(model, nn.Linear))

    assert [module.out_features for module in matches] == [3, 1]
    assert matches == [model.first, model.block[1]]


def test_iter_named_modules_of_type_preserves_model_order():
    from dendritic_modeling.utils.hooks import iter_named_modules_of_type

    model = nn.Module()
    model.first = nn.Linear(2, 3)
    model.block = nn.Sequential(nn.ReLU(), nn.Linear(3, 1))

    matches = list(iter_named_modules_of_type(model, nn.Linear))

    assert [name for name, _module in matches] == ["first", "block.1"]
    assert [module.out_features for _name, module in matches] == [3, 1]


def test_iter_modules_matching_preserves_order_and_predicate():
    from dendritic_modeling.utils.hooks import iter_modules_matching

    model = nn.Module()
    model.first = nn.Linear(2, 3)
    model.activation = nn.ReLU()
    model.block = nn.Sequential(nn.ReLU(), nn.Linear(3, 1))

    matches = list(
        iter_modules_matching(model, lambda module: isinstance(module, nn.ReLU))
    )

    assert matches == [model.activation, model.block[0]]


def test_iter_named_modules_matching_preserves_order_and_predicate():
    from dendritic_modeling.utils.hooks import iter_named_modules_matching

    model = nn.Module()
    model.first = nn.Linear(2, 3)
    model.activation = nn.ReLU()
    model.block = nn.Sequential(nn.ReLU(), nn.Linear(3, 1))

    matches = list(
        iter_named_modules_matching(
            model,
            lambda name, module: name.endswith("activation")
            or isinstance(module, nn.ReLU),
        )
    )

    assert [name for name, _module in matches] == ["activation", "block.0"]


def test_register_named_forward_hooks_preserves_order_and_predicate():
    from dendritic_modeling.utils.hooks import (
        register_named_forward_hooks,
        remove_hook_handles,
    )

    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    prepared = []
    captured = []

    def prepare(name, module):
        prepared.append((name, module.out_features))

    def hook(module, _inputs, output):
        captured.append((module.out_features, tuple(output.shape)))

    handles = register_named_forward_hooks(
        model,
        nn.Linear,
        hook,
        predicate=lambda name, _module: name == "0",
        prepare=prepare,
    )

    try:
        assert len(handles) == 1
        assert prepared == [("0", 3)]
        model(torch.ones(2, 2))
        assert captured == [(3, (2, 3))]
    finally:
        remove_hook_handles(handles)

    assert sum(len(module._forward_hooks) for module in model.modules()) == 0


def test_register_named_forward_hooks_removes_partial_registration_on_failure(
    monkeypatch,
):
    from dendritic_modeling.utils.hooks import register_named_forward_hooks

    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    original_register = nn.Linear.register_forward_hook
    call_count = 0

    def register_once_then_fail(module, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_register(module, *args, **kwargs)
        raise RuntimeError("hook registration failed")

    monkeypatch.setattr(nn.Linear, "register_forward_hook", register_once_then_fail)

    with pytest.raises(RuntimeError, match="hook registration failed"):
        register_named_forward_hooks(model, nn.Linear, lambda *_args: None)

    assert call_count == 2
    assert sum(len(module._forward_hooks) for module in model.modules()) == 0


def test_register_named_forward_hook_groups_preserves_order_and_predicate():
    from dendritic_modeling.utils.hooks import (
        register_named_forward_hook_groups,
        remove_hook_handles,
    )

    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    prepared = []
    captured = []

    def prepare(name, module):
        prepared.append((name, module.out_features))

    def register_group(name, module):
        return [
            module.register_forward_hook(
                lambda _module, _inputs, output, name=name: captured.append(
                    (name, "first", tuple(output.shape))
                )
            ),
            module.register_forward_hook(
                lambda _module, _inputs, output, name=name: captured.append(
                    (name, "second", tuple(output.shape))
                )
            ),
        ]

    handles = register_named_forward_hook_groups(
        model,
        nn.Linear,
        register_group,
        predicate=lambda name, _module: name == "0",
        prepare=prepare,
    )

    try:
        assert len(handles) == 2
        assert prepared == [("0", 3)]
        model(torch.ones(2, 2))
        assert captured == [
            ("0", "first", (2, 3)),
            ("0", "second", (2, 3)),
        ]
    finally:
        remove_hook_handles(handles)

    assert sum(len(module._forward_hooks) for module in model.modules()) == 0


def test_register_named_forward_hook_groups_removes_prior_groups_on_failure():
    from dendritic_modeling.utils.hooks import register_named_forward_hook_groups

    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

    def register_group(name, module):
        if name == "1":
            raise RuntimeError("hook group registration failed")
        return [module.register_forward_hook(lambda *_args: None)]

    with pytest.raises(RuntimeError, match="hook group registration failed"):
        register_named_forward_hook_groups(model, nn.Linear, register_group)

    assert sum(len(module._forward_hooks) for module in model.modules()) == 0


class TestShaper(unittest.TestCase):
    def test_initialization(self):
        shape = (2, 3)
        shaper = Shaper(shape)
        self.assertEqual(shaper.shape, shape)

    def test_reshape_valid_shape(self):
        shaper = Shaper((2, 3))
        x = torch.randn(6)
        reshaped_x = shaper.reshape(x)
        self.assertEqual(reshaped_x.shape, torch.Size([2, 3]))

    def test_reshape_invalid_shape(self):
        shaper = Shaper((2, 4))
        x = torch.randn(6)
        with self.assertRaises(RuntimeError):
            shaper.reshape(x)

    def test_reshape_higher_dimensions(self):
        shaper = Shaper((2, 2, 3))
        x = torch.randn(12)
        reshaped_x = shaper.reshape(x)
        self.assertEqual(reshaped_x.shape, torch.Size([2, 2, 3]))

    def test_reshape_invalid_input(self):
        shaper = Shaper((2, 3))
        x = torch.randn(7)  # Invalid size
        with self.assertRaises(RuntimeError):
            shaper.reshape(x)


class TestAccuracyScore(unittest.TestCase):
    def setUp(self):
        # Create a dummy classifier that always predicts class 1
        class DummyClassifier(nn.Module):
            def forward(self, inputs):
                batch_size = inputs.shape[0]
                logits = torch.zeros(batch_size, 2)
                logits[:, 1] = 10.0  # High logit for class 1
                return logits

        self.classifier = DummyClassifier()

    def test_accuracy_all_correct(self):
        inputs = torch.randn(10, 784)
        labels = torch.ones(10, dtype=torch.long)
        accuracy = accuracy_score(self.classifier, inputs, labels, move_device=False)
        self.assertEqual(accuracy.item(), 1.0)

    def test_accuracy_all_incorrect(self):
        inputs = torch.randn(10, 784)
        labels = torch.zeros(10, dtype=torch.long)
        accuracy = accuracy_score(self.classifier, inputs, labels, move_device=False)
        self.assertEqual(accuracy.item(), 0.0)

    def test_accuracy_half_correct(self):
        inputs = torch.randn(10, 784)
        labels = torch.cat(
            [torch.ones(5, dtype=torch.long), torch.zeros(5, dtype=torch.long)]
        )
        accuracy = accuracy_score(self.classifier, inputs, labels, move_device=False)
        self.assertAlmostEqual(accuracy.item(), 0.5)  # 50% accuracy


class TestLoadMNISTModulo10(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for caching during tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self._mnist_patch = patch(
            "dendritic_modeling.datasets.standard_datasets.load_mnist_as_datasets",
            side_effect=self._mock_mnist_as_datasets,
        )
        self._mnist_patch.start()

    def tearDown(self):
        self._mnist_patch.stop()
        # Clean up the temporary directory after tests
        self.temp_dir.cleanup()

    @staticmethod
    def _make_dataset(n_samples: int, offset: int):
        x = torch.arange(
            offset,
            offset + n_samples * 28 * 28,
            dtype=torch.float32,
        ).view(n_samples, 1, 28, 28)
        y = (torch.arange(n_samples, dtype=torch.long) + offset) % 10
        return torch.utils.data.TensorDataset(x, y)

    def _mock_mnist_as_datasets(
        self,
        train_valid_split=0.8,
        flatten=False,
        normalize=False,
        task_data_path=None,
        split_seed: int = 0,
    ):
        del normalize, task_data_path, split_seed
        total_train = 100
        train_size = int(total_train * train_valid_split)
        valid_size = total_train - train_size
        train = self._make_dataset(train_size, 0)
        valid = self._make_dataset(valid_size, 10_000)
        test = self._make_dataset(40, 20_000)
        if flatten:
            train = torch.utils.data.TensorDataset(
                train.tensors[0].view(train_size, -1), train.tensors[1]
            )
            valid = torch.utils.data.TensorDataset(
                valid.tensors[0].view(valid_size, -1), valid.tensors[1]
            )
            test = torch.utils.data.TensorDataset(
                test.tensors[0].view(40, -1), test.tensors[1]
            )
        return train, valid, test

    def test_data_loading_no_split(self):
        """Test data loading with no train/validation split."""
        data = load_mnist_modulo10(
            train_valid_split=1, task_data_path=self.temp_dir.name
        )
        self.assertIn("train", data)
        self.assertIn("test", data)
        # Check that datasets are not None and have length
        self.assertIsNotNone(data["train"])
        self.assertIsNotNone(data["test"])
        self.assertGreater(len(data["train"]), 0)
        self.assertGreater(len(data["test"]), 0)

    def test_data_loading_with_split(self):
        """Test data loading with train/validation split."""
        data = load_mnist_modulo10(
            train_valid_split=0.8, task_data_path=self.temp_dir.name
        )
        self.assertIn("train", data)
        self.assertIn("valid", data)
        self.assertIn("test", data)
        # Check that datasets are not None
        self.assertIsNotNone(data["train"])
        self.assertIsNotNone(data["valid"])
        self.assertIsNotNone(data["test"])

        # Check that train/validation split ratio is respected approximately
        total_train_valid = len(data["train"]) + len(data["valid"])
        self.assertAlmostEqual(len(data["train"]) / total_train_valid, 0.8, delta=0.05)
        self.assertGreater(len(data["test"]), 0)

    def test_shuffle_iterations_effect(self):
        """Test that different shuffle iterations produce different results."""
        # Load data with 1 shuffle iteration
        data1 = load_mnist_modulo10(
            shuffle_iterations=1,
            train_valid_split=1,
            task_data_path=self.temp_dir.name,
        )
        # Load data with 2 shuffle iterations
        data2 = load_mnist_modulo10(
            shuffle_iterations=2,
            train_valid_split=1,
            task_data_path=self.temp_dir.name,
        )

        # Check that different shuffle iterations yield different input tensors
        # For example, the 10th input tensor should be different
        self.assertFalse(torch.equal(data1["train"][9][0], data2["train"][9][0]))

    def test_modulo10_labels(self):
        """Test that labels are correctly computed with modulo 10."""
        data = load_mnist_modulo10(
            shuffle_iterations=1,
            train_valid_split=1,
            task_data_path=self.temp_dir.name,
        )
        labels = [item[1].numpy().tolist() for item in data["train"]]
        mod_status = [(item >= 0) and (item < 10) for item in labels]
        self.assertTrue(all(mod_status))


def test_mutual_information():
    config = EstimatorConfig(method="kraskov", n_neighbors=3)
    estimator = InformationEstimatorFactory.create(config)
    x = np.random.rand(100)
    y = np.random.rand(100)  # Uncorrelated
    mi = estimator.mutual_information(x.reshape(-1, 1), y.reshape(-1, 1))
    assert (
        isinstance(mi, (int, float)) and mi >= 0
    )  # MI should be non-negative, can be int or float


def test_conditional_mi():
    config = EstimatorConfig(method="kraskov", n_neighbors=3)
    estimator = InformationEstimatorFactory.create(config)
    x = np.random.rand(100)
    y = x + np.random.rand(100) * 0.1
    z = np.random.rand(100)
    cmi = estimator.conditional_mutual_information(
        x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)
    )
    assert isinstance(cmi, float)


def test_gpu_estimator():
    estimator = GPUMemoryEstimator(dtype="fp16", optimizer="adam")
    assert estimator.bytes_per_param == 2


def test_gpu_estimator_uses_default_dtype_and_optimizer_assumptions():
    estimator = GPUMemoryEstimator(dtype="unknown", optimizer="unknown")

    assert estimator.bytes_per_param == 2
    assert estimator.optimizer_state_multiplier == 2.0

    explicit = GPUMemoryEstimator(dtype="fp32", optimizer="sgd")

    assert explicit.bytes_per_param == 4
    assert explicit.optimizer_state_multiplier == 1.0


def test_gpu_recommendations_log_invalid_memory_values(caplog):
    estimator = GPUMemoryEstimator()
    breakdown = MemoryBreakdown(
        parameters=0.0,
        gradients=0.0,
        optimizer_states=0.0,
        activations=0.0,
        fsdp_overhead=0.0,
        total=1.0,
    )

    with caplog.at_level(
        logging.WARNING, logger="dendritic_modeling.utils.gpu_estimation"
    ):
        recommendations = estimator.get_gpu_recommendations(
            breakdown, safety_factor=0.0
        )

    assert "Invalid memory values" in caplog.text
    assert [rec.min_gpus_needed for rec in recommendations] == [1, 1]


def test_gpu_estimate_config_helpers_resolve_batch_size_paths():
    assert _resolve_batch_size(SimpleNamespace(), batch_size=128) == 128

    current = SimpleNamespace(
        training=SimpleNamespace(
            main=SimpleNamespace(common=SimpleNamespace(batch_size=64))
        )
    )
    assert _resolve_batch_size(current) == 64

    current_trainer = SimpleNamespace(
        training=SimpleNamespace(
            main=SimpleNamespace(trainer=SimpleNamespace(batch_size=48))
        )
    )
    assert _resolve_batch_size(current_trainer) == 48

    legacy = SimpleNamespace(
        train=SimpleNamespace(main=SimpleNamespace(trainer=SimpleNamespace()))
    )
    assert _resolve_batch_size(legacy) == 32


def test_gpu_estimate_task_config_helper_preserves_legacy_and_synthesizes_data():
    legacy_task = SimpleNamespace(dataset="mnist", parameters={"flatten": True})
    legacy_config = SimpleNamespace(task=legacy_task)

    task_cfg, task_params = _resolve_task_config(legacy_config)

    assert task_cfg is legacy_task
    assert task_params == {"flatten": True}

    data_config = SimpleNamespace(
        data=SimpleNamespace(
            dataset_name="imagenet",
            base_dir="/tmp/imagenet",
            processing={"flatten": False},
        ),
        experiment=SimpleNamespace(train_valid_split=0.7),
    )

    task_cfg, task_params = _resolve_task_config(data_config)

    assert task_cfg.dataset == "imagenet"
    assert task_cfg.data_path == "/tmp/imagenet"
    assert task_cfg.train_valid_split == 0.7
    assert task_params == {"flatten": False}


def test_gpu_estimate_fallback_input_shapes():
    assert _fallback_input_shape(SimpleNamespace(dataset="mnist"), {}) == (784,)
    assert _fallback_input_shape(SimpleNamespace(dataset="cifar10"), {}) == (3072,)
    assert _fallback_input_shape(
        SimpleNamespace(dataset="imagenet"), {"flatten": True}
    ) == (150528,)
    assert _fallback_input_shape(
        SimpleNamespace(dataset="imagenet"), {"flatten": False}
    ) == (3, 224, 224)
    assert _fallback_input_shape(
        SimpleNamespace(dataset="custom"), {"input_dim": 123}
    ) == (123,)


def test_gpu_estimate_public_input_shape_constants_preserve_values():
    assert MNIST_FLAT_INPUT_SHAPE == (784,)
    assert CIFAR10_FLAT_INPUT_SHAPE == (3072,)
    assert IMAGENET_FLAT_INPUT_SHAPE == (150528,)
    assert IMAGENET_IMAGE_INPUT_SHAPE == (3, 224, 224)


def test_gpu_estimate_model_build_helpers_preserve_parameter_paths():
    assert _flat_input_dim((784,)) == 784
    assert _flat_input_dim((3, 224, 224)) == 150528

    assert _config_params(SimpleNamespace(parameters={"a": 1})) == {"a": 1}
    assert _config_params(SimpleNamespace(params={"b": 2})) == {"b": 2}
    assert _config_params(SimpleNamespace(parameters={}, params={"c": 3})) == {"c": 3}

    legacy_core = SimpleNamespace(parameters={"input_dim": 5, "hidden_dim": 7})
    assert _core_params_from_config(legacy_core) == {"input_dim": 5, "hidden_dim": 7}

    structured_core = SimpleNamespace(
        architecture=SimpleNamespace(hidden_dim=8, shared="architecture"),
        connectivity=SimpleNamespace(num_synapses=9, shared="connectivity"),
        morphology=SimpleNamespace(use_shunting=True),
    )

    assert _core_params_from_config(structured_core) == {
        "hidden_dim": 8,
        "num_synapses": 9,
        "shared": "connectivity",
        "use_shunting": True,
    }


def test_gpu_estimate_class_count_helper_matches_dataset_defaults():
    assert (
        _resolve_n_classes("classification", SimpleNamespace(dataset="mnist"), {}) == 10
    )
    assert (
        _resolve_n_classes("classification", SimpleNamespace(dataset="imagenet"), {})
        == 1000
    )
    assert (
        _resolve_n_classes(
            "classification", SimpleNamespace(dataset="custom"), {"n_classes": 17}
        )
        == 17
    )
    assert _resolve_n_classes("regression", SimpleNamespace(dataset="mnist"), {}) == 1


def test_gpu_estimate_training_dtype_helper_resolves_current_and_legacy_paths():
    assert _resolve_training_dtype(SimpleNamespace()) == "fp16"

    current = SimpleNamespace(
        training=SimpleNamespace(
            main=SimpleNamespace(common=SimpleNamespace(dtype="bf16"))
        )
    )
    assert _resolve_training_dtype(current) == "bf16"

    current_trainer = SimpleNamespace(
        training=SimpleNamespace(
            main=SimpleNamespace(trainer=SimpleNamespace(dtype="fp32"))
        )
    )
    assert _resolve_training_dtype(current_trainer) == "fp32"

    legacy = SimpleNamespace(
        train=SimpleNamespace(
            main=SimpleNamespace(trainer=SimpleNamespace(dtype="int8"))
        )
    )
    assert _resolve_training_dtype(legacy) == "int8"


if __name__ == "__main__":
    unittest.main()

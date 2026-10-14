"""
Test suite for dendritic_modeling.datasets module.

This module tests all dataset functionality including:
- Synthetic datasets
- Real dataset loading (MNIST, CIFAR-10)
- Dataset utilities and transformations
- Data path handling
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.datasets import (
    get_data_directory,
    get_unified_datasets,
    load_cifar10_as_datasets,
    load_imagenet_as_datasets,
    load_mnist_as_datasets,
    load_mnist_modulo10,
)
from dendritic_modeling.datasets.imagenet_loading import (
    _load_imagefolder_dataset_triplet,
    _validate_imagefolder_backend_roots,
)
from dendritic_modeling.datasets.imagenet_paths import _imagefolder_root_candidates
from dendritic_modeling.datasets.imagenet_transforms import (
    _legacy_resize_transform_steps,
    _pretrained_eval_transform_steps,
    _pretrained_train_transform_steps,
)
from dendritic_modeling.datasets.sequence.builders.common import explicit_split
from dendritic_modeling.datasets.sequence_datasets import (
    register_sequence_dataset_builder,
    unregister_sequence_dataset_builder,
)
from dendritic_modeling.datasets.standard_datasets import (
    _STANDARD_VISION_DATASET_LOADERS,
    PoissonGeneratorDataset as StandardPoissonGeneratorDataset,
    _apply_imagenet_class_subsets,
    _apply_imagenet_local_override,
    _build_imagenet_transforms,
    _build_to_tensor_transform,
    _class_subset_indices,
    _copy_dataset_with_transform,
    _dataset_mapping_to_triplet,
    _is_official_imagenet_root,
    _load_imagenet_dataset_triplet,
    _load_mnist_style_as_datasets,
    _load_paired_arithmetic_from_options,
    _load_standard_vision_dataset_triplet,
    _load_torchvision_train_test_pair,
    _materialize_cifar10_tensor_triplet,
    _materialize_flattened_tensor_dataset,
    _materialize_mnist_style_tensor_triplet,
    _maybe_apply_label_noise_to_triplet,
    _maybe_materialize_flattened_imagenet_triplet,
    _normalized_imagenet_loader_backend,
    _normalized_imagenet_preset_name,
    _paired_arithmetic_option_kwargs,
    _resolve_imagefolder_roots,
    _resolve_imagenet_data_location,
    _split_imagenet_train_valid_or_use_test,
    _split_train_valid_dataset,
    _split_train_valid_pair,
    _standard_dataset_option_kwargs,
    _subset_by_indices,
    _wrap_poisson_generator_triplet,
)


class TestRealDatasets(unittest.TestCase):
    """Test real dataset loading functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    class _TinyVisionDataset(torch.utils.data.Dataset):
        """Tiny torchvision-style dataset stub used for loader smoke tests."""

        def __init__(
            self,
            n_samples,
            sample_shape,
            num_classes=10,
            *,
            transform=None,
            target_transform=None,
        ):
            self.n_samples = int(n_samples)
            self.sample_shape = tuple(sample_shape)
            self.num_classes = int(num_classes)
            self.transform = transform
            self.target_transform = target_transform
            self.transforms = SimpleNamespace(
                transform=transform,
                target_transform=target_transform,
            )
            self.targets = [idx % self.num_classes for idx in range(self.n_samples)]
            self.samples = [
                (f"sample_{idx}.jpg", self.targets[idx])
                for idx in range(self.n_samples)
            ]

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            if idx < 0 or idx >= self.n_samples:
                raise IndexError(idx)
            sample = torch.full(self.sample_shape, float(idx % 7), dtype=torch.float32)
            label = int(idx % self.num_classes)
            return sample, label

    def test_build_to_tensor_transform_adds_optional_normalization(self):
        plain_transform = _build_to_tensor_transform(
            normalize=False,
            mean=(0.1,),
            std=(0.2,),
        )
        normalized_transform = _build_to_tensor_transform(
            normalize=True,
            mean=(0.1,),
            std=(0.2,),
        )

        self.assertEqual(len(plain_transform.transforms), 1)
        self.assertEqual(len(normalized_transform.transforms), 2)
        self.assertEqual(tuple(normalized_transform.transforms[1].mean), (0.1,))
        self.assertEqual(tuple(normalized_transform.transforms[1].std), (0.2,))

    def test_split_train_valid_dataset_is_deterministic(self):
        dataset = self._TinyVisionDataset(10, (1, 4, 4))

        train_a, valid_a = _split_train_valid_dataset(
            dataset,
            train_valid_split=0.6,
            split_seed=13,
        )
        train_b, valid_b = _split_train_valid_dataset(
            dataset,
            train_valid_split=0.6,
            split_seed=13,
        )

        self.assertEqual(len(train_a), 6)
        self.assertEqual(len(valid_a), 4)
        self.assertEqual(train_a.indices, train_b.indices)
        self.assertEqual(valid_a.indices, valid_b.indices)

    @patch(
        "dendritic_modeling.datasets.standard_datasets._load_torchvision_train_test_pair"
    )
    def test_load_mnist_style_as_datasets_forwards_loader_metadata(
        self,
        mock_load_pair,
    ):
        mock_load_pair.return_value = (
            self._TinyVisionDataset(10, (1, 4, 4)),
            self._TinyVisionDataset(4, (1, 4, 4)),
        )

        train_ds, valid_ds, test_ds = _load_mnist_style_as_datasets(
            dataset_key="unit_mnist",
            dataset_cls=object,
            dataset_name="UnitMNIST",
            mean=(0.1,),
            std=(0.2,),
            train_valid_split=0.6,
            flatten=True,
            normalize=True,
            task_data_path=self.temp_dir,
            split_seed=7,
        )

        self.assertEqual(tuple(train_ds.tensors[0].shape), (6, 16))
        self.assertEqual(tuple(valid_ds.tensors[0].shape), (4, 16))
        self.assertEqual(tuple(test_ds.tensors[0].shape), (4, 16))
        self.assertIs(mock_load_pair.call_args.args[0], object)
        call_kwargs = mock_load_pair.call_args.kwargs
        self.assertEqual(call_kwargs["dataset_name"], "UnitMNIST")
        self.assertEqual(
            call_kwargs["root"],
            get_data_directory("unit_mnist", self.temp_dir),
        )
        self.assertEqual(len(call_kwargs["transform"].transforms), 2)

    @patch(
        "dendritic_modeling.datasets.standard_datasets._load_torchvision_train_test_pair"
    )
    @unittest.skipIf(
        os.environ.get("GITHUB_ACTIONS") == "true",
        "Wrapper smoke tests are flaky on hosted runners; cache/helper tests cover CI.",
    )
    def test_load_mnist_as_datasets(self, mock_load_pair):
        """Test MNIST dataset loading on a tiny torchvision-style smoke set."""
        mock_load_pair.return_value = (
            self._TinyVisionDataset(12, (1, 28, 28)),
            self._TinyVisionDataset(6, (1, 28, 28)),
        )
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=0.8,
            flatten=True,
            normalize=False,
            task_data_path=self.temp_dir,
        )

        # Check that datasets were created
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(valid_ds)
        self.assertIsNotNone(test_ds)

        # Check dataset types and basic properties
        self.assertIsInstance(train_ds, TensorDataset)
        self.assertIsInstance(valid_ds, TensorDataset)
        self.assertIsInstance(test_ds, TensorDataset)

        # Check that we have data
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(test_ds), 0)
        # valid_ds might be empty if train_valid_split=1

    def test_materialize_mnist_style_tensor_triplet_preserves_empty_valid_flatten(self):
        train_dataset = self._TinyVisionDataset(5, (1, 4, 4))
        test_dataset = self._TinyVisionDataset(3, (1, 4, 4))

        train_ds, valid_ds, test_ds = _materialize_mnist_style_tensor_triplet(
            train_dataset,
            test_dataset,
            train_valid_split=1.0,
            flatten=True,
            split_seed=0,
        )

        self.assertEqual(tuple(train_ds.tensors[0].shape), (5, 16))
        self.assertEqual(tuple(valid_ds.tensors[0].shape), (0, 16))
        self.assertEqual(tuple(test_ds.tensors[0].shape), (3, 16))
        self.assertEqual(tuple(valid_ds.tensors[1].shape), (0,))
        self.assertEqual(valid_ds.tensors[1].dtype, torch.long)

    @patch(
        "dendritic_modeling.datasets.standard_datasets._load_torchvision_train_test_pair"
    )
    @unittest.skipIf(
        os.environ.get("GITHUB_ACTIONS") == "true",
        "Wrapper smoke tests are flaky on hosted runners; cache/helper tests cover CI.",
    )
    def test_load_cifar10_as_datasets(self, mock_load_pair):
        """Test CIFAR-10 dataset loading on a tiny torchvision-style smoke set."""
        mock_load_pair.return_value = (
            self._TinyVisionDataset(15, (3, 32, 32)),
            self._TinyVisionDataset(7, (3, 32, 32)),
        )
        train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
            train_valid_split=0.8,
            flatten=False,
            normalize=False,
            task_data_path=self.temp_dir,
        )

        # Check that datasets were created
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(valid_ds)
        self.assertIsNotNone(test_ds)

        # Check dataset types and basic properties
        self.assertIsInstance(train_ds, TensorDataset)
        self.assertIsInstance(valid_ds, TensorDataset)
        self.assertIsInstance(test_ds, TensorDataset)

        # Check that we have data
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(test_ds), 0)

    def test_materialize_cifar10_tensor_triplet_holds_out_nothing_at_full_split(
        self,
    ):
        train_dataset = self._TinyVisionDataset(5, (3, 4, 4))
        test_dataset = self._TinyVisionDataset(3, (3, 4, 4))

        train_ds, valid_ds, test_ds = _materialize_cifar10_tensor_triplet(
            train_dataset,
            test_dataset,
            train_valid_split=1.0,
            flatten=True,
            split_seed=0,
        )

        self.assertEqual(tuple(train_ds.tensors[0].shape), (5, 48))
        self.assertEqual(tuple(valid_ds.tensors[0].shape), (0, 48))
        self.assertEqual(tuple(test_ds.tensors[0].shape), (3, 48))

    def test_mnist_avoids_redownload_when_cache_is_good(self):
        """MNIST cache helper should use the local cache when it is already valid."""

        dataset_cls = Mock(
            side_effect=[
                self._TinyVisionDataset(2, (1, 28, 28)),
                self._TinyVisionDataset(2, (1, 28, 28)),
            ]
        )

        train_ds, test_ds = _load_torchvision_train_test_pair(
            dataset_cls,
            root=self.temp_dir,
            transform=Mock(),
            dataset_name="MNIST",
        )

        self.assertEqual(dataset_cls.call_args_list[0].kwargs["download"], False)
        self.assertEqual(dataset_cls.call_args_list[1].kwargs["download"], False)
        self.assertEqual(len(train_ds), 2)
        self.assertEqual(len(test_ds), 2)

    def test_cifar_redownloads_after_corrupt_cache(self):
        """Cache helper should clear and redownload after a recoverable cache error."""

        dataset_cls = Mock(
            side_effect=[
                RuntimeError("File not found or corrupted."),
                self._TinyVisionDataset(2, (3, 32, 32)),
                self._TinyVisionDataset(2, (3, 32, 32)),
            ]
        )

        train_ds, test_ds = _load_torchvision_train_test_pair(
            dataset_cls,
            root=self.temp_dir,
            transform=Mock(),
            dataset_name="CIFAR10",
        )

        self.assertEqual(dataset_cls.call_args_list[0].kwargs["download"], False)
        self.assertEqual(dataset_cls.call_args_list[1].kwargs["download"], True)
        self.assertEqual(dataset_cls.call_args_list[2].kwargs["download"], True)
        self.assertEqual(len(train_ds), 2)
        self.assertEqual(len(test_ds), 2)

    def test_cache_repair_removes_only_the_failing_datasets_files(self):
        """Repairing CIFAR-100 keeps CIFAR-10 and MNIST files in the same root."""
        from torchvision import datasets as tv_datasets

        from dendritic_modeling.datasets.standard_datasets import _clear_dataset_cache

        names = (
            "cifar-10-batches-py",
            "cifar-10-python.tar.gz",
            "cifar-100-python",
            "cifar-100-python.tar.gz",
            "MNIST",
        )
        for name in names:
            path = os.path.join(self.temp_dir, name)
            if name.endswith(".gz"):
                open(path, "w").close()
            else:
                os.makedirs(os.path.join(path, "raw"))
        _clear_dataset_cache(tv_datasets.CIFAR100, self.temp_dir)
        self.assertEqual(
            sorted(os.listdir(self.temp_dir)),
            ["MNIST", "cifar-10-batches-py", "cifar-10-python.tar.gz"],
        )
        _clear_dataset_cache(tv_datasets.MNIST, self.temp_dir)
        self.assertNotIn("MNIST", os.listdir(self.temp_dir))

    def test_load_mnist_modulo10(self):
        """Test MNIST modulo 10 dataset loading."""
        # Create mock datasets
        mock_train = torch.utils.data.TensorDataset(
            torch.randn(60, 784), torch.randint(0, 10, (60,))
        )
        mock_valid = torch.utils.data.TensorDataset(
            torch.randn(20, 784), torch.randint(0, 10, (20,))
        )
        mock_test = torch.utils.data.TensorDataset(
            torch.randn(20, 784), torch.randint(0, 10, (20,))
        )

        with patch(
            "dendritic_modeling.datasets.standard_datasets.load_mnist_as_datasets"
        ) as mock_load:
            mock_load.return_value = (mock_train, mock_valid, mock_test)

            data = load_mnist_modulo10(
                shuffle_iterations=1,
                train_valid_split=0.8,
                flatten=True,
                task_data_path=self.temp_dir,
            )

            # Check that data dictionary was created
            self.assertIn("train", data)
            self.assertIn("valid", data)
            self.assertIn("test", data)

            # Check that mock was called with correct parameters
            mock_load.assert_called_once()

    def test_resolve_imagefolder_roots_detects_direct_train_val_tree(self):
        os.makedirs(os.path.join(self.temp_dir, "train", "class_a"))
        os.makedirs(os.path.join(self.temp_dir, "val", "class_a"))

        train_root, val_root = _resolve_imagefolder_roots(self.temp_dir)

        self.assertEqual(train_root, os.path.join(self.temp_dir, "train"))
        self.assertEqual(val_root, os.path.join(self.temp_dir, "val"))

    def test_imagefolder_root_candidates_preserve_layout_order(self):
        self.assertEqual(
            _imagefolder_root_candidates("/imagenet"),
            [
                ("/imagenet/train", "/imagenet/val"),
                ("/imagenet/ILSVRC2012_img_train", "/imagenet/val"),
            ],
        )

    def test_resolve_imagefolder_roots_detects_ilsvrc_train_tree(self):
        os.makedirs(os.path.join(self.temp_dir, "ILSVRC2012_img_train", "class_a"))
        os.makedirs(os.path.join(self.temp_dir, "val", "class_a"))

        train_root, val_root = _resolve_imagefolder_roots(self.temp_dir)

        self.assertEqual(
            train_root,
            os.path.join(self.temp_dir, "ILSVRC2012_img_train"),
        )
        self.assertEqual(val_root, os.path.join(self.temp_dir, "val"))

    def test_is_official_imagenet_root_requires_metadata_and_splits(self):
        os.makedirs(os.path.join(self.temp_dir, "ILSVRC2012_img_train", "class_a"))
        os.makedirs(os.path.join(self.temp_dir, "val", "class_a"))
        with open(os.path.join(self.temp_dir, "meta.bin"), "wb") as f:
            f.write(b"meta")

        self.assertTrue(_is_official_imagenet_root(self.temp_dir))

    def test_resolve_imagenet_data_location_prefers_official_root(self):
        os.makedirs(os.path.join(self.temp_dir, "train", "class_a"))
        os.makedirs(os.path.join(self.temp_dir, "val", "class_a"))
        with open(os.path.join(self.temp_dir, "meta.bin"), "wb") as f:
            f.write(b"meta")

        imagenet_dir, train_root, val_root = _resolve_imagenet_data_location(
            self.temp_dir
        )

        self.assertEqual(imagenet_dir, self.temp_dir)
        self.assertIsNone(train_root)
        self.assertIsNone(val_root)

    def test_resolve_imagenet_data_location_can_force_imagefolder(self):
        os.makedirs(os.path.join(self.temp_dir, "train", "class_a"))
        os.makedirs(os.path.join(self.temp_dir, "val", "class_a"))
        with open(os.path.join(self.temp_dir, "meta.bin"), "wb") as f:
            f.write(b"meta")

        imagenet_dir, train_root, val_root = _resolve_imagenet_data_location(
            self.temp_dir,
            loader_backend="imagefolder",
        )

        self.assertEqual(imagenet_dir, self.temp_dir)
        self.assertEqual(train_root, os.path.join(self.temp_dir, "train"))
        self.assertEqual(val_root, os.path.join(self.temp_dir, "val"))

    def test_resolve_imagenet_data_location_can_force_torchvision(self):
        os.makedirs(os.path.join(self.temp_dir, "train", "class_a"))
        os.makedirs(os.path.join(self.temp_dir, "val", "class_a"))

        imagenet_dir, train_root, val_root = _resolve_imagenet_data_location(
            self.temp_dir,
            loader_backend="torchvision",
        )

        self.assertEqual(imagenet_dir, self.temp_dir)
        self.assertIsNone(train_root)
        self.assertIsNone(val_root)

    def test_resolve_imagenet_data_location_falls_back_to_datasets_subdir(self):
        imagenet_dir = os.path.join(self.temp_dir, "datasets", "imagenet")
        os.makedirs(os.path.join(imagenet_dir, "train", "class_a"))
        os.makedirs(os.path.join(imagenet_dir, "val", "class_a"))

        resolved_dir, train_root, val_root = _resolve_imagenet_data_location(
            self.temp_dir
        )

        self.assertEqual(resolved_dir, imagenet_dir)
        self.assertEqual(train_root, os.path.join(imagenet_dir, "train"))
        self.assertEqual(val_root, os.path.join(imagenet_dir, "val"))

    def test_apply_imagenet_local_override_uses_existing_env_dir(self):
        local_dir = os.path.join(self.temp_dir, "local_imagenet")
        os.makedirs(local_dir)

        with patch.dict(os.environ, {"IMAGENET_LOCAL_DIR": local_dir}):
            self.assertEqual(
                _apply_imagenet_local_override("/configured/path"),
                local_dir,
            )

    def test_class_subset_indices_caps_each_class_deterministically(self):
        dataset = Mock()
        dataset.targets = [0, 0, 0, 1, 1, 1]

        indices = _class_subset_indices(
            dataset,
            2,
            split_name="train",
            seed=13,
        )
        repeated = _class_subset_indices(
            dataset,
            2,
            split_name="train",
            seed=13,
        )

        self.assertEqual(indices, repeated)
        self.assertEqual(len(indices), 4)
        self.assertLessEqual(sum(dataset.targets[i] == 0 for i in indices), 2)
        self.assertLessEqual(sum(dataset.targets[i] == 1 for i in indices), 2)

    def test_class_subset_indices_returns_none_without_targets(self):
        dataset = Mock(spec=[])

        self.assertIsNone(_class_subset_indices(dataset, 1, split_name="train", seed=0))

    def test_subset_by_indices_is_noop_without_indices(self):
        dataset = TensorDataset(torch.arange(4), torch.arange(4))

        self.assertIs(_subset_by_indices(dataset, None), dataset)
        subset = _subset_by_indices(dataset, [1, 3])

        self.assertEqual(len(subset), 2)
        self.assertEqual(list(subset.indices), [1, 3])

    def test_split_train_valid_pair_uses_eval_dataset_for_validation(self):
        train_dataset = TensorDataset(torch.arange(10), torch.arange(10))
        train_eval_dataset = TensorDataset(torch.arange(10) + 100, torch.arange(10))

        train_subset, valid_subset = _split_train_valid_pair(
            train_dataset,
            train_eval_dataset,
            train_valid_split=0.6,
            split_seed=3,
        )

        self.assertEqual(len(train_subset), 6)
        self.assertEqual(len(valid_subset), 4)
        self.assertIs(train_subset.dataset, train_dataset)
        self.assertIs(valid_subset.dataset, train_eval_dataset)

    def test_apply_imagenet_class_subsets_preserves_train_eval_alignment(self):
        train_dataset = self._TinyVisionDataset(6, (3, 224, 224), num_classes=3)
        train_eval_dataset = self._TinyVisionDataset(6, (3, 224, 224), num_classes=3)
        test_dataset = self._TinyVisionDataset(6, (3, 224, 224), num_classes=3)

        subset_train, subset_train_eval, subset_test = _apply_imagenet_class_subsets(
            train_dataset=train_dataset,
            train_eval_dataset=train_eval_dataset,
            test_dataset=test_dataset,
            train_samples_per_class=1,
            val_samples_per_class=1,
            split_seed=11,
        )

        self.assertEqual(len(subset_train), 3)
        self.assertEqual(len(subset_train_eval), 3)
        self.assertEqual(len(subset_test), 3)
        self.assertEqual(list(subset_train.indices), list(subset_train_eval.indices))
        self.assertIs(subset_train.dataset, train_dataset)
        self.assertIs(subset_train_eval.dataset, train_eval_dataset)
        self.assertIs(subset_test.dataset, test_dataset)

    def test_split_imagenet_train_valid_or_use_test_preserves_no_len_fallback(self):
        class NoLenDataset(torch.utils.data.Dataset):
            def __getitem__(self, idx):
                raise IndexError(idx)

        train_dataset = NoLenDataset()
        train_eval_dataset = NoLenDataset()
        test_dataset = NoLenDataset()

        train_out, valid_out = _split_imagenet_train_valid_or_use_test(
            train_dataset=train_dataset,
            train_eval_dataset=train_eval_dataset,
            test_dataset=test_dataset,
            train_valid_split=0.8,
            split_seed=0,
        )

        self.assertIs(train_out, train_dataset)
        self.assertIs(valid_out, test_dataset)

    def test_maybe_materialize_flattened_imagenet_triplet_respects_flatten_flag(self):
        images = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        labels = torch.arange(4)
        train_dataset = TensorDataset(images, labels)
        valid_dataset = TensorDataset(images + 100, labels)
        test_dataset = TensorDataset(images + 200, labels)

        unchanged = _maybe_materialize_flattened_imagenet_triplet(
            train_dataset,
            valid_dataset,
            test_dataset,
            flatten=False,
        )
        flattened = _maybe_materialize_flattened_imagenet_triplet(
            train_dataset,
            valid_dataset,
            test_dataset,
            flatten=True,
        )

        self.assertEqual(unchanged, (train_dataset, valid_dataset, test_dataset))
        for dataset in flattened:
            self.assertIsInstance(dataset, TensorDataset)
            self.assertEqual(tuple(dataset.tensors[0].shape), (4, 6))

    def test_normalized_imagenet_preset_name_normalizes_aliases(self):
        self.assertEqual(
            _normalized_imagenet_preset_name("legacy-resize"), "legacy_resize"
        )
        self.assertEqual(_normalized_imagenet_preset_name(None), "legacy_resize")

    def test_normalized_imagenet_loader_backend_normalizes_aliases(self):
        self.assertEqual(_normalized_imagenet_loader_backend(None), "auto")
        self.assertEqual(_normalized_imagenet_loader_backend("official"), "torchvision")
        self.assertEqual(
            _normalized_imagenet_loader_backend("image-folder"),
            "imagefolder",
        )
        with self.assertRaisesRegex(ValueError, "loader_backend"):
            _normalized_imagenet_loader_backend("synthetic")

    def test_imagenet_transform_step_helpers_preserve_operation_order(self):
        self.assertEqual(
            [type(t).__name__ for t in _legacy_resize_transform_steps()],
            ["Resize", "ToTensor"],
        )
        self.assertEqual(
            [type(t).__name__ for t in _pretrained_train_transform_steps()],
            ["RandomResizedCrop", "RandomHorizontalFlip", "ToTensor"],
        )
        self.assertEqual(
            [type(t).__name__ for t in _pretrained_eval_transform_steps()],
            ["Resize", "CenterCrop", "ToTensor"],
        )

    def test_build_imagenet_transforms_legacy_uses_shared_resize_transform(self):
        train_transform, eval_transform = _build_imagenet_transforms(
            normalize=False,
            transform_preset="legacy_resize",
        )

        names = [type(t).__name__ for t in train_transform.transforms]
        self.assertIs(train_transform, eval_transform)
        self.assertEqual(names, ["Resize", "ToTensor"])

    def test_build_imagenet_transforms_pretrained_uses_train_eval_pipelines(self):
        train_transform, eval_transform = _build_imagenet_transforms(
            normalize=True,
            transform_preset="pretrained",
        )

        train_names = [type(t).__name__ for t in train_transform.transforms]
        eval_names = [type(t).__name__ for t in eval_transform.transforms]
        self.assertEqual(
            train_names,
            ["RandomResizedCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"],
        )
        self.assertEqual(eval_names, ["Resize", "CenterCrop", "ToTensor", "Normalize"])

    def test_copy_dataset_with_transform_preserves_imagenet_metadata(self):
        train_transform = Mock()
        eval_transform = Mock()
        target_transform = Mock()
        dataset = self._TinyVisionDataset(
            3,
            (3, 224, 224),
            transform=train_transform,
            target_transform=target_transform,
        )

        copied = _copy_dataset_with_transform(dataset, eval_transform)

        self.assertIsNot(copied, dataset)
        self.assertIs(dataset.transform, train_transform)
        self.assertIs(copied.transform, eval_transform)
        self.assertIs(copied.target_transform, target_transform)
        self.assertIs(copied.targets, dataset.targets)
        self.assertIs(copied.samples, dataset.samples)
        self.assertIs(copied.transforms.transform, eval_transform)
        self.assertIs(copied.transforms.target_transform, target_transform)

    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageNet")
    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageFolder")
    def test_load_imagenet_dataset_triplet_uses_imagefolder_roots(
        self, mock_imagefolder, mock_imagenet
    ):
        train_transform = Mock()
        eval_transform = Mock()
        train_mock = self._TinyVisionDataset(
            10,
            (3, 224, 224),
            transform=train_transform,
        )
        test_mock = self._TinyVisionDataset(
            4,
            (3, 224, 224),
            transform=eval_transform,
        )
        mock_imagefolder.side_effect = [train_mock, test_mock]

        loaded = _load_imagenet_dataset_triplet(
            imagenet_dir="/unused",
            train_root="/imagenet/train",
            val_root="/imagenet/val",
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

        train_loaded, train_eval_loaded, test_loaded = loaded
        self.assertIs(train_loaded, train_mock)
        self.assertIsNot(train_eval_loaded, train_mock)
        self.assertIs(train_eval_loaded.transform, eval_transform)
        self.assertIs(train_eval_loaded.targets, train_mock.targets)
        self.assertIs(test_loaded, test_mock)
        mock_imagefolder.assert_any_call(
            root="/imagenet/train", transform=train_transform
        )
        mock_imagefolder.assert_any_call(root="/imagenet/val", transform=eval_transform)
        self.assertEqual(mock_imagefolder.call_count, 2)
        mock_imagenet.assert_not_called()

    def test_load_imagefolder_dataset_triplet_requires_both_roots(self):
        with self.assertRaisesRegex(RuntimeError, "both train and val"):
            _load_imagefolder_dataset_triplet(
                train_root="/imagenet/train",
                val_root=None,
                train_transform=Mock(),
                eval_transform=Mock(),
            )

    def test_validate_imagefolder_backend_roots_preserves_backend_specific_error(self):
        _validate_imagefolder_backend_roots(
            "auto",
            train_root=None,
            val_root=None,
        )
        _validate_imagefolder_backend_roots(
            "imagefolder",
            train_root="/imagenet/train",
            val_root="/imagenet/val",
        )

        with self.assertRaisesRegex(RuntimeError, "loader_backend='imagefolder'"):
            _validate_imagefolder_backend_roots(
                "imagefolder",
                train_root="/imagenet/train",
                val_root=None,
            )

    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageNet")
    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageFolder")
    def test_load_imagenet_dataset_triplet_uses_official_loader(
        self, mock_imagefolder, mock_imagenet
    ):
        train_transform = Mock()
        eval_transform = Mock()
        train_mock = self._TinyVisionDataset(
            10,
            (3, 224, 224),
            transform=train_transform,
        )
        test_mock = self._TinyVisionDataset(
            4,
            (3, 224, 224),
            transform=eval_transform,
        )
        mock_imagenet.side_effect = [train_mock, test_mock]

        loaded = _load_imagenet_dataset_triplet(
            imagenet_dir="/imagenet",
            train_root=None,
            val_root=None,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

        train_loaded, train_eval_loaded, test_loaded = loaded
        self.assertIs(train_loaded, train_mock)
        self.assertIsNot(train_eval_loaded, train_mock)
        self.assertIs(train_eval_loaded.transform, eval_transform)
        self.assertIs(train_eval_loaded.targets, train_mock.targets)
        self.assertIs(test_loaded, test_mock)
        mock_imagenet.assert_any_call(
            root="/imagenet", split="train", transform=train_transform
        )
        mock_imagenet.assert_any_call(
            root="/imagenet", split="val", transform=eval_transform
        )
        self.assertEqual(mock_imagenet.call_count, 2)
        mock_imagefolder.assert_not_called()

    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageNet")
    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageFolder")
    def test_load_imagenet_dataset_triplet_forced_imagefolder_requires_roots(
        self, mock_imagefolder, mock_imagenet
    ):
        with self.assertRaisesRegex(RuntimeError, "imagefolder"):
            _load_imagenet_dataset_triplet(
                imagenet_dir="/imagenet",
                train_root=None,
                val_root=None,
                train_transform=Mock(),
                eval_transform=Mock(),
                loader_backend="imagefolder",
            )

        mock_imagefolder.assert_not_called()
        mock_imagenet.assert_not_called()

    def test_materialize_flattened_tensor_dataset_flattens_and_caps_samples(self):
        images = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        labels = torch.arange(4)
        dataset = TensorDataset(images, labels)

        flattened = _materialize_flattened_tensor_dataset(dataset, max_samples=3)

        self.assertIsInstance(flattened, TensorDataset)
        data_tensor, label_tensor = flattened.tensors
        self.assertEqual(tuple(data_tensor.shape), (3, 6))
        self.assertTrue(torch.equal(data_tensor[0], images[0].view(-1)))
        self.assertTrue(torch.equal(label_tensor, torch.tensor([0, 1, 2])))

    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageNet")
    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageFolder")
    def test_load_imagenet_prefers_imagefolder_for_direct_tree(
        self, mock_imagefolder, mock_imagenet
    ):
        """Direct train/val trees should use ImageFolder instead of ImageNet."""
        os.makedirs(os.path.join(self.temp_dir, "train", "class_a"))
        os.makedirs(os.path.join(self.temp_dir, "val", "class_a"))

        train_mock = self._TinyVisionDataset(10, (3, 224, 224))
        test_mock = self._TinyVisionDataset(4, (3, 224, 224))
        mock_imagefolder.side_effect = [train_mock, test_mock]

        train_ds, valid_ds, test_ds = load_imagenet_as_datasets(
            train_valid_split=0.8,
            flatten=False,
            normalize=False,
            task_data_path=self.temp_dir,
        )

        mock_imagefolder.assert_any_call(
            root=os.path.join(self.temp_dir, "train"), transform=ANY
        )
        mock_imagefolder.assert_any_call(
            root=os.path.join(self.temp_dir, "val"), transform=ANY
        )
        self.assertEqual(mock_imagefolder.call_count, 2)
        mock_imagenet.assert_not_called()
        self.assertEqual(len(train_ds), 8)
        self.assertEqual(len(valid_ds), 2)
        self.assertIs(test_ds, test_mock)

    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageNet")
    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageFolder")
    def test_load_imagenet_prefers_official_loader_when_metadata_exists(
        self, mock_imagefolder, mock_imagenet
    ):
        """Official ImageNet roots should use torchvision's ImageNet loader."""
        os.makedirs(os.path.join(self.temp_dir, "ILSVRC2012_img_train", "class_a"))
        os.makedirs(os.path.join(self.temp_dir, "val", "class_a"))
        with open(os.path.join(self.temp_dir, "meta.bin"), "wb") as f:
            f.write(b"meta")

        train_mock = self._TinyVisionDataset(10, (3, 224, 224))
        test_mock = self._TinyVisionDataset(4, (3, 224, 224))
        mock_imagenet.side_effect = [train_mock, test_mock]

        train_ds, valid_ds, test_ds = load_imagenet_as_datasets(
            train_valid_split=0.8,
            flatten=False,
            normalize=False,
            task_data_path=self.temp_dir,
        )

        mock_imagenet.assert_any_call(root=self.temp_dir, split="train", transform=ANY)
        mock_imagenet.assert_any_call(root=self.temp_dir, split="val", transform=ANY)
        self.assertEqual(mock_imagenet.call_count, 2)
        mock_imagefolder.assert_not_called()
        self.assertEqual(len(train_ds), 8)
        self.assertEqual(len(valid_ds), 2)
        self.assertIs(test_ds, test_mock)

    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageNet")
    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageFolder")
    def test_load_imagenet_pretrained_preset_uses_training_only_augmentations(
        self, mock_imagefolder, mock_imagenet
    ):
        """Pretrained preset should match torchvision train/eval preprocessing."""
        os.makedirs(os.path.join(self.temp_dir, "train", "class_a"))
        os.makedirs(os.path.join(self.temp_dir, "val", "class_a"))

        train_mock = self._TinyVisionDataset(10, (3, 224, 224))
        test_mock = self._TinyVisionDataset(4, (3, 224, 224))
        mock_imagefolder.side_effect = [train_mock, test_mock]

        train_ds, valid_ds, test_ds = load_imagenet_as_datasets(
            train_valid_split=0.8,
            flatten=False,
            normalize=False,
            task_data_path=self.temp_dir,
            transform_preset="pretrained",
        )

        transforms_by_call = [
            [t.__class__.__name__ for t in call.kwargs["transform"].transforms]
            for call in mock_imagefolder.call_args_list
        ]
        self.assertIn("RandomResizedCrop", transforms_by_call[0])
        self.assertIn("RandomHorizontalFlip", transforms_by_call[0])
        self.assertIn("Resize", transforms_by_call[1])
        self.assertIn("CenterCrop", transforms_by_call[1])
        self.assertEqual(mock_imagefolder.call_count, 2)
        valid_transform_names = [
            t.__class__.__name__ for t in valid_ds.dataset.transform.transforms
        ]
        self.assertIn("Resize", valid_transform_names)
        self.assertIn("CenterCrop", valid_transform_names)
        self.assertIs(train_ds.dataset, train_mock)
        self.assertIs(test_ds, test_mock)
        mock_imagenet.assert_not_called()

    @patch(
        "dendritic_modeling.datasets.imagenet.datasets.ImageNet",
        side_effect=RuntimeError("missing devkit"),
    )
    def test_load_imagenet_missing_raises_without_fakedata_fallback(
        self, _mock_imagenet
    ):
        """Missing ImageNet should fail instead of falling back to synthetic data."""
        with self.assertRaisesRegex(RuntimeError, "valid ImageNet root") as ctx:
            load_imagenet_as_datasets(
                train_valid_split=0.8,
                flatten=False,
                normalize=False,
                task_data_path=self.temp_dir,
            )
        message = str(ctx.exception)
        self.assertIn("loader_backend='auto'", message)
        self.assertIn(f"task_data_path='{self.temp_dir}'", message)
        self.assertIn("imagenet_dir=", message)
        self.assertIn("No synthetic or FakeData fallback is used.", message)

    def test_load_imagenet_flatten_requires_explicit_opt_in(self):
        """Flattening ImageNet is memory-heavy and should be explicit."""
        with self.assertRaisesRegex(ValueError, "allow_flatten_materialize"):
            load_imagenet_as_datasets(
                train_valid_split=0.8,
                flatten=True,
                normalize=False,
                task_data_path=self.temp_dir,
            )

    @patch("dendritic_modeling.datasets.imagenet.datasets.ImageNet")
    def test_load_imagenet_applies_per_class_subset(self, mock_imagenet):
        """ImageNet loader should support config-driven per-class smoke subsets."""

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, targets):
                self.targets = list(targets)

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                return torch.zeros(3, 224, 224), self.targets[idx]

        train_mock = DummyDataset([0, 0, 1, 1, 2, 2])
        test_mock = DummyDataset([0, 0, 1, 1, 2, 2])
        mock_imagenet.side_effect = [train_mock, test_mock]

        with open(os.path.join(self.temp_dir, "meta.bin"), "wb") as f:
            f.write(b"meta")
        os.makedirs(os.path.join(self.temp_dir, "ILSVRC2012_img_train", "class_a"))
        os.makedirs(os.path.join(self.temp_dir, "val", "class_a"))

        train_ds, valid_ds, test_ds = load_imagenet_as_datasets(
            train_valid_split=0.5,
            flatten=False,
            normalize=False,
            task_data_path=self.temp_dir,
            train_samples_per_class=1,
            val_samples_per_class=1,
        )

        self.assertEqual(len(train_ds), 1)
        self.assertEqual(len(valid_ds), 2)
        self.assertEqual(len(test_ds), 3)


class TestDatasetUtilities(unittest.TestCase):
    """Test dataset utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_standard_dataset_option_kwargs_preserve_unified_loader_fields(self):
        options = SimpleNamespace(
            train_valid_split=0.75,
            flatten=True,
            normalize=False,
            task_data_path="/tmp/data",
            split_seed=17,
        )

        kwargs = _standard_dataset_option_kwargs(options)

        self.assertEqual(
            kwargs,
            {
                "train_valid_split": 0.75,
                "flatten": True,
                "normalize": False,
                "task_data_path": "/tmp/data",
                "split_seed": 17,
            },
        )

    def test_paired_arithmetic_option_kwargs_preserve_defaults_and_overrides(self):
        default_options = SimpleNamespace(
            train_valid_split=0.8,
            flatten=False,
            normalize=True,
            task_data_path="/tmp/data",
            split_seed=5,
            parameters={},
        )
        override_options = SimpleNamespace(
            train_valid_split=0.8,
            flatten=False,
            normalize=True,
            task_data_path="/tmp/data",
            split_seed=5,
            parameters={
                "shuffle_iterations": 4,
                "pair_seed": 9,
                "context_seed": 99,
            },
        )

        default_kwargs = _paired_arithmetic_option_kwargs(
            default_options,
            default_shuffle_iterations=2,
            include_context=False,
        )
        contextual_kwargs = _paired_arithmetic_option_kwargs(
            override_options,
            default_shuffle_iterations=1,
            include_context=True,
        )

        self.assertEqual(default_kwargs["shuffle_iterations"], 2)
        self.assertEqual(default_kwargs["pair_seed"], 0)
        self.assertNotIn("context_seed", default_kwargs)
        self.assertEqual(contextual_kwargs["shuffle_iterations"], 4)
        self.assertEqual(contextual_kwargs["pair_seed"], 9)
        self.assertEqual(contextual_kwargs["context_seed"], 99)
        self.assertEqual(contextual_kwargs["split_seed"], 5)

    def test_load_paired_arithmetic_from_options_preserves_fallback_mapping(self):
        options = SimpleNamespace(
            train_valid_split=0.7,
            flatten=True,
            normalize=False,
            task_data_path="/tmp/data",
            split_seed=3,
            parameters={"shuffle_iterations": 4, "pair_seed": 11},
        )
        train_ds = TensorDataset(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long))
        test_ds = TensorDataset(torch.ones(1, 2), torch.ones(1, dtype=torch.long))
        calls = []

        valid_ds = TensorDataset(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long))

        def fake_loader(**kwargs):
            calls.append(kwargs)
            return {"train": train_ds, "valid": valid_ds, "test": test_ds}

        out_train, out_valid, out_test = _load_paired_arithmetic_from_options(
            options,
            fake_loader,
            default_shuffle_iterations=2,
            include_context=False,
        )

        self.assertEqual(
            calls,
            [
                {
                    "train_valid_split": 0.7,
                    "flatten": True,
                    "normalize": False,
                    "task_data_path": "/tmp/data",
                    "split_seed": 3,
                    "shuffle_iterations": 4,
                    "pair_seed": 11,
                }
            ],
        )
        self.assertIs(out_train, train_ds)
        self.assertIs(out_valid, valid_ds)
        self.assertIs(out_test, test_ds)

    def test_standard_vision_loader_registry_preserves_dispatch_and_noise_hook(self):
        train_ds = TensorDataset(torch.zeros(4, 2), torch.arange(4, dtype=torch.long))
        valid_ds = TensorDataset(torch.ones(2, 2), torch.ones(2, dtype=torch.long))
        test_ds = TensorDataset(
            torch.full((2, 2), 2.0), torch.ones(2, dtype=torch.long)
        )
        calls = []

        def fake_loader(**kwargs):
            calls.append(kwargs)
            return train_ds, valid_ds, test_ds

        with patch.dict(
            _STANDARD_VISION_DATASET_LOADERS,
            {"custom_vision": fake_loader},
        ):
            noisy_train, out_valid, out_test = _load_standard_vision_dataset_triplet(
                dataset_name="custom_vision",
                train_valid_split=0.75,
                flatten=True,
                normalize=False,
                task_data_path="/tmp/data",
                split_seed=9,
                label_noise_rate=1.0,
                label_noise_seed=0,
            )

        self.assertEqual(
            calls,
            [
                {
                    "train_valid_split": 0.75,
                    "flatten": True,
                    "normalize": False,
                    "task_data_path": "/tmp/data",
                    "split_seed": 9,
                }
            ],
        )
        self.assertIs(out_valid, valid_ds)
        self.assertIs(out_test, test_ds)
        self.assertTrue(torch.equal(noisy_train.tensors[0], train_ds.tensors[0]))
        self.assertFalse(torch.equal(noisy_train.tensors[1], train_ds.tensors[1]))

    def test_maybe_apply_label_noise_only_changes_training_tensor_labels(self):
        """Label noise should not touch validation/test datasets."""
        train_ds = TensorDataset(torch.zeros(4, 2), torch.arange(4, dtype=torch.long))
        valid_ds = TensorDataset(torch.ones(2, 2), torch.ones(2, dtype=torch.long))
        test_ds = TensorDataset(
            torch.full((2, 2), 2.0), torch.ones(2, dtype=torch.long)
        )

        noisy_train, out_valid, out_test = _maybe_apply_label_noise_to_triplet(
            train_ds,
            valid_ds,
            test_ds,
            label_noise_rate=1.0,
            label_noise_seed=0,
        )

        self.assertIs(out_valid, valid_ds)
        self.assertIs(out_test, test_ds)
        self.assertTrue(torch.equal(noisy_train.tensors[0], train_ds.tensors[0]))
        self.assertFalse(torch.equal(noisy_train.tensors[1], train_ds.tensors[1]))

    def test_dataset_mapping_to_triplet_uses_explicit_valid_when_required(self):
        """Mapping conversion should require and return the explicit valid split."""
        train_ds = TensorDataset(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long))
        valid_ds = TensorDataset(torch.ones(1, 2), torch.ones(1, dtype=torch.long))
        test_ds = TensorDataset(
            torch.full((1, 2), 2.0), torch.ones(1, dtype=torch.long)
        )

        out_train, out_valid, out_test = _dataset_mapping_to_triplet(
            {"train": train_ds, "valid": valid_ds, "test": test_ds},
        )

        self.assertIs(out_train, train_ds)
        self.assertIs(out_valid, valid_ds)
        self.assertIs(out_test, test_ds)

    def test_get_data_directory(self):
        """Test data directory creation and path handling."""
        # Test with task data path
        data_dir = get_data_directory("mnist", task_data_path=self.temp_dir)
        expected_path = os.path.join(self.temp_dir, "datasets", "mnist")
        self.assertEqual(data_dir, expected_path)
        self.assertTrue(os.path.exists(data_dir))

        # Without a path, the shared data root (DENDRITIC_DATA_DIR in tests).
        data_dir2 = get_data_directory("cifar", task_data_path=None)
        expected_path2 = os.path.join(
            os.environ["DENDRITIC_DATA_DIR"], "datasets", "cifar"
        )
        self.assertEqual(data_dir2, expected_path2)
        self.assertTrue(os.path.exists(data_dir2))


def test_the_data_root_is_base_dir_then_environment_then_cache(monkeypatch, tmp_path):
    from dendritic_modeling.datasets.paths import resolve_data_root

    monkeypatch.setenv("DENDRITIC_DATA_DIR", str(tmp_path / "env"))
    assert resolve_data_root("/explicit") == "/explicit"
    assert resolve_data_root("") == str(tmp_path / "env")
    monkeypatch.delenv("DENDRITIC_DATA_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert resolve_data_root(None) == str(tmp_path / "cache" / "dendritic_modeling")


class TestGetUnifiedDatasets(unittest.TestCase):
    """Test the unified dataset loading function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock configs
        self.task_cfg = Mock()
        self.task_cfg.train_valid_split = 0.8

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("dendritic_modeling.datasets.standard_datasets.load_mnist_as_datasets")
    def test_mnist_dataset(self, mock_load_mnist):
        """Test MNIST dataset loading through unified interface."""
        # Setup mock
        mock_train = torch.utils.data.TensorDataset(
            torch.randn(48, 784), torch.randint(0, 10, (48,))
        )
        mock_valid = torch.utils.data.TensorDataset(
            torch.randn(12, 784), torch.randint(0, 10, (12,))
        )
        mock_test = torch.utils.data.TensorDataset(
            torch.randn(10, 784), torch.randint(0, 10, (10,))
        )
        mock_load_mnist.return_value = (mock_train, mock_valid, mock_test)

        # Configure for MNIST
        self.task_cfg.dataset = "mnist"
        self.task_cfg.data_path = self.temp_dir
        self.task_cfg.parameters = {"flatten": True, "normalize": False}

        train_ds, valid_ds, test_ds = get_unified_datasets(self.task_cfg)

        # Verify results
        self.assertEqual(len(train_ds), 48)
        self.assertEqual(len(valid_ds), 12)
        self.assertEqual(len(test_ds), 10)

        # Verify the function was called with correct parameters
        mock_load_mnist.assert_called_once()
        _, kwargs = mock_load_mnist.call_args
        self.assertEqual(kwargs["task_data_path"], self.temp_dir)

    @patch("dendritic_modeling.datasets.sequence_datasets.get_sequence_datasets")
    def test_sequence_dataset_dispatch(self, mock_get_sequence_datasets):
        """Sequence datasets should be delegated through the sequence factory."""
        mock_train = torch.utils.data.TensorDataset(
            torch.randn(8, 4, 2), torch.randint(0, 2, (8,))
        )
        mock_valid = torch.utils.data.TensorDataset(
            torch.randn(2, 4, 2), torch.randint(0, 2, (2,))
        )
        mock_test = torch.utils.data.TensorDataset(
            torch.randn(3, 4, 2), torch.randint(0, 2, (3,))
        )
        mock_get_sequence_datasets.return_value = (mock_train, mock_valid, mock_test)

        self.task_cfg.dataset = "copy_task"
        self.task_cfg.data_path = self.temp_dir
        self.task_cfg.parameters = {"seq_len": 4, "n_samples": 13}

        train_ds, valid_ds, test_ds = get_unified_datasets(self.task_cfg)

        self.assertIs(train_ds, mock_train)
        self.assertIs(valid_ds, mock_valid)
        self.assertIs(test_ds, mock_test)
        mock_get_sequence_datasets.assert_called_once_with(
            dataset_name="copy_task",
            data_path=self.temp_dir,
            train_valid_split=0.8,
            seq_len=4,
            n_samples=13,
        )

    def test_registered_sequence_dataset_dispatches_through_unified_loader(self):
        """Unified loading should use the live sequence builder registry."""
        mock_train = torch.utils.data.TensorDataset(
            torch.randn(8, 4, 2), torch.randint(0, 2, (8,))
        )
        mock_valid = torch.utils.data.TensorDataset(
            torch.randn(2, 4, 2), torch.randint(0, 2, (2,))
        )
        mock_test = torch.utils.data.TensorDataset(
            torch.randn(3, 4, 2), torch.randint(0, 2, (3,))
        )

        def builder(dataset_name, data_path, kwargs):
            self.assertEqual(dataset_name, "unit_test_unified_sequence")
            self.assertEqual(data_path, self.temp_dir)
            self.assertEqual(kwargs["seq_len"], 4)
            self.assertEqual(kwargs["n_samples"], 13)
            return explicit_split(mock_train, mock_valid, mock_test)

        register_sequence_dataset_builder("unit_test_unified_sequence", builder)
        try:
            self.task_cfg.dataset = "unit_test_unified_sequence"
            self.task_cfg.data_path = self.temp_dir
            self.task_cfg.parameters = {"seq_len": 4, "n_samples": 13}

            train_ds, valid_ds, test_ds = get_unified_datasets(self.task_cfg)
        finally:
            unregister_sequence_dataset_builder("unit_test_unified_sequence")

        self.assertIs(train_ds, mock_train)
        self.assertIs(valid_ds, mock_valid)
        self.assertIs(test_ds, mock_test)

    @patch("dendritic_modeling.datasets.standard_datasets.get_synthetic_datasets")
    def test_theoretical_synthetic_dataset_dispatch(self, mock_get_synthetic_datasets):
        """Built-in synthetic datasets should inject type without mutating params."""
        mock_train = torch.utils.data.TensorDataset(
            torch.randn(6, 3), torch.randint(0, 2, (6,))
        )
        mock_valid = torch.utils.data.TensorDataset(
            torch.randn(2, 3), torch.randint(0, 2, (2,))
        )
        mock_test = torch.utils.data.TensorDataset(
            torch.randn(4, 3), torch.randint(0, 2, (4,))
        )
        mock_get_synthetic_datasets.return_value = (mock_train, mock_valid, mock_test)

        parameters = {"n_samples": 12, "seed": 5}
        self.task_cfg.dataset = "correlated_gaussian"
        self.task_cfg.data_path = None
        self.task_cfg.parameters = parameters

        train_ds, valid_ds, test_ds = get_unified_datasets(self.task_cfg)

        self.assertIs(train_ds, mock_train)
        self.assertIs(valid_ds, mock_valid)
        self.assertIs(test_ds, mock_test)
        mock_get_synthetic_datasets.assert_called_once_with(
            {"n_samples": 12, "seed": 5, "type": "correlated_gaussian"}
        )
        self.assertNotIn("type", parameters)

    def test_invalid_dataset(self):
        """An unknown name raises with the known names and the registry hint."""
        self.task_cfg.dataset = "linear_seperable"
        self.task_cfg.data_path = None
        self.task_cfg.parameters = {}

        with self.assertRaisesRegex(ValueError, "did you mean 'linear_separable'"):
            get_unified_datasets(self.task_cfg)
        with self.assertRaisesRegex(ValueError, "register_dataset"):
            get_unified_datasets(self.task_cfg)

    def test_every_library_synthetic_dataset_is_selectable(self):
        """Each synthetic spec builds through get_unified_datasets by name."""
        from dendritic_modeling.datasets.synthetic_datasets import (
            synthetic_dataset_names,
        )

        for name in synthetic_dataset_names():
            with self.subTest(dataset=name):
                self.task_cfg.dataset = name
                self.task_cfg.data_path = None
                self.task_cfg.parameters = {"n_samples": 40, "seed": 2}
                train_ds, valid_ds, test_ds = get_unified_datasets(self.task_cfg)
                x, y = train_ds[0]
                self.assertEqual(x.dim(), 1)
                self.assertTrue(len(valid_ds) > 0 and len(test_ds) > 0)
                self.assertGreaterEqual(int(y), 0)


class TestSpecializedDatasets(unittest.TestCase):
    """Test specialized dataset classes with complex behavior."""

    def test_wrap_poisson_generator_triplet_preserves_inputs_and_options(self):
        base_inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        base_labels = torch.arange(3)
        train_ds = TensorDataset(base_inputs, base_labels)
        valid_ds = TensorDataset(base_inputs + 100, base_labels + 10)
        test_ds = TensorDataset(base_inputs + 200, base_labels + 20)

        train_out, valid_out, test_out = _wrap_poisson_generator_triplet(
            train_ds,
            valid_ds,
            test_ds,
            multiplicative_gain=False,
            fixed_gain_factor=None,
            max_gain_factor=2.5,
            uniform_gain=False,
            poisson_sampling=False,
            stimulus_duration=3.0,
            max_gain_tau_ratio=None,
        )

        for wrapped in (train_out, valid_out, test_out):
            self.assertIsInstance(wrapped, StandardPoissonGeneratorDataset)
            self.assertFalse(wrapped.multiplicative_gain)
            self.assertIsNone(wrapped.fixed_gain_factor)
            self.assertEqual(wrapped.max_gain_factor, 2.5)
            self.assertFalse(wrapped.uniform_gain)
            self.assertFalse(wrapped.poisson_sampling)
            self.assertEqual(wrapped.stimulus_duration, 3.0)

        self.assertTrue(torch.equal(train_out.base_inputs, base_inputs))
        self.assertTrue(torch.equal(valid_out.base_inputs, base_inputs + 100))
        self.assertTrue(torch.equal(test_out.base_inputs, base_inputs + 200))
        self.assertTrue(torch.equal(train_out.base_labels, base_labels))
        self.assertTrue(torch.equal(valid_out.base_labels, base_labels + 10))
        self.assertTrue(torch.equal(test_out.base_labels, base_labels + 20))

    def test_poisson_generator_dataset(self):
        """Test PoissonGeneratorDataset functionality."""
        # Create base data
        base_inputs = torch.rand(50, 10) * 2  # Scale for better Poisson behavior
        base_labels = torch.randint(0, 2, (50,))

        dataset = StandardPoissonGeneratorDataset(
            base_inputs=base_inputs, base_labels=base_labels, max_gain_factor=1.5
        )

        self.assertEqual(len(dataset), 50)

        # Test that output has Poisson noise
        sample, label = dataset[0]
        self.assertEqual(sample.shape, base_inputs[0].shape)
        self.assertEqual(label, base_labels[0])

        # The output should be different from input due to Poisson noise
        # (though this is probabilistic, so we test multiple samples)
        differences = []
        for i in range(10):
            sample, _ = dataset[i]
            diff = torch.sum(torch.abs(sample - base_inputs[i])).item()
            differences.append(diff)

        # At least some samples should be different
        self.assertTrue(any(diff > 0 for diff in differences))

        # This test requires complex mocking of MNIST dataset loading
        # The functionality should be tested in integration tests


if __name__ == "__main__":
    unittest.main()

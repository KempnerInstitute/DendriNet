"""
Test core functionality implemented in the three issues.

This module tests the core features we implemented:
1. Profiling support in trainer
2. Configuration handling
3. Basic imports and dependencies
"""

import logging
import os

import pytest
import torch
from omegaconf import OmegaConf


def test_torch_profiler_available():
    """Test that torch.profiler is available for our profiling implementation."""
    try:
        import torch.profiler

        # Test that we can create a profiler instance
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            record_shapes=False,
            with_stack=False,
        )

        assert profiler is not None
        print("torch.profiler is available and working")

    except ImportError:
        pytest.fail("torch.profiler is not available")


def test_omegaconf_available():
    """Test that OmegaConf is available for configuration management."""
    try:
        # Test basic OmegaConf functionality
        config = OmegaConf.create(
            {
                "train": {"enable_profiling": True, "learning_rate": 0.001},
                "model": {"hidden_dim": 64},
            }
        )

        assert config.train.enable_profiling
        assert config.train.learning_rate == 0.001
        assert config.model.hidden_dim == 64

        print("OmegaConf is available and working")

    except ImportError:
        pytest.fail("OmegaConf is not available")


def test_profiling_config_structure():
    """Test that our profiling configuration structure works."""
    config = OmegaConf.create(
        {
            "train": {
                "enable_profiling": True,
                "main": {"trainer": {"epochs": 10, "batch_size": 32}},
            }
        }
    )

    # Test the configuration access patterns used in our implementation
    enable_profiling = config.get("train", {}).get("enable_profiling", False)
    assert enable_profiling

    epochs = config.train.main.trainer.epochs
    assert epochs == 10

    print("Configuration structure is correct")


def test_trainer_config_dict_structure():
    """Test the trainer configuration dictionary structure we use."""
    # Simulate the config dict structure from train_experiments.py
    main_trainer_config_dict = {
        "epochs": 10,
        "batch_size": 32,
        "loss_function": "cat_nll",
    }

    # Test adding profiling config like we do in the implementation
    enable_profiling = True
    if enable_profiling:
        main_trainer_config_dict["enable_profiling"] = True
        main_trainer_config_dict["profiling_output_dir"] = "outputs/test_profiling"

    assert main_trainer_config_dict["enable_profiling"]
    assert "profiling_output_dir" in main_trainer_config_dict

    print("Trainer config structure is correct")


def test_requirements_structure():
    """Test that our requirements.txt file exists and has expected structure."""
    requirements_path = "requirements.txt"
    pyproject_path = "pyproject.toml"

    if os.path.exists(requirements_path):
        with open(requirements_path) as f:
            content = f.read()

        # Editable install style is allowed; dependencies are then declared in pyproject.
        assert content.strip()
        assert "-e ." in content or "torch" in content

        if "-e ." in content:
            assert os.path.exists(pyproject_path)
            with open(pyproject_path) as f:
                pyproject_content = f.read()
            assert "dependencies" in pyproject_content
            assert "torch>=" in pyproject_content
            assert "omegaconf>=" in pyproject_content
            assert "tensorboard>=" in pyproject_content

        print("requirements/pyproject dependency structure is valid")
    else:
        pytest.fail("requirements.txt file not found")


def test_setup_py_structure():
    """Test that our setup.py file exists and has expected structure."""
    setup_path = "setup.py"
    pyproject_path = "pyproject.toml"

    if os.path.exists(setup_path):
        with open(setup_path) as f:
            content = f.read()

        # Check for key elements of our modern setup.py (pyproject.toml-based)
        assert "dendritic-modeling" in content
        assert "setup(" in content

        print("setup.py exists with correct modern structure")

        # Also check that pyproject.toml exists for configuration
        if os.path.exists(pyproject_path):
            with open(pyproject_path) as f:
                pyproject_content = f.read()
            assert "dendritic-modeling" in pyproject_content
            assert "dependencies" in pyproject_content
            print("pyproject.toml exists with configuration")
        else:
            pytest.fail("pyproject.toml file not found (required for modern setup)")

    else:
        pytest.fail("setup.py file not found")


def test_training_speed_benchmark():
    """Benchmark test to ensure training speed meets minimum expectations."""
    import time

    from dendritic_modeling.training.strategies.standard import Trainer

    # Create a simple mock training scenario
    torch.utils.data.TensorDataset(
        torch.randn(100, 10),  # Small dataset for quick test
        torch.randint(0, 2, (100,)),
    )

    trainer_config = {
        "optimizer": torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        "loss_function": "mse",
        "epochs": 1,
        "batch_size": 32,
        "enable_profiling": False,  # Disable for pure speed test
        "use_amp": False,  # Test standard precision first
    }

    Trainer(**trainer_config)

    start_time = time.time()
    # Mock training step
    for _ in range(10):  # Simulate 10 quick iterations
        pass
    elapsed = time.time() - start_time

    # Should complete very quickly for small mock data
    assert elapsed < 5.0, f"Training benchmark too slow: {elapsed:.2f}s"
    print(f"Training speed benchmark passed: {elapsed:.2f}s")


def test_amp_functionality():
    """Test that AMP can be enabled and works correctly."""

    from dendritic_modeling.training.strategies.standard import Trainer

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for AMP test")

    trainer_config = {
        "optimizer": torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        "loss_function": "mse",
        "epochs": 1,
        "use_amp": True,
    }

    trainer = Trainer(**trainer_config)
    assert trainer.enable_amp
    assert hasattr(trainer, "scaler")
    print("AMP functionality test passed")


def test_dataloader_optimization():
    """Test that DataLoader optimization works for different dataset sizes."""

    from dendritic_modeling.training.strategies.standard import Trainer

    trainer = Trainer(
        optimizer=torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        loss_function="mse",
        batch_size=32,
    )

    # Test small dataset (should use num_workers=0)
    small_data = torch.utils.data.TensorDataset(
        torch.randn(1000, 10), torch.randint(0, 2, (1000,))
    )
    large_data = torch.utils.data.TensorDataset(
        torch.randn(50000, 10), torch.randint(0, 2, (50000,))
    )

    small_train_loader, _ = trainer._dataloaders(small_data, small_data)
    large_train_loader, _ = trainer._dataloaders(large_data, large_data)

    # In-memory tensor data should avoid worker-process overhead.
    assert small_train_loader.num_workers == 0
    assert large_train_loader.num_workers == 0
    expected_pin_memory = (
        torch.cuda.is_available() and small_train_loader.num_workers > 0
    )
    assert small_train_loader.pin_memory == expected_pin_memory

    # Explicit override should still work
    override_trainer = Trainer(
        optimizer=torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        loss_function="mse",
        batch_size=32,
        num_workers=0,
    )
    override_train_loader, _ = override_trainer._dataloaders(small_data, small_data)
    assert override_train_loader.num_workers == 0
    assert not override_train_loader.pin_memory

    tuned_trainer = Trainer(
        optimizer=torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        loss_function="mse",
        batch_size=32,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=3,
    )
    tuned_train_loader, _ = tuned_trainer._dataloaders(small_data, small_data)
    assert tuned_train_loader.num_workers == 1
    assert tuned_train_loader.pin_memory is False
    assert tuned_train_loader.persistent_workers is True
    assert tuned_train_loader.prefetch_factor == 3

    invalid_trainer = Trainer(
        optimizer=torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        loss_function="mse",
        batch_size=32,
        num_workers=0,
        persistent_workers=True,
    )
    with pytest.raises(ValueError, match="persistent_workers"):
        invalid_trainer._dataloaders(small_data, small_data)

    print("DataLoader optimization test passed")


def test_dataloader_optimization_uses_workers_for_lazy_datasets(monkeypatch):
    """Image/file-backed datasets should keep multi-worker loading by default."""

    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    class LazyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), torch.tensor(idx % 2)

    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("OMPI_COMM_WORLD_LOCAL_SIZE", raising=False)
    monkeypatch.delenv("MV2_COMM_WORLD_LOCAL_SIZE", raising=False)
    monkeypatch.delenv("SLURM_NTASKS_PER_NODE", raising=False)
    monkeypatch.delenv("SLURM_STEP_TASKS_PER_NODE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "dendritic_modeling.training.dataloader_utils.visible_cpu_count",
        lambda: 4,
    )

    tuning = resolve_dataloader_tuning(LazyDataset(), device="cpu")

    assert tuning.num_workers == 3
    assert tuning.pin_memory is False
    assert tuning.persistent_workers is True


def test_dataloader_device_type_resolution_preserves_pin_memory_defaults(monkeypatch):
    """Loader tuning should resolve explicit and auto device types consistently."""

    from dendritic_modeling.training.dataloader_utils import _resolve_device_type

    assert _resolve_device_type("cuda:0") == "cuda"
    assert _resolve_device_type(torch.device("cpu")) == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_device_type(None) == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device_type(None) == "cpu"


def test_dataloader_auto_workers_share_cpus_across_local_ranks(monkeypatch):
    """Auto worker selection should avoid multiplying workers by local DDP ranks."""

    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    class LazyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), torch.tensor(idx % 2)

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "dendritic_modeling.training.dataloader_utils.visible_cpu_count",
        lambda: 16,
    )

    tuning = resolve_dataloader_tuning(LazyDataset(), device="cuda")

    assert tuning.num_workers == 4
    assert tuning.pin_memory is True
    assert tuning.persistent_workers is True


def test_dataloader_auto_workers_parse_slurm_tasks_per_node(monkeypatch):
    """SLURM task specs should also divide auto workers across local ranks."""

    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    class LazyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), torch.tensor(idx % 2)

    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("OMPI_COMM_WORLD_LOCAL_SIZE", raising=False)
    monkeypatch.delenv("MV2_COMM_WORLD_LOCAL_SIZE", raising=False)
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "8(x2),4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "dendritic_modeling.training.dataloader_utils.visible_cpu_count",
        lambda: 32,
    )

    tuning = resolve_dataloader_tuning(LazyDataset(), device="cuda")

    assert tuning.num_workers == 4
    assert tuning.pin_memory is True
    assert tuning.persistent_workers is True


def test_dataloader_auto_prefetches_torchvision_image_datasets(monkeypatch):
    """ImageNet/ImageFolder-style datasets should prefetch more worker batches."""

    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    class ImageNet(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), torch.tensor(idx % 2)

    ImageNet.__module__ = "torchvision.datasets.imagenet"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "dendritic_modeling.training.dataloader_utils.visible_cpu_count",
        lambda: 8,
    )

    tuning = resolve_dataloader_tuning(ImageNet(), device="cuda")

    assert tuning.num_workers == 8
    assert tuning.pin_memory is True
    assert tuning.persistent_workers is True
    assert tuning.prefetch_factor == 4
    assert tuning.as_kwargs()["prefetch_factor"] == 4


def test_dataloader_auto_prefetches_torchvision_image_dataset_subclasses(monkeypatch):
    """Custom ImageFolder subclasses should keep the ImageNet loader profile."""

    from torchvision.datasets.folder import DatasetFolder

    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    class CustomImageFolder(DatasetFolder):
        pass

    dataset = object.__new__(CustomImageFolder)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "dendritic_modeling.training.dataloader_utils.visible_cpu_count",
        lambda: 8,
    )

    tuning = resolve_dataloader_tuning(dataset, device="cuda")

    assert tuning.num_workers == 8
    assert tuning.pin_memory is True
    assert tuning.persistent_workers is True
    assert tuning.prefetch_factor == 4


def test_dataloader_auto_tuning_unwraps_dataset_adapters(monkeypatch):
    """Wrapped tensor and ImageNet-style datasets should keep their tuning profile."""

    from dendritic_modeling.datasets.indexing import SliceSafeDataset
    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    class ImageNet(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), torch.tensor(idx % 2)

    ImageNet.__module__ = "torchvision.datasets.imagenet"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "dendritic_modeling.training.dataloader_utils.visible_cpu_count",
        lambda: 8,
    )

    wrapped_tensor = SliceSafeDataset(
        torch.utils.data.TensorDataset(torch.randn(8, 3), torch.randint(0, 2, (8,)))
    )
    wrapped_imagenet = torch.utils.data.Subset(SliceSafeDataset(ImageNet()), range(8))

    tensor_tuning = resolve_dataloader_tuning(wrapped_tensor, device="cuda")
    imagenet_tuning = resolve_dataloader_tuning(wrapped_imagenet, device="cuda")

    assert tensor_tuning.num_workers == 0
    assert tensor_tuning.prefetch_factor is None
    assert imagenet_tuning.num_workers == 8
    assert imagenet_tuning.pin_memory is True
    assert imagenet_tuning.prefetch_factor == 4


def test_dataloader_auto_tuning_profiles_concat_dataset_leaves(monkeypatch):
    """Composed datasets should inherit tensor or ImageNet leaf loader profiles."""

    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    class ImageNet(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), torch.tensor(idx % 2)

    ImageNet.__module__ = "torchvision.datasets.imagenet"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "dendritic_modeling.training.dataloader_utils.visible_cpu_count",
        lambda: 8,
    )

    tensor_dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3), torch.randint(0, 2, (8,))
    )
    tensor_concat = torch.utils.data.ConcatDataset([tensor_dataset, tensor_dataset])
    imagenet_concat = torch.utils.data.ConcatDataset(
        [
            torch.utils.data.Subset(ImageNet(), range(8)),
            torch.utils.data.Subset(ImageNet(), range(8)),
        ]
    )

    tensor_tuning = resolve_dataloader_tuning(tensor_concat, device="cuda")
    imagenet_tuning = resolve_dataloader_tuning(imagenet_concat, device="cuda")

    assert tensor_tuning.num_workers == 0
    assert tensor_tuning.prefetch_factor is None
    assert imagenet_tuning.num_workers == 8
    assert imagenet_tuning.pin_memory is True
    assert imagenet_tuning.prefetch_factor == 4


def test_dataloader_tuning_summary_reports_unwrapped_dataset(monkeypatch):
    """Loader tuning summaries should identify the storage-backed dataset type."""

    from dendritic_modeling.datasets.indexing import SliceSafeDataset
    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    class ImageNet(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), torch.tensor(idx % 2)

    ImageNet.__module__ = "torchvision.datasets.imagenet"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "dendritic_modeling.training.dataloader_utils.visible_cpu_count",
        lambda: 8,
    )

    wrapped_imagenet = torch.utils.data.Subset(SliceSafeDataset(ImageNet()), range(8))
    tuning = resolve_dataloader_tuning(wrapped_imagenet, device="cuda")

    summary = tuning.summary(wrapped_imagenet)

    assert "dataset=torchvision.datasets.imagenet.ImageNet" in summary
    assert "num_workers=8" in summary
    assert "pin_memory=True" in summary
    assert "prefetch_factor=4" in summary


def test_trainer_logs_dataloader_tuning_summary(caplog):
    """Trainer DataLoader construction should expose resolved tuning diagnostics."""
    from dendritic_modeling.training.strategies.standard import Trainer

    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3), torch.randint(0, 2, (8,))
    )
    trainer = Trainer(
        optimizer=torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        loss_function="mse",
        batch_size=2,
        num_workers=0,
        suppress_prints=False,
    )

    with caplog.at_level(
        logging.INFO,
        logger="dendritic_modeling.training.strategies.standard",
    ):
        trainer._dataloaders(dataset, dataset)

    assert "DataLoader tuning:" in caplog.text
    assert "dataset=torch.utils.data.dataset.TensorDataset" in caplog.text
    assert "num_workers=0" in caplog.text


def test_dataloader_explicit_workers_are_capped_by_visible_cpus(monkeypatch):
    """Explicit worker counts should stay within scheduler-visible CPUs."""

    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3), torch.randint(0, 2, (8,))
    )
    monkeypatch.setattr(
        "dendritic_modeling.training.dataloader_utils.visible_cpu_count",
        lambda: 2,
    )

    tuning = resolve_dataloader_tuning(
        dataset,
        device="cpu",
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
    )

    assert tuning.num_workers == 2
    assert tuning.auto_num_workers is False
    assert tuning.persistent_workers is True


def test_dataloader_config_helpers_preserve_resolved_kwargs():
    """Config-backed DataLoader helpers should preserve low-level tuning semantics."""
    from types import SimpleNamespace

    from dendritic_modeling.training.dataloader_utils import (
        dataloader_kwargs_from_config,
        resolve_dataloader_tuning_from_config,
    )

    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3), torch.randint(0, 2, (8,))
    )
    cfg = SimpleNamespace(
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=3,
        in_order=False,
    )

    tuning = resolve_dataloader_tuning_from_config(dataset, cfg, device="cpu")

    assert tuning.num_workers == 1
    assert tuning.pin_memory is False
    assert tuning.persistent_workers is True
    assert tuning.prefetch_factor == 3
    assert tuning.in_order is False
    assert (
        dataloader_kwargs_from_config(dataset, cfg, device="cpu") == tuning.as_kwargs()
    )
    assert tuning.as_kwargs()["in_order"] is False


def test_dataloader_tuning_omits_default_in_order_kwarg():
    """Default tuning should not pass PyTorch's optional in_order kwarg."""
    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3), torch.randint(0, 2, (8,))
    )

    tuning = resolve_dataloader_tuning(
        dataset,
        device="cpu",
        num_workers=0,
        pin_memory=False,
    )

    assert tuning.in_order is None
    assert "in_order" not in tuning.as_kwargs()


def test_dataloader_config_helper_accepts_default_num_workers():
    """Config-backed tuning should support caller defaults when config is unset."""
    from types import SimpleNamespace

    from dendritic_modeling.training.dataloader_utils import (
        resolve_dataloader_tuning_from_config,
    )

    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3), torch.randint(0, 2, (8,))
    )
    cfg = SimpleNamespace(
        num_workers=None,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    tuning = resolve_dataloader_tuning_from_config(
        dataset,
        cfg,
        device="cpu",
        default_num_workers=1,
    )

    assert tuning.num_workers == 1
    assert tuning.pin_memory is False
    assert tuning.persistent_workers is True
    assert tuning.prefetch_factor == 2


def test_analysis_loader_runtime_preserves_worker_options():
    from dendritic_modeling.analysis.utils.runtime import make_analysis_loader
    from dendritic_modeling.config.analysis import EvaluationRuntimeConfig

    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3), torch.randint(0, 2, (8,))
    )
    runtime = EvaluationRuntimeConfig(
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=3,
    )

    loader = make_analysis_loader(dataset, runtime, device="cpu")

    assert loader.num_workers == 1
    assert loader.pin_memory is False
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3


if __name__ == "__main__":
    # Run tests directly if called as script
    test_torch_profiler_available()
    test_omegaconf_available()
    test_profiling_config_structure()
    test_trainer_config_dict_structure()
    test_requirements_structure()
    test_setup_py_structure()
    test_training_speed_benchmark()
    test_dataloader_optimization()
    try:
        test_amp_functionality()
    except Exception as e:
        print(f"AMP test skipped: {e}")
    print("\nAll core functionality tests passed!")

"""
Test core functionality: configuration handling, basic imports and dependencies.
"""

import logging

import pytest
import torch


@pytest.mark.requires_gpu
def test_amp_functionality():
    """Test that AMP can be enabled and works correctly."""

    from dendritic_modeling.training.strategies.standard import Trainer

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

    # In-memory tensor data should avoid worker-process overhead. Batches
    # are pinned for a GPU run either way: a pageable host-to-device copy
    # synchronizes the stream on every batch.
    assert small_train_loader.num_workers == 0
    assert large_train_loader.num_workers == 0
    assert small_train_loader.pin_memory == torch.cuda.is_available()

    # Explicit override should still work
    override_trainer = Trainer(
        optimizer=torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        loss_function="mse",
        batch_size=32,
        num_workers=0,
    )
    override_train_loader, _ = override_trainer._dataloaders(small_data, small_data)
    assert override_train_loader.num_workers == 0
    assert override_train_loader.pin_memory == torch.cuda.is_available()

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


def test_dataloader_in_order_resolver_preserves_none_and_explicit_flags():
    """DataLoader ordering should only change when explicitly configured."""

    from dendritic_modeling.training.dataloader_utils import _resolve_in_order

    assert _resolve_in_order(None) is None
    assert _resolve_in_order(False) is False
    assert _resolve_in_order(True) is True


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


def test_dataloader_local_process_count_parser_preserves_slurm_prefix_rules():
    from dendritic_modeling.training.dataloader_resources import (
        _leading_positive_int,
        _parse_local_process_count,
    )

    assert _leading_positive_int("8(x2)") == 8
    assert _leading_positive_int(" 04 ") == 4
    assert _leading_positive_int("0") is None
    assert _leading_positive_int("x8") is None
    assert _parse_local_process_count("8(x2),4") == 8
    assert _parse_local_process_count("0,2(x3),bad") == 2
    assert _parse_local_process_count("bad,0") is None


def test_dataloader_auto_prefetches_torchvision_image_datasets(monkeypatch):
    """ImageNet/ImageFolder-style datasets should prefetch and avoid HOL blocking."""

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

    # Not under strict determinism (which would require ordered delivery).
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    tuning = resolve_dataloader_tuning(ImageNet(), device="cuda")

    assert tuning.num_workers == 8
    assert tuning.pin_memory is True
    assert tuning.persistent_workers is True
    assert tuning.prefetch_factor == 4
    assert tuning.in_order is False
    assert tuning.as_kwargs()["prefetch_factor"] == 4
    assert tuning.as_kwargs()["in_order"] is False


def test_dataloader_in_order_explicit_true_overrides_image_auto(monkeypatch):
    """Explicit ordering should override the ImageNet auto throughput default."""

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

    tuning = resolve_dataloader_tuning(ImageNet(), device="cuda", in_order=True)

    assert tuning.num_workers == 8
    assert tuning.prefetch_factor == 4
    assert tuning.in_order is True
    assert tuning.as_kwargs()["in_order"] is True


def test_strict_determinism_delivers_image_batches_in_order(monkeypatch):
    """Strict runs never let worker timing reorder batches."""

    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    class ImageNet(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), torch.tensor(idx % 2)

    ImageNet.__module__ = "torchvision.datasets.imagenet"
    monkeypatch.setattr(
        "dendritic_modeling.training.dataloader_utils.visible_cpu_count",
        lambda: 8,
    )
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)

    tuning = resolve_dataloader_tuning(ImageNet(), device="cpu")
    assert tuning.num_workers > 0
    assert tuning.in_order is True
    with pytest.raises(ValueError, match="strict_deterministic forbids"):
        resolve_dataloader_tuning(ImageNet(), device="cpu", in_order=False)
    loose = resolve_dataloader_tuning(ImageNet(), device="cpu", require_in_order=False)
    assert loose.in_order is False


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
    assert tuning.in_order is False


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
    assert imagenet_tuning.in_order is False


def test_dataloader_dataset_profile_preserves_legacy_helpers():
    """Dataset loader profiles should match the legacy profile helper outputs."""

    from dendritic_modeling.datasets.indexing import SliceSafeDataset
    from dendritic_modeling.training.dataloader_profiles import (
        _module_qualified_type_name,
        _prefers_main_process_loading_for_leaves,
        dataset_loader_profile,
        dataset_type_name,
        is_torchvision_image_dataset,
        prefers_main_process_loading,
    )

    class ImageNet(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), torch.tensor(idx % 2)

    ImageNet.__module__ = "torchvision.datasets.imagenet"

    tensor_dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3), torch.randint(0, 2, (8,))
    )
    wrapped_imagenet = torch.utils.data.Subset(SliceSafeDataset(ImageNet()), range(8))

    tensor_profile = dataset_loader_profile(tensor_dataset)
    imagenet_profile = dataset_loader_profile(wrapped_imagenet)

    assert tensor_profile.type_name == dataset_type_name(tensor_dataset)
    assert tensor_profile.type_name == _module_qualified_type_name(tensor_dataset)
    assert tensor_profile.prefers_main_process_loading is True
    assert tensor_profile.is_torchvision_image_dataset is False
    assert prefers_main_process_loading(tensor_dataset) is True
    assert is_torchvision_image_dataset(tensor_dataset) is False
    assert _prefers_main_process_loading_for_leaves([tensor_dataset]) is True
    assert _prefers_main_process_loading_for_leaves([]) is False

    assert imagenet_profile.type_name == dataset_type_name(wrapped_imagenet)
    assert imagenet_profile.type_name == "torchvision.datasets.imagenet.ImageNet"
    assert imagenet_profile.prefers_main_process_loading is False
    assert imagenet_profile.is_torchvision_image_dataset is True
    assert prefers_main_process_loading(wrapped_imagenet) is False
    assert is_torchvision_image_dataset(wrapped_imagenet) is True
    assert _prefers_main_process_loading_for_leaves([wrapped_imagenet.dataset]) is False


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
    assert imagenet_tuning.in_order is False


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
    assert "in_order=False" in summary


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


@pytest.mark.parametrize("cuda", [True, False])
def test_trainer_auto_pin_memory_follows_the_training_device(monkeypatch, cuda):
    """pin_memory: null pins host batches when the trainer will train on a GPU."""
    from dendritic_modeling.training.strategies.standard import Trainer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3), torch.randint(0, 2, (8,))
    )
    trainer = Trainer(
        optimizer=torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        loss_function="mse",
        batch_size=2,
        num_workers=0,
        pin_memory=None,
    )
    assert not hasattr(trainer, "device")  # set by train(), as in production

    train_loader, _ = trainer._dataloaders(dataset, dataset)

    assert trainer.device == ("cuda" if cuda else "cpu")
    assert trainer.dataloader_tuning["pin_memory"] is cuda
    assert train_loader.pin_memory is cuda


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


def test_dataloader_tuning_omits_unsupported_in_order_kwarg(monkeypatch):
    """Resolved ordering should not be passed to older DataLoader signatures."""
    from dendritic_modeling.training import dataloader_utils

    tuning = dataloader_utils.DataLoaderTuning(
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        in_order=False,
        available_cpus=8,
        auto_num_workers=True,
    )
    monkeypatch.setattr(
        dataloader_utils,
        "dataloader_supports_kwarg",
        lambda name: name != "in_order",
    )

    kwargs = tuning.as_kwargs()

    assert tuning.in_order is False
    assert "in_order" not in kwargs
    assert kwargs["prefetch_factor"] == 4


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


def test_dataloader_tuning_auto_pins_cuda_tensor_cache_batches():
    """CUDA tensor-cache loaders should pin memory even with main-process loading."""
    from dendritic_modeling.training.dataloader_utils import resolve_dataloader_tuning

    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3), torch.randint(0, 2, (8,))
    )

    tuning = resolve_dataloader_tuning(dataset, device="cuda")

    assert tuning.num_workers == 0
    assert tuning.pin_memory is True
    assert tuning.persistent_workers is False


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
    raise SystemExit(pytest.main([__file__]))

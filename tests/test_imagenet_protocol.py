"""Subset loaders preserve split membership, views, and configuration semantics."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from dendritic_modeling.config.conversion import dataset_specific_params
from dendritic_modeling.datasets.imagenet_protocol import (
    build_protocol_test_dataset,
    build_protocol_train_partition,
)


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.delenv("DENDRITIC_IMAGEFOLDER_CACHE", raising=False)
    for split, count in (("train", 6), ("val", 3)):
        for label in range(2):
            directory = tmp_path / split / f"class{label}"
            directory.mkdir(parents=True)
            for index in range(count):
                pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
                pixels = np.roll(pixels, index + label, axis=1)
                Image.fromarray(pixels).save(directory / f"{index}.png")
    return OmegaConf.create(
        {
            "data": {
                "dataset_name": "imagenet",
                "base_dir": str(tmp_path),
                "processing": {"flatten": False, "normalize": True},
                "dataset_params": {
                    "imagenet": {
                        "loader_backend": "imagefolder",
                        "transform_preset": "pretrained",
                    }
                },
            },
            "experiment": {"seed": 7, "split_seed": 13, "train_valid_split": 0.75},
        }
    )


@pytest.mark.parametrize("preset", ["pretrained", "legacy_resize"])
@pytest.mark.parametrize("normalize", [True, False])
def test_membership_views_and_test_split(config, preset, normalize):
    config.data.dataset_params.imagenet.transform_preset = preset
    config.data.processing.normalize = normalize
    before = torch.random.get_rng_state().clone()
    train, valid, info = build_protocol_train_partition(config)
    test, test_info = build_protocol_test_dataset(config)
    assert torch.equal(before, torch.random.get_rng_state())
    expected = torch.randperm(12, generator=torch.Generator().manual_seed(13)).tolist()
    assert train.indices == expected[:9]
    assert valid.indices == expected[9:]
    assert train.dataset is not valid.dataset
    assert train.dataset.samples == valid.dataset.samples
    assert not set(train.indices) & set(valid.indices)
    assert info["n_train"] == 9 and info["n_valid"] == 3
    assert test_info["n_samples"] == 6
    assert test_info["loader_backend"] == "imagefolder"
    assert all("/val/" in path for path, _ in test.samples)
    assert torch.equal(valid[0][0], valid[0][0])
    assert valid[0][0].shape == (3, 224, 224)
    if preset == "pretrained":
        assert "RandomResizedCrop" in repr(train.dataset.transform)
        assert "CenterCrop" in repr(valid.dataset.transform)
    config.experiment.split_seed = 99
    other_train, _, _ = build_protocol_train_partition(config)
    other_test, other_info = build_protocol_test_dataset(config)
    assert other_train.indices != train.indices
    assert other_test.samples == test.samples and other_info == test_info
    assert torch.equal(other_test[0][0], test[0][0])


@pytest.mark.parametrize("seed", [7, None])
def test_split_seed_fallback(config, seed):
    config.experiment.split_seed = None
    config.experiment.seed = seed
    train, _, info = build_protocol_train_partition(config)
    expected_seed = seed or 0
    assert info["split_seed"] == expected_seed
    assert (
        train.indices
        == torch.randperm(
            12, generator=torch.Generator().manual_seed(expected_seed)
        ).tolist()[:9]
    )


@pytest.mark.parametrize(
    "loader", [build_protocol_train_partition, build_protocol_test_dataset]
)
@pytest.mark.parametrize("invalid", ["dataset", "flatten", "subset"])
def test_rejects_protocol_changes(config, loader, invalid):
    if invalid == "dataset":
        config.data.dataset_name = "mnist"
    elif invalid == "flatten":
        config.data.processing.flatten = True
    else:
        key = (
            "train_samples_per_class"
            if loader is build_protocol_train_partition
            else "val_samples_per_class"
        )
        config.data.dataset_params.imagenet[key] = 1
    with pytest.raises(ValueError):
        loader(config)


def test_dataset_parameter_containers_and_interpolation():
    @dataclass
    class Params:
        loader_backend: str = "imagefolder"

    expected = {"loader_backend": "imagefolder"}
    for container in ({"imagenet": expected}, SimpleNamespace(imagenet=Params())):
        cfg = SimpleNamespace(dataset_name="imagenet", dataset_params=container)
        assert dataset_specific_params(cfg) == expected
    cfg = OmegaConf.create(
        {
            "backend": "imagefolder",
            "dataset_name": "imagenet",
            "dataset_params": {"imagenet": {"loader_backend": "${backend}"}},
        }
    )
    assert dataset_specific_params(cfg) == expected
    assert dataset_specific_params(SimpleNamespace(dataset_name="imagenet")) == {}


@pytest.mark.parametrize(
    "loader", [build_protocol_train_partition, build_protocol_test_dataset]
)
def test_an_empty_base_dir_uses_the_data_root_training_uses(
    config, tmp_path, monkeypatch, loader
):
    # The fixture's train/ and val/ trees are in tmp_path, which training
    # reads when data.base_dir is empty and DENDRITIC_DATA_DIR names it.
    expected = loader(config)
    monkeypatch.delenv("IMAGENET_LOCAL_DIR", raising=False)
    monkeypatch.setenv("DENDRITIC_DATA_DIR", str(tmp_path))
    config.data.base_dir = ""
    resolved = loader(config)
    assert resolved[-1]["imagenet_dir"] == str(tmp_path)
    assert resolved[-1]["loader_backend"] == "imagefolder"
    assert resolved[-1] == expected[-1]

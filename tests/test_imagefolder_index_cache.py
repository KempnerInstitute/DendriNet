"""Tests for the opt-in ImageFolder index cache (NFS scan bypass)."""

import os
import pickle

import pytest
import torch

from dendritic_modeling.datasets.imagenet_loading import (
    _imagefolder_with_index_cache,
    _index_cache_path,
)

PIL = pytest.importorskip("PIL.Image")


@pytest.fixture()
def tiny_tree(tmp_path):
    root = tmp_path / "train"
    for cls in ("class_a", "class_b"):
        (root / cls).mkdir(parents=True)
        for i in range(3):
            PIL.new("RGB", (4, 4), color=(i, 0, 0)).save(root / cls / f"img_{i}.png")
    return str(root)


def test_disabled_without_env(tiny_tree, monkeypatch):
    monkeypatch.delenv("DENDRITIC_IMAGEFOLDER_CACHE", raising=False)
    assert _index_cache_path(tiny_tree) is None
    dataset = _imagefolder_with_index_cache(tiny_tree, transform=None)
    assert len(dataset) == 6


def test_cache_roundtrip_and_bypass(tiny_tree, tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("DENDRITIC_IMAGEFOLDER_CACHE", str(cache_dir))

    first = _imagefolder_with_index_cache(tiny_tree, transform=None)
    cache_path = _index_cache_path(tiny_tree)
    assert os.path.exists(cache_path)

    second = _imagefolder_with_index_cache(tiny_tree, transform=None)
    assert second.classes == first.classes
    assert second.samples == first.samples
    assert second.class_to_idx == first.class_to_idx

    # Prove the second construction reads the index, not the directory tree:
    # a doctored cache must be honored verbatim.
    with open(cache_path, "rb") as handle:
        index = pickle.load(handle)
    index["samples"] = index["samples"][:2]
    with open(cache_path, "wb") as handle:
        pickle.dump(index, handle)
    doctored = _imagefolder_with_index_cache(tiny_tree, transform=None)
    assert len(doctored) == 2

    image, label = doctored[0]
    assert label in (0, 1)
    assert image.size == (4, 4)


def test_cached_dataset_loads_images(tiny_tree, tmp_path, monkeypatch):
    from torchvision import transforms

    monkeypatch.setenv("DENDRITIC_IMAGEFOLDER_CACHE", str(tmp_path / "cache"))
    _imagefolder_with_index_cache(tiny_tree, transform=None)
    dataset = _imagefolder_with_index_cache(tiny_tree, transform=transforms.ToTensor())
    image, _ = dataset[3]
    assert isinstance(image, torch.Tensor) and image.shape == (3, 4, 4)

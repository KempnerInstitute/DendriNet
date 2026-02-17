"""Tests for data augmentation noise functions."""

import torch

from dendritic_modeling.datasets.augmentation.noise import (
    add_frequency_noise,
    add_gaussian_noise,
    add_poisson_noise,
    add_uniform_noise,
)

# ---------------------------------------------------------------------------
# add_gaussian_noise
# ---------------------------------------------------------------------------


def test_gaussian_noise_shape_single_sample():
    x = torch.rand(4, 10)
    out = add_gaussian_noise(x, std=0.1, n_samples=1)
    assert out.shape == (1, 4, 10)


def test_gaussian_noise_shape_multi_sample():
    x = torch.rand(4, 10)
    out = add_gaussian_noise(x, std=0.1, n_samples=5)
    assert out.shape == (5, 4, 10)


def test_gaussian_noise_clamped():
    x = torch.rand(8, 20)
    out = add_gaussian_noise(x, std=1.0, n_samples=3)
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_gaussian_noise_zero_std():
    x = torch.rand(4, 10)
    torch.manual_seed(0)
    out = add_gaussian_noise(x, std=0.0, n_samples=1)
    torch.testing.assert_close(out[0], x)


# ---------------------------------------------------------------------------
# add_uniform_noise
# ---------------------------------------------------------------------------


def test_uniform_noise_shape():
    x = torch.rand(4, 10)
    out = add_uniform_noise(x, spread=0.1, n_samples=3)
    assert out.shape == (3, 4, 10)


def test_uniform_noise_clamped():
    x = torch.rand(8, 20)
    out = add_uniform_noise(x, spread=1.0, n_samples=3)
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_uniform_noise_zero_spread():
    x = torch.rand(4, 10)
    out = add_uniform_noise(x, spread=0.0, n_samples=1)
    torch.testing.assert_close(out[0], x)


# ---------------------------------------------------------------------------
# add_poisson_noise
# ---------------------------------------------------------------------------


def test_poisson_noise_shape():
    x = torch.rand(4, 10)
    out = add_poisson_noise(x, rate=10.0, n_samples=3)
    assert out.shape == (3, 4, 10)


def test_poisson_noise_clamped():
    x = torch.rand(8, 20)
    out = add_poisson_noise(x, rate=10.0, n_samples=3)
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_poisson_noise_high_rate_approx_input():
    # High rate → Poisson noise relatively small → output ≈ input
    x = torch.full((4, 10), 0.5)
    out = add_poisson_noise(x, rate=10000.0, n_samples=1)
    assert (out[0] - x).abs().mean() < 0.05


# ---------------------------------------------------------------------------
# add_frequency_noise
# ---------------------------------------------------------------------------


def test_frequency_noise_shape():
    x = torch.rand(4, 784)  # 28x28 images
    out = add_frequency_noise(
        x, std=0.1, n_samples=2, n_channels=1, image_shape=(28, 28)
    )
    assert out.shape == (2, 4, 784)


def test_frequency_noise_clamped():
    x = torch.rand(4, 784)
    out = add_frequency_noise(
        x, std=1.0, n_samples=1, n_channels=1, image_shape=(28, 28)
    )
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_frequency_noise_zero_std():
    x = torch.rand(2, 784)
    out = add_frequency_noise(
        x, std=0.0, n_samples=1, n_channels=1, image_shape=(28, 28)
    )
    torch.testing.assert_close(out[0], x, atol=1e-5, rtol=1e-5)


def test_frequency_noise_multi_channel():
    x = torch.rand(2, 3 * 16 * 16)  # 3 channels, 16x16
    out = add_frequency_noise(
        x, std=0.1, n_samples=1, n_channels=3, image_shape=(16, 16)
    )
    assert out.shape == (1, 2, 3 * 16 * 16)

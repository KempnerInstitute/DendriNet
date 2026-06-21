"""Tests for data augmentation noise functions."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

import dendritic_modeling.analysis.metrics.noise_perturbation as noise_perturbation
from dendritic_modeling.analysis.metrics.noise_perturbation import (
    NoisePerturbationAnalyzer,
)
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


def test_gaussian_noise_unclamped():
    x = torch.full((4, 10), 10.0)
    out = add_gaussian_noise(x, std=0.0, n_samples=1, clamp_range=None)
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


def test_uniform_noise_unclamped():
    x = torch.full((4, 10), -2.0)
    out = add_uniform_noise(x, spread=0.0, n_samples=1, clamp_range=None)
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


def test_poisson_noise_unclamped():
    x = torch.full((4, 10), 2.0)
    out = add_poisson_noise(x, rate=10000.0, n_samples=1, clamp_range=None)
    assert out.max() > 1.0


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


def test_frequency_noise_unclamped():
    x = torch.full((2, 784), 2.0)
    out = add_frequency_noise(
        x,
        std=0.0,
        n_samples=1,
        n_channels=1,
        image_shape=(28, 28),
        clamp_range=None,
    )
    torch.testing.assert_close(out[0], x, atol=1e-5, rtol=1e-5)


def test_frequency_noise_multi_channel():
    x = torch.rand(2, 3 * 16 * 16)  # 3 channels, 16x16
    out = add_frequency_noise(
        x, std=0.1, n_samples=1, n_channels=3, image_shape=(16, 16)
    )
    assert out.shape == (1, 2, 3 * 16 * 16)


def test_noise_perturbation_analyzer_allows_unclamped_noise():
    assert NoisePerturbationAnalyzer._normalize_clamp_range(None) is None


def test_noise_perturbation_analyzer_normalizes_clamp_pair():
    assert NoisePerturbationAnalyzer._normalize_clamp_range([0, 2]) == (0.0, 2.0)


def test_noise_perturbation_analyzer_rejects_invalid_clamp_pair():
    with pytest.raises(ValueError, match="min <= max"):
        NoisePerturbationAnalyzer._normalize_clamp_range([1, 0])


@pytest.mark.parametrize("value", [1.0, "01", b"01", {0.0, 1.0}, [0]])
def test_noise_perturbation_analyzer_rejects_non_pair_clamp_range(value):
    with pytest.raises(ValueError, match=r"\[min, max\] pair"):
        NoisePerturbationAnalyzer._normalize_clamp_range(value)


def test_noise_perturbation_analyzer_forwards_unclamped_gaussian(monkeypatch):
    params = SimpleNamespace(
        uniform_noise=False,
        gaussian_noise=True,
        gaussian_sdevs=[0.1],
        n_samples=1,
        accuracy=True,
        auc=False,
        categorical_loglikelihood=False,
        mse=False,
        cosine_similarity=False,
        clamp_range=None,
        seed=0,
    )
    analyzer = NoisePerturbationAnalyzer(params)
    captured = {}

    def fake_add_gaussian_noise(x, std, n_samples, generator, clamp_range):
        captured["clamp_range"] = clamp_range
        return x[None, ...].expand(n_samples, *x.shape)

    def fake_evaluate_accuracy(*args, **kwargs):
        return [None, torch.tensor([1.0])]

    monkeypatch.setattr(
        noise_perturbation, "add_gaussian_noise", fake_add_gaussian_noise
    )
    monkeypatch.setattr(noise_perturbation, "evaluate_accuracy", fake_evaluate_accuracy)

    model = torch.nn.Linear(2, 2)
    dataset = TensorDataset(torch.ones(3, 2), torch.zeros(3, dtype=torch.long))
    result = analyzer.analyze(model, dataset)

    assert captured["clamp_range"] is None
    assert result["gaussian"]["0.1"]["accuracy"] == [1.0]

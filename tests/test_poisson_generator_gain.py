import pytest
import torch

from dendritic_modeling.datasets.standard_datasets import PoissonGeneratorDataset


def test_uniform_gain_is_scalar_for_flattened_single_sample():
    base_inputs = torch.tensor([[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 7.0]])
    labels = torch.tensor([0, 1])
    dataset = PoissonGeneratorDataset(
        base_inputs,
        labels,
        multiplicative_gain=True,
        max_gain_factor=10.0,
        uniform_gain=True,
        poisson_sampling=False,
    )

    torch.manual_seed(0)
    sample, label = dataset[0]
    ratios = sample / base_inputs[0]

    assert label == labels[0]
    assert torch.allclose(ratios, torch.full_like(ratios, ratios[0]))
    assert 1.0 <= float(ratios[0]) <= 10.0


def test_uniform_gain_is_one_scalar_per_example_for_batched_slice():
    base_inputs = torch.tensor([[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 7.0]])
    labels = torch.tensor([0, 1])
    dataset = PoissonGeneratorDataset(
        base_inputs,
        labels,
        multiplicative_gain=True,
        max_gain_factor=10.0,
        uniform_gain=True,
        poisson_sampling=False,
    )

    torch.manual_seed(0)
    batch, batch_labels = dataset[:]
    ratios = batch / base_inputs

    assert torch.equal(batch_labels, labels)
    assert torch.allclose(ratios, ratios[:, :1].expand_as(ratios))
    assert torch.all((ratios[:, 0] >= 1.0) & (ratios[:, 0] <= 10.0))


def test_nonuniform_gain_remains_featurewise_for_flattened_sample():
    base_inputs = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    labels = torch.tensor([0])
    dataset = PoissonGeneratorDataset(
        base_inputs,
        labels,
        multiplicative_gain=True,
        max_gain_factor=10.0,
        uniform_gain=False,
        poisson_sampling=False,
    )

    torch.manual_seed(0)
    sample, _ = dataset[0]
    ratios = sample / base_inputs[0]

    assert not torch.allclose(ratios, torch.full_like(ratios, ratios[0]))
    assert torch.all((ratios >= 1.0) & (ratios <= 10.0))


def test_uniform_gain_without_explicit_factor_defaults_to_identity_gain():
    base_inputs = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    labels = torch.tensor([0])
    dataset = PoissonGeneratorDataset(
        base_inputs,
        labels,
        multiplicative_gain=True,
        uniform_gain=True,
        poisson_sampling=False,
    )

    sample, label = dataset[0]

    assert label == labels[0]
    assert torch.allclose(sample, base_inputs[0])


def test_nonuniform_gain_without_factor_is_rejected():
    base_inputs = torch.tensor([[1.0, 2.0]])
    labels = torch.tensor([0])
    with pytest.raises(ValueError, match="requires fixed_gain_factor"):
        PoissonGeneratorDataset(
            base_inputs,
            labels,
            multiplicative_gain=True,
            uniform_gain=False,
            poisson_sampling=False,
        )

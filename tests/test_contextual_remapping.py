import torch

from dendritic_modeling.datasets.contextual_remapping import (
    ContextualRemappingConfig,
    brightness_dependent_noise,
    contextual_remapping_batch,
)


def test_corruption_equation_and_lower_only_clip():
    x = torch.ones(3, 20)
    g = torch.tensor([0.05, 1.0, 20.0])
    divisor = torch.tensor([1.0, 2.0, 4.0])
    actual = brightness_dependent_noise(
        x, g, divisor, generator=torch.Generator().manual_seed(18)
    )
    z = torch.randn(x.shape, generator=torch.Generator().manual_seed(18))
    expected = (g[:, None] * x + g.sqrt()[:, None] / divisor[:, None] * z).clamp_min(0)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=4e-8)
    assert actual.max() > 1 and actual.min() >= 0


def test_context_labels_and_repeatable_clean_cues():
    config = ContextualRemappingConfig()
    images, labels = torch.zeros(47, 784), torch.arange(47)
    contexts = torch.arange(47) % 32
    x, y, c = contextual_remapping_batch(images, labels, config, contexts=contexts)
    assert x.shape == (47, 816)
    assert torch.equal(y, (labels + contexts) % 47)
    assert torch.equal(x[:, 784:].argmax(-1), c)
    assert torch.equal(x[:, :784], images)
    state = torch.get_rng_state()
    a = contextual_remapping_batch(
        torch.ones_like(images),
        labels,
        config,
        generator=torch.Generator().manual_seed(2),
    )
    b = contextual_remapping_batch(
        torch.ones_like(images),
        labels,
        config,
        generator=torch.Generator().manual_seed(2),
    )
    assert all(torch.equal(i, j) for i, j in zip(a, b))
    assert torch.equal(torch.get_rng_state(), state)

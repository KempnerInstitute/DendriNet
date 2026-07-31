import pytest
import torch

from dendritic_modeling.datasets.contextual_gain import ContextualGainDataset


def _make_dataset(
    *,
    n_samples: int = 8,
    n_features: int = 4,
    n_layered_images: int = 3,
    min_gain_factor: float = 0.01,
    max_gain_factor: float = 1.0,
    gain_sampling: str = "linear_uniform",
    poisson_sampling: bool = False,
):
    base_inputs = torch.arange(
        n_samples * n_features, dtype=torch.float32
    ).reshape(n_samples, n_features) + 1.0
    # Distinct labels so argmax-layer → label is unambiguous.
    labels = torch.arange(n_samples, dtype=torch.long)
    return ContextualGainDataset(
        base_inputs,
        labels,
        n_layered_images=n_layered_images,
        min_gain_factor=min_gain_factor,
        max_gain_factor=max_gain_factor,
        gain_sampling=gain_sampling,
        poisson_sampling=poisson_sampling,
    )


def test_scalar_and_slice_getitem_shapes():
    ds = _make_dataset(n_layered_images=2)
    x0, y0 = ds[0]
    assert x0.shape == (4,)
    assert y0.ndim == 0

    xb, yb = ds[:]
    assert xb.shape == (8, 4)
    assert yb.shape == (8,)

    xs, ys = ds[1:4]
    assert xs.shape == (3, 4)
    assert ys.shape == (3,)


def test_companion_indices_exclude_primary_and_are_unique():
    ds = _make_dataset(n_samples=10, n_layered_images=4)
    primary = torch.tensor([0, 3, 7], dtype=torch.long)
    for _ in range(20):
        layer_idx = ds._sample_layer_indices(primary, n_layers=4)
        assert layer_idx.shape == (3, 4)
        assert torch.equal(layer_idx[:, 0], primary)
        for row in layer_idx:
            assert row[0].item() not in row[1:].tolist()
            assert len(set(row.tolist())) == 4


def test_gain_sampling_uses_scaled_interval_and_batched_shape(monkeypatch):
    ds = _make_dataset(
        n_layered_images=4,
        min_gain_factor=0.08,
        max_gain_factor=0.8,
        gain_sampling="linear_uniform",
    )
    captured = {}

    def spy(min_g, max_g, *, gain_sampling, shape, dtype, device):
        captured["min"] = min_g
        captured["max"] = max_g
        captured["shape"] = shape
        captured["gain_sampling"] = gain_sampling
        return torch.full(shape, (min_g + max_g) / 2, dtype=dtype, device=device)

    monkeypatch.setattr(
        ContextualGainDataset,
        "_sample_gain_singlestage",
        staticmethod(spy),
    )

    _ = ds[:]
    assert captured["min"] == pytest.approx(0.08 / 4)
    assert captured["max"] == pytest.approx(0.8 / 4)
    assert captured["shape"] == (8, 4)
    assert captured["gain_sampling"] == "linear_uniform"

    _ = ds[2]
    assert captured["shape"] == (1, 4)


def test_label_is_class_of_max_gain_layer(monkeypatch):
    ds = _make_dataset(n_layered_images=3, poisson_sampling=False)

    def fixed_layers(primary, *, n_layers):
        # primary | primary+1 | primary+2 (mod N), all distinct for N=8
        n_pool = len(ds)
        cols = [primary]
        for k in range(1, n_layers):
            cols.append((primary + k) % n_pool)
        return torch.stack(cols, dim=1)

    def fixed_gains(min_g, max_g, *, gain_sampling, shape, dtype, device):
        gains = torch.zeros(shape, dtype=dtype, device=device)
        # Layer 2 always wins.
        gains[..., 0] = 0.1
        gains[..., 1] = 0.2
        gains[..., 2] = 0.9
        return gains

    monkeypatch.setattr(ds, "_sample_layer_indices", fixed_layers)
    monkeypatch.setattr(
        ContextualGainDataset,
        "_sample_gain_singlestage",
        staticmethod(fixed_gains),
    )

    x, y = ds[0]
    assert int(y.item()) == 2  # label of index 2
    expected = (
        0.1 * ds.base_inputs[0] + 0.2 * ds.base_inputs[1] + 0.9 * ds.base_inputs[2]
    )
    assert torch.allclose(x, expected)

    xb, yb = ds[0:3]
    assert torch.equal(yb, torch.tensor([2, 3, 4]))


def test_set_max_gain_factor_overrides_sampling_bound(monkeypatch):
    ds = _make_dataset(
        n_layered_images=2,
        min_gain_factor=0.1,
        max_gain_factor=1.0,
    )
    captured = {}

    def spy(min_g, max_g, *, gain_sampling, shape, dtype, device):
        captured["min"] = min_g
        captured["max"] = max_g
        return torch.ones(shape, dtype=dtype, device=device) * min_g

    monkeypatch.setattr(
        ContextualGainDataset,
        "_sample_gain_singlestage",
        staticmethod(spy),
    )

    ds._set_max_gain_factor(0.4)
    _ = ds[0]
    assert captured["min"] == pytest.approx(0.1 / 2)
    assert captured["max"] == pytest.approx(0.4 / 2)

    ds._clear_gain()
    _ = ds[0]
    assert captured["max"] == pytest.approx(1.0 / 2)


def test_poisson_sampling_flag():
    ds_off = _make_dataset(poisson_sampling=False, n_layered_images=2)
    ds_on = _make_dataset(poisson_sampling=True, n_layered_images=2)

    torch.manual_seed(0)
    x_off, _ = ds_off[0]
    torch.manual_seed(0)
    x_on, _ = ds_on[0]

    assert x_off.dtype == torch.float32
    assert x_on.dtype == torch.float32
    assert torch.all(x_on >= 0)
    # Poisson path should not divide by tau: values are raw Poisson counts.
    assert torch.all(x_on == x_on.round())

import pytest
import torch

from dendritic_modeling.datasets.contextual_gain import ContextualGainDataset


def _make_dataset(
    *,
    n_samples: int = 8,
    n_features: int = 4,
    n_regions: int = 3,
    min_gain_factor: float = 0.01,
    max_gain_factor: float = 1.0,
    gain_sampling: str = "linear_uniform",
    region_poisson: bool = False,
    n_context_bits: int = 0,
    context_dim: int = 50,
    context_rule: str = "bit_id",
    context_seed: int = 0,
    context_gain: bool = False,
    context_poisson: bool = False,
):
    base_inputs = torch.arange(
        n_samples * n_features, dtype=torch.float32
    ).reshape(n_samples, n_features) + 1.0
    # Distinct labels so argmax-layer → label is unambiguous.
    labels = torch.arange(n_samples, dtype=torch.long)
    return ContextualGainDataset(
        base_inputs,
        labels,
        n_regions=n_regions,
        min_gain_factor=min_gain_factor,
        max_gain_factor=max_gain_factor,
        gain_sampling=gain_sampling,
        region_poisson=region_poisson,
        n_context_bits=n_context_bits,
        context_dim=context_dim,
        context_rule=context_rule,
        context_seed=context_seed,
        context_gain=context_gain,
        context_poisson=context_poisson,
    )


def test_scalar_and_slice_getitem_shapes():
    ds = _make_dataset(n_regions=2)
    x0, y0 = ds[0]
    assert x0.shape == (8,)
    assert y0.ndim == 0

    xb, yb = ds[:]
    assert xb.shape == (8, 8)
    assert yb.shape == (8,)

    xs, ys = ds[1:4]
    assert xs.shape == (3, 8)
    assert ys.shape == (3,)


def test_companion_indices_exclude_primary_and_are_unique():
    ds = _make_dataset(n_samples=10, n_regions=4)
    primary = torch.tensor([0, 3, 7], dtype=torch.long)
    for _ in range(20):
        layer_idx = ds._sample_layer_indices(primary, n_layers=4)
        assert layer_idx.shape == (3, 4)
        assert torch.equal(layer_idx[:, 0], primary)
        for row in layer_idx:
            assert row[0].item() not in row[1:].tolist()
            assert len(set(row.tolist())) == 4


def test_gain_sampling_uses_full_interval_and_batched_shape(monkeypatch):
    ds = _make_dataset(
        n_regions=4,
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
    assert captured["min"] == pytest.approx(0.08)
    assert captured["max"] == pytest.approx(0.8)
    assert captured["shape"] == (8, 4)
    assert captured["gain_sampling"] == "linear_uniform"

    _ = ds[2]
    assert captured["shape"] == (1, 4)


def test_label_is_class_of_max_gain_layer(monkeypatch):
    ds = _make_dataset(n_regions=3, region_poisson=False)

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
    expected = torch.cat(
        [
            0.1 * ds.base_inputs[0],
            0.2 * ds.base_inputs[1],
            0.9 * ds.base_inputs[2],
        ],
        dim=-1,
    )
    assert torch.allclose(x, expected)

    xb, yb = ds[0:3]
    assert torch.equal(yb, torch.tensor([2, 3, 4]))


def test_set_max_gain_factor_overrides_sampling_bound(monkeypatch):
    ds = _make_dataset(
        n_regions=2,
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
    assert captured["min"] == pytest.approx(0.1)
    assert captured["max"] == pytest.approx(0.4)

    ds._clear_gain()
    _ = ds[0]
    assert captured["max"] == pytest.approx(1.0)


def test_region_poisson_flag():
    ds_off = _make_dataset(region_poisson=False, n_regions=2)
    ds_on = _make_dataset(region_poisson=True, n_regions=2)

    torch.manual_seed(0)
    x_off, _ = ds_off[0]
    torch.manual_seed(0)
    x_on, _ = ds_on[0]

    assert x_off.dtype == torch.float32
    assert x_on.dtype == torch.float32
    assert torch.all(x_on >= 0)
    # Poisson path should not divide by tau: values are raw Poisson counts.
    assert torch.all(x_on == x_on.round())


def test_context_appends_dims_and_seeded_projection_matches():
    ds_a = _make_dataset(
        n_regions=2,
        n_context_bits=2,
        context_dim=50,
        context_seed=7,
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
    )
    ds_b = _make_dataset(
        n_regions=2,
        n_context_bits=2,
        context_dim=50,
        context_seed=7,
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
    )
    assert torch.equal(ds_a.context_projection, ds_b.context_projection)
    assert ds_a.context_projection.shape == (2, 50)
    assert torch.all(ds_a.context_projection >= 0)
    assert torch.all(ds_a.context_projection.sum(dim=0) <= 1.0 + 1e-6)

    x, _ = ds_a[0]
    assert x.shape == (2 * 4 + 50,)


def test_context_gain_extends_gain_sample_shape(monkeypatch):
    captured = {}

    def spy(min_g, max_g, *, gain_sampling, shape, dtype, device):
        captured["shape"] = shape
        return torch.ones(shape, dtype=dtype, device=device)

    monkeypatch.setattr(
        ContextualGainDataset,
        "_sample_gain_singlestage",
        staticmethod(spy),
    )

    ds_on = _make_dataset(
        n_regions=3,
        n_context_bits=2,
        context_dim=8,
        context_gain=True,
        region_poisson=False,
        context_poisson=False,
    )
    _ = ds_on[:]
    assert captured["shape"] == (8, 4)  # n_regions + 1

    ds_off = _make_dataset(
        n_regions=3,
        n_context_bits=2,
        context_dim=8,
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
    )
    _ = ds_off[:]
    assert captured["shape"] == (8, 3)


def test_bit_id_label_remap(monkeypatch):
    ds = _make_dataset(
        n_regions=2,
        n_context_bits=2,
        context_dim=8,
        context_rule="bit_id",
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
    )

    def fixed_layers(primary, *, n_layers):
        n_pool = len(ds)
        cols = [primary]
        for k in range(1, n_layers):
            cols.append((primary + k) % n_pool)
        return torch.stack(cols, dim=1)

    def fixed_gains(min_g, max_g, *, gain_sampling, shape, dtype, device):
        gains = torch.zeros(shape, dtype=dtype, device=device)
        gains[..., 0] = 1.0  # region 0 wins -> digit = primary label
        return gains

    monkeypatch.setattr(ds, "_sample_layer_indices", fixed_layers)
    monkeypatch.setattr(
        ContextualGainDataset,
        "_sample_gain_singlestage",
        staticmethod(fixed_gains),
    )
    # bits [1, 0] -> bit_id 2
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    _, y = ds[0]
    # digit 0 + ctx 2
    assert int(y.item()) == 2


def test_xor_label_remap(monkeypatch):
    ds = _make_dataset(
        n_regions=2,
        n_context_bits=2,
        context_dim=8,
        context_rule="xor",
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
        n_samples=10,
    )
    # labels 0..9 so mod wraps meaningfully
    ds.base_labels = torch.arange(10, dtype=torch.long)
    ds.n_classes = 10

    def fixed_layers(primary, *, n_layers):
        cols = [primary]
        for k in range(1, n_layers):
            cols.append((primary + k) % len(ds))
        return torch.stack(cols, dim=1)

    def fixed_gains(min_g, max_g, *, gain_sampling, shape, dtype, device):
        gains = torch.zeros(shape, dtype=dtype, device=device)
        gains[..., 0] = 1.0
        return gains

    monkeypatch.setattr(ds, "_sample_layer_indices", fixed_layers)
    monkeypatch.setattr(
        ContextualGainDataset,
        "_sample_gain_singlestage",
        staticmethod(fixed_gains),
    )
    # bits [1, 1] -> xor 0; digit 3 -> label 3
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: torch.tensor([[1.0, 1.0]], dtype=torch.float32),
    )
    _, y0 = ds[3]
    assert int(y0.item()) == 3

    # bits [1, 0] -> xor 1; digit 3 -> label 4
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )
    _, y1 = ds[3]
    assert int(y1.item()) == 4


def test_context_poisson_only_affects_context_slice(monkeypatch):
    ds = _make_dataset(
        n_regions=2,
        n_features=4,
        n_context_bits=2,
        context_dim=8,
        context_gain=False,
        region_poisson=False,
        context_poisson=True,
    )

    def fixed_gains(min_g, max_g, *, gain_sampling, shape, dtype, device):
        return torch.ones(shape, dtype=dtype, device=device)

    monkeypatch.setattr(
        ContextualGainDataset,
        "_sample_gain_singlestage",
        staticmethod(fixed_gains),
    )
    # Deterministic bits so ctx_vec is fixed before Poisson
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: torch.ones(args[2], dtype=torch.float32),
    )

    torch.manual_seed(0)
    x, _ = ds[0]
    image = x[:8]
    ctx = x[8:]
    # Images are exact scaled concat (no region poisson); context is Poisson counts.
    assert torch.all(ctx >= 0)
    assert torch.all(ctx == ctx.round())
    assert image.dtype == torch.float32

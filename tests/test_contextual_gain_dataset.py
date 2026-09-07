from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, default_collate

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
    n_context_latents: int = 0,
    context_dim: int = 50,
    context_seed: int = 0,
    context_gain: bool = False,
    context_poisson: bool = False,
    context_distractor_scale: float = 0.0,
    region_in_label: bool = False,
    compositional_ood: bool = False,
    compositional_ood_mode: str = "region",
    labels: torch.Tensor | None = None,
):
    base_inputs = torch.arange(
        n_samples * n_features, dtype=torch.float32
    ).reshape(n_samples, n_features) + 1.0
    # Distinct labels so argmax-layer → label is unambiguous.
    if labels is None:
        labels = torch.arange(n_samples, dtype=torch.long)
    return ContextualGainDataset(
        base_inputs,
        labels,
        n_regions=n_regions,
        min_gain_factor=min_gain_factor,
        max_gain_factor=max_gain_factor,
        gain_sampling=gain_sampling,
        region_poisson=region_poisson,
        n_context_latents=n_context_latents,
        context_dim=context_dim,
        context_seed=context_seed,
        context_gain=context_gain,
        context_poisson=context_poisson,
        context_distractor_scale=context_distractor_scale,
        region_in_label=region_in_label,
        compositional_ood=compositional_ood,
        compositional_ood_mode=compositional_ood_mode,
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


def _legacy_sample_layer_indices(
    primary: torch.Tensor, n_pool: int, n_layers: int
) -> torch.Tensor:
    """Reference argsort-of-scores sampler used by the previous implementation."""
    if n_layers == 1:
        return primary.unsqueeze(1)
    scores = torch.rand(
        primary.shape[0], n_pool, device=primary.device, dtype=torch.float32
    )
    scores.scatter_(1, primary.unsqueeze(1), float("inf"))
    companions = scores.argsort(dim=1)[:, : n_layers - 1]
    return torch.cat([primary.unsqueeze(1), companions], dim=1)


def _assert_companion_contract(
    layer_idx: torch.Tensor, primary: torch.Tensor, n_layers: int
) -> None:
    assert layer_idx.shape == (int(primary.shape[0]), n_layers)
    assert torch.equal(layer_idx[:, 0], primary)
    for row in layer_idx:
        values = row.tolist()
        assert values[0] not in values[1:]
        assert len(set(values)) == n_layers


def _companion_tuple_keys(layer_idx: torch.Tensor, n_pool: int) -> torch.Tensor:
    companions = layer_idx[:, 1:]
    keys = companions[:, 0].clone()
    for col in range(1, companions.shape[1]):
        keys = keys * n_pool + companions[:, col]
    return keys


def _count_keys(keys: torch.Tensor) -> dict[int, int]:
    unique, counts = torch.unique(keys, return_counts=True)
    return {
        int(key): int(count)
        for key, count in zip(unique.tolist(), counts.tolist())
    }


def _chi_square_uniform(counts: dict[int, int], n_allowed: int, n_draws: int) -> float:
    expected = n_draws / n_allowed
    observed = list(counts.values()) + [0] * (n_allowed - len(counts))
    return sum((obs - expected) ** 2 / expected for obs in observed)


def _total_variation(
    left: dict[int, int], right: dict[int, int], n_draws: int
) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(k, 0) - right.get(k, 0)) for k in keys) / n_draws


@pytest.mark.parametrize(
    "n_samples, n_regions",
    [(10, 1), (10, 2), (10, 4), (8, 8)],
)
def test_companion_indices_exclude_primary_and_are_unique(n_samples, n_regions):
    ds = _make_dataset(n_samples=n_samples, n_regions=n_regions)
    primary = torch.tensor([0, min(3, n_samples - 1), n_samples - 1], dtype=torch.long)
    for _ in range(20):
        layer_idx = ds._sample_layer_indices(primary, n_layers=n_regions)
        _assert_companion_contract(layer_idx, primary, n_regions)
        legacy = _legacy_sample_layer_indices(primary, n_samples, n_regions)
        _assert_companion_contract(legacy, primary, n_regions)


def test_companion_k1_remap_covers_complement():
    n_pool = 6
    ds = _make_dataset(n_samples=n_pool, n_regions=2)
    primary = torch.tensor([2], dtype=torch.long)
    seen: set[int] = set()
    for _ in range(200):
        layer_idx = ds._sample_layer_indices(primary, n_layers=2)
        _assert_companion_contract(layer_idx, primary, 2)
        seen.add(int(layer_idx[0, 1].item()))
    assert seen == {0, 1, 3, 4, 5}


@pytest.mark.parametrize("n_regions, n_allowed, chi2_max", [(2, 5, 40.0), (3, 20, 80.0)])
def test_companion_distribution_matches_legacy_argsort(n_regions, n_allowed, chi2_max):
    n_pool = 6
    n_draws = 8000
    ds = _make_dataset(n_samples=n_pool, n_regions=n_regions)
    primary = torch.full((n_draws,), 2, dtype=torch.long)

    torch.manual_seed(0)
    new_idx = ds._sample_layer_indices(primary, n_layers=n_regions)
    torch.manual_seed(1)
    legacy_idx = _legacy_sample_layer_indices(primary, n_pool, n_regions)

    _assert_companion_contract(new_idx[:8], primary[:8], n_regions)
    _assert_companion_contract(legacy_idx[:8], primary[:8], n_regions)

    new_counts = _count_keys(_companion_tuple_keys(new_idx, n_pool))
    legacy_counts = _count_keys(_companion_tuple_keys(legacy_idx, n_pool))
    assert len(new_counts) == n_allowed
    assert len(legacy_counts) == n_allowed
    assert _chi_square_uniform(new_counts, n_allowed, n_draws) < chi2_max
    assert _chi_square_uniform(legacy_counts, n_allowed, n_draws) < chi2_max
    assert _total_variation(new_counts, legacy_counts, n_draws) < 0.08


def test_sample_layer_indices_does_not_allocate_pool_width(monkeypatch):
    shapes: list[tuple[int, ...]] = []
    real_rand = torch.rand
    real_randint = torch.randint

    def _spy_rand(*args, **kwargs):
        tensor = real_rand(*args, **kwargs)
        shapes.append(tuple(tensor.shape))
        return tensor

    def _spy_randint(*args, **kwargs):
        tensor = real_randint(*args, **kwargs)
        shapes.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(torch, "rand", _spy_rand)
    monkeypatch.setattr(torch, "randint", _spy_randint)

    n_pool = 32
    ds = _make_dataset(n_samples=n_pool, n_regions=3)
    primary = torch.arange(8, dtype=torch.long)
    ds._sample_layer_indices(primary, n_layers=3)
    assert shapes
    for shape in shapes:
        assert n_pool not in shape[1:]

    shapes.clear()
    _ = ds[:]
    assert shapes
    for shape in shapes:
        assert n_pool not in shape[1:]


def test_getitems_matches_batched_getitem():
    ds = _make_dataset(n_regions=2, region_poisson=False)
    idxs = [1, 2, 3]
    torch.manual_seed(0)
    batched_x, batched_y = ds[idxs]
    torch.manual_seed(0)
    collated_x, collated_y = default_collate(ds.__getitems__(idxs))
    assert torch.equal(collated_x, batched_x)
    assert torch.equal(collated_y, batched_y)


def test_dataloader_getitems_smoke():
    ds = _make_dataset(n_regions=2, region_poisson=False)
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    images, labels = next(iter(loader))
    assert images.shape[0] == 4
    assert labels.shape[0] == 4
    assert images.shape[1] == 8


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


def test_region_in_label_defaults_true():
    ds = ContextualGainDataset(
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        torch.arange(2, dtype=torch.long),
        n_regions=2,
        region_poisson=False,
    )
    assert ds.region_in_label is True


def test_label_is_class_of_max_gain_layer(monkeypatch):
    ds = _make_dataset(n_regions=3, region_poisson=False, region_in_label=False)

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
        n_context_latents=4,
        context_dim=50,
        context_seed=7,
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
    )
    ds_b = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        context_dim=50,
        context_seed=7,
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
    )
    assert torch.equal(ds_a.context_projection, ds_b.context_projection)
    assert ds_a.n_context_latents == 4
    assert ds_a.context_projection.shape == (4, 50)
    assert torch.all(ds_a.context_projection >= 0)
    assert torch.all(ds_a.context_projection < 1.0)

    x, _ = ds_a[0]
    assert x.shape == (2 * 4 + 50,)


def test_context_id_is_never_all_zero():
    ds = _make_dataset(
        n_context_latents=4,
        context_dim=8,
        region_poisson=False,
    )
    for latent_id in range(4):
        ids = torch.tensor([latent_id], dtype=torch.long)
        cid = ds._context_id(ids)
        assert cid.shape == (1, 4)
        assert torch.all((cid == 0) | (cid == 1))
        assert int(cid.sum().item()) == 1


def test_latent_id_zero_selects_w0(monkeypatch):
    ds = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        context_dim=8,
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
    )
    monkeypatch.setattr(
        ds,
        "_sample_context_ids",
        lambda winner, digit=None: torch.zeros(
            winner.shape[0], dtype=torch.long, device=winner.device
        ),
    )
    cid = ds._context_id(torch.zeros(1, dtype=torch.long))
    assert torch.equal(cid, torch.tensor([[1, 0, 0, 0]]))
    x, _ = ds[0]
    assert torch.allclose(x[-8:], ds.context_projection[0])


def test_context_stream_selects_one_w_row(monkeypatch):
    ds = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        context_dim=8,
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
    )
    monkeypatch.setattr(
        ds,
        "_sample_context_ids",
        lambda winner, digit=None: torch.full(
            (winner.shape[0],), 2, dtype=torch.long, device=winner.device
        ),
    )
    x, _ = ds[0]
    assert torch.allclose(x[-8:], ds.context_projection[2])


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
        n_context_latents=4,
        context_dim=8,
        context_gain=True,
        region_poisson=False,
        context_poisson=False,
    )
    _ = ds_on[:]
    assert captured["shape"] == (8, 4)  # n_regions + 1

    ds_off = _make_dataset(
        n_regions=3,
        n_context_latents=4,
        context_dim=8,
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
    )
    _ = ds_off[:]
    assert captured["shape"] == (8, 3)


def test_latent_id_label_remap(monkeypatch):
    ds = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        context_dim=8,
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
        region_in_label=False,
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
    monkeypatch.setattr(
        ds,
        "_sample_context_ids",
        lambda winner, digit=None: torch.full(
            (winner.shape[0],), 2, dtype=torch.long, device=winner.device
        ),
    )

    _, y = ds[0]
    # digit 0 + ctx 2
    assert int(y.item()) == 2


def test_region_only_label_adds_r_star(monkeypatch):
    ds = _make_dataset(
        n_regions=3,
        n_samples=10,
        region_poisson=False,
        region_in_label=True,
    )

    def fixed_layers(primary, *, n_layers):
        n_pool = len(ds)
        cols = [primary]
        for k in range(1, n_layers):
            cols.append((primary + k) % n_pool)
        return torch.stack(cols, dim=1)

    def fixed_gains(min_g, max_g, *, gain_sampling, shape, dtype, device):
        gains = torch.zeros(shape, dtype=dtype, device=device)
        # Region 2 wins so r* is nonzero.
        gains[..., 2] = 1.0
        return gains

    monkeypatch.setattr(ds, "_sample_layer_indices", fixed_layers)
    monkeypatch.setattr(
        ContextualGainDataset,
        "_sample_gain_singlestage",
        staticmethod(fixed_gains),
    )

    x, y = ds[0]
    # digit of index 2, r*=2 → (2 + 2) % 10
    assert int(y.item()) == 4
    assert x.shape == (3 * 4,)

    _, yb = ds[0:3]
    # i=0: d=2,r*=2 → 4; i=1: d=3,r*=2 → 5; i=2: d=4,r*=2 → 6
    assert torch.equal(yb, torch.tensor([4, 5, 6]))


def test_region_and_latents_use_path_indexed_nest(monkeypatch):
    ds = _make_dataset(
        n_regions=3,
        n_samples=10,
        n_context_latents=2,
        context_dim=8,
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
        region_in_label=True,
    )

    def fixed_layers(primary, *, n_layers):
        # Region 2 always holds index 3 so d=3 when r*=2.
        B = int(primary.shape[0])
        return torch.tensor([[0, 1, 3]], device=primary.device).expand(B, n_layers)

    def fixed_gains(min_g, max_g, *, gain_sampling, shape, dtype, device):
        gains = torch.zeros(shape, dtype=dtype, device=device)
        gains[..., 2] = 1.0
        return gains

    monkeypatch.setattr(ds, "_sample_layer_indices", fixed_layers)
    monkeypatch.setattr(
        ContextualGainDataset,
        "_sample_gain_singlestage",
        staticmethod(fixed_gains),
    )
    monkeypatch.setattr(
        ds,
        "_sample_context_ids",
        lambda winner, digit=None: torch.full(
            (winner.shape[0],), 1, dtype=torch.long, device=winner.device
        ),
    )

    _, y = ds[0]
    # d=3, r*=2, c=1 → (-((3 + 2) % 10) + 1) % 10 = 6
    assert int(y.item()) == 6


def test_context_poisson_only_affects_context_slice(monkeypatch):
    ds = _make_dataset(
        n_regions=2,
        n_features=4,
        n_context_latents=4,
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
    monkeypatch.setattr(
        ds,
        "_sample_context_ids",
        lambda winner, digit=None: torch.zeros(
            winner.shape[0], dtype=torch.long, device=winner.device
        ),
    )

    torch.manual_seed(0)
    x, _ = ds[0]
    image = x[:8]
    ctx = x[8:]
    # Images are exact scaled concat (no region poisson); context is Poisson counts.
    assert torch.all(ctx >= 0)
    assert torch.all(ctx == ctx.round())
    assert image.dtype == torch.float32


def test_context_distractor_scale_rejects_outside_unit_interval():
    with pytest.raises(ValueError, match="context_distractor_scale"):
        _make_dataset(context_distractor_scale=-0.1)
    with pytest.raises(ValueError, match="context_distractor_scale"):
        _make_dataset(context_distractor_scale=1.0)


def test_context_distractor_mixes_before_gain(monkeypatch):
    scale = 0.25
    ds = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        context_dim=8,
        context_gain=False,
        region_poisson=False,
        context_poisson=False,
        context_distractor_scale=scale,
    )
    monkeypatch.setattr(
        ds,
        "_sample_context_ids",
        lambda winner, digit=None: torch.full(
            (winner.shape[0],), 2, dtype=torch.long, device=winner.device
        ),
    )
    monkeypatch.setattr(torch, "rand_like", lambda x: torch.ones_like(x))

    x, _ = ds[0]
    expected = (1.0 - scale) * ds.context_projection[2] + scale * torch.ones(8)
    assert torch.allclose(x[-8:], expected)


def test_n_context_latents_one_disables_context_stream():
    ds_off = _make_dataset(n_regions=2, n_context_latents=0, region_poisson=False)
    ds_one = _make_dataset(
        n_regions=2,
        n_context_latents=1,
        context_dim=8,
        region_poisson=False,
    )
    x0, _ = ds_off[0]
    x1, _ = ds_one[0]
    assert ds_one.context_projection is None
    assert not ds_one._ood_enabled
    assert x1.shape == x0.shape == (8,)


def test_compositional_ood_defaults_disabled():
    ds = _make_dataset(n_regions=2, n_context_latents=4, region_poisson=False)
    assert ds.compositional_ood is False
    assert ds.compositional_ood_mode == "region"
    assert ds._ood_enabled is False
    assert ds._ood_sampling is False
    assert ds._is_ood is None


def test_compositional_ood_split_covering_invariants():
    ds = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        region_poisson=False,
        compositional_ood=True,
        compositional_ood_mode="region",
    )
    assert ds._ood_enabled is True
    assert ds._ood_sampling is False
    assert ds.compositional_ood_mode == "region"
    assert ds._is_ood.shape == (2, 4)
    assert ds._ood_ctx_idx.shape == (2, 1)
    assert ds._id_ctx_idx.shape == (2, 3)
    assert torch.equal(ds._ood_ctx_idx.squeeze(1), torch.tensor([0, 1]))
    assert ds._is_ood.any(dim=1).all()
    assert (~ds._is_ood).any(dim=1).all()
    assert not ds._is_ood.all(dim=0).any()
    n_regions, n_context = ds._is_ood.shape
    for r in range(n_regions):
        for c in range(n_context):
            if not bool(ds._is_ood[r, c].item()):
                continue
            others = [i for i in range(n_regions) if i != r]
            assert any(not bool(ds._is_ood[r_id, c].item()) for r_id in others)


def test_sample_context_ids_respects_id_and_ood_tables():
    ds = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        n_samples=16,
        region_poisson=False,
        compositional_ood=True,
        compositional_ood_mode="region",
    )
    winners = torch.tensor([0, 1] * 256, dtype=torch.long)
    ctx_id = ds._sample_context_ids(winners)
    assert torch.all(~ds._is_ood[winners, ctx_id])

    ds._set_OOD_sampling(True)
    ctx_ood = ds._sample_context_ids(winners)
    assert torch.all(ds._is_ood[winners, ctx_ood])

    ds._set_OOD_sampling(False)
    ctx_id_again = ds._sample_context_ids(winners)
    assert torch.all(~ds._is_ood[winners, ctx_id_again])
    assert ds._ood_sampling is False


def test_set_ood_sampling_is_noop_when_disabled():
    ds_flag = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        region_poisson=False,
        compositional_ood=False,
    )
    ds_flag._set_OOD_sampling(True)
    assert ds_flag._ood_sampling is False

    ds_regions = _make_dataset(
        n_regions=1,
        n_context_latents=4,
        region_poisson=False,
        compositional_ood=True,
    )
    ds_regions._set_OOD_sampling(True)
    assert ds_regions._ood_enabled is False
    assert ds_regions._ood_sampling is False

    ds_latents = _make_dataset(
        n_regions=2,
        n_context_latents=1,
        region_poisson=False,
        compositional_ood=True,
    )
    ds_latents._set_OOD_sampling(True)
    assert ds_latents._ood_enabled is False
    assert ds_latents._ood_sampling is False

    ds_mode_ignored = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        region_poisson=False,
        compositional_ood=False,
        compositional_ood_mode="region_digit",
    )
    assert ds_mode_ignored.compositional_ood_mode == "region_digit"
    assert ds_mode_ignored._ood_enabled is False
    assert ds_mode_ignored._is_ood is None
    ds_mode_ignored._set_OOD_sampling(True)
    assert ds_mode_ignored._ood_sampling is False


def test_compositional_ood_mode_invalid_raises():
    with pytest.raises(ValueError, match="compositional_ood_mode"):
        _make_dataset(
            n_regions=2,
            n_context_latents=4,
            region_poisson=False,
            compositional_ood=True,
            compositional_ood_mode="bogus",
        )


def test_region_digit_split_covering_invariants():
    ds = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        n_samples=8,
        region_poisson=False,
        compositional_ood=True,
        compositional_ood_mode="region_digit",
    )
    n_regions, n_classes, n_context = 2, 8, 4
    assert ds._ood_enabled is True
    assert ds.compositional_ood_mode == "region_digit"
    assert ds.n_classes == n_classes
    assert ds._is_ood.shape == (n_regions, n_classes, n_context)
    assert ds._ood_ctx_idx.shape == (n_regions, n_classes, 1)
    assert ds._id_ctx_idx.shape == (n_regions, n_classes, n_context - 1)

    expected_ood = (
        torch.arange(n_regions).unsqueeze(1) + torch.arange(n_classes).unsqueeze(0)
    ) % n_context
    assert torch.equal(ds._ood_ctx_idx.squeeze(-1), expected_ood)
    assert ds._is_ood.any(dim=-1).all()
    assert (~ds._is_ood).any(dim=-1).all()
    assert (~ds._is_ood).any(dim=1).all()
    assert (~ds._is_ood).any(dim=0).all()
    assert ds._is_ood.sum(dim=-1).eq(1).all()


def test_sample_context_ids_region_digit_respects_id_and_ood_tables():
    ds = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        n_samples=8,
        region_poisson=False,
        compositional_ood=True,
        compositional_ood_mode="region_digit",
    )
    winners = torch.tensor([0, 1, 0, 1] * 128, dtype=torch.long)
    digits = torch.tensor([0, 0, 3, 7] * 128, dtype=torch.long)
    ctx_id = ds._sample_context_ids(winners, digits)
    assert torch.all(~ds._is_ood[winners, digits, ctx_id])

    ds._set_OOD_sampling(True)
    ctx_ood = ds._sample_context_ids(winners, digits)
    assert torch.all(ds._is_ood[winners, digits, ctx_ood])
    assert torch.equal(
        ctx_ood, (winners + digits) % ds.n_context_latents
    )

    ds._set_OOD_sampling(False)
    ctx_id_again = ds._sample_context_ids(winners, digits)
    assert torch.all(~ds._is_ood[winners, digits, ctx_id_again])
    assert ds._ood_sampling is False


def test_region_digit_disabled_when_one_class():
    ds = _make_dataset(
        n_regions=2,
        n_context_latents=4,
        n_samples=8,
        region_poisson=False,
        compositional_ood=True,
        compositional_ood_mode="region_digit",
        labels=torch.zeros(8, dtype=torch.long),
    )
    assert ds.n_classes == 1
    assert ds._ood_enabled is False
    assert ds._is_ood is None
    ds._set_OOD_sampling(True)
    assert ds._ood_sampling is False


def test_config_default_includes_compositional_ood_false():
    from dendritic_modeling.config.config import DatasetParamsConfig

    params = DatasetParamsConfig().contextual_gain
    assert params["compositional_ood"] is False
    assert params["compositional_ood_mode"] == "region"


def test_wrap_and_options_forward_compositional_ood(monkeypatch):
    from dendritic_modeling.datasets.standard_datasets import (
        _load_contextual_gain_from_options,
        _wrap_contextual_gain_dataset,
    )
    from dendritic_modeling.datasets.unified_options import _UnifiedDatasetOptions

    class _Pair:
        def __getitem__(self, idx):
            return (
                torch.arange(32, dtype=torch.float32).reshape(8, 4),
                torch.arange(8, dtype=torch.long),
            )

    wrapped = _wrap_contextual_gain_dataset(
        _Pair(),
        n_regions=2,
        min_gain_factor=0.01,
        max_gain_factor=1.0,
        gain_sampling="linear_uniform",
        region_poisson=False,
        n_context_latents=4,
        context_dim=8,
        context_seed=0,
        context_gain=False,
        context_poisson=False,
        context_distractor_scale=0.0,
        region_in_label=True,
        compositional_ood=True,
        compositional_ood_mode="region",
    )
    assert wrapped.compositional_ood is True
    assert wrapped.compositional_ood_mode == "region"
    assert wrapped._ood_enabled is True

    wrapped_digit = _wrap_contextual_gain_dataset(
        _Pair(),
        n_regions=2,
        min_gain_factor=0.01,
        max_gain_factor=1.0,
        gain_sampling="linear_uniform",
        region_poisson=False,
        n_context_latents=4,
        context_dim=8,
        context_seed=0,
        context_gain=False,
        context_poisson=False,
        context_distractor_scale=0.0,
        region_in_label=True,
        compositional_ood=True,
        compositional_ood_mode="region_digit",
    )
    assert wrapped_digit.compositional_ood_mode == "region_digit"
    assert wrapped_digit._ood_enabled is True
    assert wrapped_digit._is_ood.ndim == 3

    captured = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return (None, None, None)

    monkeypatch.setattr(
        "dendritic_modeling.datasets.standard_datasets.load_contextual_gain_datasets",
        spy,
    )
    options = _UnifiedDatasetOptions(
        dataset_name="contextual_gain",
        task_data_path=None,
        parameters={
            "compositional_ood": True,
            "compositional_ood_mode": "region_digit",
            "n_context_latents": 4,
        },
        train_valid_split=0.8,
        flatten=True,
        normalize=False,
        base_dataset="mnist",
        poisson_sampling=True,
        stimulus_duration=1.0,
        multiplicative_gain=True,
        multiscale_gain=False,
        n_groups_per_scale=None,
        fixed_gain_factor=None,
        min_gain_factor=None,
        max_gain_factor=None,
        max_gain_tau_ratio=None,
        uniform_gain=True,
        gain_sampling="log_uniform",
        split_seed=0,
        label_noise_rate=0.0,
        label_noise_seed=0,
    )
    _load_contextual_gain_from_options(options)
    assert captured["compositional_ood"] is True
    assert captured["compositional_ood_mode"] == "region_digit"
    assert captured["n_context_latents"] == 4

    captured.clear()
    options_default = _UnifiedDatasetOptions(
        dataset_name="contextual_gain",
        task_data_path=None,
        parameters={"compositional_ood": True, "n_context_latents": 4},
        train_valid_split=0.8,
        flatten=True,
        normalize=False,
        base_dataset="mnist",
        poisson_sampling=True,
        stimulus_duration=1.0,
        multiplicative_gain=True,
        multiscale_gain=False,
        n_groups_per_scale=None,
        fixed_gain_factor=None,
        min_gain_factor=None,
        max_gain_factor=None,
        max_gain_tau_ratio=None,
        uniform_gain=True,
        gain_sampling="log_uniform",
        split_seed=0,
        label_noise_rate=0.0,
        label_noise_seed=0,
    )
    _load_contextual_gain_from_options(options_default)
    assert captured["compositional_ood_mode"] == "region"

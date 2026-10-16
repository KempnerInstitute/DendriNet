"""Allocation, initialization, and whole-model contracts for opt-in readouts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
import torch
from torch import nn

from dendritic_modeling.networks.benchmarks.parameter_budget import (
    FAMILIES,
    _IndexedAffineReadout,
    build_model,
    match_parameter_budget,
    model_report,
)


@pytest.fixture(autouse=True, scope="module")
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _spec(family="dense", **overrides):
    return {
        "family": family,
        "input_dim": 8,
        "output_dim": 3,
        "width": 8,
        "branch_factors": [2],
        "contacts_e": 4,
        "contacts_i": (
            2
            if family in {"dendritic_additive", "dendritic_shunting", "grouped_gated"}
            else 0
        ),
        "seed": 11,
        "network_depth": 1,
        **overrides,
    }


def _readout(kind, **overrides):
    return {
        "kind": kind,
        **({"topk": 3} if kind == "indexed" else {}),
        **({"rank": 2} if kind == "low_rank" else {}),
        **overrides,
    }


def _state_sha(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# Captured from the unmodified models.py before this extension, CPU FP32.
# Both complete checkpoint state and complete pre-forward report must remain
# identical, including absence of newly introduced default/receipt fields.
# The dendritic reports were re-captured on 2026-10-08, when the retired
# transfer routing keys and the biological_neuron declaration left their
# recorded production_config; their state hashes did not change.
_LEGACY = [
    (
        "dense",
        1,
        "488059c7f98ea886e9f83326d8bd53dca38c515818ac18307fb11258c5253a0d",
        "68d132dcc148c3863060e65bd04f1c90c58f579239ccd499ed1bb4839c97478f",
    ),
    (
        "dense",
        2,
        "a12b056a57105dfdbf070608ef9fd3896c2f8b371441a5d173b0a4839aef87c9",
        "f41d26a66e5e74c863f348cae480b974ce9e4da84c766923284e4d821cc6c8bd",
    ),
    (
        "sparse",
        1,
        "4eebcb86b404c84e6c68871869479546d0469992d5722e8106fb42b202712dd0",
        "f51f0d67a7e82a0243b2958b71168c83f3c807d9965cb753625a710ed17bdb5e",
    ),
    (
        "sparse",
        2,
        "5079601343ce5ef5a7e9341f4ef4d4d801d4a24989fc097bc6c10d1f22519213",
        "e6f5d093deea93b4951e13dd535b6c0596391c9dd3e6959fb982ecbaee4a7ba1",
    ),
    (
        "dendritic_signed",
        1,
        "fac8d65d3157774598b134b481d81582e9070c17ed0318c5a7b8e09f4faf4d47",
        "cfca30e042d6bf464b8df689955e6d88f614d40f9bf3adc9c69345036258a8ec",
    ),
    (
        "dendritic_signed",
        2,
        "51195a7ab4d453763c62f5890dc78ace0d720cc6dbe123d0068a5674eddb5935",
        "0a8eebfb87a441c613f494aabd0807085c744de636d4221c47f5fbededf1a25b",
    ),
    (
        "dendritic_additive",
        1,
        "13857edc6c843c3ca37dbd1af5955c75282ff524a7a757158ab9f0f8238ae58c",
        "d69d64c3d586199e4ffa3b818f17fdc08cb3eb9ea661b90724ed22496ff5e570",
    ),
    (
        "dendritic_additive",
        2,
        "fa18779ea2bb7928bc5265f79c8c6646ae3dd94d445662416c4ab4612149a291",
        "35ceaed63048b625d71aae8444537d97b30d0ce073ede241bae35a2d13ecdee1",
    ),
    (
        "dendritic_shunting",
        1,
        "fe91d49949422b20af38b0d5c409730469ef8a7f3eb59dc6753cc2b9e59854b7",
        "2962b04311fc1fd7cf221a118080c2ca6804795441fb551dbd9f22118b1e18f3",
    ),
    (
        "dendritic_shunting",
        2,
        "b58413821090d7cd41a85b437cc0208af93ff0bc1d07a812ffbc12cc62bb67b7",
        "4a5eefa89b0f16f01205a3518217f116ff12ecbc223a582461963e2a108c0fea",
    ),
    (
        "grouped_gated",
        1,
        "6570f1a62df9174c5caa37a3bec27c9e4b85b0186c746dd21288ce919c50908b",
        "31d10628b57d2f4ede81527c577b14456984cb05afceefb654450eb9a7d0d8a8",
    ),
    (
        "grouped_gated",
        2,
        "e57e2937d90f316940829d3492f88bd992615a17ca9aae9bea93890b353e03ce",
        "0f72e298f769b8200f9d5108114bbd0e255fd315a128c85b6a9b1cbb0149ba01",
    ),
]


@pytest.mark.parametrize("family,depth,state_sha,report_sha", _LEGACY)
def test_omitted_readout_preserves_exact_legacy_state_and_report(
    family, depth, state_sha, report_sha
):
    model = build_model(_spec(family, network_depth=depth))
    assert type(model.readout) is nn.Linear
    assert _state_sha(model) == state_sha
    report = model_report(model)
    # Anatomy is an additive report field; the historical fields and model
    # state remain byte-for-byte unchanged by the observed inventory.
    inventory = report.pop("dendrinet_inventory", None)
    assert (inventory is not None) == family.startswith("dendritic_")
    # Format 4 records the previously implicit exploration contract. Check
    # that declaration separately; keep all numerical/report goldens intact.
    for layer in report["layer_receipts"]:
        if "production_config" in layer:
            assert layer["production_config"].pop("biological_neuron") is False
    assert (
        hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        == report_sha
    )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ["dense", "indexed", "low_rank"])
@pytest.mark.parametrize("bias", [False, True])
def test_all_core_readout_combinations_count_train_and_roundtrip(family, kind, bias):
    spec = _spec(family, readout=_readout(kind, bias=bias))
    original = deepcopy(spec)
    rng = torch.random.get_rng_state().clone()
    model = build_model(spec)
    assert spec == original
    assert torch.equal(rng, torch.random.get_rng_state())
    report = model_report(model)
    weight_count = {"dense": 8 * 3, "indexed": 3 * 3, "low_rank": 2 * (8 + 3)}[kind]
    expected = weight_count + (3 if bias else 0)
    assert report["readout_parameters"] == expected
    assert report["total_parameters"] == sum(p.numel() for p in model.parameters())
    assert report["total_parameters"] == report["hidden_parameters"] + expected
    assert report["trainable_parameters"] == report["total_parameters"]
    receipt = report["readout"]
    assert receipt["total_parameters"] == expected
    assert receipt["parameter_bytes"] == expected * 4
    assert receipt["buffer_bytes"] == sum(
        b.numel() * b.element_size() for b in model.readout.buffers()
    )
    assert receipt["weight_transform"] == "identity"
    assert receipt["parameter_seed"] == 11
    if kind == "indexed":
        assert isinstance(model.readout, _IndexedAffineReadout)
        indices = model.readout.projection.connection_indices
        assert receipt["referenced_input_columns"] == indices.unique().numel()
        assert receipt["unreferenced_input_columns"] == 8 - indices.unique().numel()
        assert receipt["topology_seed"] == 11
        assert (
            report["indexed_projection_backends"]["readout.projection"]["requested"]
            == "eager"
        )
        assert report["support_policy"] == "static_indexed"
    elif kind == "low_rank":
        assert type(model.readout) is nn.Sequential
        assert len(model.readout) == 2 and all(
            type(m) is nn.Linear for m in model.readout
        )
        assert model.readout[0].bias is None
    else:
        assert type(model.readout) is nn.Linear
    x = torch.randn(9, 8, generator=torch.Generator().manual_seed(47))
    output = model(x)
    assert output.shape == (9, 3) and torch.isfinite(output).all()
    nn.functional.cross_entropy(output, torch.arange(9) % 3).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert any(p.grad.abs().sum() > 0 for p in model.hidden.parameters())
    assert all(p.grad.abs().sum() > 0 for p in model.readout.parameters())
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert model_report(model)["topology_sha256"] == report["topology_sha256"]
    clone = build_model(report["resolved_spec"])
    clone.load_state_dict(model.state_dict(), strict=True)
    assert _state_sha(clone) == _state_sha(model)
    torch.testing.assert_close(clone(x), model(x), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["dense", "indexed", "low_rank"])
def test_readout_parameter_seed_is_independent_of_core_rng_and_support(kind):
    readout = _readout(kind, parameter_seed=59)
    if kind == "indexed":
        readout["topology_seed"] = 61
    first = build_model(_spec("dense", readout=readout, topology_seed=67))
    different_core = build_model(
        _spec("grouped_gated", readout=readout, topology_seed=71)
    )
    different_core_seed = build_model(
        _spec("dense", readout=readout, seed=73, topology_seed=67)
    )
    assert _state_sha(first.readout) == _state_sha(different_core.readout)
    assert _state_sha(first.readout) == _state_sha(different_core_seed.readout)
    assert _state_sha(first.hidden) != _state_sha(different_core_seed.hidden)
    different_head = build_model(
        _spec("dense", readout={**readout, "parameter_seed": 79}, topology_seed=67)
    )
    assert _state_sha(first.hidden) == _state_sha(different_head.hidden)
    assert _state_sha(first.readout) != _state_sha(different_head.readout)
    assert (
        model_report(first)["readout"]["topology_sha256"]
        == model_report(different_head)["readout"]["topology_sha256"]
    )
    if kind == "indexed":
        different_support = build_model(
            _spec("dense", readout={**readout, "topology_seed": 83}, topology_seed=67)
        )
        assert _state_sha(first.hidden) == _state_sha(different_support.hidden)
        assert not torch.equal(
            first.readout.projection.connection_indices,
            different_support.readout.projection.connection_indices,
        )
        assert all(
            torch.equal(p, q)
            for p, q in zip(
                first.readout.parameters(), different_support.readout.parameters()
            )
        )


@pytest.mark.parametrize("kind", ["dense", "indexed", "low_rank"])
def test_inherited_seeds_stay_inherited_in_normalized_spec(kind):
    model = build_model(_spec(readout=_readout(kind), topology_seed=97))
    resolved = model_report(model)["resolved_spec"]
    assert "parameter_seed" not in resolved["readout"]
    assert "topology_seed" not in resolved["readout"]
    changed = build_model({**resolved, "seed": 101})
    assert model_report(changed)["readout"]["parameter_seed"] == 101
    assert _state_sha(model.readout) != _state_sha(changed.readout)
    if kind == "indexed":
        assert model_report(changed)["readout"]["topology_seed"] == 97
        assert torch.equal(
            model.readout.projection.connection_indices,
            changed.readout.projection.connection_indices,
        )


@pytest.mark.parametrize("backend", ["eager", "recompute"])
def test_indexed_readout_matches_scattered_dense_affine_forward_and_backward(backend):
    model = build_model(_spec(readout=_readout("indexed", projection_backend=backend)))
    layer = model.readout
    projection = layer.projection
    with torch.no_grad():
        layer.bias.copy_(torch.tensor([0.4, -0.2, 0.8]))
    weight = (
        torch.zeros(3, 8)
        .scatter(1, projection.connection_indices, projection.pre_w.detach())
        .requires_grad_()
    )
    bias = layer.bias.detach().clone().requires_grad_()
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(103)).requires_grad_()
    dense_x = x.detach().clone().requires_grad_()
    actual = layer(x)
    reference = nn.functional.linear(dense_x, weight, bias)
    torch.testing.assert_close(actual, reference)
    actual.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(x.grad, dense_x.grad)
    torch.testing.assert_close(
        projection.pre_w.grad, weight.grad.gather(1, projection.connection_indices)
    )
    torch.testing.assert_close(layer.bias.grad, bias.grad)
    receipt = model_report(model)["indexed_projection_backends"]["readout.projection"]
    assert receipt == {"requested": backend, "resolved": backend, "device": "cpu"}


def test_readout_backend_changes_neither_support_nor_parameters():
    eager = build_model(_spec(readout=_readout("indexed")))
    recompute = build_model(
        _spec(readout=_readout("indexed", projection_backend="recompute"))
    )
    assert _state_sha(eager) == _state_sha(recompute)
    assert (
        model_report(eager)["topology_sha256"]
        == model_report(recompute)["topology_sha256"]
    )
    assert model_report(recompute)["projection_backend"] == "eager"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ["dense", "indexed", "low_rank"])
def test_exact_whole_model_budget_matching_respects_readout_feasibility(family, kind):
    spec = _spec(family, width=17, readout=_readout(kind))
    target = sum(p.numel() for p in build_model(spec).parameters())
    search = {**spec, "width": 1}
    before = deepcopy(search)
    matched, report = match_parameter_budget(search, target, tolerance=0)
    assert search == before
    assert matched["width"] == 17
    assert report["total_parameters"] == target
    assert report["readout"]["total_parameters"] == report["readout_parameters"]
    assert sum(p.numel() for p in build_model(matched).parameters()) == target
    assert report["budget_matched"] and report["budget_relative_error"] == 0


@pytest.mark.parametrize(
    "readout",
    [
        None,
        {},
        {"kind": "dynamic"},
        {"kind": []},
        {"kind": "dense", "topk": 2},
        {"kind": "dense", "rank": 2},
        {"kind": "dense", "topology_seed": 3},
        {"kind": "dense", "projection_backend": "eager"},
        {"kind": "dense", "weight_transform": "softplus"},
        {"kind": "dense", "bias": 1},
        {"kind": "dense", "parameter_seed": True},
        {"kind": "dense", "parameter_seed": -1},
        {"kind": "dense", "init_std": 0},
        {"kind": "dense", "init_std": float("nan")},
        {"kind": "dense", "init_std": float("inf")},
        {"kind": "dense", "init_std": True},
        {"kind": "dense", "init_std": "0.02"},
        {"kind": "indexed"},
        {"kind": "indexed", "topk": 0},
        {"kind": "indexed", "topk": True},
        {"kind": "indexed", "topk": 2.0},
        {"kind": "indexed", "topk": 9},
        {"kind": "indexed", "topk": 2, "rank": 1},
        {"kind": "indexed", "topk": 2, "topology_seed": False},
        {"kind": "indexed", "topk": 2, "projection_backend": None},
        {"kind": "indexed", "topk": 2, "projection_backend": "unknown"},
        {"kind": "low_rank"},
        {"kind": "low_rank", "rank": True},
        {"kind": "low_rank", "rank": 0},
        {"kind": "low_rank", "rank": 4},
        {"kind": "low_rank", "rank": 1, "topk": 2},
    ],
)
def test_invalid_or_unsupported_readout_settings_fail_closed(readout):
    with pytest.raises(ValueError):
        build_model(_spec(readout=readout))


@pytest.mark.parametrize(
    "readout", [{"kind": "indexed", "topk": 3}, {"kind": "low_rank", "rank": 2}]
)
def test_readout_budget_is_not_silently_clipped(readout):
    with pytest.raises(ValueError):
        build_model(_spec(width=1, clip_contacts=True, readout=readout))
    with pytest.raises(ValueError, match="No width matches"):
        match_parameter_budget(_spec(width=1, clip_contacts=True, readout=readout), 1)


def test_sparse_readout_reports_unreferenced_core_outputs_without_dropping_costs():
    model = build_model(
        _spec(width=32, output_dim=1, readout={"kind": "indexed", "topk": 1})
    )
    report = model_report(model)
    assert report["readout"]["referenced_input_columns"] == 1
    assert report["readout"]["unreferenced_input_columns"] == 31
    assert report["hidden_parameters"] == 32 * (16 + 1)
    assert report["readout_parameters"] == 2
    assert report["total_parameters"] == 546


@pytest.mark.parametrize("kind", ["dense", "indexed", "low_rank"])
def test_init_std_controls_every_readout_factor_and_bias_starts_zero(kind):
    first = build_model(_spec(readout=_readout(kind, init_std=0.02)))
    second = build_model(_spec(readout=_readout(kind, init_std=0.06)))
    assert _state_sha(first.hidden) == _state_sha(second.hidden)
    for (name, a), (other_name, b) in zip(
        first.readout.named_parameters(), second.readout.named_parameters()
    ):
        assert name == other_name
        if name.endswith("bias"):
            assert torch.count_nonzero(a) == 0 and torch.count_nonzero(b) == 0
        else:
            torch.testing.assert_close(b, a * 3)

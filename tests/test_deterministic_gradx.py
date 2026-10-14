"""Deterministic-by-default grad_x for the transposed Triton indexed gather.

The two-stage sorted-CSR backward is the default grad_x of
``triton_indexed_gather_transposed`` (backend ``triton_transposed``);
``deterministic_input_gradient=False`` (backend ``triton_transposed_atomic``)
selects the atomic scatter. CPU-safe tests cover the CSR setup cache
(including in-place rewire invalidation, and the entry/byte caps under the
FSDP unshard/reshard churn pattern: same index content, fresh tensor objects
every step) and the backend wiring; the CUDA tests assert eager grad parity,
bit-identity across runs (including under churn with forced eviction), and the
atomic variant.
"""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    indexed_common,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
    triton_indexed_gather_transposed as tigt,
)


@pytest.mark.parametrize(
    "backend, deterministic",
    [("triton_transposed", True), ("triton_transposed_atomic", False)],
)
def test_each_transposed_backend_selects_its_input_gradient(
    monkeypatch, backend, deterministic
):
    calls = []

    def record(x, idx, w, *, chunk_size, deterministic_input_gradient):
        calls.append(deterministic_input_gradient)
        return x.new_zeros(x.shape[0], idx.shape[0])

    monkeypatch.setattr(tigt, "triton_sparse_gather", record)
    x = torch.rand(3, 5)
    idx = torch.zeros(4, 2, dtype=torch.int32)
    indexed_common._dispatch_indexed_sparse_projection(
        flat_x=x,
        weight=torch.rand(4, 2),
        connection_indices=idx,
        output_chunk_size=8,
        projection_backend=backend,
        backend_is_resolved=True,
        block_layout=None,
        auto_requested=False,
    )
    assert calls == [deterministic]


# ---------------------------------------------------------------------------
# CPU-safe: CSR setup cache semantics and rewire invalidation


def test_gradx_setup_matches_csr_reference():
    torch.manual_seed(0)
    IN, n_out, K = 41, 64, 9
    idx = torch.randint(0, IN, (n_out, K), dtype=torch.int32)
    idx[:, -1] = idx[:, 0]  # duplicates within rows
    perm, o_sorted, row_ptr = tigt._gradx_setup(idx, idx, IN)

    flat = idx.reshape(-1).long()
    assert torch.equal(flat[perm], flat[perm].sort().values)
    counts = torch.bincount(flat, minlength=IN)
    expected_ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    assert torch.equal(row_ptr.long(), expected_ptr)
    assert torch.equal(o_sorted.long(), perm // K)


def test_gradx_setup_cache_hits_and_invalidates_on_inplace_rewire():
    torch.manual_seed(1)
    IN, n_out, K = 23, 40, 5
    idx = torch.randint(0, IN, (n_out, K), dtype=torch.int32)

    before = tigt._gx_setup_misses
    first = tigt._gradx_setup(idx, idx, IN)
    assert tigt._gx_setup_misses == before + 1
    again = tigt._gradx_setup(idx, idx, IN)
    assert tigt._gx_setup_misses == before + 1, "second call must hit the cache"
    assert again[0] is first[0]

    # apply_rewiring writes connection_indices[rows, slots] = ... in place;
    # that bumps the tensor version and MUST invalidate the CSR entry.
    idx[0, 0] = (int(idx[0, 0]) + 1) % IN
    rewired = tigt._gradx_setup(idx, idx, IN)
    assert tigt._gx_setup_misses == before + 2, "rewire must miss the cache"
    counts = torch.bincount(idx.reshape(-1).long(), minlength=IN)
    expected_ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    assert torch.equal(rewired[2].long(), expected_ptr)

    # a distinct same-shape tensor gets its own entry, not a stale hit
    other = torch.randint(0, IN, (n_out, K), dtype=torch.int32)
    other_setup = tigt._gradx_setup(other, other, IN)
    assert tigt._gx_setup_misses == before + 3
    assert torch.equal(
        other.reshape(-1).long()[other_setup[0]],
        other.reshape(-1).long()[other_setup[0]].sort().values,
    )


# ---------------------------------------------------------------------------
# CPU-safe: bounded cache under FSDP tensor churn (entry cap + byte cap)


@pytest.fixture()
def _isolated_gx_cache():
    """Run against an empty cache; restore the shared module state after."""
    saved_cache = dict(tigt._GX_SETUP_CACHE)
    saved_bytes = tigt._gx_setup_cache_nbytes
    saved_misses = tigt._gx_setup_misses
    tigt._GX_SETUP_CACHE.clear()
    tigt._gx_setup_cache_nbytes = 0
    yield
    tigt._GX_SETUP_CACHE.clear()
    tigt._GX_SETUP_CACHE.update(saved_cache)
    tigt._gx_setup_cache_nbytes = saved_bytes
    tigt._gx_setup_misses = saved_misses


def _tracked_bytes_consistent() -> bool:
    return tigt._gx_setup_cache_nbytes == sum(
        entry[3] for entry in tigt._GX_SETUP_CACHE.values()
    )


def test_gx_setup_cache_bounded_under_fsdp_tensor_churn(
    _isolated_gx_cache, monkeypatch
):
    """FSDP unshard/reshard recreates tensors every step: same content, fresh
    objects (fresh data_ptr -> fresh key). The cache must stay under BOTH caps
    for arbitrarily many steps while every returned setup stays bit-identical
    to the unbounded computation."""
    monkeypatch.setattr(tigt, "_GX_SETUP_CACHE_MAX", 8)
    torch.manual_seed(7)
    IN, n_out, K = 37, 48, 6
    modules = [torch.randint(0, IN, (n_out, K), dtype=torch.int32) for _ in range(4)]
    # Unbounded-behaviour reference setups, computed once per module.
    refs = []
    for idx in modules:
        refs.append(tuple(t.clone() for t in tigt._gradx_setup(idx, idx, IN)))
    tigt._GX_SETUP_CACHE.clear()
    tigt._gx_setup_cache_nbytes = 0

    misses_before = tigt._gx_setup_misses
    steps = 50
    for _ in range(steps):
        for idx, ref in zip(modules, refs):
            fresh = idx.clone()  # fresh tensor object, unchanged content
            setup = tigt._gradx_setup(fresh, fresh, IN)
            for got, want in zip(setup, ref):
                assert torch.equal(
                    got, want
                ), "churned setup must be bit-identical to the unbounded one"
            stats = tigt.gx_setup_cache_stats()
            assert stats["entries"] <= stats["entry_cap"]
            assert stats["bytes"] <= stats["byte_cap"]
            assert _tracked_bytes_consistent()
    # Every churned call is a miss (new data_ptr): bounded MEMORY was bought
    # without pretending to add hits -- exactly the measured 0%-hit regime.
    assert tigt._gx_setup_misses - misses_before == steps * len(modules)


def test_gx_setup_cache_byte_cap_evicts_lru_and_keeps_hits(
    _isolated_gx_cache, monkeypatch
):
    torch.manual_seed(8)
    IN, n_out, K = 19, 32, 4
    a = torch.randint(0, IN, (n_out, K), dtype=torch.int32)
    b = torch.randint(0, IN, (n_out, K), dtype=torch.int32)
    c = torch.randint(0, IN, (n_out, K), dtype=torch.int32)
    entry = tigt._gx_entry_nbytes(a, IN)
    # Room for exactly two entries.
    monkeypatch.setattr(tigt, "_GX_SETUP_CACHE_MAX_BYTES", 2 * entry)

    tigt._gradx_setup(a, a, IN)
    setup_b = tigt._gradx_setup(b, b, IN)
    assert tigt.gx_setup_cache_stats()["entries"] == 2
    misses = tigt._gx_setup_misses
    tigt._gradx_setup(c, c, IN)  # evicts a (least recent), keeps b
    stats = tigt.gx_setup_cache_stats()
    assert stats["entries"] == 2 and stats["bytes"] <= stats["byte_cap"]
    assert _tracked_bytes_consistent()
    assert tigt._gx_setup_misses == misses + 1
    again_b = tigt._gradx_setup(b, b, IN)
    assert tigt._gx_setup_misses == misses + 1, "b must still hit after c"
    assert again_b[0] is setup_b[0]
    tigt._gradx_setup(a, a, IN)
    assert tigt._gx_setup_misses == misses + 2, "a was evicted and must miss"
    assert tigt.gx_setup_cache_stats()["bytes"] <= 2 * entry


def test_gx_setup_cache_oversized_entry_kept_alone(_isolated_gx_cache, monkeypatch):
    torch.manual_seed(9)
    IN, n_out, K = 11, 16, 3
    small = torch.randint(0, IN, (n_out, K), dtype=torch.int32)
    big = torch.randint(0, IN, (4 * n_out, K), dtype=torch.int32)
    monkeypatch.setattr(
        tigt, "_GX_SETUP_CACHE_MAX_BYTES", tigt._gx_entry_nbytes(big, IN) - 1
    )
    tigt._gradx_setup(small, small, IN)
    setup_big = tigt._gradx_setup(big, big, IN)  # over-cap: evicts all, kept alone
    stats = tigt.gx_setup_cache_stats()
    assert stats["entries"] == 1
    assert stats["bytes"] == tigt._gx_entry_nbytes(big, IN)
    assert _tracked_bytes_consistent()
    misses = tigt._gx_setup_misses
    again = tigt._gradx_setup(big, big, IN)  # its ONE live entry still hits
    assert tigt._gx_setup_misses == misses
    assert again[0] is setup_big[0]


def test_gx_setup_cache_stats_shape(_isolated_gx_cache):
    stats = tigt.gx_setup_cache_stats()
    assert stats["entries"] == 0 and stats["bytes"] == 0
    assert stats["entry_cap"] >= 1 and stats["byte_cap"] >= 1
    idx = torch.randint(0, 13, (8, 3), dtype=torch.int32)
    tigt._gradx_setup(idx, idx, 13)
    stats = tigt.gx_setup_cache_stats()
    assert stats["entries"] == 1
    assert stats["bytes"] == tigt._gx_entry_nbytes(idx, 13)
    assert stats["misses"] == tigt._gx_setup_misses


# ---------------------------------------------------------------------------
# CUDA: eager parity, bit-identity, and the atomic opt-out


def _eager_grads(x, idx, w, go):
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    out = (x[:, idx.long()] * w.unsqueeze(0)).sum(dim=-1)
    out.backward(go)
    return x.grad, w.grad


def _triton_grads(x, idx, w, go, deterministic=True):
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    out = tigt.triton_sparse_gather(
        x, idx, w, deterministic_input_gradient=deterministic
    )
    out.backward(go)
    return x.grad, w.grad


@pytest.mark.requires_gpu
def test_deterministic_backward_parity_bitstability_and_optout(monkeypatch):
    if not tigt.triton_gather_available():
        pytest.skip("Triton gather backend is unavailable")
    torch.manual_seed(0)
    dev = "cuda"
    M, IN, n_out, K = 33, 130, 512, 17
    x = torch.randn(M, IN, device=dev)
    w = 0.1 * torch.randn(n_out, K, device=dev)
    idx = torch.randint(0, IN, (n_out, K), device=dev, dtype=torch.int32)
    idx[:, -1] = idx[:, 0]  # duplicates within rows
    idx[0] = 0  # one massively hot input
    go = torch.randn(M, n_out, device=dev)

    # (a) grad parity vs eager on the DEFAULT (deterministic) path
    det_calls = []
    real_det = tigt._deterministic_gradx

    def counting_det(*args, **kwargs):
        result = real_det(*args, **kwargs)
        det_calls.append(result is not None)
        return result

    monkeypatch.setattr(tigt, "_deterministic_gradx", counting_det)
    gx, gw = _triton_grads(x, idx, w, go)
    assert det_calls == [True], "default backward must take the deterministic path"
    eager_gx, eager_gw = _eager_grads(x, idx, w, go)
    torch.testing.assert_close(gx, eager_gx, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gw, eager_gw, rtol=1e-4, atol=1e-4)

    # (b) bit-identity across 3 runs by default
    repeats = [_triton_grads(x, idx, w, go) for _ in range(3)]
    for rep_gx, rep_gw in repeats[1:]:
        assert torch.equal(repeats[0][0], rep_gx), "grad_x must be bit-identical"
        assert torch.equal(repeats[0][1], rep_gw), "grad_w must be bit-identical"
    assert torch.equal(repeats[0][0], gx)

    # rewire invalidation end-to-end: in-place index mutation must produce
    # the new topology's gradient (fresh CSR setup), still matching eager.
    idx[1, 0] = (int(idx[1, 0].item()) + 1) % IN
    gx_new, _ = _triton_grads(x, idx, w, go)
    eager_gx_new, eager_gw_new = _eager_grads(x, idx, w, go)
    torch.testing.assert_close(gx_new, eager_gx_new, rtol=1e-4, atol=1e-4)
    assert not torch.equal(gx_new, gx), "rewired topology must change grad_x"

    # (c) the atomic variant works and skips the deterministic path
    monkeypatch.setattr(
        tigt,
        "_deterministic_gradx",
        lambda *a, **k: pytest.fail(
            "the atomic input gradient must not call the deterministic path"
        ),
    )
    gx_atomic, gw_atomic = _triton_grads(x, idx, w, go, deterministic=False)
    torch.testing.assert_close(gx_atomic, eager_gx_new, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gw_atomic, eager_gw_new, rtol=1e-4, atol=1e-4)


@pytest.mark.requires_gpu
def test_gradx_bit_identical_under_fsdp_churn_with_eviction(
    _isolated_gx_cache, monkeypatch
):
    """End-to-end FSDP pattern on the real backward: fresh index tensor
    objects every step (unchanged content) with the entry cap forced BELOW
    the module count, so every step misses and evicts. grad_x must stay
    bit-identical to the first (cache-fresh) computation for every module,
    and the cache must stay under both caps for the whole run."""
    if not tigt.triton_gather_available():
        pytest.skip("Triton gather backend is unavailable")
    monkeypatch.setattr(tigt, "_GX_SETUP_CACHE_MAX", 2)
    torch.manual_seed(3)
    dev = "cuda"
    M, IN, n_out, K = 64, 256, 384, 12
    x = torch.randn(M, IN, device=dev)
    go = torch.randn(M, n_out, device=dev)
    modules = []
    for _ in range(3):  # 3 "cells" against an entry cap of 2 -> forced churn
        idx = torch.randint(0, IN, (n_out, K), device=dev, dtype=torch.int32)
        w = 0.1 * torch.randn(n_out, K, device=dev)
        modules.append((idx, w))
    refs = [_triton_grads(x, idx, w, go)[0] for idx, w in modules]
    for step in range(10):
        for (idx, w), ref in zip(modules, refs):
            fresh = idx.clone()  # FSDP unshard: new object, same content
            gx, _ = _triton_grads(x, fresh, w, go)
            assert torch.equal(gx, ref), f"grad_x changed under churn at step {step}"
            stats = tigt.gx_setup_cache_stats()
            assert stats["entries"] <= 2
            assert stats["bytes"] <= stats["byte_cap"]
            assert _tracked_bytes_consistent()

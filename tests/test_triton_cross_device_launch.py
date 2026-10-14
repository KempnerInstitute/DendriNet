"""Triton kernels must launch on their tensors' device, not the current one.

Regression test for the 2026-08-20 cross-device race: with the process's
current device at cuda:0, a module living on cuda:1 had its Triton kernels
enqueued on cuda:0's stream with pointers into cuda:1 memory (legal via
UVA, silently racing every cuda:1 consumer). Training on non-zero devices
produced deterministic stale-read forward values and nondeterministic
divergence; frozen (ATen-path) evaluation was unaffected, so only a
multi-device training test can catch it.
"""

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedSparseLinear,
)

pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2+ CUDA devices"),
]


def _train_loss_curve(device, seed=0, steps=40):
    torch.manual_seed(seed)
    torch.cuda.set_device("cuda:0")  # the bug requires current != module dev
    lin = IndexedSparseLinear(
        in_features=512,
        out_features=256,
        K=64,
        seed=seed,
        weight_transform="identity",
        projection_backend="triton_transposed",
        output_chunk_size=128,
    ).to(device)
    target_w = torch.randn(512, 256, device=device) / 512**0.5
    opt = torch.optim.Adam(lin.parameters(), lr=3e-3)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for _ in range(steps):
        x = torch.randn(128, 512, generator=gen).to(device)
        y = x @ target_w
        pred = lin(x)
        loss = (pred - y).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    torch.cuda.synchronize(device)
    return losses


def test_training_on_second_device_matches_first():
    l0 = _train_loss_curve("cuda:0")
    l1 = _train_loss_curve("cuda:1")
    # Primary race detector: identical seeds and data must give matching curves
    # (the bug gave deterministically-wrong step-0 values then divergence).
    for a, b in zip(l0, l1):
        assert abs(a - b) < 5e-3 * max(
            1.0, abs(a)
        ), f"cross-device loss mismatch: cuda:0 {a} vs cuda:1 {b}"
    # Secondary sanity: training makes progress under a deliberately loose
    # bound; the race produced increasing loss.
    assert l1[-1] < 0.9 * l1[0], f"cuda:1 loss did not descend: {l1[0]} -> {l1[-1]}"


def test_forward_values_match_across_devices():
    torch.manual_seed(1)
    torch.cuda.set_device("cuda:0")
    x = torch.randn(64, 512)
    mods = {}
    outs = {}
    for dev in ("cuda:0", "cuda:1"):
        torch.manual_seed(1)
        m = IndexedSparseLinear(
            in_features=512,
            out_features=256,
            K=64,
            seed=1,
            weight_transform="identity",
            projection_backend="triton_transposed",
            output_chunk_size=128,
        ).to(dev)
        m.train()  # trainable path = the Triton path
        for p in m.parameters():
            p.requires_grad_(True)
        outs[dev] = m(x.to(dev)).detach().cpu()
        mods[dev] = m
    torch.cuda.synchronize()
    assert torch.allclose(
        outs["cuda:0"], outs["cuda:1"], atol=1e-5
    ), "trainable forward differs across devices"

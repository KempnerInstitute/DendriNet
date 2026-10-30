"""Explicit core seeds use CPU draws regardless of the caller's default device."""

from contextlib import contextmanager
from dataclasses import asdict

import pytest
import torch

from dendritic_modeling.config import RoutedEIConfig
from dendritic_modeling.networks.architectures.excitation_inhibition.routed_ei import (
    build_routed_ei,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


@contextmanager
def cuda_default(method):
    if method == "context":
        with torch.device("cuda"):
            yield
    else:
        previous = torch.get_default_device()
        try:
            torch.set_default_device("cuda")
            yield
        finally:
            torch.set_default_device(previous)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("method", ["context", "set_default_device"])
def test_keyed_initialization_preserves_cpu_and_cuda_streams(method):
    raw = {
        "routed_ei": asdict(
            RoutedEIConfig(features=4, branches=2, branch_activation="tanh")
        ),
        "initialization_seed": 41,
    }
    reference = get_architecture("routed_ei", raw)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    with cuda_default(method):
        first = get_architecture("routed_ei", raw)
        second = get_architecture("routed_ei", raw)
        assert torch.empty(0).device.type == "cuda"
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        for name, value in reference.state_dict().items():
            for model in (first, second):
                actual = model.state_dict()[name]
                assert actual.device.type == "cpu"
                torch.testing.assert_close(actual, value, rtol=0, atol=0)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("method", ["context", "set_default_device"])
def test_unseeded_construction_keeps_default_device_and_global_draws(method):
    config = RoutedEIConfig(features=4, branches=2)
    with (
        torch.random.fork_rng(devices=[torch.cuda.current_device()]),
        cuda_default(method),
    ):
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        actual = get_architecture("routed_ei", {"routed_ei": asdict(config)})
        after = torch.cuda.get_rng_state().clone()
        assert not torch.equal(after, cuda_rng)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        torch.cuda.set_rng_state(cuda_rng)
        expected = build_routed_ei(config)
        assert torch.equal(after, torch.cuda.get_rng_state())
        for name, value in actual.state_dict().items():
            assert value.device.type == "cuda"
            torch.testing.assert_close(
                value, expected.state_dict()[name], rtol=0, atol=0
            )

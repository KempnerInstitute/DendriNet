"""Regression coverage for side-by-side indexed-kernel comparisons."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.scripts.sweeps.unified.sweep_types.unified_sweep import (
    UnifiedSweepGenerator,
)


COMPARISON_BACKENDS = {
    "control_eager",
    "iss_recompute",
    "iss_triton",
    "iss_cuda",
    "iss_auto",
    "ryan_fused",
    "ryan_triton",
    "ryan_cuda",
    "ryan_hybrid",
    "ryan_auto",
}
EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "configs" / "examples" / "indexed_backends"


def _run_cpu_backend(backend: str, state: dict[str, torch.Tensor], value: torch.Tensor):
    indices = torch.tensor([[0, 2, 4], [1, 3, 5], [0, 3, 4], [2, 4, 5]])
    layer = IndexedSparseLinear(
        6,
        4,
        3,
        connection_indices=indices,
        weight_transform="identity",
        comparison_backend=backend,
    )
    layer.load_state_dict(state)
    input_value = value.detach().clone().requires_grad_(True)
    output = layer(input_value)
    output.square().mean().backward()
    return (
        output.detach(),
        input_value.grad.detach(),
        layer.pre_w.grad.detach(),
        layer._last_resolved_projection_backend,
    )


def test_portable_comparison_backends_match_the_eager_control():
    torch.manual_seed(7)
    indices = torch.tensor([[0, 2, 4], [1, 3, 5], [0, 3, 4], [2, 4, 5]])
    seed_layer = IndexedSparseLinear(
        6,
        4,
        3,
        connection_indices=indices,
        weight_transform="identity",
    )
    state = seed_layer.state_dict()
    value = torch.randn(3, 6)
    control = _run_cpu_backend("control_eager", state, value)

    for backend in ("iss_recompute", "ryan_fused", "ryan_auto"):
        actual = _run_cpu_backend(backend, state, value)
        torch.testing.assert_close(actual[0], control[0])
        torch.testing.assert_close(actual[1], control[1])
        torch.testing.assert_close(actual[2], control[2])

    assert control[3] == "control_eager"


@pytest.mark.parametrize(
    "name",
    (
        "alexnet_fc6_large.yaml",
        "feedforward_large.yaml",
        "rnn_large.yaml",
        "transformer_olmo2_1b_large.yaml",
    ),
)
def test_kernel_comparison_examples_expand_to_all_named_conditions(name: str):
    request = OmegaConf.load(EXAMPLE_DIR / name)
    variants = UnifiedSweepGenerator().generate_configs(request)
    names = {variant._sweep_variant for variant in variants}
    assert names == COMPARISON_BACKENDS
    assert request.sweep_contract.expected_config_count == len(COMPARISON_BACKENDS)

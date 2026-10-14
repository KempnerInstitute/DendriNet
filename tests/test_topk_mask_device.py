"""Top-K masks stay on-device and retain the selected synapses."""

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import (
    VarianceTopKLinear,
)


@pytest.mark.parametrize("layer_type", [TopKLinear, VarianceTopKLinear])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mask_selection_without_cpu_row_indices(monkeypatch, layer_type, device, dtype):
    layer = layer_type(4, 2, 2, weight_transform="softplus").to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        layer.pre_w.copy_(
            torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device=device, dtype=dtype)
        )
        if layer_type is VarianceTopKLinear:
            layer.activation_var.copy_(
                torch.tensor([10, 1, 1, 0.1], device=device, dtype=dtype)
            )
    original_arange = torch.arange

    def on_device_arange(*args, **kwargs):
        if device == "cuda":
            assert torch.device(kwargs.get("device", "cpu")).type == "cuda"
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", on_device_arange)
    expected = [[0, 0, 1, 1], [1, 1, 0, 0]]
    if layer_type is VarianceTopKLinear:
        expected[0] = [1, 0, 1, 0]
    mask = layer.weight_mask()
    torch.testing.assert_close(
        mask, torch.tensor(expected, device=device, dtype=dtype), rtol=0, atol=0
    )
    assert not mask.requires_grad

import torch
import torch.nn as nn

from dendritic_modeling.analysis.utils.tensor_utils import (
    fix_tensor_shapes_for_blocklinear,
    wrap_model_for_analysis,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    EfficientBlockLinear,
)


def test_fix_tensor_shapes_returns_original_mapping_without_efficient_blocklinear():
    tensors = {"activation": torch.randn(2, 3, 4)}
    model = nn.Linear(4, 2)

    fixed = fix_tensor_shapes_for_blocklinear(tensors, model)

    assert fixed is tensors


def test_fix_tensor_shapes_flattens_extra_dims_for_efficient_blocklinear():
    tensors = {
        "activation": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        "already_flat": torch.randn(2, 12),
    }
    model = nn.Sequential(EfficientBlockLinear(in_features=12, out_features=4))

    fixed = fix_tensor_shapes_for_blocklinear(tensors, model)

    assert fixed["activation"].shape == (2, 12)
    torch.testing.assert_close(
        fixed["activation"], tensors["activation"].reshape(2, -1)
    )
    assert fixed["already_flat"] is tensors["already_flat"]


def test_wrap_model_for_analysis_flattens_efficient_blocklinear_inputs_by_batch():
    torch.manual_seed(7)
    layer = EfficientBlockLinear(in_features=12, out_features=4)
    # BlockLinear parameters are intentionally initialized by the owning
    # dendritic layer; initialize explicitly in this standalone wrapper test so
    # it never forwards uninitialized ``torch.empty`` storage.
    layer.initialize()
    wrapped = wrap_model_for_analysis(nn.Sequential(layer))
    x = torch.randn(2, 3, 4)

    actual = wrapped(x)
    expected = layer(x.reshape(2, -1))

    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected)

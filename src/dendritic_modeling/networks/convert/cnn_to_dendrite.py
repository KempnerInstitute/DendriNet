from __future__ import annotations

"""Helpers to convert ordinary *linear* Conv2d kernels into dendritic push–pull
layers so that the forward pass is (almost) equivalent.
"""

from typing import Optional

import torch
from torch import nn

from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d

__all__ = ["conv2d_to_dendrite"]


def conv2d_to_dendrite(
    conv: nn.Conv2d,
    *,
    baseline_inhib: float = 0.1,
    eps: float = 1e-6,
    copy_bias: bool = False,
) -> DendriteConv2d:
    """Return a :class:`DendriteConv2d` whose E/I weights equal the push–pull
    decomposition of ``conv``.

    Parameters
    ----------
    conv
        Trained or un-trained Conv2d to mimic.
    baseline_inhib
        Constant added to inhibitory weights to keep denominator positive.
    eps
        Numerical epsilon added before log.
    copy_bias
        If ``True`` and ``conv.bias`` is not None, the bias term is copied into
        ``layer.bias_E`` (there is no I-bias in the current implementation).
    """
    layer = DendriteConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        bias=copy_bias and (conv.bias is not None),
    )

    W = conv.weight.detach().cpu().numpy()
    E = W.clip(min=0)
    I = (-W).clip(min=0) + baseline_inhib

    with torch.no_grad():
        layer.weight_E_raw.copy_(torch.log(torch.tensor(E) + eps))
        layer.weight_I_raw.copy_(torch.log(torch.tensor(I) + eps))
        if copy_bias and conv.bias is not None:
            layer.bias_E.copy_(conv.bias)
    return layer 
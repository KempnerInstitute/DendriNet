import torch
from torch import nn
import numpy as np

from dendritic_modeling.networks.convert.cnn_to_dendrite import conv2d_to_dendrite


def test_conv2d_similarity():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 5, padding=2, bias=False)
    dend = conv2d_to_dendrite(conv, baseline_inhib=1e-4)

    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        y_conv = conv(x).flatten(1)
        y_dend = dend(x).flatten(1)

    # Pearson correlation per-sample; expect >0.8 on random weights
    corr = []
    for yc, yd in zip(y_conv, y_dend):
        v1 = yc.cpu().numpy().ravel()
        v2 = yd.cpu().numpy().ravel()
        corr.append(np.corrcoef(v1, v2)[0, 1])
    mean_corr = np.mean(corr)

    assert mean_corr > 0.8, f"Mean correlation {mean_corr}" 
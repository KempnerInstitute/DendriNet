"""Frozen ResNet-50 trunk (through avgpool) for the Adi ResNet arm.

WHY THIS EXISTS INSTEAD OF pretrained.py
========================================
The shared `networks/architectures/classical/pretrained.py` path handles ResNet-50's
module structure correctly, but it has two BatchNorm defects that have never mattered
because AlexNet has zero BatchNorm layers:

  1. `_load_backbone` never calls `.eval()` on the freshly constructed backbone, so the
     shape-inference dummy forwards run in train mode. `torch.no_grad()` does NOT stop
     BatchNorm from updating `running_mean` / `running_var`. Measured on a real BN layer:
     0.5 -> 0.45 and 2.0 -> 1.8 per forward, and the `fc` split performs two of them.

  2. `BackboneSegment` sets `requires_grad_(False)` but never overrides `train()`. When the
     trainer calls `model.train()`, a nominally "frozen" ResNet encoder switches BN to
     *batch* statistics and resumes updating its running stats — for all 50 epochs.

Either one silently corrupts the pretrained features, and neither is visible in the loss.
`pretrained.py` is shared by every AlexNet run and is deliberately NOT modified. All ResNet
work goes through this module instead, permanently. Do not "simplify" this back onto the
shared path.

The guard here is `train()`, which ignores its argument and always forces eval, so the
encoder cannot be taken out of eval by the trainer, by DDP, or by `deepcopy`+`load_state_dict`
round-trips.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

RESNET50_OUTPUT_DIM = 2048


class FrozenResNet50Encoder(nn.Module):
    """ResNet-50 conv1..avgpool, flattened to (B, 2048). Frozen and permanently in eval.

    Args:
        trunk: the ResNet-50 children up to and including avgpool.
    """

    def __init__(self, trunk: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.flatten = nn.Flatten(start_dim=1)
        self.output_dim = RESNET50_OUTPUT_DIM

        for p in self.trunk.parameters():
            p.requires_grad_(False)

        # nn.Module.__init__ defaults self.training=True. Put the WHOLE module (not just
        # the trunk) into eval at construction, so `encoder.training` is False before the
        # trainer ever calls train(). Routes through the train() override below.
        self.eval()

    def train(self, mode: bool = True):  # noqa: D102 - intentionally ignores `mode`
        # Always eval. See module docstring: this is the whole point of the class.
        super().train(False)
        self.trunk.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.trunk(x))

    def get_output_dim(self) -> int:
        return self.output_dim


def build_frozen_resnet50(weights: str = "IMAGENET1K_V1") -> tuple[FrozenResNet50Encoder, nn.Linear]:
    """Return (frozen trunk encoder -> 2048, pretrained fc 2048->1000).

    `weights` is pinned by name. Note torchvision's "DEFAULT" for ResNet-50 is V2, a
    *different* checkpoint from the V1 used by the AlexNet arm's convention, so callers
    should pass an explicit enum name rather than "DEFAULT".
    """
    resnet = torchvision.models.resnet50(weights=ResNet50_Weights[weights])
    resnet.eval()

    # children(): conv1, bn1, relu, maxpool, layer1..layer4, avgpool, fc -> drop fc
    trunk = nn.Sequential(*list(resnet.children())[:-1])
    encoder = FrozenResNet50Encoder(trunk)

    # The pretrained 2048->1000 head becomes the trainable decoder, the exact analogue of
    # AlexNet's classifier.6 (fc8) in the k1 runs. Kept OUT of the encoder so its parameters
    # land in the decoder param group (decoder_lr), not the frozen encoder group.
    decoder = resnet.fc
    for p in decoder.parameters():
        p.requires_grad_(True)

    return encoder, decoder

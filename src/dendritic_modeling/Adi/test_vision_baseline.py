"""
Sanity test: build baseline model via same config path as factory, one forward pass, assert [batch, 10].
Also asserts wrapper.output_dim equals actual encoder(x).shape[-1].
Run from repo root: PYTHONPATH=src python -m dendritic_modeling.Adi.test_vision_baseline
"""

import os
import sys

import torch

# Assume run from repo root with PYTHONPATH=src; find configs/vision
# __file__ = .../dendritic-modeling/src/dendritic_modeling/Adi/test_vision_baseline.py -> 3 levels up = repo root
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_CONFIG_PATH = os.path.join(
    _REPO_ROOT, "configs", "vision", "alexnet_cifar10_baseline.yaml"
)


def test_baseline_forward_output_shape():
    if not os.path.isfile(_CONFIG_PATH):
        raise FileNotFoundError(
            f"Config not found: {_CONFIG_PATH}. Run from repo root."
        )
    from dendritic_modeling.config import load_config
    from dendritic_modeling.scripts.script_utils.setup_utils import (
        initialize_model,
    )

    config = load_config(_CONFIG_PATH)
    model, encoder_network = initialize_model(config.model)

    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    out = model(x)
    assert out.shape == (
        batch_size,
        10,
    ), f"Expected output shape ({batch_size}, 10), got {out.shape}"

    # Wrapper output_dim must equal actual encoder feature dim
    enc_out = encoder_network.encoder(x)
    assert encoder_network.output_dim == enc_out.shape[-1], (
        f"wrapper.output_dim ({encoder_network.output_dim}) must equal "
        f"encoder(x).shape[-1] ({enc_out.shape[-1]})"
    )


if __name__ == "__main__":
    test_baseline_forward_output_shape()
    print("PASS: baseline forward output shape [batch, 10] and output_dim verified.")

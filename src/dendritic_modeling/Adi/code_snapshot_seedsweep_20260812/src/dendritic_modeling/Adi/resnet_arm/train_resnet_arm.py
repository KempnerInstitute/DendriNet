#!/usr/bin/env python
"""Entrypoint for the Adi ResNet-50 arm.

Substitutes `setup_utils.initialize_model` in-process, then hands off to the stock
`train_experiments.main()`. Nothing on disk is modified; the patch is process-local.
Everything downstream of model construction — data loading, the 0.8 split, DDP samplers,
param groups, optimizer, checkpointing, performance/final.json, final_model.pt — is
inherited from the AlexNet runs unchanged, so the training protocol is identical by reuse
rather than by re-implementation.

Usage (matches the repo's DDP convention):

    PYTHONPATH=src python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
        src/dendritic_modeling/Adi/resnet_arm/train_resnet_arm.py <config.yaml>
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Derive the src root from this file rather than hardcoding it, so the script still
# works when run from a frozen code snapshot (src/dendritic_modeling/Adi/resnet_arm
# -> up three levels = src). A hardcoded absolute path would silently pull the LIVE
# tree back in and defeat the snapshot.
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)   # for resnet_encoder / build_model
sys.path.insert(0, _SRC)

import dendritic_modeling.scripts.script_utils.setup_utils as setup_utils  # noqa: E402
from build_model import build_resnet_model  # noqa: E402
from resnet_encoder import FrozenResNet50Encoder  # noqa: E402


def _patch_initialize_model() -> None:
    """Point the single model-construction call at our builder, and prove it took."""
    original = setup_utils.initialize_model

    def _guarded(model_config, *args, **kwargs):
        model, encoder = build_resnet_model(model_config)

        # Hard failure rather than a silently wrong model. If this patch ever stops
        # applying, the run would fall through to pretrained.py, whose BackboneSegment
        # does not keep BatchNorm in eval -- the "frozen" ResNet would drift for all
        # 50 epochs with nothing visible in the loss. See resnet_encoder.py.
        enc = model.encoder_network
        if not isinstance(enc, FrozenResNet50Encoder):
            raise RuntimeError(
                f"ResNet arm: encoder is {type(enc).__name__}, expected "
                f"FrozenResNet50Encoder. The initialize_model patch did not apply, so "
                f"BatchNorm running stats would be mutated during training. Refusing to run."
            )
        if enc.training:
            raise RuntimeError(
                "ResNet arm: encoder is in train mode at construction; BatchNorm would "
                "update running stats. Refusing to run."
            )
        return model, encoder

    _guarded.__wrapped__ = original
    setup_utils.initialize_model = _guarded


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ResNet-50 arm.")
    parser.add_argument("config", help="Path to the experiment YAML.")
    # These must be forwarded to train_experiments.main(), which only applies
    # overrides when it receives an `args` object. Passing config_path alone
    # silently ignores --seed, which would make every seed run identical.
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--learning_strategy", default=None)
    args, _unknown = parser.parse_known_args()

    _patch_initialize_model()

    from dendritic_modeling.scripts.training.train_experiments import main as train_main

    train_main(config_path=args.config, args=args)


if __name__ == "__main__":
    main()

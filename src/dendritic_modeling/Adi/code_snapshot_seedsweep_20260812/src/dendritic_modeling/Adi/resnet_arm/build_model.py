"""Model assembly for the Adi ResNet-50 arm.

Substitutes for `setup_utils.initialize_model` so the rest of the training path
(data loading, DDP, param groups, optimizer, checkpointing, performance/final.json)
is inherited unchanged from the AlexNet runs.

    frozen ResNet-50 trunk (2048)  ->  core (N_E=2048)  ->  pretrained fc (2048->1000)

Nothing in the shared codebase is modified; everything here is imported.
"""

from __future__ import annotations

import logging

from dendritic_modeling.models import Classifier
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.script_utils.setup_utils import _config_to_plain_dict

from resnet_encoder import RESNET50_OUTPUT_DIM, build_frozen_resnet50

logger = logging.getLogger(__name__)


def build_resnet_model(model_config):
    """Drop-in replacement for `initialize_model`. Returns (model, encoder_network).

    Reads `model_config.core` for the core type and parameters; the backbone weight
    enum name comes from `model_config.pretrained_replacement.weights`.
    """
    weights = getattr(
        getattr(model_config, "pretrained_replacement", None), "weights", "IMAGENET1K_V1"
    )
    encoder, decoder = build_frozen_resnet50(weights=weights)

    core_params = _config_to_plain_dict(model_config.core)
    core_type = str(core_params.pop("type", "einet"))

    core = get_architecture(core_type, core_params, input_dim=RESNET50_OUTPUT_DIM)

    core_out = getattr(core, "output_dim", None)
    if core_out != decoder.in_features:
        raise ValueError(
            f"core output_dim {core_out} != decoder in_features {decoder.in_features}. "
            f"The ResNet arm uses ResNet's pretrained fc (2048->1000) as the decoder, so "
            f"the core must emit 2048. Set excitatory_layer_sizes: [2048]."
        )

    model = Classifier(encoder, core, decoder)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_core = sum(p.numel() for p in core.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    logger.info(
        "ResNet arm model: encoder=FrozenResNet50Encoder(%s params, frozen) "
        "core=%s(%s params, out=%s) decoder=Linear(%s->%s, %s params, trainable)",
        f"{n_enc:,}", type(core).__name__, f"{n_core:,}", core_out,
        decoder.in_features, decoder.out_features, f"{n_dec:,}",
    )
    return model, encoder

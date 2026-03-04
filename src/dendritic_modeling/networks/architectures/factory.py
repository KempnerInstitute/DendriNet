from typing import Any, Optional, Union

import torch.nn as nn
from omegaconf import DictConfig

from dendritic_modeling.networks.architectures.classical.autoencoder import (
    CNNAutoencoder,
    CNNVariationalAutoencoder,
    MLPAutoencoder,
    MLPVariationalAutoencoder,
)
from dendritic_modeling.networks.architectures.classical.cnn import (
    AlexNet,
    CNNDownsample,
    CNNUpsample,
)
from dendritic_modeling.networks.architectures.classical.identity import Identity
from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP,
    EffectiveMatchedParamMLP,
    MatchedParamMLP,
)
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ConfigurableEINetwork,
)


def get_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int] = None,
) -> nn.Module:
    """
    Factory method for creating network architectures.

    Args:
        type: Architecture type name
        parameters: Either flat parameters dict or structured config
        input_dim: Optional input dimension (needed for EINet from config)

    Returns:
        Initialized network module
    """
    type = type.lower()

    # Handle EINet - requires structured config
    if type == "einet":
        if not (
            isinstance(parameters, DictConfig)
            or (
                isinstance(parameters, dict)
                and "architecture" in parameters
                and "connectivity" in parameters
            )
        ):
            raise ValueError(
                "EINet now requires structured config with 'architecture', 'connectivity', etc. "
                "sections. Flat parameters are no longer supported."
            )
        if input_dim is None:
            raise ValueError(
                "input_dim required when passing structured config to EINet"
            )
        return ConfigurableEINetwork(parameters, input_dim)

    if type == "alexnet_einet_core":
        from dendritic_modeling.Adi.alexnet_einet_core import AlexNetEINetCore
        if not isinstance(parameters, DictConfig):
            raise ValueError("alexnet_einet_core requires structured DictConfig")
        if input_dim is None:
            raise ValueError("input_dim required for alexnet_einet_core")
        return AlexNetEINetCore(parameters, input_dim)

    # All other architectures expect flat parameters
    if isinstance(parameters, DictConfig):
        raise ValueError(
            f"Architecture {type} expects flat parameters, not structured config"
        )

    if type == "alexnet":
        return AlexNet(**parameters)
    elif type in ["cnn", "cnn_down", "cnndownsample", "cnn_downsample"]:
        return CNNDownsample(**parameters)
    elif type in ["cnn_up", "cnnupsample", "cnn_upsample"]:
        return CNNUpsample(**parameters)
    elif type == "mlp":
        return MLP(**parameters)
    elif type == "matchedparammlp":
        return MatchedParamMLP(**parameters)
    elif type == "effectivematchedparammlp":
        return EffectiveMatchedParamMLP(**parameters)
    elif type == "mlpautoencoder":
        return MLPAutoencoder(**parameters)
    elif type == "cnnautoencoder":
        return CNNAutoencoder(**parameters)
    elif type == "mlpvariationalautoencoder":
        return MLPVariationalAutoencoder(**parameters)
    elif type == "cnnvariationalautoencoder":
        return CNNVariationalAutoencoder(**parameters)
    elif type in ["identity", "none", None]:
        return Identity(**parameters)
    elif type == "alexnet_cifar10_baseline":
        from dendritic_modeling.Adi.vision_encoder import get_baseline_encoder
        return get_baseline_encoder(parameters)
    elif type == "alexnet_cifar10_adi":
        from dendritic_modeling.Adi.vision_encoder import get_hybrid_encoder
        return get_hybrid_encoder(parameters)
    elif type == "alexnet_fc":
        from dendritic_modeling.Adi.alexnet_fc_encoder import AlexNetFCEncoder
        return AlexNetFCEncoder(**parameters)
    elif type == "alexnet_fc8_decoder":
        from dendritic_modeling.Adi.alexnet_output_head import AlexNetFC8Decoder
        return AlexNetFC8Decoder(**parameters)
    else:
        raise ValueError(f"Invalid architecture type: {type}")
